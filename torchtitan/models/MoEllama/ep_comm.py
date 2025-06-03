from einops import rearrange
import torch


def dispatch_tokens(x, ep_mesh, ep_size, tokens_per_expert, force_float32=False):
    """
    Dispatch tokens to corresponding experts using all_to_all_single.

    Args:
        x: Tensor of shape (num_experts*tokens_per_expert, dim_in) - flattened tokens
        ep_size: int, number of EP ranks (world size)
        ep_mesh: torch.distributed process group for EP parallel
        tokens_per_expert: tensor of shape [num_experts], number of tokens per expert

    Returns:
        output: Tensor of shape (expert_per_rank, tokens_per_expert * ep_size, dim_in)
              where expert_per_rank is the number of experts on each rank
        all_tokens_per_expert: tensor of shape [ep_size, num_experts],
                              containing number of tokens per expert across all ranks
    """
    dim_in = x.shape[1]
    total_experts = len(tokens_per_expert)
    expert_per_rank = total_experts // ep_size

    if ep_size == 1:
        # Single‐rank case: just reshape and pad once.
        num_experts = tokens_per_expert.shape[0]
        max_tokens = torch.max(tokens_per_expert)

        output = torch.zeros(
            (num_experts, max_tokens, dim_in),
            device=x.device,
            dtype=x.dtype,
        )

        # Compute cumulative offsets to know where each expert's tokens begin in x
        token_offsets = torch.zeros_like(tokens_per_expert)
        token_offsets[1:] = torch.cumsum(tokens_per_expert[:-1], dim=0)

        # Build two index arrays:
        # 1. expert_ids: repeated expert index for each routed token
        expert_ids = torch.arange(total_experts, device=x.device).repeat_interleave(
            tokens_per_expert
        )

        # 2. pos_in_expert: for each token in sorted order, its row index within that expert's tree
        token_offsets = torch.zeros_like(tokens_per_expert)
        token_offsets[1:] = torch.cumsum(tokens_per_expert[:-1], dim=0)
        expert_offsets = token_offsets.repeat_interleave(tokens_per_expert)
        pos_in_expert = torch.arange(x.shape[0], device=x.device) - expert_offsets

        # Now scatter sorted tokens (x is already sorted externally before calling this)
        sorted_tokens = x  # assume x is already in the order "by expert group"
        output[expert_ids, pos_in_expert] = sorted_tokens
        return output, tokens_per_expert.view(1, -1)  # shape: [1, num_experts]

    # Synchronize tokens_per_expert across all processes
    ep_group = ep_mesh.get_group()

    tokens_per_expert_list = [
        torch.zeros_like(tokens_per_expert) for _ in range(ep_size)
    ]
    torch.distributed.all_gather(
        tokens_per_expert_list, tokens_per_expert, group=ep_group
    )
    all_tokens_per_expert = torch.stack(tokens_per_expert_list)

    # Compute offsets for each expert's tokens in the flattened tensor
    token_offsets = torch.zeros_like(tokens_per_expert)
    token_offsets[1:] = torch.cumsum(tokens_per_expert[:-1], dim=0)

    # Find maximum tokens per expert across all ranks for padding
    max_tokens = torch.stack([tpe.max() for tpe in tokens_per_expert_list]).max()

    x_reshaped = torch.zeros(
        (total_experts, max_tokens, dim_in), device=x.device, dtype=x.dtype
    )
    # Reshape input according to variable tokens per expert
    # Build expert_ids & pos_in_expert exactly as above:
    expert_ids = torch.arange(total_experts, device=x.device).repeat_interleave(
        tokens_per_expert
    )
    expert_offsets = token_offsets.repeat_interleave(tokens_per_expert)
    pos_in_expert = torch.arange(x.shape[0], device=x.device) - expert_offsets

    # Now scatter sorted tokens into a single big zero‐tensor
    x_reshaped[expert_ids, pos_in_expert] = (
        x  # assume x is already sorted by expert key before this function is called
    )

    # Reshape for all_to_all: from [total_experts, max_tokens, dim_in] to [ep_size, expert_per_rank * max_tokens, dim_in]
    x_for_a2a = x_reshaped.view(
        ep_size, expert_per_rank * max_tokens, dim_in
    ).contiguous()

    # Possibly cast to float32 if requested
    original_dtype = x_for_a2a.dtype
    if force_float32:
        x_for_a2a = x_for_a2a.to(torch.float32)

    # One all‐to‐all swap
    buffer = torch.empty_like(x_for_a2a, dtype=x_for_a2a.dtype)
    output = torch.distributed.nn.functional.all_to_all_single(
        buffer, x_for_a2a, group=ep_group
    )
    output = output.to(original_dtype)

    # Each rank now has [expert_per_rank, ep_size * max_tokens, dim_in]
    output = output.view(expert_per_rank, ep_size * max_tokens, dim_in)
    torch.distributed.barrier(ep_group)
    return output, all_tokens_per_expert


def combine_tokens(
    x: torch.Tensor,
    ep_mesh,
    ep_size: int,
    all_tokens_per_expert: torch.Tensor,
    force_float32: bool = False,
):
    """
    Combine tokens from different ranks back into a single flattened tensor.
    This is the inverse of dispatch_tokens. Inputs:

      - x: a tensor of shape (experts_per_rank, ep_size * max_tokens, dim_in)
      - all_tokens_per_expert: a tensor of shape (ep_size, total_experts)
      - ep_size: number of ranks
      - ep_mesh: the process group

    Returns:
      - a single tensor of shape (sum_of_all_tokens, dim_in), where tokens are
        concatenated in expert order.

    Steps:
      1) Perform an all-to-all to gather each expert’s padded bucket back onto one rank.
      2) Remove zero-padding from each expert’s bucket by using the token counts.
      3) Concatenate all unpadded buckets in the natural expert ordering.
    """
    # If only one rank, there’s no need for all-to-all; just unpad in place.
    if ep_size == 1:
        experts_per_rank, max_tokens, dim_in = x.shape
        tokens_per_expert = all_tokens_per_expert[0]
        total_experts = experts_per_rank

        valid_ranges = torch.arange(max_tokens, device=x.device).unsqueeze(0)
        limits = tokens_per_expert.unsqueeze(1)
        mask = valid_ranges < limits

        flat_x = x.reshape(-1, dim_in)

        expert_indices = torch.arange(total_experts, device=x.device).unsqueeze(1)
        per_expert_base = expert_indices * max_tokens
        offsets = torch.arange(max_tokens, device=x.device).unsqueeze(0)
        row_indices = per_expert_base + offsets

        flat_row_mask = mask.view(-1)
        selected_rows = row_indices.view(-1)[flat_row_mask]
        return flat_x.index_select(0, selected_rows)

    ep_group = ep_mesh.get_group()
    experts_per_rank, total_padded, dim = x.shape

    our_rank = torch.distributed.get_rank(group=ep_group)

    experts_per_rank, total_padded, dim = x.shape
    assert (
        total_padded % ep_size == 0
    ), f"total_padded ({total_padded}) is not divisible by ep_size ({ep_size})"
    L = total_padded // ep_size  # = “max_tokens per expert, on each rank”
    total_experts = experts_per_rank * ep_size
    assert total_experts == all_tokens_per_expert.size(1), (
        f"Expected all_tokens_per_expert.size(1) == {total_experts}, but got "
        f"{all_tokens_per_expert.size(1)}"
    )

    # (1) Reshape to do the inverse all_to_all
    #     x_for_a2a shape = (ep_size, experts_per_rank * L, dim)
    x_for_a2a = x.view(ep_size, experts_per_rank * L, dim).contiguous()
    orig_dtype = x_for_a2a.dtype
    if force_float32:
        x_for_a2a = x_for_a2a.to(torch.float32)

    # (2) Inverse all-to-all
    buffer = torch.empty_like(x_for_a2a, dtype=x_for_a2a.dtype)
    gathered = torch.distributed.nn.functional.all_to_all_single(
        buffer, x_for_a2a, group=ep_group
    )
    gathered = gathered.to(orig_dtype)  # back to original dtype

    # (3) Flatten in the order (expert_sub, token_pos, src_rank):
    #     gathered shape = (ep_size, experts_per_rank * L, dim)
    #     permute → (experts_per_rank * L, ep_size, dim) → view(-1, dim)
    out_flat = gathered.permute(1, 0, 2).contiguous().view(-1, dim)
    #   Expected out_flat.rows = experts_per_rank * L * ep_size = total_experts * L
    assert out_flat.shape[0] == total_experts * L, (
        f"After gather, out_flat has {out_flat.shape[0]} rows, but expected "
        f"{total_experts} * {L} = {total_experts * L}"
    )

    idx_list = []
    step = ep_size  # stride of src-rank column
    block = L * ep_size  # rows per local expert

    for g in range(total_experts):  # 0 … ep_size*experts_per_rank–1
        src = g // experts_per_rank  # host rank of this expert
        sub = g % experts_per_rank  # local index within its host
        n_here = all_tokens_per_expert[our_rank, g].item()
        if n_here == 0:
            continue

        base = sub * block + src  # first valid row for this pair
        rows = torch.arange(n_here, device=x.device, dtype=torch.long) * step + base
        idx_list.append(rows)

    idx = torch.cat(idx_list)  # (B·S·top_k) == 16 384 long
    correct_flat = out_flat.index_select(0, idx)
    return correct_flat
