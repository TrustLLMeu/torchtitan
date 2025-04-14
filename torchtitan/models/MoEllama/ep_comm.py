from einops import rearrange
import torch


def dispatch_tokens(x, ep_size, ep_mesh, tokens_per_expert):
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
        dim = x.shape[1]
        num_experts = tokens_per_expert.shape[0]
        max_tokens = torch.max(tokens_per_expert)

        output = torch.zeros(
            (num_experts, max_tokens, dim),
            device=x.device,
            dtype=x.dtype,
        )

        start = 0
        for i in range(len(tokens_per_expert)):
            n = tokens_per_expert[i]
            if n > 0:
                output[i, :n] = x[start : start + n]
                start += n
        return output, tokens_per_expert.view(1, -1)  # shape: [1, num_experts]
    # Synchronize tokens_per_expert across all processes

    tokens_per_expert_list = [
        torch.zeros_like(tokens_per_expert) for _ in range(ep_size)
    ]
    torch.distributed.all_gather(
        tokens_per_expert_list, tokens_per_expert, group=ep_mesh.get_group()
    )

    # Combine into a single tensor -> [ep_size, num_experts]
    all_tokens_per_expert = torch.stack(tokens_per_expert_list)

    # Compute offsets for each expert's tokens in the flattened tensor
    token_offsets = torch.zeros_like(tokens_per_expert)
    token_offsets[1:] = torch.cumsum(tokens_per_expert[:-1], dim=0)

    # Find maximum tokens per expert across all ranks for padding
    max_tokens = torch.stack([tpe.max() for tpe in tokens_per_expert_list]).max()

    # Reshape input according to variable tokens per expert
    expert_tensors = []

    for i in range(total_experts):
        # Extract tokens for this expert
        start_idx = token_offsets[i]
        expert_tokens = x[start_idx : start_idx + tokens_per_expert[i]]

        # Pad to max_tokens if necessary
        if expert_tokens.shape[0] < max_tokens:
            padding = torch.zeros(
                (max_tokens - expert_tokens.shape[0], dim_in),
                dtype=x.dtype,
                device=x.device,
            )

            expert_tokens = torch.cat([expert_tokens, padding], dim=0)

        expert_tensors.append(expert_tokens.unsqueeze(0))  # Add expert dimension

    # Concatenate into single tensor of shape [num_experts, max_tokens, dim_in]
    x_reshaped = torch.cat(expert_tensors, dim=0)

    # Rearrange for all-to-all
    x_for_a2a = rearrange(
        x_reshaped, "(ep er) t d -> ep (er t) d", ep=ep_size, er=expert_per_rank
    ).contiguous()

    output = torch.distributed.nn.functional.all_to_all_single(
        torch.empty_like(x_for_a2a), x_for_a2a, group=ep_mesh.get_group()
    )

    # Rearrange output to the correct final shape
    # Each rank now gets expert_per_rank experts, each with tokens from all ep_size ranks
    output = rearrange(
        output, "ep (er t) d -> er (ep t) d", er=expert_per_rank, t=max_tokens
    )

    # Return both the output tensor and the token distribution information
    return output, all_tokens_per_expert


def combine_tokens(x, ep_size, ep_mesh, all_tokens_per_expert):
    """
    Combine tokens from different ranks back to their original experts.
    This is the inverse operation of dispatch_tokens.

    Args:
        x: Tensor of shape (expert_per_rank, tokens_per_expert * ep_size, dim_in)
        ep_size: int, number of EP ranks (world size)
        ep_mesh: torch.distributed process group for EP parallel
        all_tokens_per_expert: tensor of shape [ep_size, num_experts],
                              containing number of tokens per expert across all ranks

    Returns:
        output: Tensor of shape (num_experts, tokens_per_expert, dim_in) - flattened tokens
    """
    expert_per_rank, tokens_per_all_ranks, dim_in = x.shape
    total_experts = expert_per_rank * ep_size
    max_tokens_per_expert = tokens_per_all_ranks // ep_size

    if ep_size == 1:
        tokens_per_expert = all_tokens_per_expert[0]  # [num_experts]
        output_chunks = []
        for i in range(x.shape[0]):
            n = tokens_per_expert[i].item()
            if n > 0:
                output_chunks.append(x[i, :n])
        return torch.cat(output_chunks, dim=0)  # shape: [total_routed_tokens, dim]
    # Rearrange to prepare for all-to-all communication
    x_for_a2a = rearrange(
        x, "er (ep t) d -> ep (er t) d", ep=ep_size, t=max_tokens_per_expert
    ).contiguous()

    # Perform all-to-all communication
    output_a2a = torch.distributed.nn.functional.all_to_all_single(
        torch.empty_like(x_for_a2a), x_for_a2a, group=ep_mesh.get_group()
    )

    # Rearrange to get experts back in their original form
    # (num_experts, max_tokens_per_expert, dim_in)
    experts_output = rearrange(
        output_a2a, "ep (er t) d -> (ep er) t d", er=expert_per_rank
    )

    # Get the actual number of tokens for each expert in this rank
    # We use the slice corresponding to the current rank
    rank = torch.distributed.get_rank(group=ep_mesh.get_group())
    tokens_per_expert = all_tokens_per_expert[rank]

    # Calculate total tokens to allocate the output tensor
    total_tokens = tokens_per_expert.sum().item()

    output_chunks = []
    for i in range(len(tokens_per_expert)):
        num_tokens = tokens_per_expert[i].item()
        if num_tokens > 0:
            output_chunks.append(experts_output[i, :num_tokens])

    flattened_output = torch.cat(output_chunks, dim=0)

    return flattened_output
