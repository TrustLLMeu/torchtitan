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
        # For single expert case, use rearrange to maintain consistent ordering
        return rearrange(x, "(er t) d -> er t d", er=expert_per_rank), tokens_per_expert

    # Synchronize tokens_per_expert across all processes

    tokens_per_expert_list = [
        torch.zeros_like(tokens_per_expert) for _ in range(ep_size)
    ]
    torch.distributed.all_gather(
        tokens_per_expert_list, tokens_per_expert, group=ep_mesh.get_group()
    )

    # Combine into a single tensor
    all_tokens_per_expert = torch.stack(
        tokens_per_expert_list
    )  # Shape: [ep_size, num_experts]

    # Compute offsets for each expert's tokens in the flattened tensor
    token_offsets = torch.zeros_like(tokens_per_expert)
    token_offsets[1:] = torch.cumsum(tokens_per_expert[:-1], dim=0)

    # Find maximum tokens per expert across all ranks for padding
    max_tokens = max([tpe.max().item() for tpe in tokens_per_expert_list])

    # Reshape input according to variable tokens per expert
    expert_tensors = []

    for i in range(total_experts):
        # Extract tokens for this expert
        start_idx = token_offsets[i]
        expert_tokens = x[start_idx:start_idx + tokens_per_expert[i]]

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

    # Prepare output tensor
    output = torch.empty_like(x_for_a2a)

    # Perform all-to-all communication
    torch.distributed.nn.functional.all_to_all_single(
        output, x_for_a2a, group=ep_mesh.get_group()
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
        output: Tensor of shape (num_experts*tokens_per_expert, dim_in) - flattened tokens
    """
    expert_per_rank, tokens_per_all_ranks, dim_in = x.shape
    total_experts = expert_per_rank * ep_size
    max_tokens_per_expert = tokens_per_all_ranks // ep_size

    if ep_size == 1:
        return rearrange(x, "er t d -> (er t) d")

    # Rearrange to prepare for all-to-all communication
    x_for_a2a = rearrange(
        x, "er (ep t) d -> ep (er t) d", ep=ep_size, t=max_tokens_per_expert
    ).contiguous()

    # Prepare output tensor for all-to-all
    output_a2a = torch.empty_like(x_for_a2a)

    # Perform all-to-all communication
    torch.distributed.nn.functional.all_to_all_single(
        output_a2a, x_for_a2a, group=ep_mesh.get_group()
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

    # Create the final flattened output tensor
    flattened_output = torch.zeros(
        (total_tokens, dim_in), dtype=experts_output.dtype, device=experts_output.device
    )

    # Copy tokens to the final output, removing padding
    current_idx = 0
    for i in range(len(tokens_per_expert)):
        num_tokens = tokens_per_expert[i].item()
        if num_tokens > 0:
            flattened_output[current_idx:current_idx + num_tokens] = experts_output[
                i, :num_tokens
            ]
            current_idx += num_tokens

    return flattened_output
