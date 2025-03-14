import math

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from .muon_utils import zeropower_via_newtonschulz5
from .muon import Muon
from functools import partial


class DistributedMuon(Muon):
    def __init__(
        self,
        full_params,
        model,
        **kwargs,
    ):
        super().__init__(full_params, model, **kwargs)

    def update_bucket_params(self, params, updates, start_idx, end_idx, lr, wd):
        for idx_in_bucket in range(start_idx, end_idx):
            shift = idx_in_bucket - start_idx
            p = params[idx_in_bucket]
            u = updates[shift]

            with torch.no_grad():
                if isinstance(p, DTensor) and not isinstance(u, DTensor):
                    p.data.to_local().mul_(1 - lr * wd).add_(u, alpha=-lr)
                else:
                    p.data.mul_(1 - lr * wd).add_(u, alpha=-lr)

    def calculate_shard_shape(self, shape, source_rank, world_size):
        full_dim0 = shape[0]
        base = (full_dim0 + world_size - 1) // world_size  # ceil division
        remainder = full_dim0 - base * (world_size - 1)
        if source_rank < world_size - 1:
            dim0 = base
        else:
            dim0 = remainder
        return (dim0, *shape[1:])

    def step_ddp(
        self,
        params,
        lr,
        wd,
        momentum,
        nesterov,
        ns_steps,
        bucket_size,
        total_buckets,
        world_size,
        rank,
    ):
        device = params[0].device
        cast_dtype = self.communication_dtype
        zero_tensor = partial(torch.zeros, dtype=cast_dtype, device=device)
        for bucket_idx in range(total_buckets):
            start_idx = bucket_idx * bucket_size
            end_idx = min(start_idx + bucket_size, len(params))
            current_rank_idx = start_idx + rank
            if current_rank_idx < len(params):
                p = params[current_rank_idx]
                # Step 1: Get the gradient
                g = self.update_muon_momentum_and_get_gradient(p, nesterov, momentum)

            else:
                """
                To avoid idle stream, we can randomly generate on last ranks
                """
                g = zero_tensor(params[end_idx - 1].shape)

            u = zeropower_via_newtonschulz5(g, steps=ns_steps)

            # Step 3: FOR DDP, we do all-gather
            gather_lists = [None] * world_size

            for i in range(world_size):
                param_idx = start_idx + i
                if i == rank or param_idx >= len(params):
                    gather_lists[i] = u.to(dtype=cast_dtype)
                elif param_idx < len(params):
                    p = params[start_idx + i]
                    gather_lists[i] = zero_tensor(p.shape)

            dist.all_gather(gather_lists, u)
            # Step 4: Update the parameters
            self.update_bucket_params(params, gather_lists, start_idx, end_idx, lr, wd)

    def step_fsdp(
        self,
        params,
        lr,
        wd,
        momentum,
        nesterov,
        ns_steps,
        bucket_size,
        total_buckets,
        world_size,
        rank,
    ):
        device = params[0].device
        cast_dtype = self.communication_dtype
        zero_tensor = partial(torch.zeros, dtype=cast_dtype, device=device)

        # Process each bucket
        for bucket_idx in range(total_buckets):
            start_idx = bucket_idx * bucket_size
            end_idx = min(start_idx + bucket_size, len(params))

            # Step 1: Prepare data for first all_to_all
            send_list = []
            send_shapes = []
            target_shape = None

            for rank_idx in range(world_size):
                current_rank_idx = start_idx + rank_idx

                if current_rank_idx < len(params):
                    p = params[current_rank_idx]
                    g = (
                        self.update_muon_momentum_and_get_gradient(
                            p, nesterov, momentum
                        )
                        .to_local()
                        .to(dtype=cast_dtype)
                    )

                    # Save the shape info for this parameter
                    if rank == rank_idx:
                        target_shape = p.shape
                else:
                    # Use a dummy shape for parameters beyond our range
                    p = params[end_idx - 1]
                    g = zero_tensor(p.to_local().shape)

                send_list.append(g)
                send_shapes.append(g.shape)

            # Make sure target_shape is initialized
            if target_shape is None and end_idx > 0:
                target_shape = params[start_idx].shape

            recv_shapes = [
                self.calculate_shard_shape(target_shape, rank_idx, world_size)
                for rank_idx in range(world_size)
            ]
            recv_list = [zero_tensor(shape) for shape in recv_shapes]

            # Step 3: First all_to_all - using ASYNC version
            dist.all_to_all(recv_list, send_list)

            # Step 5: Concatenate received gradients along dimension 0 and perform NS5
            # All tensors in recv_list should have the same dimensions except for dim 0

            full_g = torch.cat(recv_list, dim=0)
            u = zeropower_via_newtonschulz5(full_g, steps=ns_steps)

            # Step 6: Split the processed tensor back for second all_to_all
            split_sizes = [shape[0] for shape in recv_shapes]

            send_list = list(torch.split(u, split_sizes, dim=0))
            recv_list = [zero_tensor(shape) for shape in send_shapes]

            # Step 8: Second all_to_all - using ASYNC version
            dist.all_to_all(recv_list, send_list)
            del send_list
            # Step 10: Update parameters using the results
            self.update_bucket_params(params, recv_list, start_idx, end_idx, lr, wd)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.update_adamw()
        ############################################################
        #       Muon update for parameters with use_muon=True       #
        ############################################################

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        for group in self.param_groups:
            params = [
                p
                for p in group["params"]
                if self.state[p]["use_muon"] and p.grad is not None
            ]
            # sort params by size
            params.sort(key=lambda x: x.numel(), reverse=True)

            lr, wd = group["lr"], group["wd"]
            momentum, nesterov = group["momentum"], group["nesterov"]
            ns_steps = group["ns_steps"]

            bucket_size = world_size
            # sort params by size

            total_buckets = math.ceil(len(params) / bucket_size)

            muon_step_fn = self.step_fsdp if self.fsdp_enabled else self.step_ddp
            muon_step_fn(
                params,
                lr,
                wd,
                momentum,
                nesterov,
                ns_steps,
                bucket_size,
                total_buckets,
                world_size,
                rank,
            )

        return loss
