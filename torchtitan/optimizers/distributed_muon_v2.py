import math

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from .muon_utils import zeropower_via_newtonschulz5
from .distributed_muon import DistributedMuon
from torchtitan.tools.logging import logger
import os
from functools import partial


class DistributedMuonV2(DistributedMuon):
    def __init__(
        self,
        full_params,
        model,
        compute_streams=1,
        **kwargs,
    ):
        super().__init__(full_params, model, **kwargs)
        self.compute_streams = compute_streams

        # also try to read the env variable
        ENV_compute_streams = os.environ.get("COMPUTE_STREAMS", None)
        if ENV_compute_streams is not None:
            self.compute_streams = int(ENV_compute_streams)
        logger.info(f"DistributedMuonV2 Using {self.compute_streams} compute streams")

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

        num_streams = self.compute_streams
        current_stream = torch.cuda.current_stream()

        # Create additional streams as needed
        additional_streams = [torch.cuda.Stream() for _ in range(num_streams - 1)]

        # All streams to use
        comp_streams = [current_stream] + additional_streams

        # Make sure the streams wait for backward to complete
        default_stream = torch.cuda.current_stream()
        for stream in comp_streams:
            stream.wait_stream(default_stream)

        # Keep track of pending operations
        # Track pending work
        pending_ops = []

        for bucket_idx in range(total_buckets):
            comp_stream = comp_streams[bucket_idx % num_streams]

            with torch.cuda.stream(comp_stream):
                start_idx = bucket_idx * bucket_size
                end_idx = min(start_idx + bucket_size, len(params))
                current_rank_idx = start_idx + rank

                # Compute u using Newton-Schulz in the computation stream
                if current_rank_idx < len(params):
                    p = params[current_rank_idx]
                    g = self.update_muon_momentum_and_get_gradient(
                        p, nesterov, momentum
                    )

                else:
                    g = zero_tensor(params[end_idx - 1].shape)
                u = zeropower_via_newtonschulz5(g, steps=ns_steps)

                # Prepare gather lists
                gather_lists = [None] * world_size
                for i in range(world_size):
                    param_idx = start_idx + i
                    if i == rank or param_idx >= len(params):
                        gather_lists[i] = u.to(dtype=cast_dtype)
                    elif param_idx < len(params):
                        gather_lists[i] = zero_tensor(params[param_idx].shape)

                # Use non-blocking all_gather
                future = dist.all_gather(gather_lists, u, async_op=True)

                # Record the operation for later completion
                pending_ops.append(
                    (
                        comp_stream,
                        future,
                        params,
                        gather_lists,
                        start_idx,
                        end_idx,
                        lr,
                        wd,
                    )
                )

            # Process completed operations
            completed = []
            for i, (stream, future, p, gl, si, ei, l, w) in enumerate(pending_ops):
                if future.is_completed():
                    with torch.cuda.stream(stream):
                        self.update_bucket_params(p, gl, si, ei, l, w)
                    completed.append(i)

            # Remove processed operations
            for i in sorted(completed, reverse=True):
                pending_ops.pop(i)

        # Wait for any remaining operations
        for stream, future, p, gl, si, ei, l, w in pending_ops:
            future.wait()
            with torch.cuda.stream(stream):
                self.update_bucket_params(p, gl, si, ei, l, w)

        # Final synchronization
        for stream in comp_streams:
            stream.synchronize()

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

        num_streams = self.compute_streams
        current_stream = torch.cuda.current_stream()

        # Create additional streams as needed
        additional_streams = [torch.cuda.Stream() for _ in range(num_streams - 1)]

        # All streams to use
        comp_streams = [current_stream] + additional_streams

        # Make sure all streams wait for backward to complete
        default_stream = torch.cuda.current_stream()
        for stream in comp_streams:
            stream.wait_stream(default_stream)

        # Process each bucket in its own stream
        for bucket_idx in range(total_buckets):
            comp_stream = comp_streams[bucket_idx % num_streams]

            # Process the entire bucket in a single stream
            with torch.cuda.stream(comp_stream):
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

                # Step 2: Prepare receive buffers for first all_to_all
                recv_shapes = []
                for rank_idx in range(world_size):
                    recv_shapes.append(
                        self.calculate_shard_shape(target_shape, rank_idx, world_size)
                    )
                recv_list = [zero_tensor(shape) for shape in recv_shapes]

                # Step 3: First all_to_all - using ASYNC version
                # This will happen on the current compute stream
                first_a2a = dist.all_to_all(recv_list, send_list, async_op=True)

                # Step 4: Wait for the first all_to_all to complete
                first_a2a.wait()

                # Step 5: Concatenate received gradients along dimension 0 and perform NS5
                # All tensors in recv_list should have the same dimensions except for dim 0
                try:
                    full_g = torch.cat(recv_list, dim=0)
                    u = zeropower_via_newtonschulz5(full_g, steps=ns_steps)
                except RuntimeError as e:
                    # Print debug info for troubleshooting
                    shapes_info = [t.shape for t in recv_list]
                    logger.error(f"Bucket {bucket_idx} received shapes: {shapes_info}")
                    raise e

                # Step 6: Split the processed tensor back for second all_to_all
                split_sizes = [shape[0] for shape in recv_shapes]
                split_u = torch.split(u, split_sizes, dim=0)

                # Step 7: Prepare for second all_to_all
                second_send_list = list(split_u)  # Convert split result to list
                second_recv_list = [zero_tensor(shape) for shape in send_shapes]

                # Step 8: Second all_to_all - using ASYNC version
                second_a2a = dist.all_to_all(
                    second_recv_list, second_send_list, async_op=True
                )

                # Step 9: Wait for the second all_to_all to complete
                second_a2a.wait()

                # Step 10: Update parameters using the results
                self.update_bucket_params(
                    params, second_recv_list, start_idx, end_idx, lr, wd
                )

        # Final synchronization of all streams
        for stream in comp_streams:
            if stream != current_stream:
                stream.synchronize()

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

            logger.info(f"total buckets: {total_buckets} & blocks: {len(params)}")

            # we need to adjust the num_streams to make sure stream is not idle
            self.compute_streams = max(self.compute_streams, total_buckets)

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
