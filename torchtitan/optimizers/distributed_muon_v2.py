import math

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


@torch.compile
def fused_weight_decay_update(param, u, lr, wd):
    """Fused weight decay and parameter update."""
    # This could be implemented as a CUDA kernel for better performance
    param.mul_(1 - lr * wd).add_(u, alpha=-lr)


class DTensorHandle:
    def __init__(self, dtensor, mesh, placements):
        self.dtensor = dtensor
        self.mesh = mesh
        self.placements = placements
        # Store the local tensor to avoid extra copies
        self.local_tensor = dtensor.to_local()

    def wait(self):
        # Create a replicated DTensor
        # In a real implementation, this would be an async operation
        # that we're waiting for here
        return DTensor.from_local(
            self.local_tensor, self.mesh, placements=self.placements
        )


class DistributedMuonV2(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        full_params,
        model,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        dp_mesh=None,
        **kwargs,
    ):

        self.fsdp_enabled = dp_mesh is not None
        self.dp_mesh = dp_mesh  # DeviceMesh for DP communication

        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]

        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]
        print(
            f"Muon optimizer is enabled with dp_mesh={dp_mesh} | type={type(dp_mesh)}"
        )

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        # Pre-filter parameters for AdamW update
        self.adamw_params = []  # parameters for which we do not use Muon

        all_params = (
            list(muon_params)
            if muon_params is not None
            else [] + (list(adamw_params) if adamw_params is not None else [])
        )
        super().__init__(all_params, defaults)

        # Set state for muon parameters and adamw parameters separately
        for p in muon_params:
            assert p.ndim == 2, p.ndim
            self.state[p] = {}
            self.state[p]["use_muon"] = True

        if adamw_params is not None:
            for p in adamw_params:
                self.state[p] = {}
                self.state[p]["use_muon"] = False
                self.adamw_params.append(p)

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def gather_full_grad_async(self, g):
        """Asynchronous version of gather_full_grad using DTensor.

        Args:
            g: Gradient tensor (DTensor)

        Returns:
            Handle that can be waited on to get the gathered gradient
        """
        if not self.fsdp_enabled:
            # No sharding, create a simple handle that returns the original gradient
            class SimpleHandle:
                def __init__(self, tensor):
                    self.tensor = tensor

                def wait(self):
                    return self.tensor

            return SimpleHandle(g)

        if not isinstance(g, DTensor):
            raise RuntimeError(
                "Expected gradient to be a DTensor, but got a regular tensor."
            )

        # Ensure g is in the correct mesh
        if g.device_mesh != self.dp_mesh:
            raise RuntimeError(
                f"Gradient tensor's DeviceMesh ({g.device_mesh.mesh_dim_names}) "
                f"does not match optimizer's DeviceMesh ({self.dp_mesh.mesh_dim_names})."
            )

        return DTensorHandle(g, self.dp_mesh, [Replicate()])

    def update_adamw(self):
        ############################
        #       Optimized AdamW    #
        ############################
        for group in self.param_groups:
            params = [
                p
                for p in group["params"]
                if not self.state[p]["use_muon"] and p.grad is not None
            ]

            if not params:
                continue

            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            # Get gradients
            grads = [p.grad for p in params]

            # Initialize state if needed
            steps = []
            moment1s = []
            moment2s = []
            for p in params:
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(p.grad, device=p.device)
                    state["moment2"] = torch.zeros_like(p.grad, device=p.device)

                state["step"] += 1
                steps.append(state["step"])
                moment1s.append(state["moment1"])
                moment2s.append(state["moment2"])

            # Convert steps to tensor for bias correction
            steps_tensor = torch.tensor(
                steps, dtype=torch.float32, device=params[0].device
            )

            # Update momentum buffers using foreach operations
            torch._foreach_mul_(moment1s, beta1)
            torch._foreach_add_(moment1s, grads, alpha=1 - beta1)

            torch._foreach_mul_(moment2s, beta2)
            torch._foreach_add_(moment2s, [g * g for g in grads], alpha=1 - beta2)

            # Compute bias correction
            bias_correction1 = 1 - beta1**steps_tensor
            bias_correction2 = 1 - beta2**steps_tensor
            step_size = lr / (bias_correction1 / torch.sqrt(bias_correction2))

            # Apply weight decay
            torch._foreach_mul_(params, 1 - lr * weight_decay)

            # Update parameters using fused operation
            torch._foreach_addcdiv_(
                params,
                moment1s,
                [m.sqrt().add_(eps) for m in moment2s],
                value=-step_size,
            )

    # @torch.compile
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        """
        So, we have [params] more than world size in general.
        Hence, we distribute the params across the world_size.
        Each rank will have a subset of the params = len(params) // world_size
        """

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            bucket_size = len(params) // world_size
            start_idx = rank * bucket_size
            end_idx = (rank + 1) * bucket_size if rank < world_size - 1 else len(params)

            bucket_params = params[start_idx:end_idx]
            param_refs = []
            handles = []  # Store async handles

            u_chunks_cache = []  # Store computed `u` values
            u_shard_cache = []  # Store received `u` shards

            # generate weight updates in distributed fashion
            for id, p in enumerate(bucket_params):
                # sanity check
                g = p.grad
                if g is None:
                    continue
                assert g is not None

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # g = self.gather_full_grad(g)

                handle = self.gather_full_grad_async(g)
                handles.append(handle)
                param_refs.append(p)

            for i, handle in enumerate(handles):
                replicated_grad = handle.wait()
                p = param_refs[i]

                u = zeropower_via_newtonschulz5(
                    replicated_grad, steps=group["ns_steps"]
                )

                if isinstance(u, DTensor):
                    u_local = u.to_local()
                else:
                    u_local = u

                # Ensure `u_local` is evenly split across GPUs
                split_size = u_local.shape[0] // world_size
                remainder = u_local.shape[0] % world_size
                if remainder > 0:
                    pad = torch.zeros(
                        (world_size - remainder, u_local.shape[1]),
                        device=u_local.device,
                        dtype=u_local.dtype,
                    )
                    u_local = torch.cat([u_local, pad], dim=0)

                # Now safely split into equal-sized chunks
                u_chunks = list(torch.split(u_local, split_size, dim=0))
                u_shards = [torch.zeros_like(u_chunks[0]) for _ in range(world_size)]

                u_chunks_cache.append(u_chunks)
                u_shard_cache.append(u_shards)

            # for u_shards, u_chunks in zip(u_shard_cache, u_chunks_cache):
            #     work = dist.all_to_all(u_shards, u_chunks, async_op=True)
            #     work.wait()  # Synchronize before using u_shards

            for group in range(world_size):
                u_shards = u_shard_cache[group]
                u_chunks = u_chunks_cache[group]
                work = dist.all_to_all(u_shards, u_chunks, async_op=True)
                work.wait()  # Synchronize before using u_shards
                dist.all_to_all(u_shards, u_chunks)
                u_shard_cache[group] = u_shards

            for group, p in enumerate(bucket_params):
                u_shards = u_shard_cache[group]

                u = torch.cat(u_shards, dim=0)
                if isinstance(p.data, DTensor):
                    u = DTensor.from_local(u, p.data.device_mesh, placements=[Shard(0)])

                ############################
                # Scale update from moon-light's implementation
                # adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                if u.shape != p.data.shape:
                    raise RuntimeError(
                        f"Shape mismatch: u.shape={u.shape}, p.data.shape={p.data.shape}"
                    )

                fused_weight_decay_update(p.data, u, lr, wd)

        # For easier debug, we separate the AdamW optimizer

        self.update_adamw()

        return loss
