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
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class DistributedMuon(torch.optim.Optimizer):
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

        muon_params, adamw_params = [], []
        for name, p in model.named_parameters():
            if (
                p.ndim >= 2
                and "tok_embeddings.weight" not in name
                and "output.weight" not in name
            ):
                muon_params.append(p)
            else:
                adamw_params.append(p)

        print(f"Muon with dp_mesh={dp_mesh} | type={type(dp_mesh)}")
        print(f"Muon params: {sum(p.numel() for p in muon_params)}")
        print(f"AdamW params: {sum(p.numel() for p in adamw_params)}")

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def gather_full_grad(self, g):
        """Gathers the full gradient across all distributed processes using DTensor."""
        if not self.fsdp_enabled:
            return g  # No sharding, return the original gradient

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

        # Convert local gradients into a DTensor replicated across data parallel ranks
        replicated_grad = DTensor.from_local(
            g.to_local(), self.dp_mesh, placements=[Replicate()]
        )

        return replicated_grad

    def update_adamw(
        self,
    ):
        ############################
        #       AdamW backup       #
        ############################
        for group in self.param_groups:
            # Get all parameters that don't use muon and have gradients
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

            # Process parameters one by one - simple, reliable approach
            for p in params:
                g = p.grad
                state = self.state[p]

                # Initialize state if needed
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g, device=p.device)
                    state["moment2"] = torch.zeros_like(g, device=p.device)

                state["step"] += 1
                step = state["step"]

                # Get momentum buffers
                buf1 = state["moment1"]
                buf2 = state["moment2"]

                # Update momentum buffers - safer than lerp
                buf1.mul_(beta1).add_(g, alpha=1 - beta1)
                buf2.mul_(beta2).add_(
                    g * g, alpha=1 - beta2
                )  # More reliable than g.square()

                # Compute bias corrections
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr / (bias_correction1 / math.sqrt(bias_correction2))

                # Apply weight decay and update parameters
                with torch.no_grad():
                    p.data.mul_(1 - lr * weight_decay)
                    p.data.addcdiv_(buf1, buf2.sqrt().add_(eps), value=-step_size)

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

            # generate weight updates in distributed fashion
            for idx, p in enumerate(bucket_params):

                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                ############################
                # DISTRIBUTED MUON UPDATE #
                # Step 1: Gather full gradient across ranks

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                g = self.gather_full_grad(g)

                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                ############################
                # DISTRIBUTED MUON UPDATE #
                # Step 2: Extract the correct shard for this rank

                # Convert `u` to local tensor if it’s a DTensor
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
                # Allocate buffers for received `u` shards
                u_shards = [torch.zeros_like(u_chunks[0]) for _ in range(world_size)]
                # Use `all_to_all()` to correctly swap updates between GPUs

                # dist.all_to_all(u_shards, u_chunks)

                work = dist.all_to_all(u_shards, u_chunks, async_op=True)
                work.wait()  # Synchronize before using u_shards
                # dist.reduce_scatter(
                #     u_local, u_chunks, op=dist.ReduceOp.SUM, async_op=True
                # )

                # Reconstruct correct `u` for this rank
                u = torch.cat(u_shards, dim=0)  # Now, each GPU has the correct update

                # Convert back to DTensor if needed
                if isinstance(p.data, DTensor):
                    u = DTensor.from_local(u, p.data.device_mesh, placements=[Shard(0)])

                ############################
                # Scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                if u.shape != p.data.shape:
                    raise RuntimeError(
                        f"Shape mismatch: u.shape={u.shape}, p.data.shape={p.data.shape}"
                    )

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

                # ok question here is, other ranks does not have the updated p.data
                # so, we need to broadcast the updated p.data to all ranks

        # For easier debug, we separate the AdamW optimizer
        ############################
        #       AdamW backup       #
        ############################
        self.update_adamw()
        return loss
