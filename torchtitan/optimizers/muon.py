import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate
from .muon_utils import zeropower_via_newtonschulz5
from .muon_utils import update_adamw as real_adamw_update
from torchtitan.tools.logging import logger


"""
Before clean up     : 85.852 ms
After clean  up     : 79.852 ms
Fused AdamW         : 67.833 ms
Fused AdamW-Async:  : 63.955 ms

DDP                 : 42.206 ms
DDP  + Dist-muon     : 5.53ms
FSDP + Dist-muon    : 10.130 ms

"""


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        full_params,
        model,
        lr=1e-3,
        wd=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        world_mesh=None,
        communication_dtype="bf16",
        # == AdamW parameters ==
        async_adamw=True,
        adamw_lr=None,
        adamw_weight_decay=0.1,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        adamw_amsgrad=False,
        adamw_foreach=False,
        adamw_capturable=False,
        adamw_differentiable=False,
        adamw_fused=True,  # lets use fused AdamW by default
        adamw_maximize=False,
        **kwargs,
    ):
        self.fsdp_enabled = (
            "dp_shard" in world_mesh.mesh_dim_names
            or "dp_shard_1" in world_mesh.mesh_dim_names
        )
        self.world_mesh = world_mesh

        if communication_dtype == "bf16":
            self.communication_dtype = torch.bfloat16
        elif communication_dtype == "fp16":
            self.communication_dtype = torch.float16
        elif communication_dtype == "fp32":
            self.communication_dtype = torch.float32

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

        logger.info(f"Muon params: {sum(p.numel() for p in muon_params)}")
        logger.info(f"AdamW params: {sum(p.numel() for p in adamw_params)}")
        self.async_adamw = async_adamw

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            # == AdamW parameters ==
            adamw_lr=adamw_lr if adamw_lr is not None else lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_weight_decay=adamw_weight_decay,
            adamw_amsgrad=adamw_amsgrad,
            adamw_maximize=adamw_maximize,
            adamw_foreach=adamw_foreach,
            adamw_capturable=adamw_capturable,
            adamw_differentiable=adamw_differentiable,
            adamw_fused=adamw_fused,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params)
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim >= 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def update_muon_momentum_and_get_gradient(self, p, nestroev=False, momentum=0):
        g = p.grad
        # if g.ndim > 2:
        #     g = g.view(g.size(0), -1)
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(g)
        buf = state["momentum_buffer"]
        buf.mul_(momentum).add_(g)
        if nestroev:
            g = p.grad.add(buf, alpha=momentum)
        else:
            g = buf
        return g

    def update_adamw(self):
        if self.async_adamw:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())  # <- CRITICAL
            with torch.cuda.stream(stream):
                real_adamw_update(self)
        else:
            real_adamw_update(self)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.update_adamw()
        ############################################################
        #       Muon update for parameters with use_muon=True       #
        ############################################################

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            nestroev = group["nesterov"]
            ns_steps = group["ns_steps"]

            params = [
                p
                for p in group["params"]
                if self.state[p]["use_muon"] and p.grad is not None
            ]
            params.sort(key=lambda x: x.numel(), reverse=True)

            # generate weight updates in distributed fashion
            for p in params:
                g = self.update_muon_momentum_and_get_gradient(p, nestroev, momentum)
                ############################
                # Step 1: Gather full gradient across ranks
                if self.fsdp_enabled:
                    full_g = g.redistribute(
                        placements=[Replicate()] * g.device_mesh.ndim
                    ).to_local()
                else:
                    full_g = g
                ############################
                # Step 2: Run the Newton-Schulz iteration
                if full_g.ndim == 2:
                    u = zeropower_via_newtonschulz5(full_g, steps=ns_steps)
                else:
                    u = torch.stack(
                        [
                            zeropower_via_newtonschulz5(full_g[i], steps=ns_steps)
                            for i in range(full_g.shape[0])
                        ],
                        dim=0,
                    )

                # ############################
                # # Step 3: Extract the correct shard for this rank
                if self.fsdp_enabled:
                    u = DTensor.from_local(
                        u,
                        device_mesh=g.device_mesh,
                        placements=[Replicate()],
                    ).redistribute(placements=g.placements)

                ############################
                # Step 4: update the parameter
                with torch.no_grad():
                    # apply weight decay and update parameters
                    p.data.mul_(1 - lr * wd).add_(u, alpha=-lr)

        return loss
