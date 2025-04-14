from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from . import ep_comm
from torchtitan.models.inits import build_init_fn


class GroupedExperts(nn.Module):
    """This class implements the grouped experts layer used in Mixture of Experts. Each expert
    is a variant of the Gated Linear Units network. See more details in https://arxiv.org/pdf/2002.05202.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        num_experts (int): Number of experts in this grouped experts layer. Default is 1.
        swiglu (bool): Whether to use gated linear unit. Default is True.
        activation (nn.Module): Activation function to use. Default is F.silu.
    """

    ep_mesh = None
    ep_size = 1
    expert_per_rank = None

    def __init__(
        self,
        *,
        dim_in: int,
        dim_out: int,
        num_experts: int = 1,
        activation: Callable = F.silu,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_experts = num_experts
        self.expert_per_rank = num_experts

        self.gate_proj = nn.Parameter(torch.empty(num_experts, dim_in, dim_out))
        self.down_proj = nn.Parameter(torch.empty(num_experts, dim_in, dim_out))
        self.up_proj = nn.Parameter(torch.empty(num_experts, dim_out, dim_in))

        self.act_fn = F.silu

    def __repr__(self):
        model_str = f"GroupedExperts(dim_in={self.dim_in}, dim_hidden={self.dim_out}, num_experts={self.num_experts}, ep_size={self.ep_size} \n"
        model_str += f"\tgate_proj={self.gate_proj.shape}, \n"
        model_str += f"\tdown_proj={self.down_proj.shape}, \n"
        model_str += f"\tup_proj={self.up_proj.shape}, \n"
        model_str += ")"
        return model_str

    def setup_ep(self, ep_mesh=None, ep_size=None):
        self.ep_mesh = ep_mesh
        self.ep_size = ep_size if ep_size is not None else 1
        if ep_size is None or ep_size == 1 or ep_mesh is None:
            return

        if ep_size > self.num_experts:
            raise ValueError(
                f"ep_size {ep_size} is less than the number of experts {self.num_experts}"
            )
        if self.num_experts % ep_size != 0:
            raise ValueError(
                f"ep_size {ep_size} does not divide the number of experts {self.num_experts}"
            )

        # otherwise, we re-distribute the experts and copy the parameters from the original gate_proj, down_proj, up_proj
        self.expert_per_rank = self.num_experts // ep_size

        gate_proj = nn.Parameter(
            torch.empty(self.expert_per_rank, self.dim_in, self.dim_out)
        )

        down_proj = nn.Parameter(
            torch.empty(self.expert_per_rank, self.dim_in, self.dim_out)
        )

        up_proj = nn.Parameter(
            torch.empty(self.expert_per_rank, self.dim_out, self.dim_in)
        )

        self.register_parameter("gate_proj", gate_proj)
        self.register_parameter("down_proj", down_proj)
        self.register_parameter("up_proj", up_proj)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): with shape (experts_per_rank, tokens_per_expert, dim_in) for Expert Choice(EC).

        Returns:
            torch.Tensor: with shape (experts_per_rank, tokens_per_expert, dim_in) for Expert Choice(EC).
        """

        h = self.act_fn(torch.bmm(x, self.gate_proj))
        h = h * torch.bmm(x, self.down_proj)
        out = torch.bmm(h, self.up_proj)

        return out

    def init_weights(
            self,
            init_std: float,
            residual_div: float,
            init_gate_as_residual: bool,
            init_fn_type: str,
    ):
        init_fn = build_init_fn(init_fn_type)
        gate_init_std = (
            init_std / residual_div
            if init_gate_as_residual
            else init_std
        )

        def init_each_expert(w, init_std):
            if isinstance(w, torch.distributed.tensor.DTensor):
                local_tensor = w.to_local()
            else:
                local_tensor = w

            # Initialize the local tensor
            for e in range(local_tensor.shape[0]):
                init_fn(local_tensor[e], mean=0.0, std=init_std)
            if isinstance(w, torch.distributed.tensor.DTensor):
                w.to_local().data = local_tensor
            else:
                w.copy_(local_tensor)

        init_each_expert(self.gate_proj.data, init_std)
        init_each_expert(self.down_proj.data, gate_init_std)
        init_each_expert(self.up_proj.data, init_std / residual_div)
