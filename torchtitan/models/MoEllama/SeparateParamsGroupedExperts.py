from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from . import ep_comm
from torchtitan.models.inits import build_init_fn


class SeparateParamsGroupedExperts(nn.Module):
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
    local_rank = 0
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
        self.num_experts = num_experts
        self.expert_per_rank = num_experts
        self.dim_out = dim_out

        self.gate_proj = nn.ParameterList(
            [nn.Parameter(torch.empty(dim_in, dim_out)) for _ in range(num_experts)]
        )

        self.down_proj = nn.ParameterList(
            [nn.Parameter(torch.empty(dim_in, dim_out)) for _ in range(num_experts)]
        )

        self.up_proj = nn.ParameterList(
            [nn.Parameter(torch.empty(dim_out, dim_in)) for _ in range(num_experts)]
        )

        self.act_fn = F.silu

    def __repr__(self):
        return (
            f"Separate Params GroupedExperts(dim_in={self.dim_in}, "
            f"dim_hidden={self.dim_out}, num_experts={self.num_experts})"
        )

    def setup_ep(self, ep_mesh=None, ep_size=None):
        self.ep_mesh = ep_mesh
        self.ep_size = ep_size if ep_size is not None else 1
        if self.ep_mesh is not None:
            self.local_rank = self.ep_mesh.get_local_rank()
        if ep_size is not None:
            self.expert_per_rank = self.num_experts // ep_size
        else:
            self.expert_per_rank = self.num_experts

    def forward(
        self,
        x: torch.Tensor,
        tokens_per_expert,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): with shape (num_experts * tokens_per_expert, dim_in) for Expert Choice(EC).

        Returns:
            torch.Tensor: with shape (num_experts * tokens_per_expert, dim_in) for Expert Choice(EC).
        """
        # Expert Choice(EC) forward
        # x shape (num_experts, tokens_per_expert, dim_in)
        x, all_tokens_per_expert = ep_comm.dispatch_tokens(
            x,
            self.ep_size,
            self.ep_mesh,
            tokens_per_expert,
        )
        # This code is used for EP
        lee = self.local_rank * self.expert_per_rank
        les = (self.local_rank + 1) * self.expert_per_rank

        h = self.act_fn(torch.bmm(x, torch.stack(list(self.gate_proj[lee:les]))))
        h = h * torch.bmm(x, torch.stack(list(self.down_proj[lee:les])))
        out = torch.bmm(h, torch.stack(list(self.up_proj[lee:les])))

        out = ep_comm.combine_tokens(
            out,
            self.ep_size,
            self.ep_mesh,
            all_tokens_per_expert,
        )

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

        for expert_idx in range(self.num_experts):
            init_fn(self.gate_proj[expert_idx], mean=0.0, std=init_std)
            init_fn(self.down_proj[expert_idx], mean=0.0, std=gate_init_std)
            init_fn(self.up_proj[expert_idx], mean=0.0, std=init_std / residual_div)
