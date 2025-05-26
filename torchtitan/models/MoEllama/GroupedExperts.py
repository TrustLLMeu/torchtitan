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
    ep_rank = None
    expert_per_rank = None

    def __init__(
        self,
        *,
        dim_in: int,
        dim_out: int,
        num_experts: int = 1,
        activation: Callable = F.silu,
        moe_init_all_experts_same: bool = False,
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

        self.init_all_experts_same = moe_init_all_experts_same

    def __repr__(self):
        model_str = f"GroupedExperts(dim_in={self.dim_in}, dim_hidden={self.dim_out},\n"
        model_str += f"\tnum_experts={self.num_experts}, local_experts={self.expert_per_rank}, ep_size={self.ep_size}\n"
        model_str += f"\tgate_proj={self.gate_proj.shape}, \n"
        model_str += f"\tdown_proj={self.down_proj.shape}, \n"
        model_str += f"\tup_proj={self.up_proj.shape}, \n"
        model_str += ")"
        return model_str

    def setup_ep(self, ep_mesh, ep_size):
        self.ep_mesh = ep_mesh
        self.ep_size = ep_size
        self.ep_rank = ep_mesh.get_local_rank()
        self.expert_per_rank = self.num_experts // ep_size

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

        if isinstance(self.gate_proj, torch.distributed.tensor.DTensor):
            h = self.act_fn(torch.bmm(x, self.gate_proj.to_local()))
            h = h * torch.bmm(x, self.down_proj.to_local())
            out = torch.bmm(h, self.up_proj.to_local())

        else:
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
        gate_init_std = init_std / residual_div if init_gate_as_residual else init_std

        if self.init_all_experts_same:
            init_all_experts_same(init_fn, self.gate_proj.data, init_std)
            init_all_experts_same(init_fn, self.down_proj.data, gate_init_std)
            init_all_experts_same(init_fn, self.up_proj.data, init_std / residual_div)
        else:
            if init_fn_type != "scaled_orthogonal":
                raise ValueError(f"Unsupported init_fn_type: {init_fn_type}")

            init_all_experts_different(self, self.gate_proj.data, init_std)
            init_all_experts_different(self, self.down_proj.data, gate_init_std)
            init_all_experts_different(self, self.up_proj.data, init_std / residual_div)


def init_all_experts_same(init_fn, w, init_std):
    if isinstance(w, torch.distributed.tensor.DTensor):
        local_tensor = w.to_local()
    else:
        local_tensor = w

    init_fn(local_tensor[0], mean=0.0, std=init_std)
    for e in range(1, local_tensor.shape[0]):
        local_tensor[e].data.copy_(local_tensor[0].data)

    if isinstance(w, torch.distributed.tensor.DTensor):
        w.to_local().data = local_tensor
    else:
        w.copy_(local_tensor)


def init_all_experts_different(self, w, init_std):
    """
    here we hardcode init with the sclaed orthogonal init
    """
    fan_out, fan_in = w.shape[1], w.shape[2]
    gain = (fan_out / fan_in) ** 0.5
    gain *= init_std
    big_matrix = torch.empty([self.num_experts, fan_out, fan_in], device=w.device)
    torch.nn.init.normal_(big_matrix, mean=0.0, std=1)

    if not isinstance(w, torch.distributed.tensor.DTensor):
        dtensor_matrix = big_matrix
        # there is DDP

    elif self.ep_mesh is not None and self.ep_size > 1:
        # for experts parallel
        start_idx = self.ep_rank * self.expert_per_rank
        end_idx = start_idx + self.expert_per_rank
        dtensor_matrix = big_matrix[start_idx:end_idx, :, :]
    else:
        # for fsdp + no experts parallel
        dtensor_matrix = torch.distributed.tensor.distribute_tensor(
            big_matrix, placements=w.placements, device_mesh=w.device_mesh
        ).to_local()

    # now we do orthogonal init on the dtensor_matrix
    for e in range(dtensor_matrix.shape[0]):
        data = dtensor_matrix[e]
        if fan_out < fan_in:
            data.t_()

        q, r = torch.linalg.qr(data)
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph

        if fan_out < fan_in:
            q.t_()
        with torch.no_grad():
            data.view_as(q).copy_(q)
            data.mul_(gain)
            dtensor_matrix[e].data = data

    if isinstance(w, torch.distributed.tensor.DTensor):
        w.to_local().data.copy_(dtensor_matrix)
    else:
        w.copy_(dtensor_matrix)
