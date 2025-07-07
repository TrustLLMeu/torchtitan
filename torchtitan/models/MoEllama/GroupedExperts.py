from typing import Callable

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from torchtitan.models.inits import build_init_fn
from torchtitan.models.norms import build_norm


try:
    from grouped_gemm.ops import gmm as grouped_gemm
except ImportError:

    def mock_grouped_gemm(x, w, m_sizes):
        out = x.new_zeros(x.shape[0], w.shape[2])
        # G = m_sizes.shape[0]
        shift = 0
        for g, n_tokens in enumerate(m_sizes):
            input_tokens = x[shift:shift + n_tokens]
            out[shift:shift + n_tokens] = input_tokens @ w[g]
            shift += n_tokens
        return out

    grouped_gemm = mock_grouped_gemm


class GroupedExperts(nn.Module):
    """This class implements the grouped experts layer used in Mixture of Experts. Each expert
    is a variant of the Gated Linear Units network. See more details in https://arxiv.org/pdf/2002.05202.

    Args:
        dim_in (int): Input dimension.
        dim_hidden (int): SwiGLU hidden dimension.
        num_experts (int): Number of experts in this grouped experts layer. Default is 1.
        swiglu (bool): Whether to use gated linear unit. Default is True.
        activation (nn.Module): Activation function to use. Default is F.silu.
    """

    ep_mesh = None
    ep_size = 1
    ep_local_rank = None
    expert_per_rank = None

    def __init__(
        self,
        *,
        dim_in: int,
        dim_hidden: int,
        num_experts: int = 1,
        activation: Callable = F.silu,
        moe_init_all_experts_same: bool = False,
        norm_everywhere: bool = False,
        norm_type: str | None = None,
        norm_eps: float | None = None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.num_experts = num_experts
        self.expert_per_rank = num_experts

        self.w1 = nn.Parameter(torch.empty(num_experts, dim_in, dim_hidden))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim_hidden, dim_in))
        self.w3 = nn.Parameter(torch.empty(num_experts, dim_in, dim_hidden))

        self.act_fn = activation

        self.init_all_experts_same = moe_init_all_experts_same

        if norm_everywhere:
            assert norm_type is not None, \
                "`norm_type` needs to be passed when `norm_everywhere=True`"
            assert norm_eps is not None, "`norm_eps` needs to be passed when `norm_everywhere=True`"
            self.out_norm = build_norm(norm_type, dim=dim_hidden, eps=norm_eps)
        else:
            self.out_norm = nn.Identity()

    def __repr__(self):
        model_str = f"GroupedExperts(dim_in={self.dim_in}, hidden={self.dim_hidden},\n"
        model_str += f"\tnum_experts={self.num_experts}, local_experts={self.expert_per_rank}, "
        model_str += f"ep_size={self.ep_size}, \n"
        model_str += f"\tup_proj={self.w1.shape}, \n"
        model_str += f"\tgate_proj={self.w3.shape}, \n"
        model_str += f"\tdown_proj={self.w2.shape}, \n"
        model_str += f"\tout_norm={self.out_norm}, \n"
        model_str += ")"
        return model_str

    def setup_ep(self, ep_mesh, ep_size):
        self.ep_mesh = ep_mesh
        self.ep_size = ep_size
        self.ep_local_rank = ep_mesh.get_local_rank()
        self.expert_per_rank = self.num_experts // ep_size

    def forward(
        self,
        x: torch.Tensor,
        m_sizes: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): with shape (total_tokens, dim_in)
            m_sizes (torch.Tensor): with shape (num_experts), on CPU

        Returns:
            out (torch.Tensor): with shape (total_tokens, dim_in)
        """

        @torch.compile(fullgraph=True)
        def norm_x(norm, x):
            return norm(x)

        h = grouped_gemm(x, self.w1, m_sizes)
        h = self.act_fn(h) * grouped_gemm(x, self.w3, m_sizes)
        h = grouped_gemm(norm_x(self.out_norm, h), self.w2, m_sizes)

        return h

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
            expert_init_fn = init_all_experts_same

        else:
            expert_init_fn = init_all_experts_different

        expert_init_fn(init_fn, self.w1.data, init_std)
        expert_init_fn(init_fn, self.w3.data, gate_init_std)
        expert_init_fn(init_fn, self.w2.data, init_std / residual_div)


def init_all_experts_same(init_fn, w, init_std):
    """
    Notice that the weights are in the shape of [G, D_in, D_out]
    But we expected the weights to be [D_out, D_in] for `init_fn`
    """
    if isinstance(w, torch.distributed.tensor.DTensor):
        local_tensor = w.to_local()
    else:
        local_tensor = w

    init_fn(local_tensor[0].transpose(0, 1), mean=0.0, std=init_std)
    for e in range(1, local_tensor.shape[0]):
        local_tensor[e].data.copy_(local_tensor[0].data)

    if isinstance(w, torch.distributed.tensor.DTensor):
        w.to_local().copy_(local_tensor)
    else:
        w.copy_(local_tensor)


def init_all_experts_different(init_fn, w, init_std):
    if isinstance(w, torch.distributed.tensor.DTensor):
        local_tensor = w.to_local()
    else:
        local_tensor = w

    for e in range(local_tensor.shape[0]):
        rank = dist.get_rank()
        rand_offset = torch.randint(0, 10000, size=(), device="cpu").item()
        seed = rank * 50000 + e * 100 + rand_offset
        # for each rank, layer, expert, [w1, w2, w3], we need to set a different seed
        if w.device.type == "meta":
            rng = None
        else:
            rng = torch.Generator(device=w.device)
            rng.manual_seed(seed)

        init_fn(local_tensor[e].transpose(0, 1), mean=0.0, std=init_std, generator=rng)

    if isinstance(w, torch.distributed.tensor.DTensor):
        w.to_local().copy_(local_tensor)
    else:
        w.copy_(local_tensor)
