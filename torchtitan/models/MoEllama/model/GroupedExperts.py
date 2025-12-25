# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.activations import build_activation
from torchtitan.models.inits import build_init_fn
from torchtitan.models.moe.utils import indices_padding_wrapper
from torchtitan.models.norms import build_norm


def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    activation: Callable,
    mid_norm: nn.Module,
) -> torch.Tensor:
    # NOTE: this would incur a synchronization between device and host
    num_tokens_per_expert = num_tokens_per_expert.tolist()

    # side-effect code due to the usage of generate_permute_indices
    num_padding = x.shape[0] - sum(num_tokens_per_expert)

    # a tuple of tensors indexed by experts
    # each with shape (tokens_per_expert(varying), dim)
    x = torch.split(
        x[: sum(num_tokens_per_expert)],
        split_size_or_sections=num_tokens_per_expert,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x):
        h = activation(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = mid_norm(h)
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        # h shape (tokens_per_expert(varying), dim)
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)

    # side-effect code due to the usage of generate_permute_indices
    out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))

    return out


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    activation: Callable,
    mid_norm: nn.Module,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    h = activation(
        torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets)
    )
    h = h * torch._grouped_mm(
        x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
    )
    out = torch._grouped_mm(
        mid_norm(h), w2.bfloat16().transpose(-2, -1), offs=offsets
    ).type_as(x)

    return out


class GroupedExperts(nn.Module):
    """This class implements the grouped experts layer used in Mixture of Experts. Each expert
    is a variant of the Gated Linear Units network.
    See more details in https://arxiv.org/pdf/2002.05202.

    Args:
        dim_in (int): Input dimension.
        dim_hidden (int): SwiGLU hidden dimension.
        num_experts (int): Number of experts in this grouped experts layer. Default is 1.
        swiglu (bool): Whether to use gated linear unit. Default is True.
        activation_type (str): Activation function to use. Default is F.silu.
    """

    def __init__(
        self,
        layer_id: int,
        *,
        dim_in: int,
        dim_hidden: int,
        num_experts: int = 1,
        activation_type: str = "silu",
        norm_everywhere: bool = False,
        norm_type: str | None = None,
        norm_eps: float | None = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.num_experts = num_experts
        self.expert_per_rank = num_experts

        self.w1 = nn.Parameter(torch.empty(num_experts, dim_hidden, dim_in))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim_in, dim_hidden))
        self.w3 = nn.Parameter(torch.empty(num_experts, dim_hidden, dim_in))

        self.use_grouped_mm = True

        self.act_fn = build_activation(activation_type)

        # when ep is enabled, this is set in parallelize.py
        self.ep_enable = False
        self.expert_per_rank = num_experts
        self.ep_size = 1

        if norm_everywhere:
            assert (
                norm_type is not None
            ), "`norm_type` needs to be passed when `norm_everywhere=True`"
            assert (
                norm_eps is not None
            ), "`norm_eps` needs to be passed when `norm_everywhere=True`"
            self.mid_norm = build_norm(norm_type, dim=dim_hidden, eps=norm_eps)
        else:
            self.mid_norm = nn.Identity()

    def __repr__(self):
        model_str = f"GroupedExperts(dim_in={self.dim_in}, hidden={self.dim_hidden},\n"
        # model_str += (
        #     f"\tnum_experts={self.num_experts}, local_experts={self.expert_per_rank}, "
        # )
        # model_str += f"ep_size={self.ep_size}, \n"
        model_str += f"\tup_proj={self.w1.shape}, \n"
        model_str += f"\tgate_proj={self.w3.shape}, \n"
        model_str += f"\tdown_proj={self.w2.shape}, \n"
        model_str += f"\tmid_norm={self.mid_norm}, \n"
        model_str += ")"
        return model_str

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        if self.use_grouped_mm:
            # NOTE: If EP is not used, we need to pad the indices
            #       to prepare for grouped_mm;
            #       otherwise, EP will handle the padding.
            if (
                not isinstance(self.w1, DTensor)
                or "ep" not in self.w1.device_mesh.mesh_dim_names
            ):
                run_experts_fn = indices_padding_wrapper(_run_experts_grouped_mm)
            else:
                run_experts_fn = _run_experts_grouped_mm
            return run_experts_fn(
                w1,
                w2,
                w3,
                x,
                num_tokens_per_expert,
                activation=self.act_fn,
                mid_norm=self.mid_norm,
            )
        else:
            return _run_experts_for_loop(
                w1,
                w2,
                w3,
                x,
                num_tokens_per_expert,
                activation=self.act_fn,
                mid_norm=self.mid_norm,
            )

    def init_weights(
        self,
        init_std: float,
        residual_div: float,
        init_gate_as_residual: bool,
        init_fn_type: str,
    ):

        init_fn = build_init_fn(init_fn_type)
        gate_init_std = init_std / residual_div if init_gate_as_residual else init_std

        # lets always use different experts
        expert_init_fn = init_all_experts_different

        expert_init_fn(init_fn, self.w1.data, init_std, slot=0, layer_id=self.layer_id)
        expert_init_fn(
            init_fn, self.w3.data, gate_init_std, slot=2, layer_id=self.layer_id
        )
        expert_init_fn(
            init_fn,
            self.w2.data,
            init_std / residual_div,
            slot=1,
            layer_id=self.layer_id,
        )

        if not isinstance(self.mid_norm, nn.Identity):
            self.mid_norm.reset_parameters()


def make_seed_from_global(
    layer_id: int, slot: int, expert_id: int, total_experts: int
) -> int:
    # 3 slots per expert: w1, w2, w3
    return layer_id * (3 * total_experts) + slot * total_experts + expert_id


def init_all_experts_different(init_fn, w, init_std, slot, layer_id):
    # we should expect the experts to have same norms, rather than same weights
    assert layer_id is not None, "layer_id must be set "
    total_experts = w.shape[0]
    if isinstance(w, torch.distributed.tensor.DTensor):
        # we assume the DTensor is already sharded on dim 0
        local_tensor = w.to_local()
        shard_chunk = w.__create_chunk_list__()[0]
        offsets = shard_chunk.offsets[0]
    else:
        local_tensor = w
        offsets = 0

    for e in range(local_tensor.shape[0]):
        expert_id = e + offsets
        seed = make_seed_from_global(layer_id, slot, expert_id, total_experts)
        if local_tensor.device.type == "meta":
            rng = None
        else:
            rng = torch.Generator(device=local_tensor.device)
            rng.manual_seed(seed)

        init_fn(local_tensor[e], mean=0.0, std=init_std, generator=rng)

    if isinstance(w, torch.distributed.tensor.DTensor):
        w.to_local().copy_(local_tensor)
    else:
        w.copy_(local_tensor)
