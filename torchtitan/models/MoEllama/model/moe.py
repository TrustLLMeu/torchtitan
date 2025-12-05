# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.distributed.tensor
from torch import nn

from torchtitan.models.activations import build_activation
from torchtitan.models.inits import build_init_fn
from torchtitan.models.norms import build_norm

from .GroupedExperts import GroupedExperts


@dataclass
class MoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1
    top_k: int = 0

    # router
    scaling_factor: float | None = None

    use_grouped_mm: bool = True  # grouped mm or for-loop for the experts computation
    # TODO(JSC): Need ablation about the learning rate of the router bias
    load_balance_coeff: float | None = 1e-3
    load_balance_loss_weight: float = 0.0
    load_balance_loss_type: Literal["sequence_wise", "batch_wise"] = "sequence_wise"
    bias_update_norm_factor: str = "sign"

    _debug_force_load_balance: bool = False
    # if True, we force each experts get same amount of token via round-robin


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation_type: str = "silu",
        norm_everywhere: bool = False,
        norm_type: Optional[str] = None,
        norm_eps: Optional[float] = None,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.act_fn = build_activation(activation_type)
        self.dim = dim
        self.hidden_dim = hidden_dim

        if norm_everywhere:
            assert (
                norm_type is not None
            ), "`norm_type` needs to be passed when `norm_everywhere=True`"
            assert (
                norm_eps is not None
            ), "`norm_eps` needs to be passed when `norm_everywhere=True`"
            self.out_norm = build_norm(
                norm_type,
                dim=hidden_dim,
                eps=norm_eps,
            )
        else:
            self.out_norm = nn.Identity()

    def forward(self, x):
        return self.w2(self.out_norm(self.act_fn(self.w1(x)) * self.w3(x)))

    def init_weights(
        self,
        init_std: float,
        residual_div: float,
        init_gate_as_residual: bool,
        init_fn_type: str,
    ):
        init_fn = build_init_fn(init_fn_type)
        init_fn(self.w1.weight, mean=0.0, std=init_std)
        init_fn(self.w2.weight, mean=0.0, std=init_std / residual_div)
        gate_init_std = init_std / residual_div if init_gate_as_residual else init_std
        init_fn(self.w3.weight, mean=0.0, std=gate_init_std)

        if not isinstance(self.out_norm, nn.Identity):
            self.out_norm.reset_parameters()


class TokenChoiceTopKRouter(nn.Module):
    force_gate_on_fp32: bool = False

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        route_scale: float,
        _debug_force_load_balance: bool = False,
    ):
        super().__init__()

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.route_scale = route_scale
        self._debug_force_load_balance = _debug_force_load_balance

    def __repr__(self):
        return (
            f"Gate(experts={self.num_experts}, topk={self.top_k} | "
            f"DEBUG_FORCE_LOAD_BALANCED: {self._debug_force_load_balance})"
        )

    def init_weights(self, init_std: float, init_fn_type: str):
        # nn.init.xavier_uniform_(self.expert_embeddings)
        init_fn = build_init_fn(init_fn_type)
        init_fn(self.gate.weight, mean=0.0, std=init_std)

    def _debug_force_load_balance_routing(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment.
        Returns (selected_experts_indices [N, K] LongTensor, top_scores [N, K] FloatTensor).
        """
        n_tokens = scores.size(0)
        # Round-robin indices with exact balance
        selected_experts_indices = (
            torch.arange(
                n_tokens * self.top_k, device=scores.device, dtype=torch.int64
            ).reshape(n_tokens, self.top_k)
            % self.num_experts
        )
        top_scores = scores.gather(dim=1, index=selected_experts_indices)  # [N,K]
        return selected_experts_indices, top_scores

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.

        Returns:
            routed_input (torch.Tensor):
                Tokens grouped together by experts indices with shape ``(bs*slen*top_k,)``.
            token_indices (torch.Tensor):
                Token indices for routed_input with shape ``(bs*slen*top_k,)``.
            num_tokens_per_expert (torch.Tensor):
                Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        if self.force_gate_on_fp32:
            with torch.autocast(x.device, dtype=torch.float32):
                scores = self.gate(x)
        else:
            scores = self.gate(x)

        scores = torch.sigmoid(scores.to(torch.float32))

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        if expert_bias is not None:
            _, selected_experts_indices = torch.topk(
                scores + expert_bias, k=self.top_k, dim=1
            )
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            top_scores, selected_experts_indices = torch.topk(
                scores, k=self.top_k, dim=1
            )

        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
            ) = self._debug_force_load_balance_routing(scores)

        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)

        detached_top_scores = top_scores.detach()
        experts_entropy = (
            -(detached_top_scores * detached_top_scores.log()).sum(dim=-1).mean()
        )

        top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return (
            top_scores,
            scores,
            selected_experts_indices,
            num_tokens_per_expert,
            experts_entropy,
        )


# NOTE: the reason we make this a stateless module is to support
#       expert_tensor_parallel_degree=1 with consistent TP/EP APIs.
class TokenReorderer(nn.Module):
    """
    This module reorders token indices to match the order of experts, enabling
    efficient parallel processing of tokens by experts.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of experts each token will be routed to.
    """

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reorders token indices to match the order of experts for MoE routing.

        Args:
            top_scores (torch.Tensor): Routing scores for selected experts,
                shape (batch_size*seq_len, top_k)
            selected_experts_indices (torch.Tensor): Expert indices selected for each token,
                shape (batch_size*seq_len, top_k)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores_experts_sorted: Scores reordered to match expert ordering
                - token_indices_experts_sorted: Token indices reordered to match expert ordering
                - num_tokens_per_expert: Number of tokens assigned to each expert
        """
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )

        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        )


class MoE(nn.Module):
    experts_parallel_enabled = False

    def __init__(
        self,
        layer_id: int,
        dim: int,
        hidden_dim,
        moe_args: MoEArgs,
        activation_type: str = "silu",
        norm_everywhere: bool = False,
        norm_type: Optional[str] = None,
        norm_eps: Optional[float] = None,
    ):

        super().__init__()
        """
        match_dim_with_dense.

        Saying the dense model have the 'hidden_size' of 1024,
        Then "actual_dim * activate_experts = hidden_size" should be satisfied.
        """

        self.layer_id = layer_id

        self.num_experts = moe_args.num_experts
        self.topk = moe_args.top_k

        self.load_balance_loss_weight = (
            moe_args.load_balance_loss_weight
        )  # Loss coefficient
        self.load_balance_coeff = moe_args.load_balance_coeff
        self.bias_update_norm_factor = moe_args.bias_update_norm_factor
        self.load_balance_loss_type = moe_args.load_balance_loss_type

        # Use updated Gate with DeepSeekMoE-style routing and bias balancing
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=self.num_experts,
            top_k=self.topk,
            route_scale=moe_args.scaling_factor,
            _debug_force_load_balance=moe_args._debug_force_load_balance,
        )
        self.reorderer = TokenReorderer(num_experts=self.num_experts, top_k=self.topk)
        self.shared_experts = (
            FeedForward(
                dim,
                hidden_dim,
                activation_type=activation_type,
                norm_everywhere=norm_everywhere,
                norm_type=norm_type,
                norm_eps=norm_eps,
            )
            if moe_args.num_shared_experts > 0
            else None
        )

        # Routed Experts (only used when selected)
        assert self.num_experts > 0, "num_experts must be greater than 0"
        self.experts = GroupedExperts(
            layer_id,
            dim_in=dim,
            dim_hidden=hidden_dim,
            num_experts=self.num_experts,
            activation_type=activation_type,
            norm_everywhere=norm_everywhere,
            norm_type=norm_type,
            norm_eps=norm_eps,
        )

        self.register_buffer(
            "expert_bias", torch.zeros(self.num_experts, dtype=torch.float32)
        )
        self.register_buffer("load_balance_loss", torch.zeros(1, dtype=torch.float32))
        self.register_buffer(
            "tokens_per_expert", torch.zeros(self.num_experts, dtype=torch.int64)
        )
        self.register_buffer("router_entropy", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("acc_fwd_times", torch.zeros(1, dtype=torch.int64))

    def init_weights(
        self,
        init_std: float,
        residual_div: float,
        init_gate_as_residual: bool,
        init_fn_type: str,
        router_init_fn_type: str,
    ):
        self.experts.init_weights(
            init_std,
            residual_div=residual_div,
            init_gate_as_residual=init_gate_as_residual,
            init_fn_type=init_fn_type,
        )
        if self.shared_experts is not None:
            self.shared_experts.init_weights(
                init_std,
                residual_div=residual_div,
                init_gate_as_residual=init_gate_as_residual,
                init_fn_type=init_fn_type,
            )
        self.router.init_weights(init_std, router_init_fn_type)

        self.expert_bias.zero_()
        self.tokens_per_expert.zero_()
        self.router_entropy.zero_()
        self.acc_fwd_times.zero_()
        self.load_balance_loss.zero_()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bz, slen, dim = x.shape
        x = x.view(-1, dim)
        # TODO@JSC: check if we want to use FP32 remix
        (
            top_scores,
            sigmoid_scores,
            selected_experts_indices,
            num_tokens_per_expert,
            experts_entropy,
        ) = self.router(x, self.expert_bias)

        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)
            self.router_entropy.add_(experts_entropy)
            self.acc_fwd_times.add_(1)

        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(top_scores, selected_experts_indices)

        if self.training:
            if self.load_balance_loss_type == "sequence_wise":
                load_balance_loss = MoE.sequence_wise_aux_loss(
                    sigmoid_scores,
                    selected_experts_indices.long(),
                    bz,
                    slen,
                    self.topk,
                    self.load_balance_loss_weight,
                )
            elif self.load_balance_loss_type == "batch_wise":
                load_balance_loss = MoE.batch_wise_aux_loss(
                    sigmoid_scores,
                    num_tokens_per_expert,
                    self.topk,
                    self.load_balance_loss_weight,
                )
            else:
                raise ValueError(
                    f"Invalid load_balance_loss_type: {self.load_balance_loss_type}"
                )
            with torch.no_grad():
                # for logging only
                self.load_balance_loss.add_(load_balance_loss.detach())
        else:
            load_balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # ====

        token_indices_experts_sorted = token_indices_experts_sorted.reshape(
            -1, 1
        ).expand(-1, dim)

        # shape (bs*slen*top_k, dim)
        routed_input = torch.gather(x, dim=0, index=token_indices_experts_sorted)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        # shared expert
        # Note: we execute the shared expert before scoring the output of the routed expert
        # to "implicitly" overlap the shared expert compute with token combine communication
        if self.shared_experts is not None:
            out = self.shared_experts(x)
        else:
            out = torch.zeros_like(x)

        routed_output = (
            routed_output.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)
        ).to(x.dtype)

        out = out.scatter_add(
            dim=0, index=token_indices_experts_sorted, src=routed_output
        )

        output = out.reshape(bz, slen, dim).to(x.dtype)
        return output, load_balance_loss

    @staticmethod
    @torch.compile(fullgraph=True)
    def sequence_wise_aux_loss(
        scores: torch.Tensor,  # Shape: (B*S, N) - Raw Sigmoid Affinities (s_{i,t})
        indices: torch.Tensor,  # Shape: (B*S, K) - Selected Expert Indices
        B: int,  # Batch size
        S: int,  # Sequence length (T in the paper)
        top_k: int,  # K_r
        aux_loss_alpha: float,  # Alpha
    ) -> torch.Tensor:
        """
        Computes Sequence-Wise Auxiliary Loss (DeepSeek-V3 Equations 17-20).

        Args:
            scores: The dense affinity scores (s_{i,t}) for routed experts.
                    Should be the output of Sigmoid, shape (B*S, N).
            indices: The top-k selected expert indices. Shape (B*S, K).
        """
        if aux_loss_alpha <= 0:
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

        # N_r: Total number of routed experts
        N = scores.size(-1)

        # 1. Reshape inputs to handle each sequence separately: (B, S, N)
        #    This ensures we calculate P_i and f_i per sequence (Eq 20 & 18).
        scores_per_seq = scores.view(B, S, N)
        indices_per_seq = indices.view(B, S, top_k)

        # 2. Eq 19: Normalize affinity scores s_{i,t} to get s'_{i,t}
        #    DeepSeek-V3 uses Sigmoid, so scores don't sum to 1.
        #    Eq 19 explicitly requires dividing by the sum of all affinities.
        #    denominator shape: (B, S, 1)
        denominator = scores_per_seq.sum(dim=-1, keepdim=True) + 1e-20
        probs_per_seq = scores_per_seq / denominator  # This is s'_{i,t}

        # 3. Eq 20: Calculate P_i (Average probability per expert for each sequence)
        #    P_i = (1/T) * sum_{t=1}^T (s'_{i,t})
        #    We average over the Sequence dimension (dim=1).
        #    P_i shape: (B, N)
        P_i = probs_per_seq.mean(dim=1)

        # 4. Eq 18: Calculate f_i (Fraction of tokens selecting expert i per sequence)
        #    f_i = (N / (K * T)) * count_i

        # Flatten the top-k dimension to count hits per sequence: (B, S*K)
        flat_indices_per_seq = indices_per_seq.view(B, -1)
        selection_counts = torch.zeros((B, N), device=scores.device, dtype=scores.dtype)
        src = torch.ones_like(flat_indices_per_seq, dtype=scores.dtype)
        selection_counts.scatter_add_(1, flat_indices_per_seq, src)

        # Calculate f_i for each sequence, T (tokens in sequence) is S
        f_i = selection_counts * (N / (top_k * S))

        # 5. Eq 17: Calculate Balance Loss
        loss_per_seq = (f_i * P_i).sum(dim=1) * aux_loss_alpha

        return loss_per_seq.mean()

    @staticmethod
    @torch.compile(fullgraph=True)
    def batch_wise_aux_loss(
        scores: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        top_k: int,
        aux_loss_alpha: float,
    ) -> torch.Tensor:
        """
        Computes Batch-Wise Auxiliary Loss.
        Args:
            scores: Dense probabilities (BS, N).
            num_tokens_per_expert: Token counts (N).
            top_k: Number of experts selected per token.
            aux_loss_alpha: Scaling factor for the loss.
        """
        if aux_loss_alpha <= 0:
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

        # Total number of routed experts (N)
        N = scores.size(1)
        # Total number of tokens (T = BS * S)
        T = scores.size(0)

        P_i = scores.mean(dim=0)

        f_i = num_tokens_per_expert.to(scores.dtype) * (N / (top_k * T))

        loss = (f_i * P_i).sum() * aux_loss_alpha

        return loss
