from dataclasses import dataclass
import math
from typing import Optional, Callable

from einops import rearrange
import torch
from torch import nn
import torch.distributed.tensor
from torch.distributed.tensor import DTensor

from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.models.inits import build_init_fn
from torchtitan.models.inputs import MoEInputs, MoEInputsDict
from torchtitan.models.llama3.model import Attention, precompute_freqs_cis
from torchtitan.models.norms import build_norm
from torchtitan.protocols.train_spec import BaseModelArgs, ModelProtocol
from torchtitan.tools.logging import logger


from .GroupedExperts import GroupedExperts
from . import ep_comm
from .moe_utils import calc_gate_scaling_factor


@dataclass
class MoEModelArgs(BaseModelArgs):
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000

    max_seq_len: int = 2048
    # If `True`, then each transformer block init uses its layer ID, and
    # if `False`, each uses the total number of transformer blocks. If
    # `None`, do not apply any depth scaling.
    depth_init: Optional[bool] = True
    first_in_init_fn_type: str = "normal"
    first_in_init_std: float = 1.0
    # Exponent applied to the first input layer's input dimensionality
    # to obtain its init std factor.
    first_in_exp: float = 0.0
    router_init_fn_type: str = "trunc_normal"
    intermediate_init_fn_type: str = "trunc_normal"
    intermediate_init_std: float = 0.02
    # Exponent applied to the model's hidden dimensionality to obtain
    # intermediate layers' init std factors.
    intermediate_exp: float = 0.0
    # Whether to initialize the GLU gate as if it was a residual layer.
    init_gate_as_residual: bool = True
    final_out_init_fn_type: str = "trunc_normal"
    final_out_init_std: float = 1.0
    # Exponent applied to the final output layer's input dimensionality
    # to obtain its init std factor.
    final_out_exp: float = -0.5
    norm_type: str = "rmsnorm"
    qk_norm: bool = False
    # If this is True, it implies `qk_norm=True`.
    norm_everywhere: bool = False

    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    eos_id: int = 0
    pad_id: int = -1

    # Number of additional modules to insert for multi-token prediction.
    num_mtp_modules: int = 0

    # ==== MoE specific args ====
    n_shared_experts: int = 1
    n_routed_experts: int = 0
    activate_experts: int = 0
    moe_gate_bias_update_speed: float = 0.001  # dpskv3, 0.001
    moe_aux_loss_alpha: float = 0.01  # OLMoE, default 0.01
    moe_routed_scaling_factor = None
    # dpskv3 2.5, moonlight 2.446, set None to auto-compute
    moe_gate_use_bias_for_routing: bool = True
    moe_init_all_experts_same: bool = False

    def update_from_config(self, job_config: JobConfig, tokenizer: Tokenizer) -> None:
        for name in [
                "first_in_init_fn_type",
                "first_in_init_std",
                "first_in_exp",
                "router_init_fn_type",
                "intermediate_init_fn_type",
                "intermediate_init_std",
                "intermediate_exp",
                "init_gate_as_residual",
                "final_out_init_fn_type",
                "final_out_init_std",
                "final_out_exp",
                "norm_type",
                "use_flex_attn",
                "attn_mask_type",
        ]:
            value = getattr(job_config.model, name)
            setattr(self, name, value)
        self.vocab_size = tokenizer.n_words
        # `eos_id` is not part of the `Tokenizer` interface, so keep it
        # optional.
        if hasattr(tokenizer, "eos_id"):
            self.eos_id = tokenizer.eos_id
        # `pad_id` is not part of the `Tokenizer` interface, so keep it
        # optional.
        if hasattr(tokenizer, "pad_id"):
            self.pad_id = tokenizer.pad_id
        # Add an additional vocab element if we are explicitly
        # supporting a pad token.
        if self.pad_id >= 0:
            self.vocab_size += 1
        if job_config.model.vocab_size_multiple_of:
            vocab_divisor = job_config.model.vocab_size_multiple_of
            self.vocab_size = int(
                math.ceil(self.vocab_size / vocab_divisor) * vocab_divisor
            )
            logger.info(
                f"Padded vocab size from {tokenizer.n_words} to {self.vocab_size}."
            )
        self.max_seq_len = job_config.training.seq_len
        self.qk_norm = self.qk_norm or self.norm_everywhere

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams_not_ffn = 0
        nparams_ffn = 0

        for p_name, param in model.named_parameters():
            if "feed_forward.experts" in p_name:
                nparams_ffn += param.numel()
            else:
                nparams_not_ffn += param.numel()

        if hasattr(model, "get_sparsity_ratio"):
            sparsity_ratio = model.get_sparsity_ratio()
        else:
            sparsity_ratio = 1

        nparams_active = nparams_not_ffn + nparams_ffn * sparsity_ratio
        nparams_total = nparams_not_ffn + nparams_ffn

        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
        )

        l, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.dim // self.n_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = 6 * (nparams_active - nparams_embedding) + 12 * l * h * q * t

        return nparams_active, nparams_total, num_flops_per_token


class Gate(nn.Module):
    bias: torch.Tensor
    _accumulated_adjustment: torch.Tensor

    def __init__(
        self,
        hidden_size: int,
        experts=8,
        topk=2,
        bias=True,
        bias_update_speed=0.001,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()

        self.topk = topk
        self.experts = experts
        self.expert_embeddings = nn.Parameter(torch.zeros(experts, hidden_size))
        # Expert embeddings

        if bias and experts > 1:
            self.register_buffer("bias", torch.zeros(experts, dtype=torch.float32))
        else:
            self.bias = None

        self.routed_scaling_factor = routed_scaling_factor
        # Bias for load balancing
        self.bias_update_speed = bias_update_speed
        # Step size for updating bias dynamically

        # Cache for accumulated adjustments
        """
        TODO(JSC):
        Potential bug here.  When we use accumulated_gradient [? and PP], and load the checkpoint,
        the self._accumulated_adjustment will be reset to zero. 
        ---- 
        With buffer, the above problem [seems] will not happen. but new problem is we should always make sure
        the checkpoint is always saved right after the update.  Because we dont wand to all-reduce 
        the _accumulated_adjustment in an intermediate steps. 
        """
        self.register_buffer(
            "_accumulated_adjustment", torch.zeros(experts, dtype=torch.int32)
        )
        self.register_buffer("_num_accumulated", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("needs_reduction", torch.tensor(False), persistent=False)

    def __repr__(self):
        return f"Gate(experts={self.experts}, topk={self.topk}, bias={self.bias is not None})"

    def init_weights(self, init_std: float, init_fn_type: str):
        # nn.init.xavier_uniform_(self.expert_embeddings)
        init_fn = build_init_fn(init_fn_type)
        init_fn(self.expert_embeddings, mean=0.0, std=init_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B*S, D)
        Returns:
            weights: Normalized routing weights
            indices: Selected expert indices
            scores: Expert selection scores
        """
        # Compute expert selection scores (sigmoid-based gating)

        logits = torch.nn.functional.linear(
            x.type(torch.float32), self.expert_embeddings.type(torch.float32)
        )
        scores = torch.sigmoid(logits)

        # Optionally add bias before top-k selection
        if self.bias is not None:
            biased_scores = scores + self.bias
        else:
            biased_scores = scores

        # Select top-k experts
        top_values, indices = torch.topk(biased_scores, self.topk, dim=-1)

        # Extract and normalize weights
        weights = scores.gather(-1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Apply route scaling
        weights = weights * self.routed_scaling_factor

        if self.training and self.bias is not None:
            self.accumulate_adjustment(indices)

        return weights, indices, scores

    def accumulate_adjustment(self, indices: torch.Tensor):
        """
        Accumulate local expert counts without any all-reduce.
        """
        with torch.no_grad():
            local_expert_counts = torch.zeros(
                self.experts, device=indices.device, dtype=torch.int32
            )
            local_expert_counts.scatter_add_(
                0,
                indices.flatten(),
                torch.ones_like(indices.flatten(), dtype=torch.int32),
            )

            self._accumulated_adjustment.add_(local_expert_counts)
            self._num_accumulated.add_(1)
            self.needs_reduction.fill_(True)

    def update(self):
        """
        Apply accumulated bias adjustments and clear the cache.
        Now with all-reduce moved here (after backward).
        """
        if self._num_accumulated > 0:
            with torch.no_grad():
                if self.needs_reduction:
                    # Only do the all-reduce if needed and not already done
                    torch.distributed.all_reduce(
                        self._accumulated_adjustment, op=torch.distributed.ReduceOp.SUM
                    )
                    self.needs_reduction.fill_(False)

                # Calculate and apply adjustment
                total_usage = self._accumulated_adjustment.sum()
                avg_usage = total_usage / self.experts

                load_error = avg_usage - self._accumulated_adjustment

                adjustment = torch.sign(load_error) * self.bias_update_speed

                if isinstance(self.bias.data, DTensor):
                    update = torch.distributed.tensor.distribute_tensor(
                        adjustment.detach(),
                        device_mesh=self.bias.data.device_mesh,
                        placements=self.bias.data.placements,
                    )
                    self.bias.data.add_(update)
                else:
                    self.bias.data.add_(adjustment.detach())

                self._accumulated_adjustment.zero_()
                self._num_accumulated.zero_()


class SharedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation: Callable = torch.nn.functional.silu,
        norm_everywhere: bool = False,
        norm_type: Optional[str] = None,
        norm_eps: Optional[float] = None,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.act_fn = activation

        if norm_everywhere:
            assert norm_type is not None, \
                "`norm_type` needs to be passed when `norm_everywhere=True`"
            assert norm_eps is not None, "`norm_eps` needs to be passed when `norm_everywhere=True`"
            self.out_norm = build_norm(
                norm_type,
                dim=dim,
                eps=norm_eps,
            )
        else:
            self.out_norm = nn.Identity()

    def forward(self, x):
        h = self.act_fn(self.gate_proj(x))
        h = h * self.down_proj(x)
        out = self.out_norm(self.up_proj(h))
        return out

    def init_weights(
        self,
        init_std: float,
        residual_div: float,
        init_gate_as_residual: bool,
        init_fn_type: str,
    ):
        init_fn = build_init_fn(init_fn_type)
        init_fn(self.gate_proj.weight, mean=0.0, std=init_std)
        init_fn(self.up_proj.weight, mean=0.0, std=init_std / residual_div)
        gate_init_std = init_std / residual_div if init_gate_as_residual else init_std
        init_fn(self.down_proj.weight, mean=0.0, std=gate_init_std)


class MoE(nn.Module):
    def __init__(
        self,
        dim: int,
        multiple_of: int = 256,
        n_shared_experts: int = 2,
        n_routed_experts: int = 8,
        activate_experts: int = 2,
        ffn_dim_multiplier: Optional[float] = None,
        match_dim_with_dense: bool = True,
        use_bias_for_routing: bool = True,
        bias_update_speed: float = 0.001,  # Bias adjustment speed
        aux_loss_alpha: float = 0.001,  # Small weight for sequence-wise auxiliary loss
        routed_scaling_factor: float = 1.0,
        moe_init_all_experts_same: bool = False,
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

        if match_dim_with_dense:
            ratio = 1.0 / (activate_experts + n_shared_experts)
        else:
            ratio = 1.0

        hidden_dim = 2 * 4 * dim / 3
        if ffn_dim_multiplier is not None:
            hidden_dim = ffn_dim_multiplier * hidden_dim
        hidden_dim = int(hidden_dim * ratio)
        hidden_dim = hidden_dim - hidden_dim % multiple_of
        self.n_routed_experts = n_routed_experts
        self.topk = activate_experts
        self.n_shared_experts = n_shared_experts
        self.aux_loss_alpha = aux_loss_alpha  # Loss coefficient

        # Use updated Gate with DeepSeekMoE-style routing and bias balancing
        self.gate = Gate(
            hidden_size=dim,
            experts=n_routed_experts,
            topk=activate_experts,
            bias=use_bias_for_routing,
            bias_update_speed=bias_update_speed,
            routed_scaling_factor=routed_scaling_factor,
        )

        # Shared Experts (applies to all tokens)
        if n_shared_experts > 0:
            assert n_shared_experts == 1, "Only one shared expert is supported"
            """
            TODO(JSC):
            To make Muon work easier, we set the shared experts to either 0 or 1.
            So we dont use the GroupedExperts.
            """
            # self.shared_experts = GroupedExperts(
            #     dim_in=dim, dim_out=hidden_dim, num_experts=n_shared_experts
            # )
            self.shared_experts = SharedExperts(
                dim,
                hidden_dim,
                norm_everywhere=norm_everywhere,
                norm_type=norm_type,
                norm_eps=norm_eps,
            )

        else:
            self.shared_experts = None

        # Routed Experts (only used when selected)
        if n_routed_experts > 0:
            self.experts = GroupedExperts(
                dim_in=dim,
                dim_out=hidden_dim,
                num_experts=n_routed_experts,
                moe_init_all_experts_same=moe_init_all_experts_same,
                norm_everywhere=norm_everywhere,
                norm_type=norm_type,
                norm_eps=norm_eps,
            )

        else:
            self.experts = None

    def init_weights(
        self,
        init_std: float,
        residual_div: float,
        init_gate_as_residual: bool,
        init_fn_type: str,
        router_init_fn_type: str,
    ):
        if self.experts is not None:
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
        self.gate.init_weights(init_std, router_init_fn_type)

    def update_gate_bias(self):
        self.gate.update()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bz, slen, dim = x.shape
        x = rearrange(x, "b s d -> (b s) d")  # Flatten batch & sequence
        out = torch.zeros_like(x)

        if self.topk > 0:
            # (B*S, K), (B*S, K)
            weights, indices, scores = self.gate(x)

            with torch.no_grad():
                # [seq_len, n_routed_experts]
                cnts = indices.new_zeros((indices.shape[0], self.n_routed_experts))
                # Fill 1 to the selected experts
                cnts.scatter_(1, indices, 1)
                tokens_per_expert = cnts.sum(dim=0)
                # Token indices for each expert
                idxs = indices.view(-1).argsort()
                sorted_tokens_shape = idxs.shape + x.shape[1:]

            sorted_tokens = x[idxs // indices.shape[1]]

            experts_input, all_tokens_per_expert = ep_comm.dispatch_tokens(
                sorted_tokens,
                self.experts.ep_mesh,
                self.experts.ep_size,
                tokens_per_expert,
                force_float32=False,
            )

            experts_output = self.experts(experts_input)

            routed_output = ep_comm.combine_tokens(
                experts_output,
                self.experts.ep_mesh,
                self.experts.ep_size,
                all_tokens_per_expert,
                force_float32=False,
            )

            # Map back from sorted tokens to original positions
            original_token_indices = idxs // indices.shape[1]

            # Apply weights
            expert_weights = weights.view(-1)[idxs].unsqueeze(1)
            weighted_outputs = routed_output * expert_weights
            out.index_add_(
                0, original_token_indices, weighted_outputs.to(routed_output.dtype)
            )

        # Apply shared experts if they exist
        if self.shared_experts is not None:
            # Reshape input for shared experts: (n_shared_experts, B*S, dim)
            shared_input = torch.stack([x] * self.n_shared_experts, dim=0)
            shared_output = self.shared_experts(shared_input).sum(0)

            out = out + shared_output

        if self.training:
            aux_loss = self.sequence_wise_aux_loss(indices, scores, bz, slen)
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        if self.topk > 0:
            correct_weights = weights.detach() / self.gate.routed_scaling_factor
            routing_entropy = (
                -(correct_weights * correct_weights.log()).sum(dim=-1).mean()
            )
        else:
            routing_entropy = torch.tensor(0.0, device=x.device)

        output = rearrange(out, "(b s) d -> b s d", b=bz, s=slen)
        return output, aux_loss, routing_entropy

    def sequence_wise_aux_loss(
        self,
        indices: torch.Tensor,  # Shape (B*S, K_val), type long
        scores: torch.Tensor,  # Shape (B*S, N_r_val), type float
        B: int,  # Batch size
        S: int,  # Sequence length
        eps: float = 1e-15,
        force_flip: bool = False,  # If True, clamps f_i^(b) and P_i^(b) to [0,1]
    ) -> torch.Tensor:
        """
        Computes the Sequence-Wise Auxiliary Loss as described in DeepSeekMoE.

        Args:
            indices (torch.Tensor): Selected expert indices for each
                token, flattened across batch and sequence.
                Assumed to be of type long. Shape: (B*S, K_val).
            scores (torch.Tensor): Raw expert scores (before top-k
                selection and normalization) for each token and each
                expert. Shape: (B*S, N_r_val).
            B (int): Batch size.
            S (int): Sequence length.
            eps (float): Epsilon for numerical stability during score normalization.
            force_flip (bool): Whether to clamp intermediate f_i^(b) and P_i^(b) terms to [0,1].

        Returns:
            torch.Tensor: Computed sequence-wise auxiliary loss (scalar).

        """

        N_r = self.n_routed_experts  # Total number of routed experts (N_r in formulas)
        K_val = self.topk  # Number of experts selected per token (K in formulas)

        # Conditional returns based on your original code's logic
        # If topk == N_r, original code returned 0. This implies perfect load balancing by design.
        if K_val == N_r and N_r > 0:
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

        # If not training, alpha is zero, or no experts are selected (K_val=0)
        if not (self.aux_loss_alpha > 0 and K_val > 0):
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

        # If batch or sequence length is zero, no tokens to process.
        if B == 0 or S == 0:
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

        # Normalize scores: p_{t,i} (for token t, expert i)
        # scores shape: (B*S, N_r)
        norm_scores = scores / scores.sum(dim=-1, keepdim=True).clamp(
            min=eps
        )  # Shape: (B*S, N_r)

        # Reshape for sequence-wise processing
        # reshaped_indices shape: (B, S, K_val)
        # reshaped_norm_scores shape: (B, S, N_r)
        reshaped_indices = indices.view(B, S, K_val)
        reshaped_norm_scores = norm_scores.view(B, S, N_r)

        # 1. Compute f_i^(b) for all sequences b and experts i
        # f_i^(b) = (N_r / (S * K_val)) * sum_{s_idx, k_idx} 1{expert i is idx_{b,s_idx,k_idx}}

        # Create one-hot representation of selected expert indices
        # one_hot_indices[b, s_idx, k_idx, expert_i] = 1 if expert_i was the k_idx-th choice for token (b,s_idx)
        # Shape: (B, S, K_val, N_r)
        one_hot_indices = torch.nn.functional.one_hot(reshaped_indices, num_classes=N_r)

        # count_expert_selected_per_sequence[b, expert_i] = sum_{s_idx, k_idx} 1{...}
        # Sum over sequence tokens (S) and K_val choices
        count_expert_selected_per_sequence = one_hot_indices.sum(dim=[1, 2]).to(
            scores.dtype
        )  # Shape: (B, N_r)

        # Denominator S * K_val is non-zero due to earlier checks (S > 0, K_val > 0)
        f_i_b_all = (
            N_r / (S * K_val)
        ) * count_expert_selected_per_sequence  # Shape: (B, N_r)

        # 2. Compute P_i^(b) for all sequences b and experts i
        # P_i^(b) = (1/S) * sum_{s_idx, k_idx} p_{b,s_idx,expert_i} * 1{expert_i is idx_{b,s_idx,k_idx}}

        # Create a mask: mask_expert_selected[b,s_idx,k_idx,expert_i] is True if expert_i was selected
        # as k_idx-th choice for token (b,s_idx)
        arange_N_r = torch.arange(N_r, device=indices.device).view(
            1, 1, 1, N_r
        )  # Shape: (1,1,1,N_r) for broadcasting
        mask_expert_selected = (
            reshaped_indices.unsqueeze(-1) == arange_N_r
        )  # Shape: (B,S,K_val,N_r), boolean

        # Expand norm_scores for broadcasting: p_{b,s_idx,expert_i}
        reshaped_norm_scores_expanded = reshaped_norm_scores.unsqueeze(
            2
        )  # Shape: (B,S,1,N_r)

        # Element-wise product: p_{b,s_idx,expert_i} if expert_i was chosen, 0 otherwise
        # Product of float (norm_scores) and bool (mask) results in float.
        term_to_sum_for_P = (
            reshaped_norm_scores_expanded * mask_expert_selected
        )  # Shape: (B,S,K_val,N_r)

        # sum_val_P[b,expert_i] = sum_{s_idx,k_idx} (p_{b,s_idx,expert_i} * 1{...})
        # Sum over sequence tokens (S) and K_val choices
        sum_val_P = term_to_sum_for_P.sum(dim=[1, 2])  # Shape: (B, N_r)

        # Denominator S is non-zero due to earlier checks (S > 0)
        P_i_b_all = sum_val_P / S  # Shape: (B, N_r)

        if force_flip:
            f_i_b_all = torch.clamp(f_i_b_all, min=0, max=1)
            P_i_b_all = torch.clamp(P_i_b_all, min=0, max=1)

        # 3. Compute sum_i f_i^(b) * P_i^(b)
        loss_per_sequence = (f_i_b_all * P_i_b_all).sum(dim=1)

        # 4. Final auxiliary loss:
        aux_loss = loss_per_sequence.mean() * self.aux_loss_alpha

        return aux_loss


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    attention_cls = Attention
    feed_forward_cls = MoE

    def __init__(self, layer_id: int, model_args: MoEModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = self.attention_cls(model_args)

        if model_args.moe_routed_scaling_factor is None:
            routed_scaling_factor = calc_gate_scaling_factor(
                model_args.n_routed_experts,
                model_args.activate_experts,
                iter_times=10_000,
            )
            if layer_id == 0:
                logger.info(
                    f"Auto-computed routed_scaling_factor: {routed_scaling_factor}"
                )
        else:
            routed_scaling_factor = model_args.moe_routed_scaling_factor

        self.feed_forward = self.feed_forward_cls(
            dim=model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
            n_shared_experts=model_args.n_shared_experts,
            n_routed_experts=model_args.n_routed_experts,
            activate_experts=model_args.activate_experts,
            use_bias_for_routing=model_args.moe_gate_use_bias_for_routing,
            bias_update_speed=model_args.moe_gate_bias_update_speed,
            aux_loss_alpha=model_args.moe_aux_loss_alpha,
            match_dim_with_dense=True,
            routed_scaling_factor=routed_scaling_factor,
            moe_init_all_experts_same=model_args.moe_init_all_experts_same,
            norm_everywhere=model_args.norm_everywhere,
            norm_type=model_args.norm_type,
            norm_eps=model_args.norm_eps,
        )

        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        self.attention_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )
        self.ffn_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        self.weight_init_fn_type = model_args.intermediate_init_fn_type
        self.weight_init_std = (
            model_args.intermediate_init_std
            * model_args.dim**model_args.intermediate_exp
        )
        self.router_init_fn_type = model_args.router_init_fn_type
        if model_args.depth_init is None:
            self.residual_div = 1.0
        elif model_args.depth_init:
            self.residual_div = (2 * (self.layer_id + 1)) ** 0.5
        else:
            self.residual_div = (2 * self.num_layers) ** 0.5
        self.init_gate_as_residual = model_args.init_gate_as_residual

    def update_gate_bias(self):
        self.feed_forward.update_gate_bias()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(self.attention_norm(x), freqs_cis)

        mlp_output, moe_aux_loss, routing_entropy = self.feed_forward(self.ffn_norm(h))

        out = h + mlp_output
        return out, moe_aux_loss, routing_entropy

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(
            self.weight_init_std,
            residual_div=self.residual_div,
            init_fn_type=self.weight_init_fn_type,
        )
        self.feed_forward.init_weights(
            self.weight_init_std,
            residual_div=self.residual_div,
            init_gate_as_residual=self.init_gate_as_residual,
            init_fn_type=self.weight_init_fn_type,
            router_init_fn_type=self.router_init_fn_type,
        )

    def init_kv_cache(self, max_batch_size: int, max_seq_length: int):
        self.attention.init_kv_cache(max_batch_size, max_seq_length)


class Transformer(nn.Module, ModelProtocol):
    """
    Transformer Module

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        model_args (TransformerModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    transformer_block_cls = TransformerBlock

    def __init__(self, model_args: MoEModelArgs):
        if model_args.num_mtp_modules > 0:
            raise ValueError("currently, MTP is not supported with MoE")
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id
        self.pad_id = model_args.pad_id

        print(
            f"model_args.dim = {model_args.dim} | model_args.vocab_size = {model_args.vocab_size}"
        )

        self.tok_embeddings = nn.Embedding(
            model_args.vocab_size,
            model_args.dim,
            padding_idx=self.pad_id if self.pad_id >= 0 else None,
        )

        # TODO persistent should be set to false, since this buffer can be recomputed.
        # however, we set it to true for 2 reasons.  (1) due to pytorch/pytorch#123411,
        # compile or pipeline-tracer will not correctly handle non-persistent buffers,
        # so we need to fix that.  (2) if we initialize pipeline-parallel models from
        # a seed checkpoint rather than calling init_weights, we need freqs_cis to be
        # initialized by the checkpoint, or we need to add a separate initializer for
        # just the non-persistent buffers that is called after loading checkpoints.
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = self.transformer_block_cls(
                layer_id, model_args
            )

        self.norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.init_weights()

    def init_weights(
        self,
        buffer_device: Optional[torch.device] = None,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
        first_in_init_fn = build_init_fn(self.model_args.first_in_init_fn_type)
        first_in_std = (
            self.model_args.first_in_init_std
            * self.model_args.vocab_size**self.model_args.first_in_exp
        )
        if self.tok_embeddings is not None:
            if self.model_args.first_in_init_fn_type == "scion_normal":
                # catch cases when axis=1 is sharded
                assert self.tok_embeddings.weight.size(1) == self.model_args.dim, (
                    f"Input embedding last dim does not match model dim. "
                    f"Got shape: {self.tok_embeddings.weight.shape}"
                )
            first_in_init_fn(
                self.tok_embeddings.weight,
                mean=0.0,
                std=first_in_std,
            )
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_init_fn = build_init_fn(self.model_args.final_out_init_fn_type)
        final_out_std = (
            self.model_args.final_out_init_std
            * self.model_args.dim**self.model_args.final_out_exp
        )
        cutoff_factor = 3
        if self.output is not None:
            extra_kwargs = {}
            if self.model_args.final_out_init_fn_type == "trunc_normal":
                extra_kwargs["a"] = -cutoff_factor * final_out_std
                extra_kwargs["b"] = cutoff_factor * final_out_std
            if self.model_args.final_out_init_fn_type == "scion_normal":
                # catch cases when axis=1 is sharded
                assert self.output.weight.size(1) == self.model_args.dim, (
                    f"Output last dim does not match model dim. "
                    f"Got shape: {self.output.weight.shape}"
                )
            final_out_init_fn(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                **extra_kwargs,
            )
        if self.model_args.num_mtp_modules > 0:
            for layer in self.mtp_layers.values():
                if layer is not None:
                    layer.init_weights()

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def init_kv_cache(self, max_batch_size: int, max_seq_length: int):
        for layer in self.layers.values():
            if layer is not None:
                layer.init_kv_cache(max_batch_size, max_seq_length)

    def update_gate_bias(self):
        gates_to_update = []
        accumulated_adjustments = []
        token_selection_per_layer = []

        for layer in self.layers.values():
            if hasattr(layer.feed_forward, "gate"):
                gate = layer.feed_forward.gate
                if gate._num_accumulated > 0:
                    gates_to_update.append(gate)
                    accumulated_adjustments.append(gate._accumulated_adjustment)

        if gates_to_update:
            device = accumulated_adjustments[0].device
            all_local_counts = torch.cat(accumulated_adjustments)

            torch.distributed.all_reduce(
                all_local_counts, op=torch.distributed.ReduceOp.SUM
            )

            start_idx = 0
            for gate in gates_to_update:
                experts = gate.experts
                gate._accumulated_adjustment = all_local_counts[
                    start_idx : start_idx + experts
                ].clone()
                gate.needs_reduction.fill_(False)  # Mark as reduced
                start_idx += experts

        for layer in self.layers.values():
            if hasattr(layer, "update_gate_bias"):
                counts = layer.feed_forward.gate._accumulated_adjustment.cpu().tolist()
                token_selection_per_layer.append((layer.layer_id, counts))
                layer.update_gate_bias()
        return token_selection_per_layer

    def get_sparsity_ratio(self):
        # we assume all layers have the same number of activated and routed experts
        total_experts = self.model_args.n_routed_experts
        activated_experts = self.model_args.activate_experts
        if total_experts == 0:
            return 1
        return activated_experts / total_experts

    def forward(self, inputs: MoEInputs) -> MoEInputsDict:
        """
        Perform a forward pass through the Transformer model.

        Args:
            inputs (MoEInputs): Single tensor or dictionary containing the
                following keys and values:
                - tokens_list (Union[list[Optional[torch.Tensor]],
                  torch.Tensor]): Input token indices.
                - aux_loss (torch.Tensor): Sequence-wise auxiliary balance loss.
                - moe_entropy_per_layer (torch.Tensor): Entropy of MoE routing
                  for each MoE layer.

        Returns:
            MoEInputsDict: Dictionary containing the following keys and values:
                - tokens_list (list[torch.Tensor]): Output logits after applying
                  the Transformer model.
                - aux_loss (torch.Tensor): Sequence-wise auxiliary balance loss.
                - moe_entropy_per_layer (torch.Tensor): Entropy of MoE routing
                  for each MoE layer.

        """
        if not isinstance(inputs, dict):
            inputs = {"tokens_list": inputs}
        tokens = inputs["tokens_list"]
        total_moe_aux_loss = inputs.get("aux_loss", 0)
        moe_entropy_per_layer = inputs.get("moe_entropy_per_layer", {})
        if isinstance(tokens, list):
            tokens = tokens[0]

        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h, moe_aux_loss, routing_entropy = layer(h, self.freqs_cis)
            total_moe_aux_loss += moe_aux_loss
            moe_entropy_per_layer[layer.layer_id] = routing_entropy.detach()

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return {
            "tokens_list": [output],
            "aux_loss": total_moe_aux_loss,
            "moe_entropy_per_layer": moe_entropy_per_layer,
        }

    @classmethod
    def from_model_args(cls, model_args: MoEModelArgs) -> "Transformer":
        """
        Initialize a Transformer model from a TransformerModelArgs object.

        Args:
            model_args (TransformerModelArgs): Model configuration arguments.

        Returns:
            Transformer: Transformer model.

        """
        return cls(model_args)
