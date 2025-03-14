from dataclasses import dataclass
from functools import partial
from typing import Optional

from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.llama.model import Attention, precompute_freqs_cis
from torchtitan.models.norms import build_norm
from torchtitan.protocols.train_spec import BaseModelArgs, ModelProtocol


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
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True
    norm_type: str = "rmsnorm"
    qk_norm: bool = False

    # ==== MoE specific args ====
    # shared_experts: int = 2
    # n_routed_experts: int = 8
    # activate_experts: int = 2
    shared_experts: int = 2
    activate_experts: int = 2
    n_routed_experts: int = 8
    moe_gate_bias_update_speed: float = 0.01
    moe_aux_loss_alpha: float = 0.01


class Gate(nn.Module):
    def __init__(
        self, hidden_size: int, experts=8, topk=2, bias=True, bias_update_speed=0.01
    ):
        super().__init__()

        self.topk = topk
        self.experts = experts
        self.expert_embeddings = nn.Parameter(torch.randn(experts, hidden_size))
        # Expert embeddings
        self.bias = nn.Parameter(torch.zeros(experts)) if bias else None
        # Bias for load balancing
        self.bias_update_speed = bias_update_speed
        # Step size for updating bias dynamically

    def forward(self, x: torch.Tensor, update_bias: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B*S, D)
            update_bias: Whether to update the bias dynamically (only during training)
        Returns:
            weights: Normalized routing weights (B*S, K)
            indices: Selected expert indices (B*S, K)
        """
        # Compute expert selection scores (sigmoid-based gating)
        scores = torch.sigmoid(x @ self.expert_embeddings.T)  # (B*S, N_r)

        if self.bias is not None:
            scores = scores + self.bias  # Add bias term to improve balance

        # Select top-k highest scoring experts
        top_values, indices = torch.topk(scores, self.topk, dim=-1)

        # Extract and normalize weights
        weights = scores.gather(-1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        if update_bias and self.bias is not None:
            self.update_bias(indices)

        return weights.type_as(x), indices, scores

    def update_bias(self, indices: torch.Tensor):
        """
        Adjust bias dynamically to prevent expert overload or underload.
        If an expert is overused, decrease its bias. If underused, increase bias.
        """
        # Count how many times each expert is selected
        # Update bias (ensuring gradients track it)
        with torch.no_grad():
            expert_counts = torch.bincount(
                indices.flatten(), minlength=self.experts
            ).float()

            # Normalize the expert usage
            avg_usage = expert_counts.mean()  # Target average usage
            adjustment = (expert_counts - avg_usage) * self.bias_update_speed

            # Reduce bias for overused experts, increase for underused
            self.bias -= adjustment.detach()

            # EMA version
            # self.bias.mul_(0.9).add_(0.1 * adjustment.detach())


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class MoE(nn.Module):
    def __init__(
        self,
        dim: int,
        multiple_of: int = 256,
        shared_experts: int = 2,
        n_routed_experts: int = 8,
        activate_experts: int = 2,
        ffn_dim_multiplier: Optional[float] = None,
        match_dim_with_dense: bool = True,
        bias_update_speed: float = 0.01,  # Bias adjustment speed
        aux_loss_alpha: float = 0.01,  # Small weight for sequence-wise auxiliary loss
    ):
        super().__init__()
        """
        match_dim_with_dense.

        Saying the dense model have the 'hidden_size' of 1024, 
        Then "actual_dim * activate_experts = hidden_size" should be satisfied.
        """
        assert (
            shared_experts > 0
        ), "when moved from my libs to this one, something goes wrong for shared_experts=0"
        if match_dim_with_dense:
            total_activate_experts = shared_experts + activate_experts
            ratio = total_activate_experts / (n_routed_experts + shared_experts)
        else:
            ratio = 1.0
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3 * ratio)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim * 1.0)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        mlp_func = partial(
            FeedForward,
            dim=dim,
            hidden_dim=hidden_dim,
        )
        self.n_routed_experts = n_routed_experts
        self.topk = activate_experts
        self.shared_experts = shared_experts
        self.aux_loss_alpha = aux_loss_alpha  # Loss coefficient

        # Use updated Gate with DeepSeekMoE-style routing and bias balancing
        self.gate = Gate(
            hidden_size=dim,
            experts=n_routed_experts,
            topk=activate_experts,
            bias=True,
            bias_update_speed=bias_update_speed,
        )

        # Shared Experts (applies to all tokens)
        if shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [mlp_func() for _ in range(shared_experts)]
            )

        else:
            self.shared_experts = None

        # Routed Experts (only used when selected)
        self.experts = nn.ModuleList([mlp_func() for _ in range(n_routed_experts)])

    def init_weights(self, init_std: float):
        for expert in self.experts:
            expert.init_weights(init_std)
        if self.shared_experts is not None:
            for expert in self.shared_experts:
                expert.init_weights(init_std)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, S, D = x.size()
        x = rearrange(x, "b s d -> (b s) d")  # Flatten batch & sequence

        if self.shared_experts is not None:
            shared_outputs = torch.stack(
                [expert(x) for expert in self.shared_experts], dim=0
            )  # (num_shared, B*S, D)
            shared_outputs = shared_outputs.mean(dim=0)
        else:
            shared_outputs = torch.zeros_like(x)

        # Get routing weights and indices (update bias only during training)
        weights, indices, scores = self.gate(
            x, update_bias=self.training
        )  # (B*S, K), (B*S, K)

        # Efficient parallel expert execution
        routed_outputs = torch.zeros_like(x)
        unique_experts = indices.unique()

        for i in unique_experts:
            # Find tokens assigned to expert i
            idx = torch.where(indices == i)  # idx = (batch_indices, topk_indices)

            if idx[0].numel() > 0:  # Ensure at least one token is routed to this expert
                expert_inputs = x[idx[0]]  # Select the corresponding inputs
                expert_weights = weights[idx]  # Select corresponding weights

                # Apply expert function
                expert_output = self.experts[i](expert_inputs)

                routed_outputs[idx[0]] += expert_output * expert_weights.unsqueeze(-1)

        if self.training and self.aux_loss_alpha > 0:
            aux_loss = self.sequence_wise_aux_loss(
                indices,
                weights,
                scores,
                B,
                S,
            )
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        routing_entropy = -(weights * weights.log()).sum(dim=-1).mean()

        output = rearrange(
            shared_outputs + routed_outputs, "(b s) d -> b s d", b=B, s=S
        )
        return output, aux_loss, routing_entropy

    def sequence_wise_aux_loss(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        scores: torch.Tensor,
        B: int,
        S: int,
    ) -> torch.Tensor:
        """
        Computes the Sequence-Wise Auxiliary Loss as described in DeepSeekMoE.

        Args:
            indices (torch.Tensor): Selected expert indices (B*S, K).
            weights (torch.Tensor): Routing weights (B*S, K).
            scores (torch.Tensor): Raw expert scores before top-k selection (B*S, N_r).
            B (int): Batch size.
            S (int): Sequence length.

        Returns:
            aux_loss (torch.Tensor): Computed sequence-wise auxiliary loss.
        """
        indices = indices.view(B * S, self.topk)  # (B*S, K)
        weights = weights.view(B * S, self.topk)  # (B*S, K)
        scores = scores.view(B * S, self.n_routed_experts)  # (B*S, N_r)

        N_r = self.n_routed_experts  # Total number of routed experts
        topk = self.topk  # Number of experts per token

        # Compute expert frequency f_i (fraction of tokens assigned to expert i)
        f_i = torch.zeros(N_r, device=indices.device, dtype=weights.dtype)
        f_i.scatter_add_(
            0, indices.view(-1), torch.ones_like(indices.view(-1), dtype=weights.dtype)
        )
        f_i /= B * S * topk  # Normalize by total tokens and top-k experts

        # Compute normalized expert probabilities P_i
        sum_scores = scores.sum(
            dim=-1, keepdim=True
        )  # Sum across all experts per token (B*S, 1)
        normalized_scores = scores / sum_scores.clamp(
            min=1e-5
        )  # Avoid division by zero

        P_i = torch.zeros(N_r, device=indices.device, dtype=weights.dtype)
        for k in range(topk):
            P_i.scatter_add_(
                0,
                indices[:, k],
                normalized_scores.gather(1, indices[:, k].unsqueeze(-1)).squeeze(-1),
            )

        P_i /= B * S  # Normalize by total sequence length

        # Compute final auxiliary loss
        aux_loss = (f_i * P_i).sum() * self.aux_loss_alpha

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
        self.feed_forward = self.feed_forward_cls(
            dim=model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
            shared_experts=model_args.shared_experts,
            n_routed_experts=model_args.n_routed_experts,
            activate_experts=model_args.activate_experts,
            bias_update_speed=model_args.moe_gate_bias_update_speed,
            aux_loss_alpha=model_args.moe_aux_loss_alpha,
            match_dim_with_dense=True,
        )

        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        self.attention_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )
        self.ffn_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

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
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


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
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

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
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward(self, tokens: torch.Tensor):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens
        total_moe_aux_loss = 0

        for layer in self.layers.values():
            h, moe_aux_loss, routing_entropy = layer(h, self.freqs_cis)
            total_moe_aux_loss += moe_aux_loss

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output, total_moe_aux_loss

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
