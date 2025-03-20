from typing import Optional

import torch
import torch.nn.functional as F

from torchtitan.models.llama.model import (
    apply_rotary_emb,
    Attention,
    FeedForward,
    repeat_kv,
    Transformer,
    TransformerBlock,
    TransformerModelArgs,
)
from torchtitan.models.norms import build_norm


class BitNetAttention(Attention):
    """
    BitNet Multi-head attention module.

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.
        wo_norm (RMSNorm): Layer normalization for Attention output
            projection input.

    """

    def __init__(self, model_args: TransformerModelArgs):
        super().__init__(model_args)
        self.wo_norm = build_norm(model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps)

    def init_weights(self, init_std: float):
        self.wo_norm.reset_parameters()
        super().init_weights(init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Apply optional QK normalization
        if self.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        # we use casual mask for training
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(self.wo_norm(output))


class BitNetFeedForward(FeedForward):
    """
    BitNet FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.
        norm_type (str): Which norm to use.
        norm_eps (float): Numerical stabilizer for norms.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.
        w2_norm (RMSNorm): Layer normalization for down projection input.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        norm_type: str,
        norm_eps: float,
    ):
        super().__init__(dim, hidden_dim, multiple_of, ffn_dim_multiplier)
        self.w2_norm = build_norm(norm_type, dim=self.w2.in_features, eps=norm_eps)

    def forward(self, x):
        return self.w2(self.w2_norm(F.silu(self.w1(x)) * self.w3(x)))

    def init_weights(self, init_std: float):
        self.w2_norm.reset_parameters()
        super().init_weights(init_std)


class BitNetTransformerBlock(TransformerBlock):
    """
    BitNet TransformerBlock Module

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

    attention_cls = BitNetAttention
    feed_forward_cls = BitNetFeedForward

    def __init__(self, layer_id: int, model_args: TransformerModelArgs):
        assert model_args.norm_type.lower() == "rmsnorm", "BitNet assumes RMSNorm"

        self._init_feed_forward_builder(model_args)
        super().__init__(layer_id, model_args)

    def _init_feed_forward_builder(self, model_args: TransformerModelArgs):
        old_feed_forward_cls = self.feed_forward_cls

        def build_bitnet_feed_forward(
            dim: int,
            hidden_dim: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
        ):
            return old_feed_forward_cls(
                dim,
                hidden_dim,
                multiple_of,
                ffn_dim_multiplier,
                norm_type=model_args.norm_type,
                norm_eps=model_args.norm_eps,
            )

        self.feed_forward_cls = build_bitnet_feed_forward


class BitNetTransformer(Transformer):
    """
    BitNet Transformer Module

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

    transformer_block_cls = BitNetTransformerBlock

    def __init__(self, model_args: TransformerModelArgs):
        assert model_args.norm_type.lower() == "rmsnorm", "BitNet assumes RMSNorm"
        super().__init__(model_args)
