# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

from torch import nn

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.models.inits import parse_depth_init
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger

from .moe import MoEArgs


@dataclass
class RoPEScalingArgs:
    scaling_factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


@dataclass
class ModelInitArgs:
    # If `True`, then each transformer block init uses its layer ID, and
    # if `False`, each uses the total number of transformer blocks. If
    # `None`, do not apply any depth scaling.
    depth_init: str | None = "total_depth"
    first_in_init_fn_type: str = "normal"
    first_in_init_std: float = 1.0
    # Exponent applied to the first input layer's input dimensionality
    # to obtain its init std factor.
    first_in_exp: float = 0.0
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
    residual_scale: str = "identity"

    router_init_fn_type: str = "trunc_normal"


@dataclass
class MoEModelArgs(BaseModelArgs):
    dim: int = 4096
    intermediate_size: int | None = None
    # to explicitly set the intermediate dimension, for FFN
    moe_intermediate_size: int | None = None
    # to explicitly set the intermediate dimension, for MoE
    head_dim: int | None = None
    # to explicitly set the head dimension,
    n_layers: int = 32
    n_dense_layers: int = 0
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = -1  # If -1, then take vocab size from tokenizer.
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000

    rope_scaling_args: RoPEScalingArgs | None = None

    max_seq_len: int = 131072

    model_init_args: ModelInitArgs = field(default_factory=ModelInitArgs)
    activation_type: str = "silu"
    norm_type: str = "rmsnorm"
    qk_norm: bool = False
    # If this is True, it implies `qk_norm=True`.
    norm_everywhere: bool = False

    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    eos_id: int = 0
    pad_id: int = -1

    # MoE
    moe_args: MoEArgs = field(default_factory=MoEArgs)

    # Number of additional modules to insert for multi-token prediction.
    num_mtp_modules: int = 0

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        self.model_init_args = job_config.model.model_init_args
        self.activation_type = job_config.model.activation_type
        self.norm_type = job_config.model.norm_type
        self.qk_norm = self.qk_norm or self.norm_everywhere

        if job_config.model.moe_router_scaling_factor is not None:
            self.moe_args.scaling_factor = job_config.model.moe_router_scaling_factor

        if job_config.model.moe_router_bias_update_norm_factor is not None:
            self.moe_args.bias_update_norm_factor = (
                job_config.model.moe_router_bias_update_norm_factor
            )

        for name in ["load_balance_coeff", "load_balance_loss_weight"]:
            value = getattr(job_config.training, name)
            if value is not None:
                setattr(self.moe_args, name, value)

        self.num_mtp_modules = job_config.training.num_mtp_tokens
        assert self.num_mtp_modules >= 0

        self.model_init_args.depth_init = parse_depth_init(
            self.model_init_args.depth_init
        )

        if job_config.model.vocab_size is not None:
            self.vocab_size = job_config.model.vocab_size
        if self.vocab_size == -1:
            tokenizer = kwargs.get("tokenizer")
            assert isinstance(tokenizer, BaseTokenizer), (
                "Need a `BaseTokenizer` to be passed to `update_from_config` via the "
                "`tokenizer` keyword argument to automatically set `vocab_size` "
                "(since `vocab_size == -1`)."
            )
            self.vocab_size = tokenizer.get_vocab_size()
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
            orig_vocab_size = self.vocab_size
            vocab_divisor = job_config.model.vocab_size_multiple_of
            self.vocab_size = int(
                math.ceil(self.vocab_size / vocab_divisor) * vocab_divisor
            )
            logger.info(
                f"Padded vocab size from {orig_vocab_size} to {self.vocab_size}."
            )

        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise NotImplementedError(
                "CP support for FlexAttention is still in progress."
            )

        self.moe_args._debug_force_load_balance = (
            job_config.debug.moe_force_load_balance
        )

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, int, float]:
        return get_moe_model_nparams_and_flops(
            self,
            model,
            2 * (self.dim // self.n_heads),
            seq_len,
        )
