# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.components.tokenizer import build_hf_byte_tokenizer, build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec
from .hf_assests import setup_hf

from .infra.parallelize import parallelize_llama
from .model.args import MoEModelArgs
from .model.model import Transformer
from .model.moe import MoEArgs
from .model.state_dict_adapter import MoEllamaStateDictAdapter

__all__ = [
    "MoEArgs",
    "MoEModelArgs",
    "Transformer",
    "moe_llama3_configs",
    "MoEllamaStateDictAdapter",
]


moe_llama3_configs = {
    "debugmodel": MoEModelArgs(
        dim=512,  # beaware this if 2x then the llama3-debugmodel
        n_layers=8,
        n_heads=16,
        rope_theta=10000,
        moe_args=MoEArgs(
            num_experts=8,
            num_shared_experts=1,
            top_k=4,
        ),
        qk_norm=True,
        norm_everywhere=False,
        norm_eps=1e-30,
    ),
    "1B-7B-Proxy-8layers": MoEModelArgs(
        dim=512,
        n_layers=8,
        n_heads=4,
        n_kv_heads=2,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=1,
            top_k=8,
            scaling_factor=2.8232,  # 8 of 64 experts
        ),
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=64,
        n_dense_layers=1,
    ),
    "1B-7B-Proxy": MoEModelArgs(
        dim=512,
        n_layers=24,
        n_heads=4,
        n_kv_heads=2,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=1,
            top_k=8,
            scaling_factor=2.8232,  # 8 of 64 experts
        ),
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=64,
    ),
    "1B-7B": MoEModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=1,
            top_k=8,
            scaling_factor=2.8232,  # 8 of 64 experts
        ),
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=256,
    ),
    "test": MoEModelArgs(
        dim=256,
        n_layers=8,
        n_heads=2,
        n_kv_heads=1,
        ffn_dim_multiplier=1,
        multiple_of=64,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=1,
            top_k=8,
            scaling_factor=2.8232,  # 8 of 64 experts
        ),
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
    ),
}

register_train_spec(
    "MoEllama3",
    TrainSpec(
        model_cls=Transformer,
        model_args=moe_llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=MoEllamaStateDictAdapter,
        hf_assets_setup_fn=setup_hf.copy_and_overwrite_model_config,
    ),
)

register_train_spec(
    "byte_MoEllama3",
    TrainSpec(
        model_cls=Transformer,
        model_args=moe_llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_byte_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=MoEllamaStateDictAdapter,
        hf_assets_setup_fn=setup_hf.copy_and_overwrite_model_config,
    ),
)
