from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.datasets.tokenizer.byte_tokenizer import build_byte_tokenizer
from torchtitan.datasets.tokenizer.tiktoken import build_tiktoken_tokenizer
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .pipeline_MoEllama import pipeline_llama
from .parallelize_MoEllama import parallelize_llama
from .moellama import Transformer, MoEModelArgs


__all__ = [
    "MoEModelArgs",
    "Transformer",
    "moe_llama3_configs",
]


moe_llama3_configs = {
    "debugmodel": MoEModelArgs(
        dim=512,  # beaware this if 2x then the llama3-debugmodel
        n_layers=8,
        n_heads=16,
        rope_theta=10000,
        n_shared_experts=1,
        activate_experts=4,
        n_routed_experts=8,
        qk_norm=True,
        norm_everywhere=False,
        depth_init=False,
        norm_eps=1e-30,
    ),
    "1B-7B-Proxy": MoEModelArgs(
        dim=512,
        n_layers=24,
        n_heads=4,
        n_kv_heads=2,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init=False,
        init_gate_as_residual=False,
        norm_type="np_rmsnorm",
        norm_everywhere=False,
        multiple_of=64,
        # MoE specific args
        moe_gate_bias_update_speed=0.001,
        moe_aux_loss_alpha=0.01,
        moe_routed_scaling_factor=None,  # will be set automatically
        moe_gate_use_bias_for_routing=True,
        moe_init_all_experts_same=False,
    ),
    "1B-7B": MoEModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init=False,
        init_gate_as_residual=False,
        norm_type="np_rmsnorm",
        norm_everywhere=False,
        multiple_of=256,
        # MoE specific args
        moe_gate_bias_update_speed=0.001,
        moe_aux_loss_alpha=0.01,
        moe_routed_scaling_factor=None,  # will be set automatically
        moe_gate_use_bias_for_routing=True,
        moe_init_all_experts_same=False,
    ),
    "1B-7B-norm-everywhere": MoEModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
        qk_norm=True,
        norm_eps=1e-20,
        rope_theta=10000,
        depth_init=False,
        init_gate_as_residual=False,
        norm_type="np_rmsnorm",
        norm_everywhere=True,
        multiple_of=256,
        # MoE specific args
        moe_gate_bias_update_speed=0.001,
        moe_aux_loss_alpha=0.01,
        moe_routed_scaling_factor=None,  # will be set automatically
        moe_gate_use_bias_for_routing=True,
        moe_init_all_experts_same=False,
    ),
}


register_train_spec(
    TrainSpec(
        name="MoEllama3",
        cls=Transformer,
        config=moe_llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_tiktoken_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)

register_train_spec(
    TrainSpec(
        name="byte_MoEllama3",
        cls=Transformer,
        config=moe_llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_byte_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
