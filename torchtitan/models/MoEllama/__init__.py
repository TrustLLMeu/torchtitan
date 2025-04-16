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
        rope_theta=500000,
        n_shared_experts=1,
        activate_experts=2,
        n_routed_experts=8,
    ),
    "1B-7B": MoEModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=16,
        n_kv_heads=8,
        rope_theta=500000,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
        qk_norm=True,
    ),
    "3B": MoEModelArgs(
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        rope_theta=500000,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
    ),
    "8B": MoEModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
    ),
    "8B_qk": MoEModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        qk_norm=True,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
    ),
    "70B": MoEModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
    ),
    "405B": MoEModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
        n_shared_experts=1,
        activate_experts=8,
        n_routed_experts=64,
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
