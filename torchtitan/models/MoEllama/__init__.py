from torchtitan.components.loss import cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.datasets.hf_datasets import build_hf_dataloader
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
        dim=256,
        n_layers=8,
        n_heads=16,
        rope_theta=500000,
        shared_experts=1,
        n_routed_experts=4,
        activate_experts=1,
    ),
    "8B": MoEModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
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
    ),
    "70B": MoEModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": MoEModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
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
        loss_fn=cross_entropy_loss,
    )
)
