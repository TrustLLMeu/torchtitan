# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.expert_parallel import ExpertParallel, TensorParallel
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.experiments.llama4.infra.parallelize import apply_fsdp
from torchtitan.models.llama3.infra.parallelize import (
    apply_ddp,
    PrepareMidNormInputOutput,
)
from torchtitan.models.llama3.model.bitnet_model import BitNetTransformerBlock
from torchtitan.tools.logging import logger


def parallelize_llama(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    world_mesh = parallel_dims.world_mesh
    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    use_flex_attn = getattr(model.model_args, "use_flex_attn", False)
    if job_config.parallelism.context_parallel_degree > 1 and use_flex_attn:
        raise NotImplementedError("CP support for FlexAttention is still in progress.")

    if parallel_dims.tp_enabled:
        if "bitnet" in job_config.model.converters:
            raise RuntimeError("BitNet currently does not support tensor parallelism")

        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise
        if enable_float8_tensorwise_tp:
            # TODO(jianiw): This branch needs to be tested and enabled
            raise NotImplementedError(
                "Currently, float8 tensorwise TP is not tested for deepseekv3"
            )

        enable_approx_mid_norm_for_tensor_parallel = (
            job_config.parallelism.enable_approx_mid_norm_for_tensor_parallel
        )
        tp_only_attention = job_config.parallelism.tensor_parallel_only_attention
        apply_non_moe_tp(
            model,
            world_mesh["tp"],
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            tensor_parallel_only_attention=tp_only_attention,
            enable_approx_mid_norm_for_tensor_parallel=enable_approx_mid_norm_for_tensor_parallel,
        )
        maybe_enable_async_tp(job_config, world_mesh["tp"])

    # I dont think we need to apply TP for MOE?

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=world_mesh["tp"] if parallel_dims.tp_enabled else None,
            ep_mesh=world_mesh["ep"] if parallel_dims.ep_enabled else None,
            ep_tp_mesh=(
                world_mesh["ep", "tp"]
                if parallel_dims.tp_enabled
                and parallel_dims.ep_enabled
                and parallel_dims.etp_enabled
                else None
            ),
            etp_enabled=parallel_dims.etp_enabled,
            tp_only_attention=tp_only_attention,
        )

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled,
            use_flex_attn,
        )

    if model_compile_enabled:
        apply_compile(model, ep_enabled=parallel_dims.ep_enabled)

    dp_mesh: DeviceMesh | None = None

    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)

        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]

        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
        dp_mod_ep_mesh_dim_names = []
        if parallel_dims.ep_enabled:
            if parallel_dims.dp_replicate_enabled:
                dp_mod_ep_mesh_dim_names.append("dp_replicate")
            dp_mod_ep_mesh_dim_names.append("dp_shard_mod_ep")

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            dp_mod_ep_mesh=(
                world_mesh[tuple(dp_mod_ep_mesh_dim_names)]
                if parallel_dims.ep_enabled
                else None
            ),
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        dp_mesh = world_mesh
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=model_compile_enabled,
            enable_compiled_autograd=job_config.parallelism.enable_compiled_autograd,
        )
    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    tensor_parallel_only_attention: bool = False,
    enable_approx_mid_norm_for_tensor_parallel: bool = False,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears with tensorwise scaling.
    if enable_float8_tensorwise_tp:
        # TODO(vkuzo): add the items below to __init__.py of torchao.float8 and import from there
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": colwise_parallel(),
            "attention.wk": colwise_parallel(),
            "attention.wv": colwise_parallel(),
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
        }
        if not isinstance(transformer_block.attention.o_norm, nn.Identity):
            if enable_approx_mid_norm_for_tensor_parallel:
                layer_plan["attention.o_norm"] = SequenceParallel(sequence_dim=-1)
            else:
                layer_plan["attention.o_norm"] = PrepareMidNormInputOutput()

        if isinstance(transformer_block, BitNetTransformerBlock):
            layer_plan.update(
                {
                    "attention.wo_norm": SequenceParallel(),
                    "feed_forward.w2_norm": SequenceParallel(),
                }
            )
        # dont want to bother the Mid-norm for now
        if not transformer_block.moe_enabled and not tensor_parallel_only_attention:
            layer_plan.update(
                {
                    "ffn_norm": SequenceParallel(),
                    "feed_forward": prepare_module_input(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
                    "feed_forward.w3": colwise_parallel(),
                }
            )

            if not isinstance(transformer_block.moe.out_norm, nn.Identity):
                if enable_approx_mid_norm_for_tensor_parallel:
                    layer_plan["feed_forward.out_norm"] = SequenceParallel(
                        sequence_dim=-1
                    )
                else:
                    layer_plan["feed_forward.out_norm"] = PrepareMidNormInputOutput()

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        "Tensor Parallelism to the model"
    )


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    ep_tp_mesh: DeviceMesh | None,
    etp_enabled: bool,
    tp_only_attention: bool = False,
    enable_approx_mid_norm_for_tensor_parallel: bool = False,
):
    """
    I will disble moe EP_TP for now.
    To make it work, we need
    1) Takeing the mid-norm things same same dense model
    2) We need to name the MoE layer as "moe" and dense layer as "feed_forward", we cannot use the
       same name "feed_forward" for both.

    """
    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            continue

        if tp_only_attention:
            tp_mesh = None

        if tp_mesh is not None:
            # TODO(JSC): DO we really want to run TP for MoE?
            raise NotImplementedError(
                "Maybe we dont need TP for MoE, ->use --parallelism.tensor_parallel_only_attention"
            )

            # moe_layer_plan = {
            #     # input / output sharding on the seqlen dim
            #     # all-gather for input, reduce-scatter for output
            #     "feed_forward": PrepareModuleInputOutput(
            #         input_layouts=(Shard(1),),
            #         desired_input_layouts=(Replicate(),),
            #         use_local_input=True,
            #         output_layouts=(Partial(),),
            #         desired_output_layouts=(Shard(1),),
            #     ),
            #     # replicate computation for the router
            #     "feed_forward.router.gate": NoParallel(),
            #     # input Replicate, output Partial
            #     "feed_forward.shared_experts": TensorParallel(),
            # }
            # parallelize_module(
            #     module=transformer_block,
            #     device_mesh=tp_mesh,
            #     parallelize_plan=moe_layer_plan,
            # )

        # if ep_mesh is not None:
        experts_mesh, experts_plan = None, None
        if ep_mesh is None:  # TP only
            experts_mesh = tp_mesh
            # input Replicate, output Partial
            experts_plan = TensorParallel()
        elif tp_mesh is None:  # EP only
            experts_mesh = ep_mesh
            # input / output sharding on the batch / tokens dim
            experts_plan = ExpertParallel()
            transformer_block.moe.experts.ep_enable = True
            total_experts = transformer_block.moe.experts.num_experts
            ep_world_size = ep_mesh.size()
            ep_per_rank = total_experts // ep_world_size
            transformer_block.moe.experts.expert_per_rank = ep_per_rank
            transformer_block.moe.experts.ep_size = ep_world_size

        elif etp_enabled:  # EP + TP
            # DONT THINK WE NEED THIS
            # TODO(JSC): DO we really want to run TP for MoE?
            raise NotImplementedError("Maybe we dont need TP for MoE")
            # experts_mesh = ep_tp_mesh
            # experts_plan = ExpertTensorParallel(tp_mesh=tp_mesh, ep_mesh=ep_mesh)
        else:
            experts_mesh = ep_mesh
            # input / output sharding on the batch / tokens dim
            experts_plan = ExpertParallel()
            transformer_block.moe.experts.ep_enable = True
            total_experts = transformer_block.moe.experts.num_experts
            ep_world_size = ep_mesh.size()
            ep_per_rank = total_experts // ep_world_size
            transformer_block.moe.experts.expert_per_rank = ep_per_rank
            transformer_block.moe.experts.ep_size = ep_world_size

        if experts_mesh:
            parallelize_module(
                module=transformer_block.moe.experts,
                device_mesh=experts_mesh,
                parallelize_plan=experts_plan,
            )


def apply_compile(model: nn.Module, ep_enabled: bool):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """

    for layer_id, transformer_block in model.layers.named_children():
        # we dont do compile when EP enabled
        fullgraph = True and not ep_enabled
        # if transformer_block.moe_enabled:
        #     fullgraph = False
        transformer_block = torch.compile(transformer_block, fullgraph=fullgraph)
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")
