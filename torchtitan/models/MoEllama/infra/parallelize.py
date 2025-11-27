# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
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
from torchtitan.config.job_config import Compile as CompileConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.expert_parallel import ExpertParallel, TensorParallel
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama3.infra.parallelize import (
    apply_ddp,
    PrepareMidNormInputOutput,
)
from torchtitan.models.llama4.infra.parallelize import apply_fsdp
from torchtitan.models.MoEllama.model import (
    GroupedExperts as grouped_experts_module,
    moe as moe_module,
)
from torchtitan.tools.logging import logger

# for selective op activation checkpointing
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops._c10d_functional.all_to_all_single.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
}


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

    tp_only_attention = job_config.parallelism.tensor_parallel_only_attention

    if parallel_dims.tp_enabled:
        if "bitnet" in job_config.model.converters:
            raise RuntimeError("BitNet currently does not support tensor parallelism")

        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        enable_approx_mid_norm_for_tensor_parallel = (
            job_config.parallelism.enable_approx_mid_norm_for_tensor_parallel
        )
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
            model_compile_enabled=model_compile_enabled,
            use_flex_attn=use_flex_attn,
            op_sac_save_list=_op_sac_save_list,
            base_folder=job_config.job.dump_folder,
        )

    if model_compile_enabled:
        if parallel_dims.ep_enabled:
            apply_compile(model, compile_config=job_config.compile)
        else:
            apply_compile_wo_ep(model, compile_config=job_config.compile)

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
                layer_plan["attention.mid_norm"] = SequenceParallel(sequence_dim=-1)
            else:
                layer_plan["attention.mid_norm"] = PrepareMidNormInputOutput()

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
                    layer_plan["feed_forward.mid_norm"] = SequenceParallel(
                        sequence_dim=-1
                    )
                else:
                    layer_plan["feed_forward.mid_norm"] = PrepareMidNormInputOutput()

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
        elif tp_mesh is None or not etp_enabled:  # EP only
            experts_mesh = ep_mesh
            # input / output sharding on the batch / tokens dim
            experts_plan = ExpertParallel()

            # below is for logging and debugging
            transformer_block.moe.experts.ep_enable = True
            total_experts = transformer_block.moe.experts.num_experts
            ep_world_size = ep_mesh.size()
            ep_per_rank = total_experts // ep_world_size
            transformer_block.moe.experts.expert_per_rank = ep_per_rank
            transformer_block.moe.experts.ep_size = ep_world_size

        else:  # EP + TP
            # TODO(JSC): DO we really want to run TP for MoE?
            raise NotImplementedError("Maybe we dont need TP for MoE")
            # experts_mesh = ep_tp_mesh
            # experts_plan = ExpertTensorParallel()

        if experts_mesh:
            parallelize_module(
                module=transformer_block.moe.experts,
                device_mesh=experts_mesh,
                parallelize_plan=experts_plan,
            )


def apply_compile_wo_ep(model: nn.Module, compile_config: CompileConfig):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(
            transformer_block, backend=compile_config.backend, fullgraph=True
        )
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")


def apply_compile(model: nn.Module, compile_config: CompileConfig):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    # NOTE: This flag is needed for torch.compile to avoid graph breaking on dynamic shapes in token-choice MoE
    # but it is experimental.
    torch._dynamo.config.capture_scalar_outputs = True
    # Workaround for https://github.com/pytorch/pytorch/issues/166926
    torch._C._dynamo.eval_frame._set_lru_cache(False)
    for layer_id, transformer_block in model.layers.named_children():
        if transformer_block.moe_enabled:
            # If it is a MoE layer, FSDP(GroupedExperts) will cause a graph break
            # So we must weave compile wrappers around those FSDP hooks to
            # prevent AC from falling back the whole graph to eager.
            # TODO: Fix Compile(AC(graph break))

            if isinstance(transformer_block, CheckpointWrapper):
                # TODO: Make CheckpointWrapper a transparent wrapper
                # unwrap so that .named_children() works
                block = transformer_block._checkpoint_wrapped_module
            else:
                block = transformer_block

            for attr_name, submod in block.named_children():
                assert getattr(block, attr_name) == getattr(
                    transformer_block, attr_name
                )

                if isinstance(submod, moe_module.MoE):
                    # avoid graph breaking on the GroupedExperts' FSDP hooks
                    # by wrapping each submod's forward instead of their __call__
                    moe = submod
                    for attr_name, submod in moe.named_children():
                        if attr_name == "experts":
                            # NOTE: We don't compile token dispatch and token combine due to an issue on B200:
                            # https://github.com/pytorch/torchtitan/issues/1940
                            continue
                        setattr(
                            moe,
                            attr_name,
                            torch.compile(
                                submod, backend=compile_config.backend, fullgraph=True
                            ),
                        )
                else:
                    setattr(
                        block,
                        attr_name,
                        torch.compile(
                            submod, backend=compile_config.backend, fullgraph=True
                        ),
                    )

        else:
            # If it's not a MoE layer, there is no FSDP(GroupedExperts)
            # So we can compile the whole block
            transformer_block = torch.compile(
                transformer_block,
                backend=compile_config.backend,
                fullgraph=True,
            )

        model.layers.register_module(layer_id, transformer_block)

    grouped_experts_module._run_experts_grouped_mm = torch.compile(
        grouped_experts_module._run_experts_grouped_mm,
        backend=compile_config.backend,
        fullgraph=True,
    )

    # NOTE: We don't compile for loop code path due to an issue with unbacked symints:
    # https://github.com/pytorch/pytorch/issues/166460

    logger.info("Compiling each TransformerBlock with torch.compile")
