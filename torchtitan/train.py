# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import functools
import importlib
import os
import time
from datetime import timedelta
from typing import Any, Iterable, Optional

import torch

from torch.distributed.elastic.multiprocessing.errors import record

import torchtitan.components.ft as ft
import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.metrics import (
    build_metrics_processor,
    ensure_pp_loss_visible,
)
from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)


class Trainer(torch.distributed.checkpoint.stateful.Stateful):
    job_config: JobConfig
    gc_handler: utils.GarbageCollection

    parallel_dims: ParallelDims
    train_spec: train_spec_module.TrainSpec
    world_mesh: torch.distributed.DeviceMesh

    dataloader: train_spec_module.BaseDataLoader
    metrics_processor: train_spec_module.MetricsProcessor
    checkpointer: CheckpointManager

    model_parts: list[torch.nn.Module]
    optimizers: train_spec_module.OptimizersContainer
    lr_schedulers: train_spec_module.LRSchedulersContainer

    pp_has_first_stage: bool
    pp_has_last_stage: bool

    device: torch.device

    # states
    step: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, job_config: JobConfig):
        self.job_config = job_config

        logger.info(f"Starting job: {job_config.job.description}")

        if job_config.experimental.custom_import:
            importlib.import_module(job_config.experimental.custom_import)

        if job_config.job.print_args:
            logger.info(f"Running with args: {job_config.to_dict()}")

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)
        ft_manager = ft.init_ft_manager(job_config)

        # init distributed
        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism
        if not ft_manager.enabled:
            self.parallel_dims = parallel_dims = ParallelDims(
                dp_shard=parallelism_config.data_parallel_shard_degree,
                dp_replicate=parallelism_config.data_parallel_replicate_degree,
                cp=parallelism_config.context_parallel_degree,
                tp=parallelism_config.tensor_parallel_degree,
                pp=parallelism_config.pipeline_parallel_degree,
                world_size=world_size,
                enable_loss_parallel=not parallelism_config.disable_loss_parallel,
            )
        else:
            self.parallel_dims = parallel_dims = ft.FTParallelDims(
                dp_shard=parallelism_config.data_parallel_shard_degree,
                dp_replicate=parallelism_config.data_parallel_replicate_degree,
                cp=parallelism_config.context_parallel_degree,
                tp=parallelism_config.tensor_parallel_degree,
                pp=parallelism_config.pipeline_parallel_degree,
                world_size=world_size,
                enable_loss_parallel=not parallelism_config.disable_loss_parallel,
                ft_manager=ft_manager,
            )
        dist_utils.init_distributed(job_config)

        # build meshes
        self.world_mesh = world_mesh = parallel_dims.build_mesh(device_type=device_type)
        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            world_mesh,
            self.device,
            job_config.training.seed,
            job_config.training.deterministic,
        )
        self.train_spec = train_spec_module.get_train_spec(job_config.model.name)

        # verify batch sizes
        if job_config.training.global_batch_size is None:
            job_config.training.global_batch_size = \
                job_config.training.batch_size * dp_degree
        assert job_config.training.global_batch_size > 0
        assert (
            job_config.training.global_batch_size
            % (job_config.training.batch_size * dp_degree)
            == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({job_config.training.global_batch_size} "
            f"% ({job_config.training.batch_size} * {dp_degree}) != 0)"
        )

        self.gradient_accumulation_steps = (
            job_config.training.global_batch_size
            // (job_config.training.batch_size * dp_degree)
        )
        assert self.gradient_accumulation_steps > 0

        unwrapped_loss_fn = self.train_spec.loss_fn

        @functools.wraps(unwrapped_loss_fn)
        def accumulated_loss_fn(*args, **kwargs):
            loss = unwrapped_loss_fn(*args, **kwargs)
            return loss / self.gradient_accumulation_steps

        self.train_spec.loss_fn = accumulated_loss_fn

        # build dataloader
        tokenizer = self.train_spec.build_tokenizer_fn(job_config)

        # If TorchFT is enabled, the dp_rank and dp_degree, which are used for
        # dataloader must be changed.
        if ft_manager.enabled:
            dp_degree, dp_rank = ft_manager.get_dp_info(dp_degree, dp_rank)

        self.dataloader = self.train_spec.build_dataloader_fn(
            dp_world_size=dp_degree,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
            job_config=job_config,
        )

        # build model (using meta init)
        model_cls = self.train_spec.cls
        model_config = self.train_spec.config[job_config.model.flavor]
        # set the model configs from training inputs:
        # 1. norm type to decide which norm layer to use
        # 2. vocab size from tokenizer
        # 3. max_seq_len base on inputs
        model_config.norm_type = job_config.model.norm_type
        model_config.vocab_size = tokenizer.n_words
        if job_config.model.vocab_size_multiple_of:
            vocab_divisor = job_config.model.vocab_size_multiple_of
            model_config.vocab_size = int(
                math.ceil(model_config.vocab_size / vocab_divisor)
                * vocab_divisor
            )
            logger.info(
                f"Padded vocab size from {tokenizer.n_words} to {model_config.vocab_size}."
            )
        model_config.max_seq_len = job_config.training.seq_len

        logger.info(
            f"Building {self.train_spec.name} {job_config.model.flavor} with {model_config}"
        )
        with torch.device("meta"):
            model = model_cls.from_model_args(model_config)

        logger.info(f"model: {model}")

        # Build the collection of model converters. No-op if `model.converters` empty
        model_converters = build_model_converters(job_config, parallel_dims)
        model_converters.convert(model)

        # metrics logging
        build_metrics_processor_fn = (
            build_metrics_processor
            if self.train_spec.build_metrics_processor_fn is None
            else self.train_spec.build_metrics_processor_fn
        )
        self.metrics_processor = build_metrics_processor_fn(job_config, parallel_dims)
        color = self.metrics_processor.color

        # log model size
        model_param_count = utils.get_num_params(model)
        self.metrics_processor.num_flop_per_token = utils.get_num_flop_per_token(
            utils.get_num_params(model, exclude_embedding=True),
            model_config,
            job_config.training.seq_len,
        )

        logger.info(
            f"{color.blue}Model {self.train_spec.name} {job_config.model.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        if job_config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif job_config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = device_type
        else:
            init_device = device_type
            buffer_device = None

        # apply parallelisms and initialization
        if parallel_dims.pp_enabled:
            if not self.train_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {self.train_spec.name} "
                    f"does not support pipelining"
                )

            # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = self.train_spec.pipelining_fn(
                model,
                world_mesh,
                parallel_dims,
                job_config,
                self.device,
                model_config,
                self.train_spec.parallelize_fn,
                self.train_spec.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    m.init_weights(buffer_device=buffer_device)
                m.train()

            # confirm that user will be able to view loss metrics on the console
            ensure_pp_loss_visible(parallel_dims, job_config, color)
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            model = self.train_spec.parallelize_fn(
                model, world_mesh, parallel_dims, job_config
            )

            model.to_empty(device=init_device)
            with torch.no_grad():
                model.init_weights(buffer_device=buffer_device)
            model.train()

            self.model_parts = [model]

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        muon_kwargs = {"rank": dp_rank, "world_size": dp_degree}

        if parallel_dims.dp_shard_enabled:
            muon_kwargs["dp_mesh"] = world_mesh["dp_shard_cp"]

        # build optimizer after applying parallelisms to the model
        self.optimizers = self.train_spec.build_optimizers_fn(
            self.model_parts, job_config, ft_manager, muon_kwargs
        )
        self.lr_schedulers = self.train_spec.build_lr_schedulers_fn(
            self.optimizers, job_config
        )
        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
        # where it issues a single all-reduce for all parameters at once for better performance
        self.optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(
                self.model_parts
            )
        )
        self.metrics_processor.optimizers = self.optimizers

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0

        self.checkpointer = CheckpointManager(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            job_config=job_config,
            ft_manager=ft_manager,
        )

        if job_config.checkpoint.create_seed_checkpoint:
            assert (
                world_size == 1
            ), "Must create seed checkpoint using a single device, to disable sharding"
            assert (
                job_config.checkpoint.enable_checkpoint
            ), "Must enable checkpointing when creating a seed checkpoint"
            self.checkpointer.save(curr_step=0, force=True)
            logger.info("Created seed checkpoint")
            return

        self.checkpointer.load(step=job_config.checkpoint.load_step)

        self.train_context = dist_utils.get_train_context(
            parallel_dims.loss_parallel_enabled,
            parallelism_config.enable_compiled_autograd,
        )

        logger.info(
            "Trainer initialized. "
            f"Training starts at step {self.step + 1}, "
            f"with local batch size {job_config.training.batch_size}, "
            f"global batch size {job_config.training.global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {job_config.training.seq_len}, "
            f"total steps {job_config.training.steps} "
            f"(warmup {job_config.lr_scheduler.warmup_steps})"
        )

    def next_batch(self, data_iterator: Iterable) -> tuple[torch.Tensor, torch.Tensor]:
        data_load_start = time.perf_counter()
        batch = next(data_iterator)
        input_ids, labels = batch
        self.metrics_processor.ntokens_since_last_log += labels.numel()
        self.metrics_processor.data_loading_times.append(
            time.perf_counter() - data_load_start
        )

        device_type = utils.device_type
        input_ids = input_ids.to(device_type)
        labels = labels.to(device_type)
        return input_ids, labels

    def batch_backward(self, inputs: torch.Tensor, labels: torch.Tensor):
        model_parts = self.model_parts
        world_mesh = self.world_mesh
        parallel_dims = self.parallel_dims

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=world_mesh["cp"],
                cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                cp_no_restore_buffers={inputs, labels},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            with self.train_context(optional_context_parallel_ctx):
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(inputs, target=targets, losses=losses)
                else:
                    self.pp_schedule.step(target=targets, losses=losses)

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            loss = (
                torch.mean(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor([-1.0], device=self.device)
            )
        else:
            # Non-PP forward / backward
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                output = model_parts[0](inputs)
                if isinstance(output, tuple):
                    pred = output[0]
                    aux_loss = output[1]
                else:
                    pred = output
                    aux_loss = None
                loss = self.train_spec.loss_fn(pred, labels)
                if aux_loss is not None:
                    loss += aux_loss / self.gradient_accumulation_steps

                # pred.shape=(bs, seq_len, vocab_size)
                # need to free to before bwd to avoid peaking memory
                del pred
                loss.backward()
        return loss, aux_loss

    def train_step(self, data_iterator: Iterable):
        self.optimizers.zero_grad()

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        model_parts = self.model_parts
        world_mesh = self.world_mesh
        parallel_dims = self.parallel_dims

        for microbatch in range(self.gradient_accumulation_steps):
            inputs, labels = self.next_batch(data_iterator)
            loss, aux_loss = self.batch_backward(inputs, labels)
            self.metrics_processor.accumulated_losses.append(loss.detach())
            if aux_loss is not None:
                self.metrics_processor.accumulated_aux_losses.append(loss.detach())

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=self.world_mesh["pp"] if parallel_dims.pp_enabled else None,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        loss = torch.sum(torch.stack(self.metrics_processor.accumulated_losses))
        self.metrics_processor.accumulated_losses.clear()
        if len(self.metrics_processor.accumulated_aux_losses) > 0:
            aux_loss = torch.sum(torch.stack(self.metrics_processor.accumulated_aux_losses))
            self.metrics_processor.accumulated_aux_losses.clear()

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return

        if (
            parallel_dims.dp_replicate_enabled
            or parallel_dims.dp_shard_enabled
            or parallel_dims.cp_enabled
        ):
            loss = loss.detach()
            global_avg_loss, global_max_loss = (
                dist_utils.dist_mean(loss, world_mesh["dp_cp"]),
                dist_utils.dist_max(loss, world_mesh["dp_cp"]),
            )
            if aux_loss is not None:
                aux_loss = dist_utils.dist_mean(aux_loss, world_mesh["dp_cp"])
        else:
            global_avg_loss = global_max_loss = loss.item()

        extra_log_data = {
            "optim/grad_norm": grad_norm,
        }
        if aux_loss is not None:
            extra_log_data["loss_metrics/aux_loss"] = aux_loss

        color = self.metrics_processor.color
        extra_print_data = (
            f"  {color.green}gradnorm: {grad_norm:7.4f}{color.reset}"
        )

        self.metrics_processor.log(
            self.step, global_avg_loss, global_max_loss, extra_log_data, extra_print_data,
        )

    @record
    def train(self):
        job_config = self.job_config
        with maybe_enable_profiling(
            job_config, global_step=self.step
        ) as torch_profiler, maybe_enable_memory_snapshot(
            job_config, global_step=self.step
        ) as memory_profiler:
            data_iterator = iter(self.dataloader)
            while self.step < job_config.training.steps:
                self.step += 1
                self.gc_handler.run(self.step)
                self.train_step(data_iterator)
                self.checkpointer.save(
                    self.step, force=(self.step == job_config.training.steps)
                )

                # signal the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(
                            seconds=job_config.comm.train_timeout_seconds
                        ),
                        world_mesh=self.world_mesh,
                    )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        self.metrics_processor.close()
        logger.info("Training completed")

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]

    def close(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()


if __name__ == "__main__":
    init_logger()
    config = JobConfig()
    config.maybe_add_custom_args()
    config.parse_args()
    trainer: Optional[Trainer] = None

    try:
        trainer = Trainer(config)
        trainer.train()
    finally:
        if trainer:
            trainer.close()

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Process group destroyed.")
