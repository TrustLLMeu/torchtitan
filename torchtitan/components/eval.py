import enum
from multiprocessing import get_context
import os
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from torchtitan.config_manager import JobConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger


def _get_batch(data_iterator, device_type):
    # get batch
    data_load_start = time.perf_counter()
    batch = next(data_iterator)
    input_ids, labels = batch
    batch_ntokens = labels.numel()
    data_loading_time = time.perf_counter() - data_load_start

    input_ids = input_ids.to(device_type)
    labels = labels.to(device_type)

    return input_ids, labels, data_loading_time, batch_ntokens


class EvalModel:
    pass


class IntervalType(enum.Enum):
    SECONDS = enum.auto()
    STEPS = enum.auto()


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class Terminate:
    pass


class EvalDone:
    pass


def eval_mp(recv, send, cpu_only):
    init_logger()
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 2)
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    if not cpu_only:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group()
    try:
        while True:
            logger.debug("Evaluation background process is done.")
            send.put(EvalDone())
            logger.debug("Wait for the new state_dict.")
            obj = recv.get()
            logger.debug("Received the new state_dict.")
            if isinstance(obj, Terminate):
                logger.info("Terminating the evaluation background process.")
                return
            assert isinstance(obj, tuple)
            begin = time.monotonic()
            state, evaluation_id = obj
            # TODO replace with eval function
            # dcp.save(state, checkpoint_id=checkpoint_id)
            logger.info(
                "Finish evaluating in the background process in "
                f"{time.monotonic() - begin:.2f} seconds."
            )
    finally:
        logger.info("Destroying the process group.")
        dist.destroy_process_group()


class EvaluationManager:
    def __init__(
        self,
        job_config: JobConfig,
    ) -> None:
        eval_config = job_config.evaluation
        self.enable_evaluation = eval_config.enable_evaluation

        if not self.enable_evaluation:
            return

        self.cpu_only = eval_config.cpu_only
        # self.folder = os.path.join(job_config.job.dump_folder, eval_config.folder)
        self.interval_type = (
            IntervalType.SECONDS
            if eval_config.interval_type == "seconds"
            else IntervalType.STEPS
        )
        self.interval = eval_config.interval
        self.begin_time = 0
        self.time_sync_work = None
        self.time_sync_result = None
        self.pg = dist.new_group(backend="gloo")

        self.mp = None
        self.cpu_offload_state_dict = None
        async_mode = eval_config.async_mode.lower()
        if async_mode == AsyncMode.DISABLED:
            self.async_mode = AsyncMode.DISABLED
        elif async_mode == AsyncMode.ASYNC:
            self.async_mode = AsyncMode.ASYNC
            self.async_future = None
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
            ctx = get_context("spawn")
            self.mp_queue_send = ctx.Queue()
            self.mp_queue_recv = ctx.Queue()
            self.mp = ctx.Process(
                target=eval_mp,
                args=(
                    self.mp_queue_send,
                    self.mp_queue_recv,
                ),
                daemon=True,
            )
            self.mp.start()
            self.staging = False
            self.staging_id = None
            self.staging_stream = torch.cuda.Stream()
        else:
            raise ValueError(f"Unkown evaluation async_mode {eval_config.async_mode}")

        logger.info("Evaluation active")

    def __del__(self):
        if self.enable_evaluation and self.mp and self.mp.is_alive():
            self.mp_queue_send.put(Terminate())
            self.mp.join()

    def reset(self) -> None:
        self.begin_time = time.monotonic()

    def _create_evaluation_id(self, step: int) -> str:
        return f"evaluation_step-{step}"

    def _eval_last_step(
            self,
            curr_step: int,
            model,
            data_iterator,
            loss_fn,
            eval_context,
            parallel_dims,
            world_mesh,
            dp_mesh,
            pp_schedule,
            eval_steps,
            metric_logger,
            logger,
    ) -> None:
        logger.info(f"Running an evaluation at last step, step {curr_step}.")

        self.evaluate(
            curr_step,
            model,
            data_iterator,
            loss_fn,
            eval_context,
            parallel_dims,
            world_mesh,
            dp_mesh,
            pp_schedule,
            eval_steps,
            metric_logger,
            logger,
        )
        self.reset()

    def _should_evaluate(self, curr_step: int, force: bool = False) -> bool:
        if not self.enable_evaluation:
            return False

        if not force:
            if self.interval_type == IntervalType.STEPS and not (
                curr_step % self.interval == 0
            ):
                return False
            if self.interval_type == IntervalType.SECONDS:
                time_sync_result = (time.monotonic() - self.begin_time) >= self.interval
                self.time_sync_result = torch.tensor(int(time_sync_result))
                if self.time_sync_work is None:
                    self.time_sync_work = dist.all_reduce(
                        self.time_sync_result, group=self.pg, async_op=True
                    )
                    return False
                elif curr_step % 5 == 4:
                    self.time_sync_work.wait()
                    self.time_sync_work = None
                    time_sync_result = self.time_sync_result.item()
                    self.time_sync_result = None
                    if time_sync_result == 0:
                        return False
                else:
                    return False

        if self.time_sync_work:
            self.time_sync_work.wait()
            self.time_sync_work = None
            self.time_sync_result = None

        return True

    def _async_wait(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            logger.debug(
                f"Waiting for the evaluation background process to finish, {time.monotonic()=}.:.2f"
            )
            if not self.mp.is_alive():
                raise RuntimeError("The evaluation background process is dead.")
            _ = self.mp_queue_recv.get()
        elif self.async_mode == AsyncMode.ASYNC:
            if self.async_future is not None:
                self.async_future.result()

    def _async_with_pinned_memory(self, evaluation_id: str) -> None:
        try:
            from torch.distributed._state_dict_utils import (
                _copy_state_dict,
                _create_cpu_state_dict,
            )
        except ImportError as e:
            raise ImportError(
                "Please install the latest PyTorch nightly to use async evaluation with pinned memory."
            ) from e
        state_dict = dcp.state_dict_saver._stateful_to_state_dict(self.states)
        # TODO somehow make this loadable on CPU, ideally without initializing a new model
        if self.cpu_offload_state_dict is None:
            logger.debug(f"Preparing the CPU memory, {time.monotonic()=}.:.2f")
            self.cpu_offload_state_dict = _create_cpu_state_dict(
                state_dict, pin_memory=True, share_memory=True
            )

        logger.debug(f"Staging the state_dict, {time.monotonic()=}.:.2f")
        with torch.cuda.stream(self.staging_stream):
            self.cpu_offload_state_dict = _copy_state_dict(
                state_dict,
                self.cpu_offload_state_dict,
                non_blocking=True,
            )
            self.staging = True
            self.staging_id = evaluation_id

    def maybe_wait_for_staging(self) -> None:
        if (
            self.enable_evaluation
            and self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
            and self.staging
        ):
            if not self.staging_stream.query():
                self.staging_stream.synchronize()

            def sync_func():
                # TODO maybe need to add CPU model here
                self.mp_queue_send.put_nowait(
                    (self.cpu_offload_state_dict, self.staging_id)
                )

            # This may be a faster way to do zero-overhead checkpointing staging
            # checkpointing but we need more thorough investigation before
            # swithing to this method.
            # self.my_thread = threading.Thread(target=func).start()
            sync_func()
            self.staging = False

    def _batch_loss(
            self,
            model_parts,
            input_ids,
            labels,
            train_spec,
            eval_context,
            parallel_dims,
            world_mesh,
            pp_schedule,
            has_first_stage,
            has_last_stage,
            device,
            job_config,
    ):
        with torch.no_grad():
            # apply context parallelism if cp is enabled
            # ensure CP handles the separate freqs_cis buffer for each pp stage
            optional_context_parallel_ctx = (
                dist_utils.create_context_parallel_ctx(
                    cp_mesh=world_mesh["cp"],
                    cp_buffers=[input_ids, labels] + [m.freqs_cis for m in model_parts],
                    cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                    cp_no_restore_buffers={input_ids, labels},
                    cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                )
                if parallel_dims.cp_enabled
                else None
            )

            if parallel_dims.pp_enabled:
                # Pipeline Parallel forward / backward inside step() call
                with eval_context(optional_context_parallel_ctx):
                    targets, losses = (labels, []) if has_last_stage else (None, None)
                    if has_first_stage:
                        pp_schedule.step(input_ids, target=targets, losses=losses)
                    else:
                        pp_schedule.step(target=targets, losses=losses)

                # accumulate losses across pipeline microbatches
                # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                loss = (
                    torch.mean(torch.stack(losses)).to(device)
                    if has_last_stage
                    else torch.tensor([-1.0], device=device)
                )
            else:
                model = model_parts[0]

                # Non-PP forward
                with eval_context(optional_context_parallel_ctx):
                    pred = model(input_ids)
                    loss = train_spec.loss_fn(pred, labels)
                    # pred.shape=(bs, seq_len, vocab_size)
                    # need to free to before bwd to avoid peaking memory
                    del pred
        return loss

    def evaluate(
            self,
            curr_step,
            model_parts,
            data_iterator,
            train_spec,
            eval_context,
            parallel_dims,
            world_mesh,
            dp_mesh,
            pp_schedule,
            has_first_stage,
            has_last_stage,
            device,
            job_config,
            eval_steps,
            device_memory_monitor,
            gpu_peak_flops,
            num_flop_per_token,
            metric_logger,
            logger,
    ):
        """Note that the `data_iterator` needs to be reset from the
        outside if `eval_steps < sum(1 for _ in data_iterator)` in order
        to iterate through the same data each time.
        """
        for model_part in model_parts:
            model_part.eval()

        color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color

        losses = []
        ntokens_since_last_log = 0
        data_loading_times = []
        time_last_log = time.perf_counter()
        device_memory_monitor.reset_peak_stats()

        for eval_step in range(eval_steps):
            (
                input_ids,
                labels,
                data_loading_time,
                batch_ntokens,
            ) = _get_batch(data_iterator)
            ntokens_since_last_log += batch_ntokens
            data_loading_times.append(data_loading_time)

            loss = self._batch_loss(
                model_parts,
                input_ids,
                labels,
                train_spec,
                eval_context,
                parallel_dims,
                world_mesh,
                pp_schedule,
                has_first_stage,
                has_last_stage,
                device,
                job_config,
            )
            losses.append(loss.item())

        avg_loss, max_loss = sum(losses) / len(losses), max(losses)
        if parallel_dims.dp_enabled:
            global_avg_loss, global_max_loss = (
                dist_utils.dist_mean(avg_loss, dp_mesh),
                dist_utils.dist_max(max_loss, dp_mesh),
            )
        else:
            global_avg_loss, global_max_loss = avg_loss, max_loss

        time_delta = time.perf_counter() - time_last_log

        # tokens per second per device, abbreviated as tps
        tps = ntokens_since_last_log / (
            time_delta * parallel_dims.non_data_parallel_size
        )
        # model FLOPS utilization
        # For its definition and calculation, please refer to the PaLM paper:
        # https://arxiv.org/abs/2204.02311
        mfu = 100 * num_flop_per_token * tps / gpu_peak_flops

        time_end_to_end = time_delta / eval_steps
        time_data_loading = sum(data_loading_times) / len(data_loading_times)
        time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

        device_mem_stats = device_memory_monitor.get_peak_stats()

        metrics = {
            "eval/loss_metrics/global_avg_loss": global_avg_loss,
            "eval/loss_metrics/global_max_loss": global_max_loss,
            "eval/throughput(tps)": tps,
            "eval/mfu(%)": mfu,
            "eval/time_metrics/end_to_end(s)": time_end_to_end,
            "eval/time_metrics/data_loading(s)": time_data_loading,
            "eval/time_metrics/data_loading(%)": time_data_loading_pct,
            "eval/memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "eval/memory/max_active(%)": device_mem_stats.max_active_pct,
            "eval/memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "eval/memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
            "eval/memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "eval/memory/num_ooms": device_mem_stats.num_ooms,
        }
        metric_logger.log(metrics, step=curr_step)
        logger.info(
            f"evaluation:  "
            f"{color.cyan}step: {curr_step:2}  "
            f"{color.green}loss: {global_avg_loss:7.4f}  "
            f"{color.yellow}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}tps: {round(tps):,}  "
            f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
        )

        device_memory_monitor.reset_peak_stats()
        for model_part in model_parts:
            model_part.train()

        return global_avg_loss

    def run(
            self,
            curr_step: int,
            model,
            data_iterator,
            loss_fn,
            eval_context,
            parallel_dims,
            world_mesh,
            dp_mesh,
            pp_schedule,
            eval_steps,
            metric_logger,
            logger,
            force: bool = False,
    ) -> None:
        """
        force = True will force the evaluation to be ran, even if the interval
        has not been reached.
        This only happens when train_state.step == job_config.training.steps, or
        for initial evaluation.
        """
        if not self._should_evaluate(curr_step, force):
            return

        begin = time.monotonic()
        evaluation_id = self._create_evaluation_id(curr_step)
        self._async_wait()
        if force:
            self._eval_last_step(
                curr_step,
                model,
                data_iterator,
                loss_fn,
                eval_context,
                parallel_dims,
                world_mesh,
                dp_mesh,
                pp_schedule,
                eval_steps,
                metric_logger,
                logger,
            )
        elif self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self._async_with_pinned_memory(evaluation_id)
        elif self.async_mode == AsyncMode.ASYNC:
            # TODO async eval function
            # self.async_future = dcp.async_save(
            #     self.states, checkpoint_id=checkpoint_id, process_group=self.pg
            # )
            assert False, "not supported"
        else:
            self.evaluate(
                curr_step,
                model,
                data_iterator,
                loss_fn,
                eval_context,
                parallel_dims,
                world_mesh,
                dp_mesh,
                pp_schedule,
                eval_steps,
                metric_logger,
                logger,
            )
        self.reset()

        logger.info(
            "Finished running the evaluation (or staging if async is enabled)"
            f"in {time.monotonic() - begin:.2f} seconds."
        )
