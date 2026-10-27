"""Fully asynchronous rollout generation.

A persistent background worker keeps up to ``rollout_batch_size`` prompt groups in
flight at all times; each training step only drains already-completed groups from the
data buffer (see ``fully_async_data_buffer.py``). Rollout production and training
consumption run in parallel, so per-iteration wall time moves from
``rollout_time + train_time`` toward ``max(rollout_time, train_time)``.

Selected by ``train_async.py --fully-async``, which requires the class-based
rollout API (the default; incompatible with ``MILES_USE_LEGACY_ROLLOUT_V1=1``).

Evaluation targets whatever ``GenerateState`` ``RolloutManager`` passes via
``RolloutFnEvalInput.generate_state`` (see ``miles/rollout/checkpoint_eval.py``
for how the dedicated-fleet state is built). When unset, eval shares the
rollout engines, pausing producer submissions for the duration of the
(blocking) eval.
"""

import asyncio
import copy
import logging
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
    compute_kv_cache_namespace,
    stamp_kv_cache_namespace,
)
from miles.rollout.filter_hub.base_types import iter_samples  # [fork fix-048]
from miles.rollout.fully_async_data_buffer import (
    DataBuffer,
    DataBufferConstructorInput,
    DataBufferInput,
    DefaultDataBuffer,
    DefaultMultiDataBuffer,
    Group,
    UnusedReason,
    add_data_buffer_arguments,
    first_sample,
)
from miles.rollout.generate_utils.sample_utils import reward_log_summary, sample_text_preview
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils.audit_utils.sample_ownership.recorder import SampleOwnershipRecorder
from miles.utils.function_registry import load_function
from miles.utils.simple_checkpointer import load_simple_checkpoint, save_simple_checkpoint
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class FullyAsyncRolloutFn(BaseRolloutFn):
    """Continuous rollout generation decoupled from training steps.

    The worker runs as a long-lived task on the shared rollout event loop, created
    lazily on the first train call. Which finished groups reach training is the
    data buffer's call (see ``fully_async_data_buffer.py``); this class assembles
    what it hands back into a batch.
    """

    add_arguments = staticmethod(add_data_buffer_arguments)

    def __init__(self, input: RolloutFnConstructorInput):
        super().__init__(input)
        self.args = input.args
        assert self.args.buffer_filter_path is None, "fully-async rollout does not support buffer_filter_path"
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        # default to sample level backfill for fully async rollout
        self._scheduler = make_submission_scheduler(input.args, default="sample")
        assert input.args.async_unused_samples_handler in ("retry", "drop")
        # applied to every group we do not train on; "drop" discards instead of recycling
        self._handle_unused = (
            self._recycle if input.args.async_unused_samples_handler == "retry" else self._drop_unused
        )
        self._sample_filter = load_function(input.args.rollout_sample_filter_path)
        self._worker: asyncio.Task | None = None
        self._worker_error_cancel_targets: set[asyncio.Task] = set()
        self._eval_prompt_dataset_cache: dict = {}
        self._curr_kv_cache_namespace: str | None = None
        self._producer_resumed = asyncio.Event()
        self._producer_resumed.set()
        default_buffer_cls = (
            DefaultMultiDataBuffer if resolve_megatron_config(self.args).is_multi_policy else DefaultDataBuffer
        )
        buffer_cls = load_function(self.args.custom_async_data_buffer_path) or default_buffer_cls
        self._output: DataBuffer = buffer_cls(
            DataBufferConstructorInput(args=self.args, unused_handler_fn=self._handle_unused)
        )
        self._retry_buffer: deque[list[Sample]] = deque()
        # [fork fix-048] begin: a try regenerated on its own gets neither its group's reward nor its own seed
        assert input.args.async_retry_aborted_tries >= 0, "--async-retry-aborted-tries must be 0 or more"
        assert not input.args.async_retry_aborted_tries or not (
            getattr(input.args, "group_rm", False)
            or getattr(input.args, "sglang_enable_deterministic_inference", False)
        ), "--async-retry-aborted-tries runs each try alone: no --group-rm or deterministic inference"
        self._retried_tries = 0
        self._try_rewards: dict[bool, list[float]] = {False: [], True: []}
        # [fork fix-048] end
        self._running_tasks: list[_RunningTask] = []
        # [fork fix-004] begin
        self._single_pass = input.args.async_single_pass
        if self._single_pass:
            self.data_source.epoch_handed_out()  # fails at startup for a source without epochs
        self._pass_finished = asyncio.Event()  # every prompt handed out, nothing in flight or to retry
        self._group_recycled = asyncio.Event()
        # [fork fix-004] end

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        self._curr_kv_cache_namespace = compute_kv_cache_namespace(self.args, input)
        if self._worker is None:
            assert (
                input.weight_version is not None or self.args.debug_rollout_only or self.args.debug_skip_weight_update
            ), (
                "the orchestration publishes the restored weight version before the producer starts, or every "
                "group a checkpoint restored is filtered out as stale"
            )
            self._worker = asyncio.create_task(self._worker_loop())
            self._worker.add_done_callback(self._on_worker_error)
            logger.info("Started fully-async rollout worker")
        return await self._drain(input)

    async def dispose(self) -> None:
        if (worker := self._worker) is None:
            return
        await _end_worker(worker)

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnOutput:
        if input.generate_state is not None:
            results = await run_eval_datasets(
                input.generate_state,
                self._eval_prompt_dataset_cache,
                kv_cache_namespace=compute_kv_cache_namespace(self.args, input),
            )
            return RolloutFnEvalOutput(data=results)

        logger.info("Pausing fully-async producer submissions for shared-engine eval")
        self._producer_resumed.clear()
        try:
            results = await run_eval_datasets(
                self.state,
                self._eval_prompt_dataset_cache,
                kv_cache_namespace=compute_kv_cache_namespace(self.args, input),
            )
        finally:
            self._producer_resumed.set()
            logger.info("Resumed fully-async producer submissions after eval")
        return RolloutFnEvalOutput(data=results)

    # -------------------------- producer --------------------------

    def _max_in_flight_groups(self) -> int:
        if (x := self.args.async_max_concurrent_samples) is not None:
            # Whole groups are submitted, so the sample budget floors to a group count.
            return max(1, x // self.args.n_samples_per_prompt)
        return self.args.rollout_batch_size

    def _submit_one_group(self) -> asyncio.Task:
        samples = [self._retry_buffer.popleft()] if self._retry_buffer else self.data_source.get_samples(1)
        stamp_kv_cache_namespace(samples, namespace=self._curr_kv_cache_namespace)
        self._scheduler.on_submit(samples)
        [prompt_group] = samples
        task = asyncio.create_task(self._generate_group(prompt_group))
        self._running_tasks.append(_RunningTask(prompt_group=prompt_group, task=task))
        return task

    async def _generate_group(self, prompt_group: list[Sample]) -> DataBufferInput:
        if self.args.async_retry_aborted_tries:  # [fork fix-048]
            return DataBufferInput(prompt_group=prompt_group, group=await self._generate_tries(prompt_group))
        result = await generate_and_rm_group(
            self.state,
            prompt_group,
            sampling_params=self.state.sampling_params.copy(),
            evaluation=False,
            sample_done_callback=self._scheduler.sample_done_callback,
        )
        return DataBufferInput(prompt_group=prompt_group, group=result)

    # [fork fix-048] begin
    async def _generate_tries(self, prompt_group: list[Sample]) -> list:
        """Run each try of the group on its own, so an aborted one is regenerated at once while its siblings go on."""
        tasks = [asyncio.create_task(self._generate_try(sample)) for sample in prompt_group]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            # as generate_and_rm_group: no try outlives its group
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _generate_try(self, prompt_sample: Sample) -> Sample | list[Sample]:
        attempt = prompt_sample
        for retry in range(self.args.async_retry_aborted_tries + 1):
            if retry:
                logger.info(f"Regenerating aborted try: index={prompt_sample.index}, retry {retry}")
                [attempt] = _copy_and_reset_for_retry([prompt_sample])
                stamp_kv_cache_namespace(attempt, namespace=self._curr_kv_cache_namespace)  # the reset cleared it
                self._scheduler.on_submit([[attempt]])  # the retry frees its own in-flight slot when done
            [item] = await generate_and_rm_group(
                self.state,
                [attempt],
                sampling_params=self.state.sampling_params.copy(),
                evaluation=False,
                sample_done_callback=self._scheduler.sample_done_callback,
            )
            if not _has_aborted(item):
                rewards = (sample.get_reward_value(self.args) for sample in iter_samples([item]))
                self._try_rewards[bool(retry)].extend(r for r in rewards if r is not None)
                break
        self._retried_tries += retry
        return item

    def _try_metrics(self) -> dict[str, float]:
        """Tries regenerated so far, and the mean reward of tries finished since the last step, first try vs retry."""
        prefix = "rollout/fully_async/"
        metrics = {f"{prefix}retried_tries": self._retried_tries}
        for retried, name in ((False, "first_try_reward_mean"), (True, "retried_try_reward_mean")):
            if rewards := self._try_rewards[retried]:
                metrics[prefix + name] = sum(rewards) / len(rewards)
        self._try_rewards = {False: [], True: []}
        return metrics

    # [fork fix-048] end

    async def _worker_loop(self) -> None:
        active: set[asyncio.Task] = set()
        while True:
            await self._producer_resumed.wait()
            # [fork fix-004] begin: once the pass is out, wait on the groups only (the scheduler's
            # sample wakeup is armed only by a failed capacity check) and then idle until a recycle
            if self._pass_handed_out():
                if not active:
                    await self._wait_for_recycled_group()
                    continue
                done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            else:
                while not self._pass_handed_out() and self._scheduler.has_capacity(
                    pending_groups=len(active), group_budget=self._max_in_flight_groups()
                ):
                    active.add(self._submit_one_group())
                done, active = await self._scheduler.wait_for_progress(active)
            # [fork fix-004] end
            for task in done:
                prompt_group = next(x.prompt_group for x in self._running_tasks if x.task is task)
                entry = self._collect_group_result(task, prompt_group)
                await self._output.put(entry)
                self._running_tasks = [x for x in self._running_tasks if x.task is not task]

    # [fork fix-004] begin
    def _pass_handed_out(self) -> bool:
        """--async-single-pass: every prompt of the data source's current epoch is out and none waits for a retry."""
        return self._single_pass and not self._retry_buffer and self.data_source.epoch_handed_out()

    async def _wait_for_recycled_group(self) -> None:
        logger.info("--async-single-pass: every prompt has been handed out once and has finished")
        self._group_recycled.clear()
        self._pass_finished.set()
        await self._group_recycled.wait()
        self._pass_finished.clear()

    # [fork fix-004] end

    def _collect_group_result(self, task: asyncio.Task, prompt_group: list[Sample]) -> DataBufferInput:
        if not task.cancelled():
            return task.result()

        logger.warning(
            "Rollout group was cancelled; marking samples aborted: indices=%s",
            [sample.index for sample in prompt_group],
        )
        return DataBufferInput(
            prompt_group=prompt_group,
            group=[replace(sample, status=Sample.Status.ABORTED) for sample in prompt_group],
        )

    # -------------------------- consumer --------------------------

    def _on_worker_error(self, worker: asyncio.Task) -> None:
        # The worker loop never returns normally, so any completion fails the waiting step.
        logger.error("fully-async rollout worker ended", exc_info=None if worker.cancelled() else worker.exception())
        for target in list(self._worker_error_cancel_targets):
            target.cancel()

    async def _drain(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        assert not self._worker.done(), "the fully-async rollout worker has ended"
        self._worker_error_cancel_targets.add(asyncio.current_task())
        get = self._get_from_pass if self._single_pass else self._output.get  # [fork fix-004]
        entries = await get(
            current_version=input.weight_version,
            num_groups=args.rollout_batch_size,
            trainer_model_id=input.trainer_model_id,
        )
        self._worker_error_cancel_targets.discard(asyncio.current_task())

        data: list[Group] = []
        do_print = True

        for entry in entries:
            assert len(entry.group) == args.n_samples_per_prompt

            if do_print:
                sample = first_sample(entry.group)
                logger.info(
                    "First rollout sample: text_preview=%s, label=%s, reward_summary=%s",
                    sample_text_preview(sample),
                    str(sample.label)[:100],
                    reward_log_summary(sample.reward),
                )
                do_print = False

            data.append(entry.group)

        sample = first_sample(data[-1])
        logger.info(
            "Finish rollout: text_preview=%s, label=%s, reward_summary=%s",
            sample_text_preview(sample),
            str(sample.label)[:100],
            reward_log_summary(sample.reward),
        )

        data.sort(key=lambda group: first_sample(group).index)

        before_filter = SampleOwnershipRecorder.flatten_samples(data)
        if self._sample_filter is not None:
            self._sample_filter(args, data)
        SampleOwnershipRecorder.log_dropped_groups(
            args=args, before=before_filter, after=data, reason="rollout_sample_filter"
        )

        metrics = self._output.get_metrics(input.trainer_model_id)
        if self.args.async_retry_aborted_tries:  # [fork fix-048]
            metrics = {**metrics, **self._try_metrics()}
        return RolloutFnTrainOutput(samples=data, metrics=metrics)

    # [fork fix-004] begin
    async def _get_from_pass(self, *, num_groups: int, **context) -> list[DataBufferInput]:
        """--async-single-pass: ``num_groups`` groups, or once the pass has finished, the groups it left."""
        while True:
            get = asyncio.create_task(self._output.get(num_groups=num_groups, **context))
            finished = asyncio.create_task(self._pass_finished.wait())
            try:
                await asyncio.wait({get, finished}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                finished.cancel()
                get.cancel()  # no-op once it returned; get() takes its groups without awaiting in between
                await asyncio.wait({get, finished})
            if not get.cancelled():
                return get.result()
            if not self._pass_finished.is_set():
                continue  # a recycled group reopened the pass
            if not (ready := self._output.num_ready(**context)):
                raise RuntimeError(
                    "--async-single-pass: every prompt was handed out once and has finished, and no group is "
                    "left for this rollout step; --num-rollout x --rollout-batch-size exceeds one pass over the data"
                )
            logger.info(f"--async-single-pass: the pass has finished; this step takes its last {ready} groups")
            return await self._output.get(num_groups=ready, **context)

    # [fork fix-004] end

    def _recycle(self, prompt_group: list[Sample], reason: UnusedReason) -> None:
        for sample in prompt_group:
            sample.reset_for_retry()
        self._retry_buffer.append(prompt_group)
        self._group_recycled.set()  # [fork fix-004]

    def _drop_unused(self, prompt_group: list[Sample], reason: UnusedReason) -> None:
        if reason is UnusedReason.UNRESTORED:  # [fork fix-049] never trained on, so regenerate it under "drop" too
            return self._recycle(prompt_group, reason)
        SampleOwnershipRecorder.log_dropped_samples(args=self.args, samples=prompt_group, reason=reason.value)

    def save(self, directory: Path) -> None:
        state = _FullyAsyncRolloutState(
            retry_buffer=list(self._retry_buffer),
            running=[_copy_and_reset_for_retry(x.prompt_group) for x in self._running_tasks],
            output=self._output.state_dict(),
        )
        save_simple_checkpoint(directory=directory, data=state)

    def load(self, directory: Path) -> None:
        assert self._worker is None, "restore the fully async rollout state before producer startup"
        state: _FullyAsyncRolloutState = load_simple_checkpoint(directory=directory)
        assert not self._retry_buffer, "restore the fully async rollout state before any group was recycled"
        self._retry_buffer = deque([*state.retry_buffer, *state.running])
        self._output.load_state_dict(state.output)


@dataclass(frozen=True)
class _RunningTask:
    prompt_group: list[Sample]
    task: asyncio.Task


@dataclass(frozen=True)
class _FullyAsyncRolloutState:
    retry_buffer: list[list[Sample]]
    running: list[list[Sample]]
    output: Any


def _copy_and_reset_for_retry(prompt_group: list[Sample]) -> list[Sample]:
    # [fork fix-046] reset a shallow copy first so the deep copy skips the generated outputs
    copied = [copy.copy(sample) for sample in prompt_group]
    for sample in copied:
        sample.reset_for_retry()
    return copy.deepcopy(copied)


# [fork fix-048]
def _has_aborted(item: Sample | list[Sample]) -> bool:
    return any(sample.status == Sample.Status.ABORTED for sample in iter_samples([item]))


async def _end_worker(worker: asyncio.Task) -> None:
    worker.cancel()
    await asyncio.wait({worker})
