# [fork fix-004] --async-single-pass: each prompt handed out once, the step waits for every group

import asyncio
from collections import Counter

import pytest

from tests.fast.rollout.test_fully_async_rollout import N_SAMPLES_PER_PROMPT, make_args, make_fn, train_input

from miles.rollout.data_source import RolloutDataSource
from miles.utils.types import Sample


class _Dataset:
    def __init__(self, num_prompts: int):
        self.samples = [Sample(prompt=f"task {i}", label="ok") for i in range(num_prompts)]

    def __len__(self) -> int:
        return len(self.samples)

    def shuffle(self, epoch_id: int) -> None:
        pass


def make_data_source(args, num_prompts: int) -> RolloutDataSource:
    """The real data source (its epoch wrap is what the mode must not reach), without the tokenizer load."""
    source = object.__new__(RolloutDataSource)
    source.args = args
    source.epoch_id = source.sample_group_index = source.sample_index = source.sample_offset = 0
    source.metadata = {}
    source.dataset = _Dataset(num_prompts)
    return source


def single_pass_args(**overrides):
    return make_args(async_single_pass=True, rollout_shuffle=False, **overrides)


def prompts_of(groups) -> list[str]:
    return [group[0].prompt for group in groups]


class Generate:
    """Completes every group with reward 1; prompts in ``hold`` wait for ``release``."""

    def __init__(self, hold=(), status_of=None):
        self.hold = set(hold)
        self.release = asyncio.Event()
        self.status_of = status_of or (lambda prompt, attempt: Sample.Status.COMPLETED)
        self.attempts: Counter[str] = Counter()

    async def __call__(self, state, group, sampling_params, evaluation=False, sample_done_callback=None):
        prompt = group[0].prompt
        self.attempts[prompt] += 1
        if prompt in self.hold:
            await self.release.wait()
        await asyncio.sleep(0)
        for sample in group:
            sample.status = self.status_of(prompt, self.attempts[prompt])
            sample.reward = 1
            if sample_done_callback is not None:
                sample_done_callback()
        return group


async def test_an_eval_hands_each_prompt_out_once_and_waits_for_the_slow_ones(monkeypatch):
    """Run 4's shape: one step of all prompts, fewer slots than prompts, one slow task."""
    args = single_pass_args(rollout_batch_size=5, async_max_concurrent_samples=2 * N_SAMPLES_PER_PROMPT)
    source = make_data_source(args, num_prompts=5)
    generate = Generate(hold={"task 0"})
    fn = make_fn(monkeypatch, args, source, generate=generate)

    step = asyncio.create_task(fn(train_input(rollout_id=0)))
    await asyncio.sleep(0.05)
    # every other task finished long ago; nothing was handed out a second time and the step still waits
    assert sorted(generate.attempts) == [f"task {i}" for i in range(5)]
    assert set(generate.attempts.values()) == {1}
    assert source.epoch_id == 0
    assert not step.done()

    generate.release.set()
    output = await asyncio.wait_for(step, timeout=5)

    assert sorted(prompts_of(output.samples)) == [f"task {i}" for i in range(5)]
    assert all(len(group) == N_SAMPLES_PER_PROMPT for group in output.samples)
    assert set(generate.attempts.values()) == {1}
    assert source.epoch_id == 0


async def test_without_the_flag_the_producer_wraps_into_the_next_epoch(monkeypatch):
    """The behavior the flag turns off: fast prompts' second groups fill the step."""
    args = make_args(
        rollout_batch_size=5, async_max_concurrent_samples=2 * N_SAMPLES_PER_PROMPT, rollout_shuffle=False
    )
    source = make_data_source(args, num_prompts=5)
    generate = Generate(hold={"task 4"})
    fn = make_fn(monkeypatch, args, source, generate=generate)

    output = await asyncio.wait_for(fn(train_input(rollout_id=0)), timeout=5)

    assert "task 4" not in prompts_of(output.samples)
    assert source.epoch_id >= 1
    generate.release.set()
    await fn.dispose()


async def test_a_dropped_group_ends_the_step_with_the_groups_that_are_left(monkeypatch):
    """A filtered group never reaches the buffer; the step must not wait for it forever."""
    args = single_pass_args(rollout_batch_size=4)
    source = make_data_source(args, num_prompts=4)

    def aborted(prompt, attempt):
        return Sample.Status.ABORTED if prompt == "task 2" else Sample.Status.COMPLETED

    fn = make_fn(monkeypatch, args, source, generate=Generate(status_of=aborted))

    output = await asyncio.wait_for(fn(train_input(rollout_id=0)), timeout=5)

    assert sorted(prompts_of(output.samples)) == ["task 0", "task 1", "task 3"]
    assert output.metrics["rollout/fully_async/aborted_groups_filtered"] == 1


async def test_a_retried_group_is_still_part_of_the_pass(monkeypatch):
    args = single_pass_args(rollout_batch_size=4, async_unused_samples_handler="retry")
    source = make_data_source(args, num_prompts=4)

    def abort_first(prompt, attempt):
        return Sample.Status.ABORTED if prompt == "task 2" and attempt == 1 else Sample.Status.COMPLETED

    generate = Generate(status_of=abort_first)
    fn = make_fn(monkeypatch, args, source, generate=generate)

    output = await asyncio.wait_for(fn(train_input(rollout_id=0)), timeout=5)

    assert sorted(prompts_of(output.samples)) == [f"task {i}" for i in range(4)]
    assert generate.attempts == Counter({"task 0": 1, "task 1": 1, "task 2": 2, "task 3": 1})
    assert source.epoch_id == 0


async def test_steps_split_one_pass_and_a_step_past_it_fails(monkeypatch):
    args = single_pass_args(rollout_batch_size=3)
    source = make_data_source(args, num_prompts=5)
    fn = make_fn(monkeypatch, args, source, generate=Generate())

    first = await asyncio.wait_for(fn(train_input(rollout_id=0)), timeout=5)
    second = await asyncio.wait_for(fn(train_input(rollout_id=1)), timeout=5)

    assert len(first.samples) == 3 and len(second.samples) == 2
    assert sorted(prompts_of(first.samples + second.samples)) == [f"task {i}" for i in range(5)]
    with pytest.raises(RuntimeError, match="exceeds one pass"):
        await asyncio.wait_for(fn(train_input(rollout_id=2)), timeout=5)
