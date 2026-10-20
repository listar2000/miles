import asyncio
import dataclasses
import json
import threading
from pathlib import Path

import numpy as np
import pytest
from miles_ext.rollout import group_log
from tests.fast.rollout.test_fully_async_rollout import FakeDataSource, make_args, make_fn, make_group

from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput

BUFFER = "miles_ext.rollout.group_log.GroupLogDataBuffer"


def _fn(monkeypatch, save: Path, load: Path | None = None, **overrides):
    args = make_args(
        save=str(save),
        load=str(load or save),
        rollout_batch_size=1,
        custom_async_data_buffer_path=BUFFER,
        async_checkpoint_buffer="log",
        **overrides,
    )
    return make_fn(monkeypatch, args, FakeDataSource())


async def _put(buffer, index: int, max_expert: int = 5) -> None:
    await buffer.put(DataBufferInput(prompt_group=make_group(index), group=_r3_group(index, max_expert)))
    await asyncio.gather(*buffer._writing)


def _r3_group(index: int, max_expert: int) -> list:
    group = make_group(index)
    for i, sample in enumerate(group):
        sample.tokens = list(range(10 + i))
        sample.rollout_log_probs = [-0.1] * (10 + i)
        sample.rollout_routed_experts = np.full((9 + i, 3, 2), max_expert, dtype=np.int32)
    return group


def _files(save: Path) -> set[str]:
    return {p.name for p in (save / group_log.GROUPS_DIRNAME).glob("*.safetensors")}


def _stored_dtype(path: Path) -> str:
    header = path.read_bytes()
    return json.loads(header[8 : 8 + int.from_bytes(header[:8], "little")])["rollout_routed_experts.0"]["dtype"]


@pytest.mark.parametrize("max_expert,stored", [(383, "U16"), (70_000, "I32")])
async def test_resume_restores_buffered_groups_exactly(monkeypatch, tmp_path: Path, max_expert, stored):
    """A save records a manifest; a resume from another run's save reads the groups back and keeps a copy."""
    old, new = tmp_path / "old", tmp_path / "new"
    fn = _fn(monkeypatch, old)
    await _put(fn._output, 1, max_expert)
    group = _r3_group(1, max_expert)
    fn.save(old / "state")
    [name] = _files(old)
    assert _stored_dtype(old / group_log.GROUPS_DIRNAME / name) == stored

    restored = _fn(monkeypatch, new, load=old)
    restored.load(old / "state")
    [entry] = await asyncio.wait_for(restored._output.get(num_groups=1), timeout=5)
    for got, want in zip(entry.group, group, strict=True):
        assert got.rollout_routed_experts.dtype == np.int32
        assert np.array_equal(got.rollout_routed_experts, want.rollout_routed_experts)
        assert dataclasses.replace(got, rollout_routed_experts=None) == dataclasses.replace(
            want, rollout_routed_experts=None
        )
    assert _files(new) == {name}


async def test_a_group_without_its_file_is_regenerated(monkeypatch, tmp_path: Path):
    fn = _fn(monkeypatch, tmp_path)  # unused samples "drop", as in the recipes
    await _put(fn._output, 1)
    fn.save(tmp_path / "state")
    for path in (tmp_path / group_log.GROUPS_DIRNAME).glob("*.safetensors"):
        path.unlink()

    restored = _fn(monkeypatch, tmp_path)
    restored.load(tmp_path / "state")
    assert restored._output.num_ready() == 0
    [prompts] = restored._retry_buffer
    assert [s.index for s in prompts] == [10, 11] and all(s.response == "" for s in prompts)


async def test_save_deletes_only_group_files_no_checkpoint_names(monkeypatch, tmp_path: Path):
    fn = _fn(monkeypatch, tmp_path)
    buffer = fn._output

    await _put(buffer, 1)
    await _put(buffer, 2)
    fn.save(tmp_path / "rollout" / "0" / "generate_rollout")  # published, names groups 1 and 2
    named = await asyncio.wait_for(buffer.get(num_groups=2), timeout=5)
    await _put(buffer, 3)
    await asyncio.wait_for(buffer.get(num_groups=1), timeout=5)  # group 3: consumed, never named
    await _put(buffer, 4)
    fn.save(tmp_path / "unpublished")
    buffer._writer.shutdown(wait=True)
    live = await asyncio.wait_for(buffer.get(num_groups=1), timeout=5)
    assert _files(tmp_path) == {f"{e.uid}.safetensors" for e in (*named, *live)}


async def test_a_group_is_not_visible_until_its_file_is_written(monkeypatch, tmp_path: Path):
    """put returns before the write, so the producer never waits for storage; a save meanwhile keeps the prompts."""
    started, release = threading.Event(), threading.Event()
    write = group_log.write_group
    monkeypatch.setattr(
        group_log, "write_group", lambda group, path: (started.set(), release.wait(5), write(group, path))
    )
    fn = _fn(monkeypatch, tmp_path)
    buffer = fn._output

    await buffer.put(DataBufferInput(prompt_group=make_group(1), group=_r3_group(1, 5)))
    await asyncio.to_thread(started.wait, 5)
    assert buffer.num_ready() == 0
    fn.save(tmp_path / "state")
    release.set()
    await asyncio.gather(*buffer._writing)
    assert buffer.num_ready() == 1 and len(_files(tmp_path)) == 1

    restored = _fn(monkeypatch, tmp_path)
    restored.load(tmp_path / "state")
    [prompts] = restored._retry_buffer
    assert [s.index for s in prompts] == [10, 11]


async def test_single_pass_put_returns_once_the_group_is_ready(monkeypatch, tmp_path: Path):
    """The pass may end once put returns; until then the group is still a running group, not a manifest entry."""
    started, release = threading.Event(), threading.Event()
    write = group_log.write_group
    monkeypatch.setattr(
        group_log, "write_group", lambda group, path: (started.set(), release.wait(5), write(group, path))
    )
    args = make_args(save=str(tmp_path), rollout_batch_size=1, async_single_pass=True, async_checkpoint_buffer="log")
    buffer = group_log.GroupLogDataBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=print))

    put = asyncio.create_task(buffer.put(DataBufferInput(prompt_group=make_group(1), group=_r3_group(1, 5))))
    await asyncio.to_thread(started.wait, 5)
    assert not put.done() and buffer.state_dict().entries == []
    release.set()
    await asyncio.wait_for(put, timeout=5)
    assert buffer.num_ready() == 1


async def test_a_default_buffer_checkpoint_loads_and_saves_again(monkeypatch, tmp_path: Path):
    fn = _fn(monkeypatch, tmp_path)
    fn._output.load_state_dict([DataBufferInput(prompt_group=make_group(1), group=_r3_group(1, 5))])
    fn.save(tmp_path / "state")  # the group has no file, so a later resume regenerates it

    restored = _fn(monkeypatch, tmp_path)
    restored.load(tmp_path / "state")
    [prompts] = restored._retry_buffer
    assert [s.index for s in prompts] == [10, 11]


async def test_a_log_checkpoint_refuses_another_mode(monkeypatch, tmp_path: Path):
    fn = _fn(monkeypatch, tmp_path)
    await _put(fn._output, 1)
    fn.save(tmp_path / "state")

    other = make_fn(monkeypatch, make_args(save=str(tmp_path), async_checkpoint_buffer="prompts"), FakeDataSource())
    with pytest.raises(AssertionError, match="resume with log"):
        other.load(tmp_path / "state")
