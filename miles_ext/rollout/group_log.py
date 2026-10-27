"""Write-ahead group log: a fully async buffer that saves each group as it enters, so a checkpoint names it."""

import asyncio
import copy
import dataclasses
import logging
import os
import pickle
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import safetensors.numpy
import torch

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.ray.rollout.rollout_executor import compute_rollout_checkpoint_dir
from miles.rollout.fully_async_data_buffer import (
    MULTI_POLICY_CHECKPOINT_UNSUPPORTED,
    DataBufferConstructorInput,
    DataBufferInput,
    DefaultDataBuffer,
    Group,
    UnusedReason,
)
from miles.rollout.session.samples.codec import SAMPLES_VALUE_SPEC, _asarray_wire
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

FORMAT = 1
GROUPS_DIRNAME = "rollout_groups"
_SUFFIX = ".safetensors"
_SKELETON_KEY = "_skeleton"
_ROUTED_EXPERTS = "rollout_routed_experts"
_MANIFEST_SCAN_MAX_BYTES = 100 * 2**20  # a larger state.pt holds whole groups (default buffer), not a manifest
# per-token fields stored as arrays, with the sample codec's dtypes; everything else is pickled with the sample
_ARRAY_FIELDS = {f: spec for f, spec in SAMPLES_VALUE_SPEC.items() if spec.codec in ("tensor", "tensor_list")}


@dataclasses.dataclass
class _LoggedInput(DataBufferInput):
    uid: str | None = None  # None: the write failed, so a resume regenerates this group from its prompts


@dataclasses.dataclass(frozen=True)
class GroupLogManifest:
    format: int
    entries: list[tuple[str | None, list[Sample]]]  # (uid, reset prompt group), in buffer order


class GroupLogDataBuffer(DefaultDataBuffer):
    def __init__(self, input: DataBufferConstructorInput):
        super().__init__(input)
        assert not resolve_megatron_config(self._args).is_multi_policy, MULTI_POLICY_CHECKPOINT_UNSUPPORTED
        self._dir = Path(self._args.save) / GROUPS_DIRNAME if self._args.save else None
        self._writer = ThreadPoolExecutor(max_workers=2, thread_name_prefix="group-log")
        self._known: set[str] = set()  # uids with a file in self._dir that no GC has deleted
        self._writing: dict[asyncio.Task, DataBufferInput] = {}  # accepted groups whose file is being written
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)
            for tmp in self._dir.glob(".tmp-*"):  # left by a write that never finished
                tmp.unlink()
            self._known = {p.name.removesuffix(_SUFFIX) for p in self._dir.glob(f"*{_SUFFIX}")}

    async def put(self, input: DataBufferInput) -> None:
        if not self._preput_filter(input):
            return
        if self._dir is None:
            return await self._append(input, uid=None)
        # a group being written counts toward capacity; the producer waits for room, and for the write only
        # with --async-single-pass
        async with self._cond:
            while len(self._buffer) + len(self._writing) >= self._capacity:
                await self._cond.wait()
        task = asyncio.create_task(self._write_and_append(input))
        self._writing[task] = input
        if self._args.async_single_pass:  # the pass may end once put returns: the group must be ready by then
            await task

    async def _write_and_append(self, input: DataBufferInput) -> None:
        # the entry becomes visible only once its file is written, so get() never hands out a group being read
        uid = await self._write(input.group)
        async with self._cond:
            del self._writing[asyncio.current_task()]
            if uid is not None:
                self._known.add(uid)  # with the append, so a GC never sees a written file no entry names
            self._buffer.append(_LoggedInput(prompt_group=input.prompt_group, group=input.group, uid=uid))
            self._cond.notify_all()

    async def _append(self, input: DataBufferInput, *, uid: str | None) -> None:
        async with self._cond:  # DefaultDataBuffer.put's capacity wait and append
            while len(self._buffer) >= self._capacity:
                await self._cond.wait()
            self._buffer.append(_LoggedInput(prompt_group=input.prompt_group, group=input.group, uid=uid))
            self._cond.notify_all()

    async def _write(self, group: Group) -> str | None:
        uid = uuid.uuid4().hex
        start = time.monotonic()
        try:
            await asyncio.get_running_loop().run_in_executor(self._writer, write_group, group, self._path(uid))
        except Exception:
            logger.exception(f"group log: writing group {uid} failed; a resume regenerates it from its prompts")
            return None
        size_mb = self._path(uid).stat().st_size / 2**20
        logger.info(f"group log: wrote group {uid} ({size_mb:.0f} MiB) in {time.monotonic() - start:.1f}s")
        return uid

    def _path(self, uid: str, root: Path | None = None) -> Path:
        return (root or self._dir) / f"{uid}{_SUFFIX}"

    def state_dict(self) -> GroupLogManifest | list[DataBufferInput]:
        if self._dir is None:
            return super().state_dict()
        entries = [(entry.uid, _reset_copy(entry.prompt_group)) for entry in self._buffer]
        live = {uid for uid, _ in entries}
        # a group still being written has no file yet: a resume regenerates it from its prompts. Under
        # --async-single-pass its put is still awaited, so the group is saved with the running groups instead
        if not self._args.async_single_pass:
            entries += [(None, _reset_copy(entry.prompt_group)) for entry in self._writing.values()]
        self._writer.submit(self._collect_garbage, candidates=self._known - live)
        return GroupLogManifest(format=FORMAT, entries=entries)

    def load_state_dict(self, state: GroupLogManifest | list[DataBufferInput]) -> None:
        if not isinstance(state, GroupLogManifest):  # saved by the default buffer: groups without files
            return super().load_state_dict([_LoggedInput(e.prompt_group, e.group) for e in state])
        assert not self._buffer and state.format == FORMAT, (len(self._buffer), state.format)
        root = Path(getattr(self._args, "lora_resume_dir", None) or self._args.load) / GROUPS_DIRNAME
        with ThreadPoolExecutor(max_workers=8) as pool:
            groups = list(pool.map(lambda e: self._restore(e[0], root), state.entries))
        for (uid, prompt_group), group in zip(state.entries, groups, strict=True):
            if group is None:
                self._unused_handler_fn(prompt_group, UnusedReason.UNRESTORED)
            else:
                self._buffer.append(_LoggedInput(prompt_group=prompt_group, group=group, uid=uid))
        logger.info(f"group log: restored {len(self._buffer)} of {len(state.entries)} buffered groups from {root}")

    def _restore(self, uid: str | None, root: Path) -> Group | None:
        if uid is None or not (path := self._path(uid, root)).is_file():
            logger.warning(f"group log: group {uid} has no file under {root}; regenerating it from its prompts")
            return None
        if self._dir is not None and path != self._path(uid):  # resumed from another run: keep the group here
            shutil.copyfile(path, self._path(uid))
            self._known.add(uid)
        return read_group(path)

    def _collect_garbage(self, candidates: set[str]) -> None:
        try:
            named = _uids_in_published_manifests(Path(self._args.save))
        except Exception:
            logger.exception("group log: reading the saved manifests failed; keeping every group file")
            return
        for uid in candidates - named:
            self._path(uid).unlink(missing_ok=True)
            self._known.discard(uid)
        logger.info(f"group log: deleted {len(candidates - named)} group files no checkpoint names")


def write_group(group: Group, path: Path) -> None:
    tmp = path.with_name(f".tmp-{path.name}")
    safetensors.numpy.save_file(encode_group(group), str(tmp))
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_group(path: Path) -> Group:
    return decode_group(safetensors.numpy.load_file(str(path)))


def encode_group(group: Group) -> dict[str, np.ndarray]:
    """One safetensors dict per group: per-token fields as arrays, the rest of each sample pickled."""
    tensors: dict[str, np.ndarray] = {}
    skeleton = []
    for i, sample in enumerate(_flatten(group)):
        light = copy.copy(sample)
        for field, spec in _ARRAY_FIELDS.items():
            value = getattr(sample, field)
            if value is None:
                continue
            arr = np.asarray(value) if spec.strict else _asarray_wire(field, value, spec.dtype)  # raises on loss
            assert arr.dtype == spec.dtype, (field, arr.dtype)
            if field == _ROUTED_EXPERTS and arr.size and arr.min() >= 0 and arr.max() < 2**16:
                # departure from the codec's int32: expert ids fit 16 bits (384 experts in V4.1), half the bytes;
                # read_group casts back to int32
                arr = arr.astype(np.uint16)
            tensors[f"{field}.{i}"] = np.ascontiguousarray(arr)
            setattr(light, field, None)
        skeleton.append(light)
    tensors[_SKELETON_KEY] = np.frombuffer(pickle.dumps(_unflatten(group, skeleton)), dtype=np.uint8)
    return tensors


def decode_group(tensors: dict[str, np.ndarray]) -> Group:
    group = pickle.loads(tensors[_SKELETON_KEY].tobytes())
    for i, sample in enumerate(_flatten(group)):
        for field, spec in _ARRAY_FIELDS.items():
            if (arr := tensors.get(f"{field}.{i}")) is None:
                continue
            arr = arr.astype(spec.dtype, copy=False)
            setattr(sample, field, arr.tolist() if spec.codec == "tensor_list" else arr)
    return group


def _flatten(group: Group) -> list[Sample]:
    return [s for item in group for s in (item if isinstance(item, list) else [item])]


def _unflatten(group: Group, flat: list[Sample]) -> Group:
    it = iter(flat)
    return [[next(it) for _ in item] if isinstance(item, list) else next(it) for item in group]


def _reset_copy(prompt_group: list[Sample]) -> list[Sample]:
    # reset a shallow copy first so the deep copy skips the generated outputs (as fix-046's prompts mode does)
    copied = [copy.copy(sample) for sample in prompt_group]
    for sample in copied:
        sample.reset_for_retry()
    return copy.deepcopy(copied)


def _uids_in_published_manifests(save: Path) -> set[str]:
    named: set[str] = set()
    for state_file in compute_rollout_checkpoint_dir(save, rollout_id=0).parent.glob("*/generate_rollout/state.pt"):
        if state_file.parts[-3].startswith(".") or state_file.stat().st_size > _MANIFEST_SCAN_MAX_BYTES:
            continue
        output = torch.load(state_file, weights_only=False).output
        if isinstance(output, GroupLogManifest):
            named |= {uid for uid, _ in output.entries if uid is not None}
    return named
