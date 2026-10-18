"""[fork fix-010] [fork fix-021] The NVMe store's fp32 mains: written once by a load path,
or built by the first step on a fresh start."""

import os
from types import SimpleNamespace

import pytest
import torch

from miles_plugins.optimizers import nvme_stream
from miles_plugins.optimizers.nvme_stream import NVMeOptimizerStateStore, _Bucket, _Entry, _resize, _Stager

SIZES = (1000, 3000)
DTYPES = {segment: torch.float32 for segment in ("main", "exp_avg", "exp_avg_sq")}


def _cpu_stager(nbytes: int = 64 * 1024) -> _Stager:
    stager = object.__new__(_Stager)
    stager._buf = torch.empty(nbytes, dtype=torch.uint8)
    stager._bytes = stager._buf.numpy()
    stager._device_buf = None
    return stager


def _store(tmp_path, name: str):
    torch.manual_seed(0)
    params, entries, ranges = [], [], {}
    for numel in SIZES:
        param = torch.nn.Parameter(torch.randn(numel).to(torch.bfloat16))
        main = torch.empty(numel, dtype=torch.float32)
        _resize(main, 0)  # what Megatron's deferred main-param initialization leaves
        ranges[param] = {"param": SimpleNamespace(start=0, end=numel, size=numel)}
        params.append(param)
        entries.append(_Entry(param, main, 0))
    (tmp_path / name).mkdir()
    adam = SimpleNamespace(param_groups=[{}], state={})
    bucket = _Bucket(str(tmp_path / name / "bucket00000.bin"), entries, adam, _cpu_stager(), DTYPES)
    store = object.__new__(NVMeOptimizerStateStore)
    store.dist_opt = SimpleNamespace(_get_model_param_range_map=lambda p: ranges[p])
    store.buckets, store.dtypes = [bucket], DTYPES
    store._fp32_adam, store._allow_fresh_state = None, False
    store._rank, store._instance, store.uid = 0, 0, 0
    store._main_ready = False
    store._main_from_model = False
    return store, params, bucket


def _mains_on_disk(bucket) -> list[torch.Tensor]:
    out = []
    for entry, offset in zip(bucket.entries, bucket.offsets["main"], strict=True):
        nbytes = entry.main_param.numel() * 4
        out.append(torch.frombuffer(bytearray(os.pread(bucket.fd, nbytes, offset)), dtype=torch.float32))
    return out


def _count_writes(monkeypatch) -> list[int]:
    written, real = [0], os.pwritev

    def counting(fd, buffers, offset):
        written[0] += (n := real(fd, buffers, offset))
        return n

    monkeypatch.setattr(nvme_stream.os, "pwritev", counting)
    return written


def _initialize(store):
    store.initialize_main_from_model_params()


def _refresh(store):
    store.refresh_main_from_model_params(lambda: None)  # only the flag matters here, not the values


@pytest.mark.parametrize("writer", [_initialize, _refresh])
def test_mains_are_not_written_again_after_a_load_path_wrote_them(tmp_path, monkeypatch, writer):
    store, _, bucket = _store(tmp_path, "a")
    writer(store)
    written = _count_writes(monkeypatch)

    store._ensure_main()

    assert written[0] == 0
    os.close(bucket.fd)


def test_mains_are_filled_from_the_model_params_if_no_load_path_did(tmp_path):
    store, params, bucket = _store(tmp_path, "a")

    store._ensure_main()

    for main, param in zip(_mains_on_disk(bucket), params, strict=True):
        assert torch.equal(main, param.detach().float())
    os.close(bucket.fd)


def test_resume_from_saved_state_does_not_rewrite_the_mains(tmp_path, monkeypatch):
    saved, params, saved_bucket = _store(tmp_path, "a")
    saved.save_to(str(tmp_path / "ckpt"))
    resumed, _, bucket = _store(tmp_path, "b")
    written = _count_writes(monkeypatch)

    assert resumed.load_from(str(tmp_path / "ckpt"))
    resumed._ensure_main()

    assert written[0] == 0
    for main, param in zip(_mains_on_disk(bucket), params, strict=True):
        assert torch.equal(main, param.detach().float())
    os.close(saved_bucket.fd)
    os.close(bucket.fd)


# [fork fix-021] begin
def _steppable(store, bucket):
    """step() with Megatron's grad and copy-back plumbing stubbed; Adam adds 1 to every main and records what it saw."""
    seen = []

    def adam_step():
        seen.append([e.main_param.clone() for e in bucket.entries])
        for entry in bucket.entries:
            entry.main_param.add_(1)
            state = bucket.adam.state.setdefault(entry.main_param, {})
            for segment in ("exp_avg", "exp_avg_sq"):  # created by the first step, like Adam's
                state.setdefault(segment, torch.zeros_like(entry.main_param))

    bucket.adam.step = adam_step
    store.dist_opt.config = SimpleNamespace(clip_grad=0.0)
    store._grad_norm = None
    for name in ("_sync_lr_wd", "_attach_grads", "_detach_grads", "_copy_main_to_model_params"):
        setattr(store, name, lambda *args: None)
    return seen


def _reads(monkeypatch) -> list[int]:
    read, real = [0], os.preadv

    def counting(fd, buffers, offset):
        read[0] += (n := real(fd, buffers, offset))
        return n

    monkeypatch.setattr(nvme_stream.os, "preadv", counting)
    return read


def test_the_first_step_builds_the_mains_from_the_model_params_without_reading_them(tmp_path, monkeypatch):
    eager, _, eager_bucket = _store(tmp_path, "a")
    eager.initialize_main_from_model_params()
    eager_seen = _steppable(eager, eager_bucket)
    eager.step()
    lazy, params, lazy_bucket = _store(tmp_path, "b")
    lazy.defer_main_to_model_params()
    lazy_seen = _steppable(lazy, lazy_bucket)
    read = _reads(monkeypatch)

    lazy.step()

    assert read[0] == 0
    assert all(torch.equal(a, b) for a, b in zip(lazy_seen[0], eager_seen[0], strict=True))
    assert all(
        torch.equal(a, b) for a, b in zip(_mains_on_disk(lazy_bucket), _mains_on_disk(eager_bucket), strict=True)
    )
    lazy.step()  # the second step reads what the first one wrote
    assert read[0] > 0
    assert all(torch.equal(main, param.detach().float() + 1) for main, param in zip(lazy_seen[1], params, strict=True))
    os.close(eager_bucket.fd)
    os.close(lazy_bucket.fd)


def test_a_save_before_the_first_step_writes_the_mains(tmp_path):
    store, params, bucket = _store(tmp_path, "a")
    store.defer_main_to_model_params()

    store.save_to(str(tmp_path / "ckpt"))

    for main, param in zip(_mains_on_disk(bucket), params, strict=True):
        assert torch.equal(main, param.detach().float())
    os.close(bucket.fd)


# [fork fix-021] end
