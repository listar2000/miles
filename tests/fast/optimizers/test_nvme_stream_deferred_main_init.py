"""[fork fix-010] The NVMe store writes the fp32 mains once, from the loaded params, not first from the init params.

setup used to write every main from the freshly initialised model params, and the checkpoint load then wrote them all
again (reload_model_params) or copied them over (load_from): one wasted pass over the whole state, ~127 GB per rank
for DeepSeek-V4.1 on 16 GPUs. Now the load path is the only writer, and step() / save_to() fill the mains from the
model params if no load path did. A CPU stager stands in for the pinned one.
"""

import os
from types import SimpleNamespace

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
    return store, params, bucket


def _mains_on_disk(bucket) -> list[torch.Tensor]:
    out = []
    for entry, offset in zip(bucket.entries, bucket.offsets["main"], strict=True):
        nbytes = entry.main_param.numel() * 4
        out.append(torch.frombuffer(bytearray(os.pread(bucket.fd, nbytes, offset)), dtype=torch.float32))
    return out


class _WriteCounter:
    def __init__(self, monkeypatch):
        self.bytes = 0
        real = os.pwritev

        def counting(fd, buffers, offset):
            n = real(fd, buffers, offset)
            self.bytes += n
            return n

        monkeypatch.setattr(nvme_stream.os, "pwritev", counting)


def test_load_path_is_the_only_main_writer(tmp_path, monkeypatch):
    store, params, bucket = _store(tmp_path, "a")
    writes = _WriteCounter(monkeypatch)
    with torch.no_grad():
        for p in params:  # the checkpoint load overwrites the init values
            p.copy_(torch.randn(p.numel()).to(torch.bfloat16))

    written = store.initialize_main_from_model_params()  # what the bound reload_model_params() runs
    store._ensure_main()  # what step() / save_to() run first

    assert writes.bytes == written == sum(SIZES) * 4, "the mains must be written exactly once"
    for main, param in zip(_mains_on_disk(bucket), params, strict=True):
        assert torch.equal(main, param.detach().float()), "the mains must be the loaded params"
    os.close(bucket.fd)


def test_mains_are_filled_from_the_model_params_if_no_load_path_did(tmp_path):
    store, params, bucket = _store(tmp_path, "a")

    store._ensure_main()

    for main, param in zip(_mains_on_disk(bucket), params, strict=True):
        assert torch.equal(main, param.detach().float())
    os.close(bucket.fd)


def test_resume_from_saved_state_does_not_rewrite_the_mains(tmp_path, monkeypatch):
    saved, params, saved_bucket = _store(tmp_path, "a")
    saved.save_to(str(tmp_path / "ckpt"))  # fills the mains first, then copies the files
    resumed, _, bucket = _store(tmp_path, "b")
    writes = _WriteCounter(monkeypatch)

    assert resumed.load_from(str(tmp_path / "ckpt"))
    resumed._ensure_main()

    assert writes.bytes == 0, "load_from copies the files; the mains must not be written again"
    for main, param in zip(_mains_on_disk(bucket), params, strict=True):
        assert torch.equal(main, param.detach().float())
    os.close(saved_bucket.fd)
    os.close(bucket.fd)
