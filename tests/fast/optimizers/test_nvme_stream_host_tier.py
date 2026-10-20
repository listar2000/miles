"""[fork fix-045] A bucket in the host tier holds the same bytes as one on disk, and their checkpoint files swap."""

from types import SimpleNamespace

import pytest
import torch

from miles_plugins.optimizers.nvme_stream import SEGMENTS, _Bucket, _Entry

SIZES = (1000, 3000)
DTYPES = {"main": torch.float32, "exp_avg": torch.bfloat16, "exp_avg_sq": torch.bfloat16}


def _stager():
    from miles_plugins.optimizers.nvme_stream import _Stager

    stager = object.__new__(_Stager)
    stager._buf = torch.empty(4096, dtype=torch.uint8)  # smaller than one segment, so copies are chunked
    stager._bytes = stager._buf.numpy()
    stager._device_buf = None
    return stager


def _bucket(tmp_path, name: str, in_host: bool, seed: int = 0):
    torch.manual_seed(seed)
    entries = [_Entry(torch.nn.Parameter(torch.zeros(n)), torch.randn(n), 0) for n in SIZES]
    adam = SimpleNamespace(param_groups=[{}], state={})
    for entry in entries:
        adam.state[entry.main_param] = {s: torch.randn(entry.main_param.numel()) for s in SEGMENTS[1:]}
    return _Bucket(str(tmp_path / f"{name}.bin"), entries, adam, _stager(), DTYPES, in_host)


def _state(bucket) -> list[torch.Tensor]:
    return [tensor.clone() for segment in SEGMENTS for tensor, _ in bucket._tensors(segment)]


def test_round_trip_matches_the_disk_tier(tmp_path):
    disk, host = _bucket(tmp_path, "disk", False), _bucket(tmp_path, "host", True)
    disk.flush(), host.flush()
    disk.fetch(), host.fetch()
    for a, b in zip(_state(disk), _state(host), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("src_host,dst_host", [(True, False), (False, True)])
def test_checkpoint_files_are_tier_independent(tmp_path, src_host, dst_host):
    src, dst = _bucket(tmp_path, "src", src_host), _bucket(tmp_path, "dst", dst_host, seed=1)
    src.flush()
    src.fetch()
    expected = _state(src)
    assert not all(torch.equal(a, b) for a, b in zip(expected, _state(dst), strict=True))
    src.flush()
    src.save_file(str(tmp_path / "saved.bin"))
    dst.flush()
    dst.load_file(str(tmp_path / "saved.bin"))
    dst.fetch()
    for a, b in zip(expected, _state(dst), strict=True):
        assert torch.equal(a, b)


def test_buckets_fill_the_host_budget_in_order(tmp_path, monkeypatch):
    from miles_plugins.optimizers import nvme_stream

    monkeypatch.setattr(nvme_stream, "BUCKET_NUMEL_LIMIT", 1000)
    params = [torch.nn.Parameter(torch.zeros(1000)) for _ in range(4)]
    store = object.__new__(nvme_stream.NVMeOptimizerStateStore)
    store.dist_opt = SimpleNamespace(
        model_float16_groups=[params],
        shard_fp32_from_float16_groups=[[torch.zeros(1000) for _ in params]],
        model_param_gbuf_map={p: (0, torch.bfloat16, i) for i, p in enumerate(params)},
        optimizer=SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.0}]),
        config=SimpleNamespace(decoupled_weight_decay=True),
    )
    store.dir, store.dtypes, store._stager = str(tmp_path), DTYPES, _stager()
    bucket_bytes = nvme_stream._layout([nvme_stream._Entry(params[0], torch.zeros(1000), 0)], DTYPES)[1]
    store._host_budget = 2 * bucket_bytes + 1  # room for two buckets, not three
    monkeypatch.setattr(store, "_adam_for", lambda _: SimpleNamespace(param_groups=[{}], state={}))
    assert [b.host is not None for b in store._build_buckets()] == [True, True, False, False]
