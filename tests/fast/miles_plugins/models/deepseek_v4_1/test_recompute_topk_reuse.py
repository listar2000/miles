"""[fork fix-047] A full-recompute replay of an index-source layer reuses its forward's top-k."""

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from miles_plugins.models.deepseek_v4_1 import deepseek_v4_1 as v41


class _Indexer:
    def __init__(self):
        self.calls = 0

    def __call__(self, step):
        self.calls += 1
        return torch.full((1, 4, 2), step, dtype=torch.int64), torch.full((1, 4, 1), -step, dtype=torch.int16)


def _attention(pp_size=2):
    indexer = _Indexer()
    attn = SimpleNamespace(
        _recompute_topk=deque(),
        core_attention=SimpleNamespace(indexer=indexer),
        config=SimpleNamespace(pipeline_model_parallel_size=pp_size),
        layer_id=3,
    )
    return attn, indexer


def _run(attn, step, monkeypatch, *, checkpointing, grad):
    monkeypatch.setattr(v41, "is_checkpointing", lambda: checkpointing)
    with torch.set_grad_enabled(grad):
        return v41.DeepSeekV41Attention._index_topk(attn, step)


def test_recompute_replays_forwards_in_order(monkeypatch):
    attn, indexer = _attention()
    forwards = [_run(attn, step, monkeypatch, checkpointing=True, grad=False) for step in (1, 2)]
    replays = [_run(attn, step, monkeypatch, checkpointing=True, grad=True) for step in (7, 8)]
    assert indexer.calls == 2 and not attn._recompute_topk
    for (topk, cand), (rtopk, rcand) in zip(forwards, replays, strict=True):
        assert rtopk.dtype == topk.dtype and torch.equal(rtopk, topk) and torch.equal(rcand, cand)


def test_forward_only_and_disabled_paths_run_the_indexer(monkeypatch):
    attn, indexer = _attention()
    _run(attn, 1, monkeypatch, checkpointing=False, grad=False)  # log-prob pass: eval mode, no checkpointing
    attn._recompute_topk = None
    _run(attn, 2, monkeypatch, checkpointing=True, grad=True)
    assert indexer.calls == 2


def test_stash_deeper_than_the_pipeline_fails(monkeypatch):
    attn, _ = _attention(pp_size=1)
    _run(attn, 1, monkeypatch, checkpointing=True, grad=False)
    with pytest.raises(AssertionError):
        _run(attn, 2, monkeypatch, checkpointing=True, grad=False)


def test_train_mode_drops_a_stale_stash(monkeypatch):
    attn = object.__new__(v41.DeepSeekV41Attention)
    torch.nn.Module.__init__(attn)
    attn._recompute_topk = deque([("stale", torch.int64, None)])  # a step that died before its recompute
    attn.train()
    assert not attn._recompute_topk and attn.training
