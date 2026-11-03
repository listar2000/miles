"""[fork fix-072] A full-recompute replay of a GLM-5.3 DSA layer reuses its forward's selected rows."""

import importlib
import sys
import types
from collections import deque
from types import SimpleNamespace
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

_STUBBED = (
    "megatron",
    "megatron.core",
    "megatron.core.parallel_state",
    "megatron.core.tensor_parallel",
    "megatron.core.tensor_parallel.mappings",
    "megatron.core.tensor_parallel.random",
    "megatron.core.transformer",
    "megatron.core.transformer.enums",
    "megatron.core.transformer.module",
    "megatron.core.transformer.moe",
    "megatron.core.transformer.moe.moe_utils",
)
DSA = "miles_plugins.models.glm5_next.dsa"


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return mock.MagicMock(name=f"{self.__name__}.{name}")


@pytest.fixture(scope="module")
def dsa():
    """dsa.py on CPU without Megatron: stubbed imports, a plain Module as the base class; sys.modules restored."""
    glm5 = types.ModuleType("miles_plugins.models.glm5.glm5")
    glm5.DSAMLASelfAttention = torch.nn.Module
    with pytest.MonkeyPatch.context() as patch:
        for name in _STUBBED:
            patch.setitem(sys.modules, name, _Stub(name))
        patch.setitem(sys.modules, glm5.__name__, glm5)
        patch.delitem(sys.modules, DSA, raising=False)
        package = importlib.import_module(DSA.rpartition(".")[0])
        patch.setattr(package, "dsa", None, raising=False)  # undone below, so the stubbed module stays private
        module = importlib.import_module(DSA)
    if sys.modules.get(DSA) is module:
        del sys.modules[DSA]
    return module


class _Selector:
    def __init__(self):
        self.calls = 0

    def __call__(self, step):
        self.calls += 1
        return SimpleNamespace(tokens=torch.full((4, 64), step, dtype=torch.int32))


def _attention(dsa, pp_size=2):
    attn = object.__new__(dsa.Glm5NextDSAAttention)
    torch.nn.Module.__init__(attn)
    attn._recompute_topk = deque()
    attn.config = SimpleNamespace(pipeline_model_parallel_size=pp_size)
    attn.layer_number = 3
    attn._kpool_select = _Selector()
    return attn


def _run(dsa, attn, step, monkeypatch, *, checkpointing, grad):
    monkeypatch.setattr(dsa, "is_checkpointing", lambda: checkpointing)
    with torch.set_grad_enabled(grad):
        return attn._index_rows(step)


def test_recompute_replays_forwards_in_order(dsa, monkeypatch):
    attn = _attention(dsa)
    forwards = [_run(dsa, attn, step, monkeypatch, checkpointing=True, grad=False) for step in (1, 2)]
    replays = [_run(dsa, attn, step, monkeypatch, checkpointing=True, grad=True) for step in (7, 8)]
    assert attn._kpool_select.calls == 2 and not attn._recompute_topk
    for rows, replayed in zip(forwards, replays, strict=True):
        assert replayed is rows


def test_forward_only_and_replay_paths_run_the_indexer(dsa, monkeypatch):
    attn = _attention(dsa)
    _run(dsa, attn, 1, monkeypatch, checkpointing=False, grad=False)  # log-prob pass: eval mode, no checkpointing
    attn._recompute_topk = None  # indexer replay
    _run(dsa, attn, 2, monkeypatch, checkpointing=True, grad=True)
    assert attn._kpool_select.calls == 2


def test_stash_deeper_than_the_pipeline_fails(dsa, monkeypatch):
    attn = _attention(dsa, pp_size=1)
    _run(dsa, attn, 1, monkeypatch, checkpointing=True, grad=False)
    with pytest.raises(AssertionError):
        _run(dsa, attn, 2, monkeypatch, checkpointing=True, grad=False)


def test_train_mode_drops_a_stale_stash(dsa):
    attn = _attention(dsa)
    attn._recompute_topk.append("stale")  # a step that died before its recompute
    attn.train()
    assert not attn._recompute_topk and attn.training
