"""CPU tests for [fork fix-007]: candidate block ids select exactly what the pre-fix dense mask did.

Megatron and the GPU kernels are stubbed; only the torch selection code runs.
"""

import importlib
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

_STUBBED = (
    "megatron",
    "megatron.core",
    "megatron.core.extensions",
    "megatron.core.extensions.transformer_engine",
    "megatron.core.transformer",
    "megatron.core.transformer.module",
    "megatron.core.transformer.transformer_config",
    "miles.kernels.attention.dsa",
    "miles.kernels.attention.dsa.topk",
    "miles.kernels.position.rope",
    "miles.kernels.quant.fake_quant",
    "miles.utils.replay_base",
    "miles_plugins.models.deepseek_v4.ops.norm",
)


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__init__": lambda self, *a, **k: None})


INDEXER = "miles_plugins.models.deepseek_v4_1.ops.indexer"


@pytest.fixture(scope="module")
def ops():
    with pytest.MonkeyPatch.context() as mp:
        for name in _STUBBED:
            mp.setitem(sys.modules, name, _Stub(name))
        mp.delitem(sys.modules, INDEXER, raising=False)
        yield importlib.import_module(INDEXER)


def _reference_keep(logits, compress_lens, topk_blocks, block_size):
    """select_candidate_blocks before fix-007 (dense per-position mask)."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def _scores(b, q, n_kv, lens, seed):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(b, q, n_kv, generator=g)
    s = s.masked_fill(torch.arange(n_kv) >= lens.unsqueeze(-1), -torch.inf)  # what indexer_logits_sbhd writes
    return s


CASES = [
    # (b, q, n_kv, topk_blocks, block_size)
    (1, 64, 1024, 16, 8),  # fewer kept blocks than blocks
    (1, 64, 1027, 16, 8),  # partial last block
    (2, 33, 200, 64, 8),  # topk_blocks > num_blocks
    (1, 128, 4096, 2048, 8),  # the real topk_blocks
]


@pytest.mark.parametrize("b,q,n_kv,topk_blocks,bs", CASES)
def test_masked_scores_match_dense_reference(ops, b, q, n_kv, topk_blocks, bs):
    lens = torch.randint(1, n_kv + 1, (q,), generator=torch.Generator().manual_seed(3))
    source = _scores(b, q, n_kv, lens, seed=4)
    consumer = _scores(b, q, n_kv, lens, seed=5)
    ids = ops.select_candidate_blocks(source, lens.unsqueeze(-1), topk_blocks, bs)
    assert ids.shape == (b, q, min(topk_blocks, -(-n_kv // bs)))  # the carry stores [b, q, k]
    got = consumer.clone()
    ops.mask_to_candidates_(got, ids, bs)
    ref_mask = _reference_keep(source, lens.unsqueeze(-1), topk_blocks, bs)
    assert torch.equal(got, consumer.masked_fill(~ref_mask, -torch.inf))


def test_mask_on_a_strided_view(ops):
    """indexer_select masks scores[..., :n_kv] of a row-padded buffer (deep_select alignment)."""
    n_kv, pad = 1027, 13
    lens = torch.full((16,), n_kv)
    buf = _scores(1, 16, n_kv + pad, torch.full((16,), n_kv + pad), seed=6)
    ref = buf.clone()
    ids = ops.select_candidate_blocks(_scores(1, 16, n_kv, lens, seed=7), lens.unsqueeze(-1), 32, 8)
    ops.mask_to_candidates_(buf[..., :n_kv], ids, 8)
    ref_view = ref[..., :n_kv]
    ref_view.masked_fill_(~_reference_keep(_scores(1, 16, n_kv, lens, seed=7), lens.unsqueeze(-1), 32, 8), -torch.inf)
    assert torch.equal(buf, ref)  # the padded tail columns are untouched


def test_block_ids_above_int16_range(ops):
    """Block ids >= 32768 survive the int16 storage."""
    bs, n_kv = 8, 60_000 * 8
    logits = torch.full((1, 1, n_kv), -1.0)
    logits[0, 0, 50_000 * bs + 3] = 5.0
    ids = ops.candidate_ids(ops.select_candidate_blocks(logits, torch.tensor([[n_kv]]), 4, bs)).flatten().tolist()
    assert 50_000 in ids and (n_kv - 1) // bs in ids


def test_too_many_blocks_for_uint16_ids_asserts(ops):
    n_kv = 70_000 * 8
    with pytest.raises(AssertionError):
        ops.select_candidate_blocks(torch.zeros(1, 1, n_kv), torch.tensor([[n_kv]]), 4, 8)
