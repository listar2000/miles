"""[fork fix-052] Check split indexer top-k and candidate parity across CPU ranks."""

import importlib
import os
import socket
import sys
import types
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

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


def _import_indexer():
    """The indexer with stubbed imports and the reference scoring kernel; sys.modules is restored on return."""
    with pytest.MonkeyPatch.context() as patch:
        for name in _STUBBED:
            patch.setitem(sys.modules, name, _Stub(name))
        patch.delitem(sys.modules, INDEXER, raising=False)
        ix = importlib.import_module(INDEXER)
        patch.setitem(sys.modules, INDEXER, ix)
    ix.indexer_logits_sbhd = _logits
    return ix


def _logits(q, k, w, starts, ends):
    # reference for indexer_logits_sbhd: q [s, b, h, d], k [n, b, d], w [s, b, h] -> [b, s, n], -inf past each end
    scores = torch.einsum("sbhd,nbd->sbhn", q.float(), k.float()).relu()
    scores = (scores * w.unsqueeze(-1)).sum(2).transpose(0, 1)
    cols = torch.arange(scores.size(-1))
    return scores.masked_fill((cols < starts[:, None]) | (cols >= ends[:, None]), -torch.inf)


def _select(ix, split_group, candidates_from=None):
    g = torch.Generator().manual_seed(0)
    bsz, seqlen, heads, dim, n_kv = 1, 37, 2, 8, 19
    q = torch.randn(bsz, seqlen, heads, dim, generator=g)
    index_k = torch.randn(bsz, n_kv, dim, generator=g)
    weights = torch.randn(bsz, seqlen, heads, generator=g)
    lens = (torch.arange(seqlen) + 1) // 2
    return ix.indexer_select(
        q, index_k, weights, lens, candidates_from,
        is_candidate_source=candidates_from is None, uses_candidates=candidates_from is not None,
        candidate_topk_blocks=3, candidate_block_size=4, topk=5,
        topk_fn=lambda s, k: s.topk(k, dim=-1).indices, allow_deep_select=False, query_chunk=8,
        split_group=split_group,
    )  # fmt: skip


def _worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=15))
    ix = _import_indexer()
    idx, cand = _select(ix, dist.group.WORLD)
    consumer_idx, _ = _select(ix, dist.group.WORLD, candidates_from=cand)
    if rank == 0:
        torch.save((idx, cand, consumer_idx), out)
    dist.destroy_process_group()


def test_split_matches_unsplit(tmp_path):
    ix = _import_indexer()
    ref_idx, ref_cand = _select(ix, None)
    ref_consumer, _ = _select(ix, None, candidates_from=ref_cand)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = tmp_path / "split.pt"
    mp.spawn(_worker, args=(3, port, str(out)), nprocs=3)
    idx, cand, consumer = torch.load(out)
    assert idx.dtype == ref_idx.dtype and torch.equal(idx, ref_idx)
    assert cand.dtype == torch.int16 and torch.equal(cand, ref_cand)
    assert torch.equal(consumer, ref_consumer)
