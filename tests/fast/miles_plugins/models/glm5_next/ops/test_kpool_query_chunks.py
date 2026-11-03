"""[fork fix-070] The kpool indexer's query-row chunks give the top-k of one whole-sequence call."""

import pytest

torch = pytest.importorskip("torch")

import miles.kernels.attention.dsa as dsa_kernels  # noqa: E402
from miles.kernels.attention.dsa.kpool import SPARSE_MLA_BLOCK, pool_boundaries  # noqa: E402
from miles_plugins.models.glm5_next.ops import kpool_indexer  # noqa: E402

KPOOL = 4
TOPK = 8
SEQ_LENS = [37, 13, 22, 50]


def reference_logits(q, k, weights, ks, ke):
    """indexer_logits on CPU: [T, T_kv] fp32, -inf outside each query's [ks, ke)."""
    scores = (torch.einsum("thd,nd->thn", q.float(), k.float()).relu() * weights.float().unsqueeze(-1)).sum(1)
    cols = torch.arange(k.shape[0])
    return scores.masked_fill((cols < ks[:, None]) | (cols >= ke[:, None]), -torch.inf)


def reference_select_expand_tail(pool_logits, seq_token_base, pool_base, local_positions, topk, kpool):
    """select_expand_tail on CPU, row by row as its triton kernel does."""
    width = -(-(topk + kpool - 1) // SPARSE_MLA_BLOCK) * SPARSE_MLA_BLOCK
    group_topk = min(topk // kpool, pool_logits.shape[1])
    scores, pools = torch.topk(pool_logits.float(), group_topk, dim=-1)
    out = torch.full((pool_logits.shape[0], width), -1, dtype=torch.int32)
    rows = zip(seq_token_base.tolist(), pool_base.tolist(), local_positions.tolist(), strict=True)
    for t, (base, pbase, lpos) in enumerate(rows):
        if lpos + 1 <= topk:
            out[t, : lpos + 1] = torch.arange(base, base + lpos + 1)
            continue
        for g in range(group_topk):
            if torch.isfinite(scores[t, g]):
                first = base + (int(pools[t, g]) - pbase) * kpool
                out[t, g * kpool : (g + 1) * kpool] = torch.arange(first, first + kpool)
        tail = (lpos + 1) % kpool
        tail_start = base + (lpos + 1) // kpool * kpool
        out[t, topk : topk + tail] = torch.arange(tail_start, tail_start + tail)
    return out


def select_inputs(seq_lens=SEQ_LENS, heads=2, dim=8):
    g = torch.Generator().manual_seed(0)
    cu_seqlens = torch.tensor([0, *seq_lens]).cumsum(0).to(torch.int32)
    num_tokens = int(cu_seqlens[-1])
    pool_cu_seqlens = pool_boundaries(cu_seqlens, KPOOL)
    return dict(
        index_q=torch.randn(num_tokens, heads, dim, generator=g),
        pooled_k=torch.randn(int(pool_cu_seqlens[-1]), dim, generator=g),
        head_weights=torch.randn(num_tokens, heads, generator=g),
        cu_seqlens=cu_seqlens,
        pool_cu_seqlens=pool_cu_seqlens,
        index_topk=TOPK,
        kpool=KPOOL,
    )


@pytest.fixture
def cpu_kernels(monkeypatch):
    monkeypatch.setattr(dsa_kernels, "indexer_logits", reference_logits)
    monkeypatch.setattr(kpool_indexer, "select_expand_tail", reference_select_expand_tail)


@pytest.mark.parametrize("chunk", [1, 7, 64, 121])
def test_chunks_match_one_call(cpu_kernels, monkeypatch, chunk):
    inputs = select_inputs()
    monkeypatch.setattr(kpool_indexer, "_QUERY_CHUNK", 10**9)
    whole = kpool_indexer.kpool_select_topk(**inputs)
    monkeypatch.setattr(kpool_indexer, "_QUERY_CHUNK", chunk)
    chunked = kpool_indexer.kpool_select_topk(**inputs)
    assert whole.shape == (sum(SEQ_LENS), 1, SPARSE_MLA_BLOCK) and whole.dtype == torch.int32
    assert (whole >= 0).any(dim=-1).all()  # every query keeps at least itself
    assert torch.equal(chunked, whole)
