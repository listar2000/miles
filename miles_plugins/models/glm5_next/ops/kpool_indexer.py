from dataclasses import dataclass  # [fork fix-071]

import torch
import torch.distributed as dist  # [fork fix-071]

from miles.kernels.attention.dsa.kpool import (  # [fork fix-071] SPARSE_MLA_BLOCK
    SPARSE_MLA_BLOCK,
    append_tail_and_pad,
    pool_topk_to_token_fn,
    select_expand_tail,
)
from miles.utils.replay_base import indexer_replay_manager

# [fork fix-070] query rows scored at once: their dense [rows, T_pool] fp32 scores are the indexer's transient
_QUERY_CHUNK = 8192


@dataclass(frozen=True)
class QueryRows:
    """[fork fix-071] The token ids one rank selected, chunk by chunk, and where those chunks sit."""

    tokens: torch.Tensor  # [rows of this rank, width] int32
    starts: range  # first query row of each of this rank's chunks
    chunk: int
    num_tokens: int
    split_group: dist.ProcessGroup | None  # None: tokens holds every query row, in order


def kpool_select_topk(
    index_q: torch.Tensor,
    pooled_k: torch.Tensor,
    head_weights: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pool_cu_seqlens: torch.Tensor,
    index_topk: int,
    kpool: int,
    split_group: dist.ProcessGroup | None = None,  # [fork fix-071]
) -> torch.Tensor:
    # [fork fix-071] each rank of split_group selects its share of the query rows; the rows are then assembled
    rows = kpool_select_rows(
        index_q, pooled_k, head_weights, cu_seqlens, pool_cu_seqlens, index_topk, kpool, split_group=split_group
    )
    return gather_query_rows(rows)


def kpool_select_rows(
    index_q: torch.Tensor,
    pooled_k: torch.Tensor,
    head_weights: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pool_cu_seqlens: torch.Tensor,
    index_topk: int,
    kpool: int,
    split_group: dist.ProcessGroup | None,
) -> QueryRows:
    """[fork fix-071] kpool_select_topk's body: the rows of every chunk this rank owns."""
    num_tokens = index_q.shape[0]
    device = index_q.device
    token_ids = torch.arange(num_tokens, device=device)
    seq_indices = torch.searchsorted(cu_seqlens, token_ids, right=True) - 1
    seq_token_base = cu_seqlens[seq_indices].to(torch.int32)
    pool_base = pool_cu_seqlens[seq_indices].to(torch.int32)
    local_positions = (token_ids - seq_token_base).to(torch.int32)
    eligible_pools = torch.div(local_positions + 1, kpool, rounding_mode="floor")

    # [fork fix-070] Indexer replay requires one whole-sequence selection per call.
    chunk = num_tokens if indexer_replay_manager.enabled else _QUERY_CHUNK
    starts = range(0, num_tokens, chunk)
    # [fork fix-071] the ranks of split_group hold the same queries: each scores every world-th of world-times
    # smaller chunks (round-robin, so the causal cost stays balanced). Not with indexer replay, which records and
    # replays one whole-sequence top-k per rank.
    world = 1 if split_group is None or indexer_replay_manager.enabled else dist.get_world_size(split_group)
    if world > 1:
        chunk = -(-chunk // (world * SPARSE_MLA_BLOCK)) * SPARSE_MLA_BLOCK  # aligned chunk starts
        starts = range(0, num_tokens, chunk)[dist.get_rank(split_group) :: world]
    width = -(-(index_topk + kpool - 1) // SPARSE_MLA_BLOCK) * SPARSE_MLA_BLOCK  # as select_expand_tail pads
    parts = []
    for start in starts:
        rows = slice(start, min(start + chunk, num_tokens))
        parts.append(
            _select_rows(
                index_q[rows],
                pooled_k,
                head_weights[rows],
                seq_token_base[rows],
                pool_base[rows],
                local_positions[rows],
                eligible_pools[rows],
                index_topk,
                kpool,
            )
        )
    if not parts:  # [fork fix-071] this rank has no chunk of a short sequence
        parts.append(index_q.new_empty((0, width), dtype=torch.int32))
    tokens = parts[0] if len(parts) == 1 else torch.cat(parts)
    assert tokens.shape[1] == width, (tokens.shape, width)
    return QueryRows(tokens, starts, chunk, num_tokens, split_group if world > 1 else None)


def gather_query_rows(rows: QueryRows) -> torch.Tensor:
    """[fork fix-071] Every query's token ids, [T, 1, width]. Split rows are summed into place with one integer
    all-reduce: each row has exactly one writer and is zero on the other ranks, so the sum is exact."""
    if rows.split_group is None:
        return rows.tokens.unsqueeze(1)
    full = rows.tokens.new_zeros((rows.num_tokens, rows.tokens.shape[1]))
    offset = 0
    for start in rows.starts:
        end = min(start + rows.chunk, rows.num_tokens)
        full[start:end] = rows.tokens[offset : offset + end - start]
        offset += end - start
    dist.all_reduce(full, group=rows.split_group)
    return full.unsqueeze(1)


def _select_rows(
    index_q, pooled_k, head_weights, seq_token_base, pool_base, local_positions, eligible_pools, index_topk, kpool
):
    """[fork fix-070] Upstream's scoring and selection, for one chunk of query rows."""
    num_tokens = index_q.shape[0]
    device = index_q.device
    if pooled_k.shape[0] > 0:
        # tilelang is GPU-only; importing it here keeps this module importable on CPU,
        # where tests/fast pins the per-sequence pool invariants.
        from miles.kernels.attention.dsa import indexer_logits

        with torch.no_grad():
            pool_logits = indexer_logits(
                index_q,
                pooled_k,
                head_weights,
                pool_base.to(torch.int32),
                (pool_base + eligible_pools).to(torch.int32),
            )
    else:
        pool_logits = torch.full((num_tokens, 1), float("-inf"), dtype=torch.float32, device=device)

    if indexer_replay_manager.enabled:
        topk_fn = indexer_replay_manager.get_topk_fn(
            pool_topk_to_token_fn(seq_token_base, pool_base, local_positions, kpool),
            return_probs=False,
        )
        tokens = topk_fn(pool_logits, index_topk)
        shortcut = (local_positions + 1) <= index_topk
        tokens = append_tail_and_pad(tokens, seq_token_base, local_positions, shortcut, kpool)
    else:
        tokens = select_expand_tail(pool_logits, seq_token_base, pool_base, local_positions, index_topk, kpool)
    return tokens
