import torch

from miles.kernels.attention.dsa.kpool import append_tail_and_pad, pool_topk_to_token_fn, select_expand_tail
from miles.utils.replay_base import indexer_replay_manager

# [fork fix-070] query rows scored at once: their dense [rows, T_pool] fp32 scores are the indexer's transient
_QUERY_CHUNK = 8192


def kpool_select_topk(
    index_q: torch.Tensor,
    pooled_k: torch.Tensor,
    head_weights: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pool_cu_seqlens: torch.Tensor,
    index_topk: int,
    kpool: int,
) -> torch.Tensor:
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
    parts = []
    for start in range(0, num_tokens, chunk):
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
    tokens = parts[0] if len(parts) == 1 else torch.cat(parts)
    return tokens.unsqueeze(1)


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
