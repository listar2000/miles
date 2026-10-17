# fix-007: DeepSeek-V4.1 indexer candidates as block ids, not a dense mask

## Why
The candidate-source layer (`candidate_source_layer_id`, 20 in V4.1-Flash) picks the top 2048 blocks of 8
compressed keys per query; the later index-source layers (24, 28, 32, 36) may only select among them. Upstream
passed that choice on as a dense bool `[b, local_q, n_kv]`: 32 GiB per GPU at 256k context, built from 8192-query
chunks and concatenated (another 32 GiB), re-expanded densely at every later layer, and packed in its block form
(~4 GiB per layer at 256k) into the inter-layer carry that full recompute saves once per layer. On PP stage 1 this
is what ran out of memory: dsv41-tb4-overfit16-3 (256k) asked for 32 GiB at `indexer.py:120` in the logprob pass,
and dsv41-tb4-overfit16-4 (192k) had 221 GiB allocated on the first training microbatch.

## Decision
Carry the kept block ids, `[b, local_q, 2048]`, and let each consumer build a block mask for its own
8192-query chunk. The kept set, the masked scores and the indexer's top-512 are unchanged (CPU test against the
pre-fix code). Alternatives that lost:
- Keep the dense mask but free it earlier: still 32 GiB per transient at 256k and 72 GiB at 384k.
- Bit-pack the block mask (n_kv/64 bytes per query): 8x smaller than the old carry but still grows as q x kv;
  the ids are fixed at 2048 per query (4 KiB per query whatever the context).
- Recompute the candidates in every consumer: costs a full indexer pass per consumer layer.
Ids are stored as uint16 bits in an int16 tensor (the carry is bf16 bits), `0xFFFF` marks an unused slot, so the
number of blocks must stay below 65,535: n_kv up to 524,280 compressed keys. Beyond that
`select_candidate_blocks` asserts instead of silently wrapping. The carry also stops after the last consumer
(layer 36), not the last compressed layer (38).

## How it is applied
`miles_plugins/models/deepseek_v4_1/ops/indexer.py` (`select_candidate_blocks`, `candidate_ids`,
`mask_to_candidates_`, the `uses_candidates` branch of `indexer_select`) and
`miles_plugins/models/deepseek_v4_1/deepseek_v4_1.py` (`_bits_to_bf16` int16 branch, `V41CarryPlan.candidates`,
pack/unpack of `_CARRY_CAND`); every spot is marked `[fork fix-007]`. Verify:
`tests/fast/miles_plugins/models/deepseek_v4_1/ops/test_candidate_block_ids.py` (CPU), and on GPU the per-token
train log-probs of a patched and an unpatched run on the same batch must be bit-identical (`--dump-details`). On a
sync that touches these files, re-apply the same representation to the new code; drop it if upstream stops
materialising the dense mask.

## Remove when
The upstream DeepSeek-V4.1 plugin no longer builds a dense `[q, n_kv]` candidate mask or carries it per layer.
Status 2026-10-17: upstream main e1a791d3e3 unchanged here.
