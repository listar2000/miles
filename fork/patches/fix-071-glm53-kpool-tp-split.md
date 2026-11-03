# fix-071: GLM-5.3 kpool indexer query chunks split across tensor-parallel ranks

## Why
GLM-5.3 DSA gathers `index_q` and the head weights over sequence parallelism, so every TP rank scores every
query and selects the same top-k. At TP8 seven eighths of the indexer work and its transient memory are
duplicate: at 256k one rank spends 0.50 s per DSA layer call (B200) on rows seven other ranks also compute.

## Decision
The DeepSeek-V4.1 design (fix-052), always on except with indexer replay (fix-059: a fix has no switch).
Each TP rank scores every tp-th query chunk (chunks of `8192 / tp` rows, rounded up to 64 rows so chunk starts
stay aligned; round-robin so the causal cost stays balanced) and the token ids are assembled with one int32
sum all-reduce over the TP group: each row has exactly one writer and is zero elsewhere, so the sum is exact.
`kpool_select_topk` becomes `kpool_select_rows` (this rank's rows, a `QueryRows`) followed by
`gather_query_rows`; fix-072 stashes the `QueryRows` between them. No chunk-size agreement check as in fix-052:
the chunk is a constant here, not an env var. Indexer replay records and replays a whole-sequence top-k per
rank, so with replay every rank selects all rows. Alternatives that lost:
- all-gather of each rank's rows: unequal parts and reordering of round-robin chunks; the all-reduce places
  disjoint rows directly (fix-052's reasoning);
- splitting the heads of the indexer: the top-k needs the sum over all heads, so every rank would all-reduce
  the fp32 scores.

## How it is applied
`miles_plugins/models/glm5_next/ops/kpool_indexer.py` (`QueryRows`, `kpool_select_rows`, `gather_query_rows`)
and `miles_plugins/models/glm5_next/dsa.py` (`_kpool_select` passes `pg_collection.tp`), marked
`[fork fix-071]`. Cost per DSA layer call: one `[T, 2112]` int32 buffer (2.06 GiB at 256k, the size of the
top-k the attention needs anyway) and its all-reduce, whose time was not measured. Verify:
`tests/fast/miles_plugins/models/glm5_next/ops/test_kpool_query_chunks.py` (2 and 3 gloo ranks against the
unsplit top-k). On GPU (same B200 check as fix-070, eight ranks played in one process with the all-reduce done
as a sum): assembled token ids bit-identical to the unsplit ones at 16k, 32k, 64k and 256k; at 256k one rank
selects 32768 rows in 0.072 s instead of 0.50 s for all rows, peak 2.32 GiB above inputs including the
2.06 GiB assembled buffer. Not yet run with real NCCL ranks.

## Remove when
Upstream's GLM-5.3 kpool indexer shares its work across tensor-parallel ranks; upstream status: not filed.
