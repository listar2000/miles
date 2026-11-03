# fix-070: GLM-5.3 kpool indexer scores and selects in query-row chunks

## Why
`kpool_select_topk` asked `indexer_logits` for the dense fp32 scores of every query against every pooled key,
`[T, T_pool]` with `T_pool = T / kpool`, before taking the top-k. At one 256k-token sample (kpool 4) that is
262144 x 65536 x 4 B = 64 GiB per DSA layer call, on every TP rank; measured on one B200, the whole call peaks
at 67.6 GiB above its inputs. Full-parameter GLM-5.3-Flash on one B300:8 node cannot hold it.

## Decision
Score and select 8192 query rows at a time (`_QUERY_CHUNK`, a constant) and concatenate the selected token ids,
the design of the DeepSeek-V4.1 query chunks (`INDEXER_QUERY_CHUNK`, fix-007 / fix-008). Every row is scored
and selected on its own, so the top-k is the same; the transient is `8192 x T_pool x 4 B` (2 GiB at 256k).
Indexer replay records and replays one top-k per call, so with replay the call stays one chunk (upstream's
dense path). Alternatives that lost:
- a fused score-and-select kernel that never writes the scores: a new kernel for a memory problem chunking solves;
- chunking only past a length threshold: two code paths for one behavior.

## How it is applied
`miles_plugins/models/glm5_next/ops/kpool_indexer.py`: `kpool_select_topk` loops over row chunks and
`_select_rows` holds upstream's scoring and selection for one chunk; marked `[fork fix-070]`. Verify:
`tests/fast/miles_plugins/models/glm5_next/ops/test_kpool_query_chunks.py` (CPU reference scorer and selector,
chunks of 1 to 121 rows against one call). On GPU (one B200, the `miles-mobius:main` image with this branch
overlaid, GLM-5.3 indexer shape: 32 heads x 128, top-k 2048, kpool 4, random inputs): token ids bit-identical
to the single call for packs of 16k, 32k (one and three sequences) and 64k tokens and for one 256k sequence;
peak above inputs 4.13 GiB chunked vs 67.57 GiB whole at 256k (2.06 GiB of it is the `[T, 2112]` int32
output), 1.06 vs 4.89 GiB at 64k; time 0.50 s vs 0.52 s at 256k. On a sync that touches this file, keep the
row chunks around upstream's scoring and selection.

## Remove when
Upstream's GLM-5.3 kpool indexer no longer builds the `[T, T_pool]` scores for the whole sequence;
upstream status: not filed.
