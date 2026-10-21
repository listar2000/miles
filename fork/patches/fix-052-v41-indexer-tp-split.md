# fix-052: DeepSeek-V4.1 indexer query chunks split across tensor-parallel ranks

## Why
The V4.1 lightning indexer runs before attention on activations every tensor-parallel (TP) rank holds in full, so
all four TP ranks of the V4.1 LoRA overfit layout (TP4 CP2 EP8 on one B300 node) score every query and get the same
top-k: three quarters of the indexer work is repeated. In that run's trainer, attention plus indexer is the largest
kernel family of the train pass (27% on rank 4 of a full-model profile). With fix-047 the indexer runs only in the
forward, which is where the duplicate sits.

## Decision
`--dsv4-indexer-tp-split` (default off). `indexer_select` cuts the queries into chunks of `query_chunk / tp` rows
and each TP rank scores every tp-th chunk (round-robin, so the causal cost stays balanced). Each index-source call
assembles the top-k rows with one integer sum all-reduce over the TP group; the candidate-source layer does a second
all-reduce for its candidate block ids. Each row has exactly one writer and is zero on the other ranks, so the sums
are exact. Candidate ids travel as int32 (NCCL has no int16) and are narrowed back bit for bit. The first split call
checks once (MIN all-reduce) that the TP ranks agree on the configured query chunk, which comes from a per-process
env var. Refused with indexer replay, which records and replays whole-sequence top-k per rank.
Alternatives that lost:
- all-gather of each rank's rows: needs unequal parts handled and round-robin chunks reordered; the all-reduce
  assembles disjoint rows directly. Its own time was not isolated in the profile.
- keeping 8192-row chunks per rank (`MILES_DSV41_INDEXER_QUERY_CHUNK=32768`): same speed as 2048-row chunks.

## How it is applied
`miles_plugins/models/deepseek_v4_1/ops/indexer.py` (`indexer_select`, `_gather_query_chunks`,
`DeepSeekV41Indexer`), `miles_plugins/models/deepseek_v4_1/deepseek_v4_1.py` (`DeepSeekV41Attention`,
`get_dsv41_spec`) and `miles_plugins/models/deepseek_v4/arguments.py`, every spot marked `[fork fix-052]`. Cost: per
index-source call, one `[bsz, seqlen, topk]` int64 buffer and one all-reduce; the candidate-source call adds a full
int32 candidate buffer and a second all-reduce. Recompute repeats this unless top-k reuse (fix-047) is on. Verify:
`tests/fast/miles_plugins/models/deepseek_v4_1/ops/test_indexer_tp_split.py` (3 gloo ranks, split vs unsplit
top-k, candidates and the consumer top-k).

On GPU (author-run trainer-only replays of two saved batches of the 16-step V4.1 LoRA overfit run,
16 samples each, that run's image and args plus `--skip-actor-forward-only`, one B300 node per arm):
- top-k: 704 of 704 split calls identical to the unsplit result computed in the same process;
- step-0 loss bit-identical. Step-1 loss (all arms with `--skip-actor-forward-only`) ranged -0.025704 to -0.025621
  split and -0.025681 to -0.025667 unsplit; the backward is not bit-deterministic, and these runs do not establish
  equivalence after an update;
- peak allocated GPU memory unchanged at 211.52 GiB (max across ranks); peak reserved 219.48 GiB unsplit vs
  222.71 GiB split (+3.2 GiB);
- speed, three arms started together, micro-batches 2-16 of the second batch: forward 113.1 s unsplit vs 95.0 and
  88.6 s split, forward+backward 241.8 s vs 224.3 and 217.7 s (-7 to -10%). These exclude
  micro-batch 1 (the longest, which also pays one-time host-memory pinning that varied 3-10 min between machines).
  An earlier replay with 8192-row chunks per rank agrees (226.0 s forward+backward).
- one outlier: one uninstrumented replay of the same split code and args ran micro-batches 2-16 in 338.5 s against
  250.3 s unsplit (+35%), while the two later replays were fast; the cause was not found (a later split arm with
  NCCL INIT logging was fast).

## Remove when
Upstream's DeepSeek-V4.1 indexer shares its work across tensor-parallel ranks. Status 2026-10-21: not filed.
