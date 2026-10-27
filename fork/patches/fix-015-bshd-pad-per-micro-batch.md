# fix-015: bshd pads each micro-batch to its own longest sample

## Why
With `--qkv-format bshd`, `get_rollout_data` padded every sample of the rollout batch to the batch's longest
sample (`max_seq_lens = [max(total_lengths)] * n`; upstream marks it "TODO: micro-batch wise dynamic"). Each
micro-batch then runs forward, recompute and backward over the full padded length, whatever its own length. On the
DeepSeek-V4.1-Flash overfit run (overfit-1018-a2, 2 nodes of 8 B300, TP4 PP2 CP2 EP8, micro-batch size 1, 64
samples, mean 139k tokens, longest 210k) forward and backward took 24.8 of the 46 min training step, and a third
of the tokens it processed were padding.

## Decision
Pad each micro-batch to its own longest sample, rounded up to the same pad multiple as before; every sample of a
micro-batch shares that length. Micro-batches are the contiguous `--micro-batch-size` slices `DataIterator` hands
out, or the explicit `micro_batch_indices` when the rollout side schedules them. Every consumer already reads the
length per sample (`max_seq_lens[i]`) or per micro-batch (`max_seq_lens[0]`), and Miles always runs Megatron with
`variable_seq_lengths`, so pipeline stages exchange each micro-batch's own shape. Padding sits after the last real
token, so real tokens see the same causal context; with context or sequence parallelism the shorter length moves
the split points between ranks, so results match to float rounding rather than bit for bit.

Each step also visits its micro-batches longest first (`get_data_iterator`, a stable sort by each micro-batch's longest
sample). In data order, every micro-batch longer than all before it made the GPU allocator and ext-020's pinned host
pool grow again: on the 2-node replay of overfit-1018-a2's batch such a forward took 25-45 s instead of 2-4 s on both
pipeline stages, about 1 min per new longest micro-batch, 8 times in that batch, which cancelled most of the gain.
Longest first, both reach their largest size on micro-batch 1, as with batch-max padding, and an OOM shows up there.
Samples stay in their step; only the order of gradient accumulation within a step changes (float rounding).
`aggregate_forward_results` now restores sample order whenever the iterator has explicit micro-batch indices, not
only with dynamic batch size.

Two side effects of many distinct lengths are handled in the same change:
- The DeepSeek-V4 rope table is cached per length (`lru_cache(32)` of GPU tables, up to 64 MiB each at 256k).
  `wrapped_precompute_freqs_cis` now builds the next power of two and slices it; rows depend only on position,
  so the values are bit-identical and the cache holds a few tables instead of up to 32.
- ext-020's pinned host pool now drops too-small free buffers before growing (see its page).

## How it is applied
`get_rollout_data` in `miles/backends/training_utils/data/rollout.py` (`_micro_batch_groups`,
`_micro_batch_max_seq_lens`), and `get_data_iterator` there (`_longest_micro_batch_first`); the order restore in
`aggregate_forward_results` (`miles/backends/training_utils/metrics/log_utils.py`). Always on
(fix-059 removed `MILES_BSHD_PAD_PER_MICRO_BATCH`). The rope change is in
`miles_plugins/models/deepseek_v4/ops/rope.py`. Verify:
`tests/fast/backends/training_utils/test_bshd_pad_per_micro_batch.py` (per-sample and per-micro-batch lengths,
longest-first order within each step, forward results back in sample order, rope tables bit-identical)
and `miles_ext/tests/test_ckpt_offload.py` (GPU: the pool stays at the working set).
On a sync, check that `get_rollout_data` still sets `max_seq_lens` and that upstream has not done the TODO itself.

## Remove when
Upstream pads bshd per micro-batch. Status 2026-10-18: upstream pads to the rollout batch's longest sample.
