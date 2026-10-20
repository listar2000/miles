# fix-047: DeepSeek-V4.1 full recompute reuses the index-source layer's forward top-k

## Why
With `--recompute-granularity full` every V4.1 layer runs its forward twice: once under no_grad inside Megatron's
checkpoint, once under grad in the backward. An index-source layer re-runs the lightning indexer in the second
pass, though the indexer is no_grad and, in deterministic mode, returns the same integers. On a four-layer slice of
the 256k DeepSeek-V4.1 overfit recipe (eight B200 GPUs), `indexer_fwd_kernel` was 4.0 s of 28.8 s train-pass kernel
time, half of it in recompute. Under all-gather context parallelism the rank holding the later half of the sequence
does about three times the causal indexer work, so rank 0 waited 8 s in the mixture-of-experts token-count all-gather.

## Decision
`--dsv4-reuse-indexer-topk` (default off). Each index-source attention keeps a FIFO of (top-k as int32, its dtype,
candidate blocks) from its checkpointed forwards; the recompute pops the oldest instead of calling the indexer.
1F1B recomputes micro-batches in forward order, so the FIFO never holds more than the pipeline depth (asserted).
`train()` empties it, so a train pass that died between a forward and its recompute leaves nothing behind. The
forward-only log-prob pass runs in eval mode without checkpointing and never touches it. Refused with indexer replay,
which pops its own top-k in the recompute. Not wired for TE's checkpoint (fp8/fp4), where the flag has no effect.
Alternatives that lost:
- Megatron `recompute_modules` (selective recompute): too much activation memory at 256k.
- `--use-indexer-replay`: still runs the scoring kernel, and needs ~185 GB of pinned host RAM per GPU.

## How it is applied
`miles_plugins/models/deepseek_v4_1/deepseek_v4_1.py` (`DeepSeekV41Attention._index_topk`, `train`) and
`miles_plugins/models/deepseek_v4/arguments.py`, every spot marked `[fork fix-047]`. Cost: one int32 top-k per
index-source layer and in-flight micro-batch on the GPU (~256 MiB per layer at 128k local tokens). Verify:
`tests/fast/miles_plugins/models/deepseek_v4_1/test_recompute_topk_reuse.py`. On GPU (author-run, logs not in the
repo): a same-container A/B of the four-layer slice, whose verify arm re-ran the indexer in every recompute and found
576 of 576 top-k identical; steady train pass 33.1 s without reuse, 24.8 s with it.

## Remove when
Upstream's DeepSeek-V4.1 recompute skips the indexer. Status 2026-10-20: not filed.
