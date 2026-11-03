# fix-072: GLM-5.3 full recompute reuses each DSA layer's forward top-k

## Why
With `--recompute-granularity full` every GLM-5.3 layer runs its forward twice, once under no_grad inside
Megatron's checkpoint and once under grad in the backward. Each of the 11 DSA layers re-runs the kpool indexer
in the second pass, though it is no_grad and returns the same integers: at 256k that is 0.50 s per layer on a
B200 without fix-071 (0.072 s per rank with it), plus the all-reduce, repeated for every micro-batch.

## Decision
The DeepSeek-V4.1 design (fix-047), always on except with indexer replay. Each DSA attention keeps a FIFO of the
`QueryRows` its checkpointed forwards selected; the recompute pops the oldest instead of scoring again and
repeats only fix-071's all-reduce. The stash holds this rank's rows, not the assembled top-k: at 256k and TP8
that is 264 MiB per DSA layer and in-flight micro-batch (2.9 GiB for 11 layers) instead of 2.06 GiB per layer
(22.7 GiB). 1F1B recomputes micro-batches in forward order, so the FIFO never holds more than the pipeline
depth (asserted). `train()` empties it, so a pass that died between a forward and its recompute leaves nothing
behind. The forward-only log-prob pass runs in eval mode without checkpointing and never touches it. Refused
with indexer replay, which pops its own top-k in the recompute. Not wired for TE's checkpoint (fp8/fp4).
Alternatives that lost:
- stash the assembled top-k as fix-047 does: 8x the memory at TP8 to save one all-reduce per layer;
- selective recompute: too much activation memory at 256k.

## How it is applied
`miles_plugins/models/glm5_next/dsa.py` (`__init__`, `_kpool_select`, `forward`, `train`, `_index_rows`), marked
`[fork fix-072]`. Verify: `tests/fast/miles_plugins/models/glm5_next/test_kpool_recompute_topk.py` (CPU, Megatron
stubbed). On GPU, in the `miles-mobius:main` image: Megatron's `tensor_parallel.checkpoint` reports
`is_checkpointing()` True with grad off in the forward and True with grad on in the recompute, which is what
the FIFO keys on, and `dsa.py` imports with the real Megatron. A training step at 256k has not run yet.

## Remove when
Upstream's GLM-5.3 recompute skips the kpool indexer; upstream status: not filed.
