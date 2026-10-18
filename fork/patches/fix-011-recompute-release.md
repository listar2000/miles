# fix-011: release full-recompute inputs after each checkpoint's backward

## Why
With `--recompute-granularity full`, Megatron's `CheckpointFunction.backward` recomputes each layer from a
detached copy of the layer input that requires grad. Modules keep tensors from that recompute after the
backward: the alltoall dispatcher's `self.probs` (`token_dispatcher.py:670`, reset only by the overlap
schedule), the mHC layer's `mlp_norm_manager`. Their autograd graph ends in the `AccumulateGrad` node of the
detached input, and that node holds the input and so its `.grad`. Each layer therefore keeps two copies of its
input alive from its backward until the next forward through it. On DeepSeek-V4.1 the layer input is the
4 x 5120 hidden plus the carry columns, so at 192k context this is ~3.6 GiB per layer and ~72 GiB on a
20-layer pipeline stage. Evidence: on a 4-layer DeepSeek-V4.1 test model (one node of 8 B300, 192k tokens), a count
of live GPU tensors after the first microbatch found `[24576, 1, 20489]` bf16 x7 and `[24576, 1, 33801]` x2 (2 per
layer + the next input, 9.7 GiB). On the full model across 2 nodes, the second pipeline stage (layers 20-39) held
+71.9 GiB per GPU after the first microbatch and ran out of memory in the second.

## Decision
Wrap `CheckpointFunction.backward`: record the grad-requiring tensors the recompute is called with, and after
the original backward set their `.grad` to `None` and rebind their `.data` to an empty tensor. The engine already
holds the returned grads, so no value changes (CPU test: parameter grads bit-identical). Alternatives that lost:
- Reset the dispatcher's transient state after backward: fixes one holder; `mlp_norm_manager` and any future
  module attribute would bring it back.
- Copy Megatron's backward with the release added: same effect, but drifts from upstream on every sync.
- Resize the storage to 0 instead of rebinding: an input passed to several checkpoints would lose the storage
  their saved copies still need.
Not covered: `te_checkpoint` (FP8/FP4 training) and `CheckpointWithoutOutput`.

## How it is applied
`_release_recompute_inputs()` in `miles/backends/megatron_utils/initialize.py`, called from `init()` after the
seeds are set; on by default for every Megatron trainer, `MILES_RECOMPUTE_RELEASE=0` turns it off. Verify:
`tests/fast/backends/megatron_utils/test_initialize.py::TestReleaseRecomputeInputs` (CPU, real
`CheckpointFunction`: 8 retained activation storages without the fix, 0 with it, identical grads). On a sync,
check that `CheckpointFunction.backward` still calls `ctx.run_function(*detached_inputs)`.

## Remove when
Upstream Megatron releases the recompute inputs itself, or no module keeps graph-carrying tensors from the
recompute. Status 2026-10-18: Megatron-LM 9df207f unchanged here.
