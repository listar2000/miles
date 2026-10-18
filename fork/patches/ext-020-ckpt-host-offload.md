# ext-020: full-recompute activations in pinned host memory (layer inputs, then the recomputed layer)

## Why
With `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`, each layer keeps its
input on the GPU until its backward. On DeepSeek-V4.1-Flash at 256k (TP4 CP2, sequence parallel), one checkpoint
(hidden states plus carry columns) is 1.25 GiB per GPU. A 20-layer pipeline stage holds 20 per in-flight microbatch,
and 1F1B keeps two in flight on stage 0. The 2-node trainer OOMs at <=192k (investigations/trainer BELIEFS B18).
Megatron's `cpu_offloading` refuses both PP>1 and recompute.

A second term is the layer being recomputed: its backward re-runs the forward with grad and keeps 38-49 GiB per GPU
at 256k (saved-tensor census, slice). That memory is spread over HyperConnection (~10 GiB), attention (~8-10), MoE
(6-12) and fp32 HC copies, and backward consumes it almost in reverse save order: a ~5 GiB reverse-order prefetch
window covers it. PCIe is 43-54 GiB/s per GPU with all 8 copying at once, against ~6 s of compute per layer and
microbatch.

## Decision
- `torch.autograd.graph.saved_tensors_hooks` around Megatron's `tensor_parallel.checkpoint`, active only inside
  `TransformerBlock._checkpointed_forward` with grad enabled. Only the checkpoint's first argument (the hidden states)
  is moved. Masks and the RoPE tables are shared by every layer and stay on the GPU. Nested checkpoints inside a
  layer are not touched. The checkpoint is a pure copy, so results are bit-identical.
- The device-to-host copy runs on a side stream right after the save. In backward each checkpoint is copied back one
  layer ahead, so the GPU holds at most two.
- Host buffers come from an exact-size pool registered with `cudaHostRegister` and reused across steps. The pinned
  caching allocator rounds every block up to a power of two, which can double the host RAM.
- Plugged in with Miles' existing `--custom-megatron-init-path`, so no upstream file changes.
  - Rejected: patching `CheckpointFunction` itself, which is more invasive and would touch every caller.
  - Rejected: Megatron's fine-grained activation offload, whose hooks sit in the standard attention and MLP modules
    that the V4.1 plugin bypasses.
- bf16 only: the fp8/fp4 path (`te_checkpoint`) is not wrapped.
- `recompute_offload` (installs ckpt_offload too) wraps `CheckpointFunction.backward` in saved-tensor hooks. It
  enters them there because backward runs on autograd's thread. Every saved non-parameter tensor of >= 32 MiB is
  copied to host as it is saved. Copies come back in reverse save order with `MILES_RECOMPUTE_OFFLOAD_WINDOW_GIB`
  (default 8) prefetched ahead; an out-of-order unpack is copied on demand. A tensor object saved twice gets one copy.
  This was preferred over nested sub-module checkpoints because it is exact without re-running the indexer or relying
  on the runtime's side effects being idempotent, and it trades PCIe time for memory.
- The newest `MILES_RECOMPUTE_OFFLOAD_RESIDENT_GIB` (default: the window) of saved activations never leave the GPU,
  because backward needs them first. On the 128k slice this cut the warm train step from 16.7 s to 13.2 s for 1.1 GiB.
- Copies back are allocated on the compute stream and written on the side stream. Allocated under the side stream,
  their blocks went to that stream's pool, which backward cannot reuse; at 256k reserved memory then ran ~40 GiB
  above allocated.

## How it is applied
Pass `--custom-megatron-init-path miles_ext.memory.ckpt_offload.install` (layer inputs only) or
`miles_ext.memory.recompute_offload.install` (both). To check it is active, the log prints
"ckpt offload: N checkpoints, X GiB per microbatch; Y GiB pinned in total" once per shape.

Host RAM needed per GPU is that per-microbatch figure times the in-flight microbatches. Size the container's
`memory=` request for it. Tests: `miles_ext/tests/test_ckpt_offload.py` (CUDA; bit-identity, memory freed, buffer
reuse, pool best fit, window bound, dedupe, resident tail, copies back in the compute stream's pool).

On an upstream sync, check that `TransformerBlock._checkpointed_forward` still calls `tensor_parallel.checkpoint`
with the hidden states first.

## Remove when
Megatron supports offloading full-recompute checkpoints with PP>1, or the trainer no longer uses full recompute.
