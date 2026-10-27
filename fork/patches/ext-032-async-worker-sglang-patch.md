# ext-032: Megatron's persistent async-save worker starts with sglang's torch patch applied

## Why
Miles' weight sync applies sglang's `monkey_patch_torch_reductions` in the trainer
(`miles/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py`). After that, every CUDA tensor
the trainer pickles names sglang's `_rebuild_cuda_tensor_modified`, which calls
`reductions._rebuild_cuda_tensor_original`, an attribute only that patch sets. With `--async-save`, the save
request travels to Megatron's persistent worker, a freshly spawned process that never ran the patch. On
2026-10-19 (two H100:8 nodes, Qwen3.5-9B, app ap-CXnQdniKzjtkXS3sc5lCF2), the workers died at the first
save with `AttributeError: module 'torch.multiprocessing.reductions' has no attribute
'_rebuild_cuda_tensor_original'`, and the trainer waited forever for their copy to the host.

## Decision
The worker's entry point (`PersistentAsyncCaller.async_loop`, read when Megatron spawns it) becomes
`miles_ext.checkpoint.async_worker._loop`, which applies the patch and then runs Megatron's loop. The
change reaches only that worker. Rejected: undoing the patch around each save (the queue pickles in a
background thread after `put` returns, so the undo can race it). Rejected: wrapping the reducer for every
process (the weight-sync receivers would depend on `miles_ext`). Rejected: Megatron's temporal (forked)
worker (`--async-save` requires `--use-persistent-ckpt-worker`).

## How it is applied
- `--custom-megatron-save-context-path miles_ext.checkpoint.async_worker.context` (hook-007). Each save calls
  `install()` first, and the first save spawns the worker. With staged saves (ext-031) it is not needed:
  they do not use `--async-save`.
- Probe: `modal run fork/probes/ckpt/ckpt_async_probe.py --only async-sglang-patched` (one H100:8, 8 GiB per
  GPU, the patch applied in each rank and the worker started through this module).

## Results
- 2026-10-19, one H100:8, `async-sglang-patched`: both saves reloaded correctly. The second save blocked
  for 0.6 s and finished in the background in 25 s; peak host RAM was 99 GiB.
- 2026-10-19, two H100:8 nodes (`smoke-ckpt-a2b`, app ap-O1uzknDfpQ0mY7PcP6ZhxO): the patched rebuild
  resolved, but the workers died one step later in `UntypedStorage._new_shared_cuda` with
  `RuntimeError: pidfd_getfd: Operation not permitted`. The trainer runs with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and the probe did not.
- 2026-10-19, one H100:8, `async-sglang-patched-expandable`: the same failure on all 8 ranks, so
  expandable segments are the cause.

## Status
Parked by user decision (2026-10-19), not merged. Megatron's async save cannot run under the Miles
trainer on Modal, so runs use staged saves ([ext-031](ext-031-staged-save.md)). The run gets past this
fix, then the persistent worker fails with `RuntimeError: pidfd_getfd: Operation not permitted`. The
trainer's `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` makes CUDA IPC fetch the sender's fd
with `pidfd_getfd`, which Modal's sandbox forbids. This was confirmed on one H100:8 (app
ap-qf6e8w2ngaDKCNay3NHwm3), where the same case passed with the default allocator. The workarounds
fail too: turning off expandable segments risks the memory fit, host shared memory is capped at
320 GiB, and Megatron turns `--async-save` off without the persistent worker. Off Modal (a host
that allows `pidfd_getfd`), fix-043 plus ext-032 would likely work; rerun the probes there first.

## Remove when
Megatron's persistent worker inherits the parent's tensor reducers, or sglang's patch stops being needed
(pytorch/pytorch#149248); upstream status: not filed.
