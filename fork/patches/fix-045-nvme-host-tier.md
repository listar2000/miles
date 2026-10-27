# fix-045: NVMe-streamed optimizer state in host memory up to a per-GPU budget

## Why
With `--stream-optimizer-state-to-disk` every optimizer step writes the whole fp32 main + moment state to the
node-local disk. On DeepSeek-V4.1-Flash (2x B300:8, PP2 EP8) that is 253 GB per rank, 2 TB per node, and the
container's disk takes it at ~1.4 GB/s: the 256k overfit run's step 0 spent 1454-1522 s per rank in the optimizer, out of
a 3055 s actor_train. The disk is the cap (~2.6 GB/s per container whatever the parallelism), so overlapping its I/O
cannot recover this; only fewer disk bytes can. B300:8 nodes accept memory requests up to 2,560 GiB.

## Decision
`--stream-optimizer-state-host-gb` (default 0) gives each GPU a budget of page-locked host memory. Buckets fill it in
order and the rest stay on disk. A host bucket holds the file's bytes in the file's layout (`_layout`), so
`save_to` / `load_from` write and read the same files in either tier and a checkpoint does not record the tier.
Copies between the GPU and a host bucket are one DMA per tensor (same dtype) or per staging chunk (the bf16
moments, cast on the GPU first, as `transfer` does). The Adam math is untouched: a bench run of the real store with
the same seeds gives byte-identical mains, moments and model params with and without the tier.
Host buckets are registered once with `cudaHostRegister` and stay pinned for the life of the trainer process (the
store lives as long as the optimizer); `pin_memory=True` was not used because the caching host allocator rounds each
buffer up to a power of two. A positive budget needs `--stream-optimizer-state-to-disk` with Adam: Muon's file-backed
offloader buffers do not read it, so the argument check refuses the combination.
Alternatives that lost:
- Pipelining disk reads and writes with Adam: the disk is already at its cap (fix-023 was dropped for this).
- Lower-precision mains or moments: changes the numerics.

## How it is applied
`miles_plugins/optimizers/nvme_stream.py` and `miles/utils/arguments.py`, every spot marked `[fork fix-045]`. The
store's startup line reports how many GB are in host memory. Size the budget to what the node's memory request
leaves after the other host users (ckpt_offload's pinned pool, Ray, the trainer processes). Verify:
`tests/fast/optimizers/test_nvme_stream_host_tier.py` (round trip equals the disk tier; checkpoint files swap between
tiers). On GPU (author-run, bench not in the repo): the real store over full-size DeepSeek-V4.1 expert shapes on
eight B200 GPUs gave byte-identical state and a slowest-rank steady step of 1,541 s on disk vs 1,064 s with
100 GiB/GPU in host memory; with a 600 GiB stand-in for the trainer's other host memory, 1,428 of the 1,536 GiB
request was in use.

## Remove when
Upstream `nvme_stream` gains a host-memory tier. Status 2026-10-20: not filed.

## Removed 2026-10-27

The flag prune removed this change (fix-060 for fixes and hooks, ext commits for extensions). The page
records what it did; the code is in git history and in images built before the prune.
