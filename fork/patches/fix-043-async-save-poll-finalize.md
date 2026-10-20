# fix-043: Miles finalizes Megatron async saves in short rounds every rank joins

## Why
With `--async-save`, Miles waits for the previous save at the next save, on a forced
sync and on a hot restart (`_finalize_pending_async_save`, `miles/backends/megatron_utils/actor.py`),
using Megatron's blocking finalize. Each rank waits for its own writer, then enters a single
all-reduce (`sync_all_async_calls`, `megatron/core/dist_checkpointing/strategies/async_utils.py`).
A rank that finishes early waits there for the slowest one. At 64 GiB per GPU to a Modal Volume
(probe `async-mcore-persistent-64`, 2026-10-19), one rank finished more than 10 minutes before the
rest, and the all-reduce hit the process group timeout (`ran for 600026 ms`).

## Decision
Repeat Megatron's own non-blocking finalize (`maybe_finalize_async_save(blocking=False)`) every
`ASYNC_SAVE_POLL_SECS` (5 s) until the async queue is empty, then make the blocking call, which
now only reaps deletion processes. Every rank joins each round, so no collective waits on the
writers. The queue empties on every rank in the same round, because "done" is the all-reduced value.
Rejected: a longer `--distributed-timeout-minutes` (it only moves the limit, and also loosens every
other collective). Rejected: patching Megatron's `is_current_async_call_done` (the same
change, but in a dependency).

## How it is applied
- Only the Miles call site changes; Megatron is untouched. Off unless `--async-save` is set.
- Async recipes also need `--async-strategy mcore --use-persistent-ckpt-worker` (nvrx + persistent
  hangs, see [ext-031](ext-031-staged-save.md)).
- `save_model` used to finalize inside the rank-0-only post-save-hook branch, which hangs any collective
  finalize. Every rank now finalizes before that branch, and the call inside it finds the queue empty.
- Probe: `modal run fork/probes/ckpt/ckpt_async_probe.py --only async-64-nccl2-poll,async-64-nccl2`.
  Both cases use a 2-minute process group timeout. The worker compares its own copy of this polling
  loop with Megatron's blocking finalize; it does not call the Miles actor method.
- Results (2026-10-19, H100:8, 64 GiB per GPU, 2 saves each, reload checked): polling blocked for
  92.5 s and 3.6 s, then ran 255 s and 250 s in the background (2.17 GB/s); the blocking finalize took
  72.2 s and 3.4 s, then 220 s and 235 s (2.31 GB/s). Neither timed out, because the ranks finished within
  2 minutes of each other: the runs show that both variants complete and reload. They do not reproduce
  the timeout or test the Miles path. Peak host RAM was about 548 GiB per node (the async host copy of
  the save). Megatron logs that its mcore async strategy is deprecated in favor of nvrx.
- The Miles path (uneven finishes, save and resume): a two-node run, pending.

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
Megatron's blocking finalize stops parking finished ranks in a collective; upstream status: not filed.
