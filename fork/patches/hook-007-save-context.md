# hook-007: `--custom-megatron-save-context-path`, a context every trainer rank enters around each Megatron save

## Why
A DeepSeek-V4.1 run (2 B300 trainer nodes, ~64 GiB per GPU, ~1 TB per save) died at its first
weight save: written straight to the results Volume at ~0.6 GB/s in total, the save outlasted
NCCL's 10 min timeout. Probes on 2026-10-19 (`fork/probes/ckpt/`, one H100:8
container): Megatron's torch_dist sync save of 512 GiB per node reached the Volume at 0.39-0.46
GB/s (1 or 8 threads) and node-local disk at 2.2-2.9 GB/s (3-4 min). Copying those local files to
the Volume afterwards ran at 0.45 GB/s in one test and 2.4 GB/s in the next, cause unknown.
Megatron's own async save hung inside `save()` on this image (nvrx and persistent worker, at 8
and 64 GiB per GPU). Nothing in Miles lets a caller choose where a save is written.

## Decision
A context entered on every rank around `save_checkpoint` (and `save_checkpoint_with_lora`) in
`miles.backends.megatron_utils.model.save`. The caller decides what it does; the first user
(ext-031, `miles_ext.checkpoint.staged_save`) points `args.save` at node-local disk for the save and
marks the rank done, and a per-node uploader copies the files to the Volume while training goes
on. Rejected: `--custom-megatron-post-save-hook-path` (runs on rank 0 only, after the save has
already landed). Rejected: pointing `--save` itself at local disk (the rollout state, the event
log and the tracker the launcher watches all live under `--save`).

## How it is applied
The flag in `miles/utils/arguments.py` next to `--custom-megatron-post-save-hook-path`; in
`save()` a `with ExitStack()` block around the save call (`[fork hook-007]` markers), so the
context's exit also runs, with the exception, when a save fails. Unset, the stack is empty. On a
sync: re-indent whatever upstream's save call becomes into the block.

## Remove when
Upstream lets the caller stage or redirect Megatron checkpoint saves; upstream status: not filed.
