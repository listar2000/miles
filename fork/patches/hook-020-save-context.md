# hook-020: `--custom-megatron-save-context-path`, a context every trainer rank enters around each Megatron save

## Why
Full-parameter GLM-5.3-Flash on one B300:8 trainer node (a two-step split-mode run, 2026-11-03) paused training for
1,281 s to save 627 GB of weights straight to the results volume (about 0.5 GB/s). Probes on 2026-11-04
(`fork/probes/ckpt/`) show the same save landing on node-local disk several times faster, and a B300:8
Megatron sync save of 8 x 78 GiB to the volume took 816 s on an otherwise idle node. Nothing in Miles lets a
caller choose, per save, where a save is written or whether it carries the optimizer state.

This is hook-007 (retired with ext-031 by fix-060 on 2026-10-27, because no recipe used it) again, under a new
id, for ext-056.

## Decision
A context entered on every rank around `save_checkpoint` (and `save_checkpoint_with_lora`) in
`miles.backends.megatron_utils.model.save`. The caller decides what it does; ext-056 points `args.save` at
node-local disk and sets `args.no_save_optim` per save. Rejected: `--custom-megatron-post-save-hook-path` (rank 0
only, after the save has landed). Rejected: pointing `--save` itself at local disk (the rollout state, the event
log and the tracker the launcher watches all live under `--save`).

## How it is applied
The flag in `miles/utils/fork_arguments.py` (hook-014's file); in `save()` a `with ExitStack()` block around the
save call (`[fork hook-020]` markers), so the context's exit also runs, with the exception, when a save fails.
Unset, the stack is empty. On a sync: re-indent whatever upstream's save call becomes into the block.

## Remove when
Upstream lets the caller stage or redirect Megatron checkpoint saves; upstream status: not filed.
