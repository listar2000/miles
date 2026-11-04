# hook-020: `--custom-megatron-save-context-path`, a context every trainer rank enters around each Megatron save

## Why
Nothing in Miles lets a caller change a Megatron save per save. Full-parameter GLM-5.3-Flash needs that: Adam's state
is ~2.6 TB, and a two-node save with it took 2,608 s on 2026-11-04, so a run should carry it in only some saves.
Work on where a save lands (local disk, background writes) needs the same place to act.

This is hook-007 (retired with ext-031 by fix-060 on 2026-10-27, because no recipe used it) again, under a new
id, for ext-056.

## Decision
A context entered on every rank around `save_checkpoint` (and `save_checkpoint_with_lora`) in
`miles.backends.megatron_utils.model.save`. The caller decides what it does; ext-056 sets `args.no_save_optim` per
save. Rejected: `--custom-megatron-post-save-hook-path` (rank 0 only, after the save has landed).

## How it is applied
The flag in `miles/utils/fork_arguments.py` (hook-014's file); in `save()` a `with ExitStack()` block around the
save call (`[fork hook-020]` markers), so the context's exit also runs, with the exception, when a save fails.
Unset, the stack is empty. Verify: `tests/fast/backends/megatron_utils/test_lora_model_branches.py` (needs a GPU
container). On a sync: re-indent whatever upstream's save call becomes into the block.

## Remove when
Upstream lets the caller stage or redirect Megatron checkpoint saves; upstream status: not filed.
