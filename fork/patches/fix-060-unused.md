# fix-060: remove fork code no run uses

## Why
Three fork changes to upstream code served options no recipe or host run turned on (flag prune plan, user decision
2026-10-27), and each kept a flag and a branch in upstream files that every sync has to carry.

## Decision
- Retired: fix-045 (`--stream-optimizer-state-host-gb`, the pinned host tier of the NVMe optimizer store),
  hook-007 (`--custom-megatron-save-context-path`, used only by ext-031's staged saves) and hook-008 (the
  checkpoint-buffer log hook behind `--async-checkpoint-buffer`).
- fix-046 keeps its cheaper reset copy; its flag and the `prompts` mode go. Fully async saves always use the
  write-ahead group log (ext-034), which recipes always pass.
- The matching extension code goes in the same PR (ext-031 staged saves, ext-020's recompute offload, four
  recipe keys).
- Rejected: keeping them off by default. Unused switches still cost a reader and a sync.

## How it is applied
The upstream spots are back to their base form; `Retires: fix-045, hook-007, hook-008` and `Releases: fix-046 ...`
tell `ledger.py`, which accepts a removal that only restores base lines in place without a marker of its own.

## Remove when
Never: it only removes our own code.
