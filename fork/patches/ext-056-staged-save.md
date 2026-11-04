# ext-056: staged Megatron saves with a parallel upload, Adam's state on every k-th save, retention while training

## Why
Full-parameter GLM-5.3-Flash (run glm53-gpud-full-1103-a12, one B300:8 trainer) paused training 1,281 s to
save 627 GB of BF16 weights to the results volume. Adam's state would add ~2.3 TB per save. Modal
containers live at most 24 h and GPU functions can be preempted, so long runs need regular saves, and RL
runs need older steps to backtrack to. The launcher applied retention only at the end of a run.

Probes, 2026-11-04 (`fork/probes/ckpt/`, volume v2):

| What | Rate |
| --- | --- |
| CPU container, 64 GiB as 8 files to the volume (raw / torch.save pattern) | 2.0 / 1.8 GB/s |
| B300:1 container, the same | 4.3 / 4.9 GB/s; node-local disk 13 GB/s |
| Megatron sync save, one B300 rank, 64 GiB, to the volume / to local disk | 0.79 / 2.8 GB/s (host preload 4 s) |
| Megatron sync save, B300:8, 8 x 78 GiB to the volume, nothing else running | 816 s = 0.8 GB/s (preload 31 s) |

## Decision
- `checkpoint.stage_dir`: ext-031 (removed 2026-10-27, unused) again, on hook-020. Each rank saves to node-local
  disk; one uploader thread per node copies the save to the volume while training goes on, and the tracker
  moves only when every node has uploaded (never backwards). The upload copies 1 GiB pieces of each shard
  with 32 workers, because a torch_dist shard is one file per rank.
- `checkpoint.optimizer_every` (k): with `save_optimizer`, Adam's state goes in every k-th save; those go
  straight to the volume, because Adam's ~2.3 TB does not fit on the 3 TiB disk beside the disk-streamed
  optimizer, and carry `iter_N/.optimizer`. A resume from a save without it passes `--no-load-optim`.
- `checkpoint.keep_every`: retention also keeps saves at multiples of it. The launcher applies retention each time
  the tracker moves (newest `keep`, the tracker's save, the newest save with Adam's state, and `keep_every` steps).
- Rejected: Megatron `--async-save` (ext-031's page: hung or timed out on this image at 64 GiB per GPU).
  Rejected: more writer threads per rank for direct saves (the volume took 0.8 GB/s from 8 ranks at once).

## How it is applied
`miles_ext/checkpoint/staged_save.py` (`context` for hook-020, `Uploader`, `with_optimizer`, `has_optimizer`),
the launcher's uploaders and retention thread (`miles_ext/launch/modal_app.py`), the supervisor's pending-upload wait,
`miles_ext/platform/checkpoints.py` (`keep_every`, the Adam-save pin), recipe keys in `miles_ext/recipes/recipe.py`.
Verify: the trainer log prints `save: iteration N to the node-local stage (...) in S s`, then
`staged save: node K uploaded iteration N (... GiB/s)` and `tracker -> N`; a relaunch with the same run id resumes.
Needs `cluster.disk_gib` for one staged save per node beside anything else on local disk. Tests:
`miles_ext/tests/test_staged_save.py`.

## Remove when
Megatron's async save works on the image and a direct save reaches the volume at local-disk speed; upstream
status: not filed.
