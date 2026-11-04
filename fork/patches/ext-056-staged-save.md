# ext-056: staged Megatron saves with a parallel upload, Adam's state on every k-th save, retention while training

## Why
Full-parameter GLM-5.3-Flash (a two-step split-mode run on one B300:8 trainer node, 2026-11-03) paused training 1,281 s to
save 627 GB of BF16 weights to the results volume. Adam's state would add ~2.3 TB per save. Modal
containers live at most 24 h and GPU functions can be preempted, so long runs need regular saves, and RL
runs need older steps to backtrack to. The launcher applied retention only at the end of a run.

Measurements, 2026-11-04 (probes in `fork/probes/ckpt/`, volume v2):

| What | Result |
| --- | --- |
| B300:1 container, 64 GiB as 8 files to the volume (raw / torch.save pattern) | 4.3 / 4.9 GB/s; node-local disk 13 GB/s (short writes: buffered) |
| Megatron sync save, one B300 rank, 64 GiB, to the volume / to local disk | 0.79 / 2.8 GB/s (host preload 4 s) |
| Megatron sync save, idle B300:8, 8 x 78 GiB to the volume | 816 s = 0.8 GB/s (preload 31 s) |
| GLM-5.3-Flash trainer, one B300:8, 627 GB weights save to local disk (`stage_dir`) | 939 s; the upload then took 2,722 s beside the disk-streamed optimizer |
| GLM-5.3-Flash trainer, two B300:8 (Adam in host memory), one save with weights and Adam's state (3,410 GB) to the volume | 2,608 s = 1.3 GB/s (~0.65 per node) |
| Qwen3.5-9B, one H200:8, staged weights-only save / save with Adam's state to the volume | training paused 5-16 s / 59-74 s (different contents) |

A container sustains ~0.65-0.8 GB/s of long checkpoint writes to the volume, and its local NVMe under the GLM
trainer is no faster, so staging does not shorten a GLM save on one node; it does for models whose save fits
the page cache. More trainer nodes divide a save between their containers.

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
- Not done here: Megatron `--async-save` (ext-031's page: hung or timed out at 64 GiB per GPU with the 10 min
  collective timeout); it would also hold a full host copy of the weights during the write.
  Rejected: more writer threads per rank for direct saves (the volume took 0.8 GB/s from 8 ranks at once).

## How it is applied
`miles_ext/checkpoint/staged_save.py` (`context` for hook-020, `Uploader`, `with_optimizer`, `has_optimizer`),
the launcher's uploaders and retention thread (`miles_ext/launch/modal_app.py`), the supervisor's pending-upload wait,
`miles_ext/platform/checkpoints.py` (`keep_every`, the Adam-save pin), recipe keys in `miles_ext/recipes/recipe.py`.
Verify: the trainer log prints `save: iteration N to the node-local stage (...) in S s`, then
`staged save: node K uploaded iteration N (... GiB/s)` and `tracker -> N`; a relaunch with the same run id resumes.
Needs `cluster.disk_gib` for one staged save per node beside anything else on local disk. Tests:
`miles_ext/tests/test_staged_save.py`.

## Rerunning the probes
- `modal volume create --version=2 miles-ckpt-probe` once; the probes write fake data only there.
- `CKPT_PROBE_GPU=B300:1 modal run fork/probes/ckpt/ckpt_stream_probe.py`: N parallel files to the volume (raw,
  torch.save, multi-thread pwrite), node-local disk, local-to-volume copies. `CKPT_PROBE_GIB`, `CKPT_PROBE_STREAMS`
  and `CKPT_PROBE_VOL_ONLY=1` (vol-N cases only, with a 30 s rate ticker) size a long sustained write.
- `CKPT_PROBE_GPU=B300:8 CKPT_PROBE_MEMORY_GIB=2560 CKPT_PROBE_IMAGE=starlimobius/miles-mobius:main modal run
  fork/probes/ckpt/ckpt_async_probe.py --only sync-t1-78`: a real-size Megatron sync save; the CASE line times
  the host preload apart.

## Remove when
Megatron's async save works on the image and a direct save reaches the volume at local-disk speed; upstream
status: not filed.
