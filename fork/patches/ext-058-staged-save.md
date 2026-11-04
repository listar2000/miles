# ext-058: staged Megatron saves with a parallel upload (draft: does not help one-node GLM)

## Why
Full-parameter GLM-5.3-Flash (a two-step split-mode run on one B300:8 trainer node, 2026-11-03) paused training 1,281 s to
save 627 GB of BF16 weights to the results volume. Adam's state would add ~2.3 TB per save. Modal
containers live at most 24 h and GPU functions can be preempted, so long runs need regular saves, and RL
runs need older steps to backtrack to.

Measurements, 2026-11-04 (probes in `fork/probes/ckpt/`, volume v2):

| What | Result |
| --- | --- |
| B300:1 container, 64 GiB as 8 files to the volume (raw / torch.save pattern) | 4.3 / 4.9 GB/s; node-local disk 13 GB/s (short writes: buffered) |
| Megatron sync save, one B300 rank, 64 GiB, to the volume / to local disk | 0.79 / 2.8 GB/s (host preload 4 s) |
| Megatron sync save, idle B300:8, 8 x 78 GiB to the volume | 816 s = 0.8 GB/s (preload 31 s) |
| GLM-5.3-Flash trainer, one B300:8, 627 GB weights save to local disk (`stage_dir`) | 939 s; the upload then took 2,722 s beside the disk-streamed optimizer |
| GLM-5.3-Flash trainer, two B300:8 (Adam in host memory), one save with weights and Adam's state (3,410 GB) to the volume | 2,608 s = 1.3 GB/s (~0.65 per node) |
| Qwen3.5-9B, one H200:8, staged weights-only save / save with Adam's state to the volume | training paused 5-16 s / 59-74 s (different contents) |

In these configurations a container wrote long checkpoints to the volume at ~0.65-0.8 GB/s, and its local NVMe
under the GLM trainer was no faster, so staging did not shorten a one-node GLM save. Whether it shortens a smaller
model's save is not measured like for like (untested: saves that fit the page cache). Two trainer nodes split a
save between their containers.

## Decision
- `checkpoint.stage_dir`: ext-031 (removed 2026-10-27, unused) again, on hook-020. Each rank saves to node-local
  disk; one uploader thread per node copies the save to the volume while training goes on, and the tracker
  moves only when every node has uploaded (never backwards). The upload copies 1 GiB pieces of each shard
  with 32 workers, because a torch_dist shard is one file per rank.
- Not done here: Megatron `--async-save` (ext-031's page: hung or timed out at 64 GiB per GPU with the 10 min
  collective timeout); it would also hold a full host copy of the weights during the write.
  Rejected: more writer threads per rank for direct saves (the volume took 0.8 GB/s from 8 ranks at once).

## How it is applied
`miles_ext/checkpoint/staged_save.py` (`context` for hook-020, `Uploader`), the launcher's uploaders
(`miles_ext/launch/modal_app.py`), the supervisor's pending-upload wait, `checkpoint.stage_dir` in `miles_ext/recipes/recipe.py`.
A save carrying Adam's state (ext-056) goes straight to the volume: it does not fit beside the disk-streamed optimizer.
Verify: the trainer log prints `save: iteration N to the node-local stage (...) in S s`, then
`staged save: node K uploaded iteration N (... GiB/s)` and `tracker -> N`; a relaunch with the same run id resumes.
Needs `cluster.disk_gib` for one staged save per node beside anything else on local disk. With `stage_dir`,
trainer rank 0 must run in container 0 (whose uploader writes the tracker and gates the exit record); the launcher
sets `MILES_EXT_SAVE_NODE` per container and a save refuses to run otherwise. Tests:
`miles_ext/tests/test_staged_save.py`.

## Remove when
Megatron's async save works on the image and a direct save reaches the volume at local-disk speed; upstream
status: not filed.
