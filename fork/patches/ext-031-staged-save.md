# ext-031: Megatron saves land on node-local disk; each node uploads them to the results volume in the background

## Why
A DeepSeek V4.1 run on two B300:8 trainer nodes died at its first save: about 1 TB written
straight to the results volume at about 0.6 GB/s outlasted NCCL's 10-minute timeout. Probes on 2026-10-19
(`fork/probes/ckpt/`, one H100:8 container, 64 GiB per GPU = 512 GiB):

| What | Rate | 512 GiB takes |
| --- | --- | --- |
| Megatron torch_dist sync save to the volume (1 or 8 writer threads) | 0.39-0.46 GB/s | ~20 min |
| the same save to node-local disk | 2.2-2.9 GB/s | 3-4 min |
| copying those files to the volume, 16 at a time (two runs) | 0.45 and 2.4 GB/s | 20 and 4 min |
| Megatron `--async-save`, 8 GiB per GPU, mcore or temporal worker | 2nd save blocks <1 s | ~25 s in background |
| Megatron `--async-save`, nvrx + persistent worker (Megatron's default pair) | hung in `save()` | never |
| Megatron `--async-save`, mcore + persistent, 64 GiB per GPU | 51 s, then NCCL timeout | one rank finished >10 min before the rest; finalize all-reduce timed out |

CPU containers write to the same volume at 2.5-2.9 GB/s each. Why the copy rate swung 5x between
two back-to-back runs is not known.

## Decision
Save to node-local disk, upload in the background. Training blocks for the local save (3-4 min
for 512 GiB per node) plus, when a save comes before the previous upload is done, the rest of that
upload; the upload overlaps the next steps (one staged copy per node: next to
Adam's on-disk state on a 3 TiB disk). The tracker on the volume moves only after every node has
uploaded and committed its shards, so a resume never sees a half-uploaded checkpoint.

Rejected: Megatron async save with `--async-strategy mcore`. It works at 8 GiB per GPU, but at
64 GiB per GPU each rank writes to the volume at its own uneven rate, and the finalize all-reduce
timed out after 10 minutes waiting for the slowest rank (the same failure as the direct save). It
also holds about one more checkpoint of host RAM per node. Rejected: more writer threads (no faster to the
volume). Rejected: `--save` pointed at local disk outright (the rollout state, event log and tracker
the launcher watches live under `--save`, and a lost container loses them).

## How it is applied
- `checkpoint.stage_dir` in the recipe (e.g. `/tmp/miles-ckpt-stage`; None = off, the default)
  adds `--custom-megatron-save-context-path miles_ext.checkpoint.staged_save.context` (hook-007)
  and sets `MILES_EXT_SAVE_STAGE` in the trainer env.
- `context`: each rank waits for its node's previous upload, saves to the stage, marks
  `iter_N/.saved-rank<R>`.
- `Uploader` (one thread per node, started by `miles_ext/launch/modal_app.py`): copies each
  marked save to `<run>/checkpoints/iter_N`, commits, reports `(node, N, ranks)` on the completion
  queue (partition `upload-<run_id>`, one count per node, so a retried upload is not counted
  twice). Node 0 writes the tracker when the reports cover all trainer ranks; the supervisor
  commits it as for any save. Like the unstaged path, this assumes rank 0 runs on node 0, whose
  mount the supervisor watches.
- The supervisor takes `pending` (`Uploader.busy`): a save it requested counts as done, and the
  exit record reads the tracker, only once that save is in the tracker (up to
  `UPLOAD_TIMEOUT_SECS`, 30 min, heartbeating meanwhile). Head and workers then drain the
  uploader before their last commit: a drain returns False if the thread is still copying after
  the timeout (that copy goes on in the background); a copy the drain starts itself runs to the end.
- Verify: the trainer log prints `staged save: node K uploaded iteration N (... GiB/s)` per node,
  then `tracker -> N`; a relaunch with the same run id resumes from N.
- Needs `cluster.disk_gib` large enough for one save per node plus anything else on local disk.
- A segment's deadline does not count the upload: leave the function timeout room for it.
- Unit tests: `miles_ext/tests/test_staged_save.py`.
- Two-node check (2026-10-19, 2 x H200:8, Qwen3.5-9B, 117 GiB per save over 12 trainer ranks): four
  saves, training paused only 8-10 s each (waiting for the previous upload), each node uploaded in
  1-2.5 min (0.35-0.83 GiB/s), and a relaunch resumed from the uploaded checkpoint.

## Rerunning the probes
- `modal volume create --version=2 miles-ckpt-probe` once; every probe writes fake data only there.
- `modal run fork/probes/ckpt/ckpt_save_probe.py [--quick]`: CPU containers write `streams` files
  of `file_gb` GiB each to the volume (one thread per file, fsync per file, commit timed apart),
  several containers at once, then a local-disk-to-volume copy; `CKPT_PROBE_VOLUME` picks another
  volume, and `CKPT_PROBE_GCS_BUCKET=<bucket>` writes to a GCS bucket mount instead (HMAC key in
  the Modal secret `miles-gcs-hmac`). One JSON line per case and a SUMMARY.
- `modal run fork/probes/ckpt/ckpt_async_probe.py [--only <case>]`: one H100:8 node pushes random
  bf16 shards through Megatron's dist_checkpointing (the `--save` path) via
  `ckpt_async_worker.py` under torchrun. Cases: `sync-t1`/`sync-t8` (writer threads) and
  `sync-t1-local`; `async-{nvrx,mcore}-{persistent,temporal}` at 8 GiB per GPU, plus `async-mcore-persistent-64` and fix-043's `async-64-nccl2[-poll]` (nvrx +
  persistent hangs; Megatron defaults to nvrx and `--async-save` needs the persistent worker, so
  async recipes need `--async-strategy mcore`); `local-then-copy[-diskload]` at 64 GiB per GPU.
  One CASE JSON line per case (blocking and background seconds, peak host RAM, `/dev/shm`).
  py-spy cannot attach on Modal, and a case timeout does not reach torchrun's ranks (own process
  groups).

## Remove when
Megatron's async save works on the image and the volume takes GPU-container writes at local-disk
speed; upstream status: not filed.
