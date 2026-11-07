# ext-074: weight sync on a mount of its own

## Why
Weight sync shared `/results` with checkpoints. A commit on a mirrored `/results` (ext-072) uploads a save in progress
too, so a sync waits on it, and mirroring `/results` would also move every checkpoint to the bucket. We want to try GCS
for weights while checkpoints stay on the Modal volume.

## Decision
`MILES_WEIGHT_SYNC_MOUNT` names the root for `<run_id>/gpu_delta` and `<run_id>/adapters`; unset, it is `/results`, and
argv and behavior are unchanged. Both roles must set it. The trainer's commit watcher and the rollout agent use that
mount's `Storage`. With `MILES_GCS_MOUNTS` mapping it, a rollout node needs no shared `/results` for weights. Rejected:
filtering sync paths inside the mirror, since those files would still sit on the Modal volume and be committed there.

## How it is applied
`recipe.weight_sync_root()` owns the path. `modal_app` carries both env vars into its image through
`contract.storage_env`; credentials stay in the run's secret. Verify: `pytest miles_ext/tests/test_node_gcs.py
miles_ext/tests/test_split_gpu_delta.py`. Not run on GPUs, or with real engines reading from the mirror.

## Remove when
Never while a caller uses it. Not an upstream candidate.
