# ext-072: a mount mirrored to a GCS prefix

## Why
We want weight sync through GCS instead of a Modal volume, and rollout nodes off Modal need shared storage that is not
one. RFC 0005 left buckets to the caller's mount, but a container may not be able to mount (we run under gVisor on Modal),
and `write_sync` needs a rename that bucket mounts handle badly or not at all. Neither limit was tested here. This
supersedes ext-062's "no client for a bucket" for GCS.

## Decision
`MILES_GCS_MOUNTS` maps a mount to `gs://bucket/prefix`; `for_mount` returns `GcsMirror` for it and `MILES_STORAGE`'s
adapter for the rest. The mount is a local directory. `commit` uploads what changed by size and mtime, and deletes what
this node removed and knew about. `refresh` downloads newer generations and removes what vanished, keeping uncommitted
edits to tracked files; a file it never tracked yields to the bucket. `remove` deletes a tree on the bucket. Files over 256 MiB move in concurrent chunks. A commit is not atomic,
so a tracker uploads last. One mirror per mount in a process, so every committing thread shares what it knows.

Rejected: gcsfuse (needs FUSE, and directory renames are not atomic on a flat bucket), and `gcloud storage rsync`
(deletes what another node wrote, and adds a CLI to the image).

## How it is applied
`node/adapters/gcs.py`, loaded only when a mount is listed; the client is pinned in `stack.py`. Credentials are in
`fork/guide/image.md`. Verify: `pytest miles_ext/tests/test_node_gcs.py`; with a bucket, set `MILES_GCS_TEST_URI` (and
`MILES_GCS_TEST_TOKEN=$(gcloud auth print-access-token)` on a dev box with no ADC). Ran against
one of our buckets with a user token. A service account key and ADC have not run against a real bucket.

## Remove when
Never while a caller uses GCS. Not an upstream candidate.
