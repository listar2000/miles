# ext-062: node-side storage goes through `miles_ext.node.storage`

## Why
RFC 0005: node code called `volume.commit()` and `reload()` on Modal volumes. A caller on another provider mounts a
shared path or a bucket instead.

## Decision
`Storage` has `commit`, `refresh` and `remove`; `for_mount(mount)` picks the adapter from `MILES_STORAGE`: `posix`
(default, no-ops, since every node sees the same files), `modal-volume` (`node/adapters/modal_volume.py`, volume names
in `MILES_MODAL_VOLUMES`) or a dotted path. Rejected: storage URIs and a Python client for each bucket; a caller mounts
buckets, and the image only needs the flush rule. Call sites keep the order they had; `shards`, `supervisor` and
`compile_cache` already took a `commit` callable, so only what they get changed.

## How it is applied
`modal_app` sets both env vars on its image and calls `for_mount` inside the node. `split.agent` and
`remove_other_node_shards` take a `Storage`. Verify: `test_node_storage.py`, `test_shards.py`, `test_supervisor.py`;
a 2-step run with a resume. Caller tools (`sft/modal_*`, `obs/sandbox_bench`, `gpu_delta/run_glm_bench`) keep their own
volumes: they leave with the launcher.

## Remove when
Never while the image has callers. Not an upstream candidate.
