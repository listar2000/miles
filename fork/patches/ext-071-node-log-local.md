# ext-071: a node keeps its log off storage that is not live

## Why
ext-070's `python -m miles_ext.node trainer|rollout` kept `tee` open on `/results/<run_id>/logs/`. Modal documents that
a volume's `reload()` fails while files on it are open, and a rollout node reloads on every weight sync, so that command
on `modal-volume` would fail its first sync. `modal_app` calls the roles directly, so no run hit this. Not reproduced on
Modal: the reload failure is from Modal's docs and the old docstring here.

The same reading found node jobs that wrote or read without the storage: `prepare-weights` and `convert-fsdp` never
committed, `convert-fsdp`, `sft-prepare` and `sft-train` never refreshed, the compile cache restored with no refresh, and
`MILES_STATE=file` was accepted where a caller cannot see the files.

## Decision
`Storage.live` says every node sees writes as they happen and an open file blocks nothing: true for `posix`, false for
`modal-volume`. A role on storage that is not live writes its log under `/tmp/miles-logs`, then appends it to
`logs/<name>.log` and commits on exit. A hard kill loses the copy on storage; the node's stdout stays the live copy.
Rejected: copying on a timer, which reopens the race with `reload()`. `MILES_STATE=file` on such storage is refused at
the command line, before any work.

## How it is applied
`runtime.node_log` wraps the role in `__main__`; `runtime.restore_compile_cache` refreshes first and never fails a run.
Verify: `pytest miles_ext/tests/test_node_state.py miles_ext/tests/test_node_contract.py`. A caller's own adapter must
now set `live`.

## Remove when
Never while the image has callers. Not an upstream candidate.
