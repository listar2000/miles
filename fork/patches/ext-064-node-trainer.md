# ext-064: the trainer role with no Modal

## Why
RFC 0005: `modal_app`'s `_head` and `_worker` are generic apart from `modal.Cluster`, a `modal.Dict` and a `modal.Queue`.

## Decision
- `miles_ext/node/trainer.py` is their body, moved as it was. Rank and addresses come from `MILES_NODE_RANK`,
  `MILES_NNODES`, `MILES_HEAD_ADDR`, `MILES_NODE_IP`; a public session URL from `MILES_SESSION_PUBLIC_URL`. The
  deadline is `MILES_SEGMENT_DEADLINE`.
- Worker release and shard commit tokens go through `miles_ext/node/gang.py`, an HTTP server on rank 0 on the gang's
  private network. `shards` is unchanged: it got a dict and a queue, and still does. Rejected: Ray's key-value store (a
  driver that loses the GCS may exit before a worker's final commit, and it needs private API to go without a driver).
- A worker that cannot reach rank 0 for 5 min fails, where it used to wait on a queue for the function's whole timeout.
  The result of a run stays readable and each worker confirms it, so a lost reply costs a poll; rank 0 stays up at most 60 s for confirmations.
- Run state stays dict-shaped. `MILES_STATE`: `file` (`<run>/state/*.json`, the default a non-Modal caller gets),
  `modal-dict` (the progress Dict, so `miles_ext.platform` reads it as before).
- `platform.progress`, `layout`, `checkpoints`, `images` import Modal only inside the functions that use it, and sandbox
  adapters move to `miles_ext/sandboxes/cleanup.py`, so all of this imports with no Modal client.
- The completion queue is gone. A running segment of an older image still has the queue; nothing here touches it.

## How it is applied
`modal_app.train` maps `Cluster.from_context()` to the env, opens tunnels on rank 0, then calls `trainer.run`.
Verify: `test_node_gang.py`, `test_node_state.py`; a 2-node Megatron run saves and resumes.

## Remove when
Never while the image has callers. Not an upstream candidate.
