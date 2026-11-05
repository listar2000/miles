# ext-063: split's rollout role with no Modal

## Why
RFC 0005: the engine container read and wrote a `modal.Dict` and opened its own `modal.forward`, so a rollout node
could only run on Modal, next to a trainer that shared that Dict.

## Decision
- `miles_ext/node/rollout.py` (was `launch/split/engine.py`) starts the engines, the agent and the proxy, then beats to
  `miles_ext/node/registry.py` on rank 0. A beat is also the registration, so a live node joins a relaunched trainer's
  new registry by itself. Rollout state on the node is only the engines; the caller watches phases through `on_state`.
- The caller gives the node `MILES_ROLLOUT_PUBLIC_URL`, `MILES_POOL_TOKEN`, and `MILES_TRAINER_URL` or
  `MILES_TRAINER_URL_FILE`. The file form exists for relaunches: rank 0's URL can change and the node outlives it.
- A complete run tells each node through its agent's `/release`; a node that misses it reads `done` on its next beat,
  and otherwise leaves at `cluster.engine.idle_minutes`. The Dict's reliable `done` is the one thing given up: the idle
  limit is now the backstop when rank 0 dies right after success.
- Rejected: rendezvous through files on `/results`. A commit per beat on a Modal volume, and bucket caches elsewhere,
  are too slow and too costly for liveness; HTTP is already what the engines speak. The trainer still drives one node
  through fixed `--rollout-external-engine-addrs` and relays; many nodes come with the elastic pool.

## How it is applied
`modal_app`: `engine` opens the tunnel and carries the trainer's URL from the Dict into the file; `_head` starts the
`Registry` and publishes its tunnel. Worker ranks read rank 0's registry over the cluster network. `miles_ext.platform`
exports its API only where Modal imports, so its node-side modules (`compile_cache`) import in a node with no Modal
client. Verify: `test_node_rollout.py` runs the whole
exchange with fake engines; a split smoke with a trainer relaunch checks it for real.

## Remove when
Never while the image has callers. Not an upstream candidate yet; it could become one with no Modal in it.
