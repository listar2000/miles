# ext-066: elastic rollout pool

## Why
RFC 0005: rollout capacity should scale separately from the trainer, possibly on spot GPUs of another provider, so
nodes join and leave a running trainer.

## Decision
- `cluster.engine.elastic: true` swaps the fixed `--rollout-external-engine-addrs` for upstream's
  `--custom-inference-engine-provider-path` over the registry (`node/pool_provider.py`). Upstream already reconciles
  cells mid-run, and on a new engine set it reconnects the sync protocol and re-registers adapters, so no new hook.
- Each engine is one cell at a local slot, and each trainer node's relay follows the registry (`launch/split/relay.py`).
  Slots are not reused in a segment. All split runs use this relay; a first node's slots equal the fixed addresses, so
  non-elastic argv is unchanged.
- A sync goes to the nodes behind Miles' cell snapshot, through shared storage. A node that fails `/push` or `/reload`
  is evicted and exits: it serves stale weights otherwise. A late joiner takes the next sync before it serves (every step
  in `--update-weights-interval 1`), so there is no separate catch-up. If no node takes a sync, the segment fails as a
  lost engine did before.
- Load (each engine's running and waiting requests) comes with each beat and goes to run state under `pool:<run>`.
  Scaling stays with the caller; `modal_app --add-rollout-node` stands in for it.
- LoRA only: `elastic` with `rollout.weight_sync: gpu-delta` is refused, since a late node has no base for deltas.

## How it is applied
`recipe.py` (`cluster.engine.elastic`, `_split_args`), `node/registry.py`, `node/pool_provider.py`, `launch/split/relay.py`,
`launch/split/volume_sync.py`, `node/trainer.py`. On a sync: read upstream's `BaseWorkerProvider`, `PollingReconcileLoop`
and `reconnect_if_needed`. Verify: `test_node_rollout.py`, `test_node_pool_provider.py`; a run where a second node joins
mid-run and one is killed, with the step still moving.

## Remove when
Never while the image has callers. A candidate for upstream once proven, since the provider has no Modal in it.
