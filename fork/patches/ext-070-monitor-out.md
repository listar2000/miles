# ext-070: the run monitor leaves the fork

## Why
RFC 0005: the monitor starts nothing in a node, so it does not belong in the image, and it scrapes Modal (app list, app
logs, the Volume API, a launch's stdout), which stops working with nodes on other providers.

## Decision
- `miles_ext/monitor/` moved to [`miles-viewer`](https://github.com/listar2000/miles-viewer), on the launcher side. Only `launch/manifest.py` stays: it is
  what the stand-in caller writes for the viewer, and it goes with `modal_app`.
- `MILES_STATE=http`: a node puts run state on `<MILES_STATE_URL>/<run_id>/<key>` with `MILES_STATE_TOKEN`, as W&B takes
  metrics, so the viewer can run anywhere. No new interface: it is another store behind `node/state.py`, and the
  supervisor already carries on when a store fails.
- `python -m miles_ext.node trainer|rollout` copies the node's output to `/results/<run_id>/logs/<role>.log`, for a viewer
  to tail. Rejected for `modal_app`: an open file on a Modal volume blocks `reload`, and its launch log still exists.
- W&B stays the metrics store; the viewer is the ops view.

## How it is applied
`node/state.py`, `node/runtime.py` (`tee_logs`), `node/__main__.py`, `launch/manifest.py`. Verify: `test_node_state.py`,
`test_launch_manifest.py`. On a sync: nothing.

## Remove when
Never while the image has callers. Not an upstream candidate.
