# ext-036 / ext-037: split placement (`cluster.placement: split`) and the adapter sync through the volume

## Why
In one cluster the engines are Ray actors on the trainer's `@modal.clustered` gang: a trainer failure
takes the engines down and they reload (~25 min for DeepSeek-V4.1-Flash), every node pays
the trainer's RAM and disk, and the gang needs RDMA between nodes. Spindle and Stitch (Modal) run the
engines as their own Modal function and reach them over the network.

## Decision
- One recipe key, `cluster.placement: one_cluster | split` (default one_cluster, unchanged: the golden
  test records every recipe's argv and shape). The mode is decided in the recipe (argv, shape) and in
  `modal_app` (which functions exist); nothing else reads it.
- Only orchestration differs. The engine container starts the engines with Miles' own
  `compute_engine_launch_cmd` on the run's parsed argv (no second SGLang flag list); the trainer drives
  them through Miles' `--rollout-external-engine-addrs` and its own router; the adapter goes through
  the same exporter and session frame, with the same `--check-lora-weight-equal` checksums.
- Addresses: the engines sit behind the session proxy on one TLS `modal.forward` tunnel with a random
  path token; the trainer reaches engine i at `127.0.0.1:31000+i`, a local relay that adds the tunnel
  and token. Miles' parser takes only `http://host:port`, so the relay is what keeps Miles unchanged.
  Rejected: an `@app.server` (the address parser drops its https and path; the probe saw ~136 ms per
  ping through the gateway against ~57 ms through a tunnel, both across regions).
- The engine function outlives a failed trainer: a relaunch with the same run id reconnects to it.
  Modal refuses `nonpreemptible` for GPU functions, so a preempted engine ends the trainer's segment, as a
  lost gang node does today. It exits when the trainer reports success, or after `cluster.engine.idle_minutes`.
- Split refuses what it cannot sync: merged LoRA, full-parameter training, PP > 1, more than one
  trainer node.

## How it is applied
`miles_ext/recipes/recipe.py` (key, checks, `shape`, `_split_args`), `miles_ext/launch/modal_app.py`
(`engine` function, `_ensure_engine`, the relay before `ray start`), `miles_ext/launch/split/`
(`engine`, `relay`), recipe `dsv41-flash-lora-split.yaml`. Verify: the split
smoke (DeepSeek-V4.1-Flash LoRA, 1 trainer + 1 engine node of 8x B300, 2026-10-25): a relaunched
trainer reused the live engines without a reload. Teardown: stop the run's apps by exact name and id (the engine
call can live in the app of the first launch).

## Remove when
When the launcher no longer needs Miles-side orchestration for separate engine placement (for example, a
platform launcher that places engines itself and reconnects trainers). Not an upstream candidate: it is
Modal orchestration.
