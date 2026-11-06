# ext-068: split placement on part of a node

## Why
`cluster.placement: split` required whole nodes for the trainer and always gave the engine container 8 GPUs, so the
smallest split run cost two nodes. A GLM-5.3-Flash LoRA run fits 4 trainer + 4 engine GPUs (the one-cluster run of 2026-11-05), and
split at that size costs the same as one cluster while measuring the volume sync and the tunnel.

## Decision
- `cluster.engine.gpus` (default 8) sizes the engine container. `cluster.train_gpus` may be a divisor of 8 when
  `cluster.nodes` is 1. Other recipes' argv and shape are unchanged (placement golden test).
- Such a trainer is a plain Modal function. Modal refuses `@modal.clustered(size=1)` below a full node: "Multi-node
  functions must use the full number of GPU devices per node; requested 4, but the selected GPU type requires 8"
  (checked 2026-11-06). With no cluster, Ray runs on 127.0.0.1. Rejected: a size-1 cluster, which Modal refuses.
- `gpus_per_node` and `engine.gpus` enter the shape only when they differ from 8, so existing shapes stay as recorded.

## How it is applied
`miles_ext/recipes/recipe.py` (`gpus_per_node`, `rollout_gpus`, `_validate_split`, `shape`, argv),
`miles_ext/launch/modal_app.py` (`NODE_GPUS`, `CLUSTERED`, `_gang`, `train`). Tests: `miles_ext/tests/test_split.py`.
The run that uses the other value: `experiments/glm53_stak_overfit`.

## Remove when
With ext-036.
