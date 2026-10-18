# hook-002: `--custom-placement-hook-path`, an extension sees which GPUs train and which serve rollouts

## Why
W&B samples GPUs only in processes that call `wandb.init`, and each samples every GPU on its node.
In Miles those are the driver, the rollout executor and the trainer ranks, so a node that only
serves rollouts never reaches W&B and each trainer node is sampled once per rank, all under
`modal-<id>` labels. In overfit16 attempt 4 the W&B "GPU Utilization" chart showed 0% on every
series for 2+ hours of rollout while both engine nodes ran about 50 requests each: the chart held
only the two trainer nodes. Nothing in Miles tells an extension where each role's GPUs landed:
`_create_placement_group` knows the node of every bundle and `create_placement_groups` knows the
train/rollout split, and neither is exposed.

## Decision
A function-path flag in the style of Miles' other `--custom-*-path` flags. The driver calls it once,
after the placement group exists, with `(args, pgs)`: the same `PlacementGroupInfo` dict
`create_placement_groups` returns. Unset (the default), nothing changes. First user, ext-015:
`miles_ext.obs.gpu_telemetry.start` (one W&B observer per node and role, labelled
`<role>-node<k>`). Rejected: widening `PlacementGroupInfo` with node IPs (rewrites upstream return
values and unpacking); a second call site after the routers are up for the engines' request
metrics (kept out until that scrape is needed; `WANDB__DISABLE_STATS` also mutes upstream's
`--sglang-enable-metrics` scrape, which no recipe sets today).

## How it is applied
`miles/utils/arguments.py` (flag) and `create_placement_groups` in `miles/ray/placement_group.py`
(the call before `return ans`, plus the `load_function` import); all spots are marked. Not called
on the Kubernetes backend, which creates no placement group. On a sync, re-insert the two lines if
upstream reshapes `create_placement_groups`.

## Remove when
Upstream labels W&B system metrics by role, or offers an equivalent extension point. Status: not
filed.
