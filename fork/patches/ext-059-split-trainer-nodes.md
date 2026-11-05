# ext-059: split placement with N trainer nodes

## Why
`cluster.placement: split` refused `cluster.nodes` above 1 in `_validate_split`, so a recipe could not choose it.
Two trainer nodes hold all of Adam in host memory: a GLM-5.3-Flash run (2x B300:8, TP8 EP16 DP2, one cluster,
trainer only, synthetic data) took 98 to 103 s per step against 630 s on one node. The check was the only block in the recipe, but
the launcher and the sync assumed one node in four places.

## Decision
- `cluster.train_gpus` must equal `cluster.nodes * 8` in split. One node stays the default and every existing recipe's
  argv is unchanged (placement golden test). The engines stay one container.
- Relays and the commit watcher run on every trainer container, before `ray start` (`_split_links`). A worker starts
  them before it waits for the head's port, since the head opens it only after the engines serve. The heartbeat stays on
  rank 0, which also writes the release; a worker's beat could overwrite it.
- GPU-delta: every rank's owner shard is on its own node's view of the volume until that node commits. The post-write
  hook has every rank ask its node's watcher for a commit, gathers the result, and only then has rank 0 call `/reload`.
  The watcher answers all requests it sees with one commit, so a node's 8 ranks do not commit 8 times. Rejected: a
  per-node leader chosen by a hostname gather (more collectives, same result).
- `optimizer_host_gib` charges each node `copies * served_gib / nodes`. With PP 1 the owners are spread over all ranks
  (`_ordinary_owners` over `tp_dp_cp`; experts on their own EP rank). At 2,560 GiB and GLM-5.3-Flash this is 125 GiB/GPU
  on 2 nodes against 67 on one. `TRAINER_HOST_RESERVE_GIB` is unchanged.
- Retention: rank 0's `rmtree` reaches only its own view, so a dropped save's other-node shards stayed on the volume.
  `remove_other_node_shards` removes the directory on the volume after rank 0's commit, on any multi-node run.
- The supervisor and the `.optimizer` marker assume Miles' rank 0 is on the head container. Miles orders ranks by node IP,
  and Modal gave rank 0 the lowest IP in that 2-node run (10.100.0.1, worker .2). This PR does not change that.

## How it is applied
`miles_ext/recipes/recipe.py` (`_validate_split`, `optimizer_host_gib`), `miles_ext/node/trainer.py` (`_split_links`,
`_worker`, `_head`), `miles_ext/platform/checkpoints.py`, `miles_ext/launch/split/relay.py`, `miles_ext/launch/split/volume_sync.py`.
Tests: `miles_ext/tests/test_split.py`, `test_split_gpu_delta.py`, `test_split_volume_sync.py`,
`test_optimizer_host_gib.py`, `test_platform.py`. A multi-node split run has not run yet; see the PR for its status.

## Remove when
With ext-036.
