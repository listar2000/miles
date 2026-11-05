# Running the image without Modal

The image holds Miles, Harbor and the node commands. Anything that can start a container with GPUs can run it. The
contract is [RFC 0005](../rfcs/0005-provider-agnostic-image.md), and `miles_ext/node/contract.py` owns its names. This
page is the walk through. `miles_ext.launch.modal_app` is one caller of this contract, and the only one tested so far:
no run has started these commands off Modal yet.

## 1. Ask what the run needs

Write the spec, `{"run_id": ..., "recipe": {...}}` with the recipe resolved, and read the manifest:

    python -m miles_ext.node manifest --spec /run/spec.json

It lists each role's GPUs, node count, ports, the mounts and the secrets. `nodes: null` on the rollout role means as
many as you want to run. Mount `/models`, `/results` and the rest as it says. Trainer and rollout nodes must see the
same `/results`; a bucket mounted through a CSI driver or a provider volume is fine. Set `MILES_STORAGE` if writes need
flushing (`miles_ext/node/storage.py`), and put your own adapter's dotted path there if none fits.

## 2. Start the trainer

Trainer nodes of one run must be one cluster with a fast private interconnect (RDMA for more than one node): never
across providers or regions. On every trainer node, rank 0 first:

    MILES_NODE_RANK=0 MILES_NNODES=2 MILES_HEAD_ADDR=10.0.0.1 MILES_NODE_IP=10.0.0.1 \
        python -m miles_ext.node trainer --spec /run/spec.json

Rank 0 opens the ports the manifest lists on the trainer role. Expose them. In-sandbox harnesses also need
`MILES_SESSION_PUBLIC_URL`, the public URL of the session proxy port. Exit code 0 means done or segment end: run the
same command again and it resumes. Otherwise read `/results/<run_id>/exit.json`.

## 3. Start rollout nodes (split recipes)

Rollout nodes need not match each other or the trainer, within what the manifest requires: `gpus_per_engine`, `precision`
and, today, 8 GPUs per node. `gpu_advice` is only the recipe's choice. Generate one `MILES_POOL_TOKEN` per run and give
it to both roles. Then, on each rollout node:

    MILES_POOL_TOKEN=... MILES_TRAINER_URL=https://<rank 0 registry> MILES_ROLLOUT_PUBLIC_URL=https://<this node> \
        python -m miles_ext.node rollout --spec /run/spec.json

If rank 0 can move (a relaunch on a new host), use `MILES_TRAINER_URL_FILE` and rewrite that file instead: the node
rereads it on each beat. A node leaves when the trainer says done, when it is evicted for missing a sync, or after
`cluster.engine.idle_minutes` with no trainer.

With `cluster.engine.elastic: true` you can start and stop rollout nodes at any time. Read the load under
`pool:<run_id>` in run state to decide when (`MILES_STATE`, `miles_ext/node/state.py`).

## Other jobs

`python -m miles_ext.node --help` lists them: `check`, `prepare-weights`, `convert-fsdp`, `sft-prepare`, `sft-train`.
