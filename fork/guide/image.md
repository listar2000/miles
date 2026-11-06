# Running the image on your own GPUs

For the people who provide GPUs and launch training from outside this repository. The image holds Miles, Harbor and
the node commands. You start containers; you never need a checkout of this repo. The contract is
[RFC 0005](../rfcs/0005-provider-agnostic-image.md), and `miles_ext/node/contract.py` owns its names.

**State:** early. Modal is the only place these roles have run, and it calls them in process. See
[What has not run](#what-has-not-run). Please report what breaks as a GitHub issue on listar2000/miles, with the
command, the node's log and `/results/<run_id>/exit.json`.

## The image

Each push to the `migration` branch publishes `docker.io/starlimobius/miles-mobius:migration-<sha10>` and moves
`:migration` to it. Pin the digest for a run: all of a run's nodes use the same image.

## 1. Write the spec, read the manifest

Inside the image (`docker run` or any one-off container), from your recipe file:

    python -m miles_ext.node spec --recipe recipe.yaml --run-id my-run-1 > spec.json
    python -m miles_ext.node manifest --spec spec.json

Recipes that exist are in `miles_ext/recipes/configs/`, and `fork/guide/recipes.md` explains them. The manifest gives you:

| Field | What to do |
| --- | --- |
| `roles.trainer`, `roles.rollout` | GPUs, CPU, memory, disk and node count per role. `nodes: null` means as many as you like |
| `roles.*.ports` | Ports to expose publicly |
| `roles.trainer.placement`, `interconnect`, `open_between_nodes` | With more than one trainer node: see step 3 |
| `roles.rollout.gpu_advice`, `gpus_per_engine`, `precision` | What a rollout node must satisfy, with enough GPU memory for the model; the GPU type is only the recipe's choice |
| `mounts` | Paths to mount; `/results` must be the same storage on every node of both roles |
| `inputs` | Where the model, prompts and tasks are read from; prompts and tasks may instead sit under `inputs.fallback` |
| `secrets` | Env vars to set. The image never reads a `.env` |

## 2. Storage

`/models`, `/results` and the others are plain mounts: NFS, a provider volume, or a bucket through a CSI driver. If your
mount needs a flush after writing, write an adapter with `commit`, `refresh` and `remove`
(`miles_ext/node/storage.py`) and set `MILES_STORAGE` to its dotted path. Weights can be fetched with
`python -m miles_ext.node prepare-weights --hf-repo R --revision C --path <inputs.model_dir>`, and likewise for
`inputs.draft_checkpoint` when the manifest names one.

## 3. Start the trainer

**Trainer nodes of one run must be one cluster** with a fast private interconnect (RDMA for more than one node): never
across providers or regions. They reach each other on any port, since Ray and NCCL pick ports at run time. Give each
node the manifest's `gpus_per_node`. Each node gets its own rank from 0 to N-1 and its own IP; `MILES_NNODES` and
`MILES_HEAD_ADDR` (rank 0's IP) are the same on all. Two nodes, rank 0 first:

    MILES_NODE_RANK=0 MILES_NNODES=2 MILES_HEAD_ADDR=10.0.0.1 MILES_NODE_IP=10.0.0.1 WANDB_API_KEY=... \
        python -m miles_ext.node trainer --spec spec.json
    MILES_NODE_RANK=1 MILES_NNODES=2 MILES_HEAD_ADDR=10.0.0.1 MILES_NODE_IP=10.0.0.2 WANDB_API_KEY=... \
        python -m miles_ext.node trainer --spec spec.json

A node missing something names it and exits before doing anything. Exit code 0 means done or segment end: run the same
command again and it resumes from the newest save. SIGTERM saves and exits, and so does `MILES_SEGMENT_DEADLINE` (epoch
seconds). Otherwise read `/results/<run_id>/exit.json`. Each node's log is in `/results/<run_id>/logs/`. Heartbeat and exit are in `/results/<run_id>/state/`, one JSON file per key with the name URL-encoded; write
`lease%3A<run_id>.json` there as `{"expires_at": <epoch seconds>}` to bound a run.

In-sandbox harnesses also need `MILES_SESSION_PUBLIC_URL` on rank 0: the public URL of port 29999.

To send run state to a viewer instead, W&B style, set `MILES_STATE=http`, `MILES_STATE_URL` and `MILES_STATE_TOKEN` on every
node. It gets `PUT`, `GET` and `DELETE` on `<url>/<run_id>/<key>`, and a failed call never stops training. [`miles-viewer`](https://github.com/listar2000/miles-viewer) is
such a viewer, and it also reads `/results/<run_id>/` directly when it can mount that storage.

## 4. Start rollout nodes (split recipes)

Rollout nodes need not match each other or the trainer, and can come from anywhere, within what the manifest requires.
Give each the manifest's `gpus_per_node`. Set `MILES_GPU` if a node's GPU is not the recipe's, so it keeps its own compiled kernels.
Generate one `MILES_POOL_TOKEN` per run, give it to both roles, and expose rank 0's port 30999:

    MILES_POOL_TOKEN=... MILES_TRAINER_URL=https://<rank 0 port 30999> MILES_ROLLOUT_PUBLIC_URL=https://<this node port 32999> \
        WANDB_API_KEY=... python -m miles_ext.node rollout --spec spec.json

If rank 0 can move (a relaunch on a new host), use `MILES_TRAINER_URL_FILE` and rewrite that file: the node rereads it on
each beat. A node exits when the trainer says done, when it is evicted for failing a sync, or after
`cluster.engine.idle_minutes` with no trainer. A relaunched trainer finds live nodes by itself.

Without `cluster.engine.elastic: true`, a run uses one rollout node with the recipe's engine count. With it, start and
stop nodes at any time (LoRA recipes only); read `state/pool%3A<run_id>.json` for each engine's load.

## Other jobs

`python -m miles_ext.node --help` lists them: `check` (a dry check of the recipe, on one GPU; `--no-gpu` still checks inputs
and markers but Megatron's checks fail there), `prepare-weights`, `convert-fsdp`, `sft-prepare`, `sft-train`.

## What has not run

- Any of these commands on a GPU node outside Modal, and any provider but Modal.
- Split on real engines through the rollout role, including a trainer relaunch (over real tunnels with fake engines only).
- Elastic join and leave on real engines, and the load numbers in `pool%3A<run_id>.json`.
- Storage other than Modal volumes on GPUs: `posix` and `file` state are unit-tested only.

What has: one-node and two-node save and resume through `modal_app`, which calls the same trainer role. In the image on a
CPU container with Modal's env stripped, `spec` and `manifest` ran, and `trainer` went through its inputs, Ray and an
exit record until Miles needed a GPU.
