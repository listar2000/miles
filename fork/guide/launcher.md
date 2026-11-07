# Building a launcher: what the image does and what you do

For whoever builds the system that places Miles nodes on GPUs: a SkyPilot-like launcher, a platform, a script. Read
this page first for the boundary and the reasons behind it. [image.md](image.md) is the step-by-step, and
[RFC 0005](../rfcs/0005-provider-agnostic-image.md) is the contract. `miles_ext/node/contract.py` owns every name. This
page links to those three instead of repeating their tables.

## The split in one paragraph

This repository builds one image, and every Miles node runs it. Inside a run, the nodes coordinate everything
themselves: the training loop, the trainer gang, which rollout engines exist, weight syncs, saves, and the exit record.
That is **internal orchestration**, and it is the image's job. Everything that happens to a node from outside is
**external orchestration**, and it is yours: getting GPUs, starting and stopping containers, wiring the two roles
together, choosing storage and secrets, deciding when to relaunch, scale or give up, and watching cost. The image never
asks for a GPU, starts a node, or creates an app, volume or secret. You never parse a recipe or reach into a running
node.

The reason is provider independence. Rollout and trainer nodes may come from different providers and scale
separately, and Modal is one launcher among several.

## Who does what

| Concern | The image (inside nodes) | The launcher (outside) |
| --- | --- | --- |
| What a run needs | `spec` and `manifest` turn a recipe into a demand list | Reads the manifest, provisions what it lists |
| Trainer nodes | One gang: Ray, NCCL, rank 0 runs Miles under a supervisor | Provides N nodes in **one cluster** with RDMA, all ports open between them, rank and IPs in env |
| Rollout nodes | Start engines, register with rank 0, apply weight syncs, leave when told | Provides as many as it wants, from anywhere that meets the manifest's rollout role |
| Wiring | Opens fixed ports behind tokens; nodes find each other through the URLs you give | Exposes the listed ports, hands rollout nodes the trainer's URL and one pool token per run |
| Storage | Reads and writes fixed paths; commits and refreshes through an adapter | Picks the backend, mounts it (or sets the GCS mirror), sets `MILES_STORAGE` |
| Secrets | Reads env vars named by the manifest | Injects them |
| Membership | Drops a rollout node that stops beating or fails a sync, retries its samples | Decides when to add or remove nodes, and replaces dead ones if it wants |
| Saves and resume | Saves on schedule, on SIGTERM, before the deadline or lease; resumes from the newest save | Sends SIGTERM, sets the deadline or lease, relaunches the same command |
| Ending | Writes `exit.json` with the reason; tells rollout nodes "done"; terminates the run's task sandboxes | Reads `exit.json`, tears everything down, decides on a relaunch |
| Watching | Writes run state (heartbeat, exit, pool load) and logs | Reads them, or receives them over HTTP, and shows them (`miles-viewer`) |

## The demand list

`python -m miles_ext.node manifest --spec spec.json` is the only thing you read from a recipe. A launcher that
provisions what the manifest lists, starts each node with its role and env, and wires the roles together should not
need to know anything else about Miles. [image.md, step 1](image.md#1-write-the-spec-read-the-manifest) explains
each field.

Treat the manifest as binding in these places:

- `roles.trainer`: the node count, `gpus_per_node`, and, with more than one node, `placement: one_cluster` and
  `interconnect: rdma`. Never spread one gang across providers or regions. Miles' trainer runs NCCL collectives
  every step, and they assume a fast private link.
- `roles.rollout`: `gpus_per_node`, `gpus_per_engine`, `precision`, and enough GPU memory for the model. The GPU type
  (`gpu_advice`) is the recipe's choice, not a requirement, so rollout nodes may differ from each other and from the
  trainer.
- `mounts`, `secrets`, `ports`.

The manifest does not say these yet, so a launcher needs them from this page:

- **How many rollout nodes to start.** `nodes: null` means the image accepts any number. Without
  `cluster.engine.elastic: true` in the recipe, start exactly one. With it, start as many as you like and change the
  count during the run.
- **The order.** Start rollout nodes first, and start the trainer once a rollout container is running, before its
  engines are ready. Starting earlier bills trainer GPUs while rollout GPUs are still queued. Waiting for ready
  wastes the 6 to 16 minutes both sides can spend loading at once (ext-069).

## Wiring a split run

    launcher ──provisions──> trainer gang (one cluster)        rollout nodes (anywhere, any number)
                              rank 0: registry :30999  <───────── each registers its public URL + beats
                                      session proxy :29999  <──── task sandboxes (in-sandbox harnesses)
                              ranks: Ray/NCCL, any port, private      proxy :32999 ───> its engines + sync agent
                              all nodes, both roles: the same /results storage

Give every node of both roles the same `MILES_POOL_TOKEN`. Each rollout node needs a public URL for its port 32999
(`MILES_ROLLOUT_PUBLIC_URL`) and the trainer's registry URL (`MILES_TRAINER_URL`). Use `MILES_TRAINER_URL_FILE` if a
relaunched trainer can come back at another address: rewrite the file and the nodes follow. Ports and env names are in
[image.md](image.md) and `contract.py`.

## Storage is your choice, with one rule

Both roles must see **the same `/results`**, because weights travel through it. The trainer writes each sync there and
the rollout nodes read it. There is no separate weight-sync channel. NFS, a provider volume, a bucket through a CSI
driver, or the built-in GCS mirror all work. If writes need a flush to become visible, the node calls its storage
adapter's `commit` and `refresh`. Pick the adapter with `MILES_STORAGE`, or write one
(`miles_ext/node/storage.py`). Put the storage in the trainer's region. Every rollout node reads each sync, so
rollout nodes far from it pay for that traffic.

## A run's life

A run is a series of **segments** with one run id. Each segment is one trainer gang from start to exit. Relaunching the
same command with the same spec resumes from the newest save, and a run that is already complete exits at once
without training. Rollout nodes can outlive a trainer segment, and a relaunched trainer finds them.

At the end of every segment, rank 0 writes `/results/<run_id>/exit.json`. Act on it:

| `exit.json` | Meaning | Usual launcher action |
| --- | --- | --- |
| `complete: true` | every step trained | tear down all nodes of both roles |
| `reason: segment_end` or `lease_expired` | saved and stopped on time | relaunch the trainer (same spec); keep rollout nodes |
| `reason: nccl_timeout` (`retryable: true`) | a failure worth one retry | relaunch the trainer |
| any other reason | Miles failed; `note` and `tail` say why | stop; a relaunch would repeat it |

What fails, and who notices first:

| Failure | The image | The launcher |
| --- | --- | --- |
| A trainer node dies | the gang fails; rank 0 writes `error` if it can | sees the node die first; stop the rest of the gang, then relaunch or stop |
| A rollout node dies | dropped after missed beats; its samples are retried | replace it if the pool should stay that size |
| No rollout node ever registers | trainer stops itself after `MILES_FIRST_ROLLOUT_NODE_TIMEOUT_SECS` (1 h by default) | stop the trainer sooner when a node it started dies before registering |
| Every rollout node dies later in the run | nothing: the trainer waits for engines | notice it (pool load, no step progress) and stop the run or bring nodes back |
| Out of time or budget | saves before `MILES_SEGMENT_DEADLINE` or the lease, or on SIGTERM | set one of them; SIGTERM is the immediate one |

Rules that keep a run sound:

- **One trainer gang per run id at a time.** Two gangs writing one `/results/<run_id>` corrupt it. The image does
  not stop you.
- **One image digest per run**, on every node of both roles, for every segment.
- **The gang is all or nothing.** If any trainer node is lost, stop the whole gang and relaunch it as a unit.
- **Tear down rollout nodes yourself** when the run ends or you give up. They leave on the trainer's "done", but a
  trainer that crashed never sends it. They then exit only after `cluster.engine.idle_minutes`, with GPUs held
  until then.

## Scaling the rollout pool

With `cluster.engine.elastic: true` (LoRA recipes only), start and stop rollout nodes at any time. A new node reads the
newest weights before it reports ready, so it never serves stale ones. Read each engine's load from the `pool:<run_id>`
key in run state, and decide when to scale from it. The image never asks for nodes.

## Watching

Nodes write run state (heartbeat, exit, pool load, and the lease you set) through `MILES_STATE`. With `file`, it sits
under `/results/<run_id>/state/`. With `http`, nodes push it to your endpoint with a token, as W&B does.
Each node's log goes to `/results/<run_id>/logs/`. Metrics go to W&B with the key you inject. The run monitor is
outside this repository, as [`miles-viewer`](https://github.com/listar2000/miles-viewer). Cost is yours: only the
launcher knows what it rented.

## The reference launcher

`miles_ext/launch/modal_app.py` is the only launcher so far, and it stands in for yours while the migration runs. It
calls the same node roles on Modal, and it shows each duty above: the launch order (`_ensure_engine`, then the trainer
once the rollout container runs), the trainer URL file for relaunches, segments with a deadline, and waiting on the
rollout node after a trainer fails. It leaves this repository once a real launcher exists. Until then, copy its
behaviour, not its code.

## What is not settled

- Nothing has run off Modal yet. [image.md](image.md#what-has-not-run) keeps that list.
- The manifest does not yet carry the starting rollout count, the elastic flag or the launch order. This page states
  them until it does.
- Node sizes are fixed per recipe: `gpus_per_node` in the manifest is 8 unless the recipe sets a smaller split
  (ext-068: a one-node trainer on part of a node, `cluster.engine.gpus` for rollout nodes). A pool of differently sized
  rollout nodes is not supported yet.
