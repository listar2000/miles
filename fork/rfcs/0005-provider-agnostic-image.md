# RFC 0005: This repository is the source of one image; callers place its nodes

Status: accepted 2026-11-05 (owner). Work lands on the long-lived branch `migration`, by stacked PRs per phase;
`main` is untouched until the owner decides otherwise.

## Decision

The fork stops doing two jobs. What runs inside a node stays here. What starts a node, picks GPUs, or creates apps,
volumes or secrets belongs to a caller, whose final home is out of this repository's concern. The two roles, rollout
and trainer, may come from different providers and scale separately, as under a SkyPilot-like launcher.

While the migration runs, `miles_ext.launch.modal_app` is the stand-in caller. All phase tests run through it, on
Modal. Modal is one special case of the contract below, not an assumption of the code behind it.

## The contract

`miles_ext/node/contract.py` owns the names; this table says what they mean. An entry joins the code in the phase
that first uses it.

    python -m miles_ext.node trainer  --spec /run/spec.json   # one per trainer node; rank 0 runs Miles
    python -m miles_ext.node rollout  --spec /run/spec.json   # one per rollout node (split)
    python -m miles_ext.node check    --spec /run/spec.json   # dry check
    python -m miles_ext.node manifest --spec /run/spec.json   # what the run needs, as JSON

The spec is `{"run_id": ..., "recipe": {...}}`, the recipe resolved. `manifest` gives GPUs and node count per role,
mounts, ports and secret keys; `cluster.*` reaches the caller there as advice, and placement keys stay in the recipe
so old runs reproduce.

| Kind | Name | Meaning |
| --- | --- | --- |
| Gang | `MILES_NODE_RANK`, `MILES_NNODES`, `MILES_HEAD_ADDR`, `MILES_NODE_IP` | Trainer nodes, as torchrun does |
| Pool | `MILES_TRAINER_URL` or `MILES_TRAINER_URL_FILE`, `MILES_POOL_TOKEN`, `MILES_ROLLOUT_PUBLIC_URL` | Split: a rollout node registers its public URL with rank 0 and beats. The file form is reread on each beat, because a relaunched trainer can move |
| Ingress | `MILES_SESSION_PUBLIC_URL` | In-sandbox harnesses: reaches rank 0's session proxy |
| Storage | `/models` ro, `/results` rw, `/datasets` ro, `/compile-cache` rw optional; `MILES_STORAGE` | The caller mounts them. Trainer and rollout nodes share `/results`; S3, GCS, Azure or a provider volume is the caller's business. The adapter names the commit rule: `posix` or `modal-volume` |
| Run state | `MILES_STATE` | Heartbeat, lease and exit for the caller: `file` under `/results`, or `modal-dict` |
| Secrets | `WANDB_API_KEY`, `SAIL_API_KEY`, ... | Env vars; `manifest` names them. The image never reads a `.env` |
| Time | `MILES_SEGMENT_DEADLINE`, SIGTERM | Save and exit before the deadline or on the signal |

One trainer gang is one cluster with a fast private interconnect (RDMA for more than one node), never split across
providers or regions. Rollout nodes can be heterogeneous and come from anywhere. Miles' trainer ranks form one
torch.distributed group whose collectives run every step, while engines are independent replicas behind a router.

A node gives back: rank 0's session proxy (29999) and registry (30999), a rollout node's proxy (32999) with `/healthz`;
under `/results/<run_id>/` the checkpoints, HF exports, `exit.json`, `exits.jsonl` and launch records; exit code 0 for
done or segment end (relaunch resumes), non-zero with a reason in `exit.json`.

The image never creates apps, volumes or secrets, never picks GPUs, and never starts a node. The one provider SDK it
may call is the sandbox provider the recipe names, plus the adapters below, loaded only when selected.

## Interfaces

1. **Storage:** `commit()` and `refresh()`, no-ops on a shared POSIX path. The Modal adapter makes today's calls in
   today's order, per node, so durability does not change.
2. **Weights:** through the shared `/results` storage only. The trainer writes a numbered version, commits it and tells
   each registered rollout node; a node refreshes, reads and applies it, and a late joiner reads the newest before it
   reports ready. The image does not know buckets: the storage interface is the whole abstraction.
3. **Membership:** rollout nodes register with rank 0 over HTTP and beat, on upstream's
   `--custom-inference-engine-provider-path`. A missed node is dropped and its in-flight samples retried. When to add or
   drop nodes is the caller's decision; the image reports load and accepts whoever registers.
4. **Run state:** `get`, `put`, `delete` for what the caller reads. Coordination inside the trainer gang moves to a small
   HTTP service on rank 0, on the gang's private network like Ray's ports, and split lifecycle rides the registry link.
   Ray's key-value store was the first plan; a driver that loses the GCS can take its process down before a worker's
   last commit, and these calls are simple enough not to need it.

Adapters run inside a node and launch nothing, so they stay in the image and load by dotted path. Endpoints need no
interface: nodes listen on fixed ports behind tokens, and the caller exposes them.

## Fork

All of it is Ext and Infra under `miles_ext/` and `fork/`, so syncs do not get harder. The caller-facing surface
becomes this contract; a change to it is flagged in the stable tag's note, as platform changes are today. Patch pages,
the ledger and SGLang patches are unchanged. CI keeps building the image in a Modal CPU sandbox: Modal there is a
build host.

## Phases

Each is a PR stacked towards `migration`. After each one `modal_app` still works, `test_placement_golden.py` is
unchanged, and one smoke passes.

| Phase | Work | Smoke |
| --- | --- | --- |
| 0 | This RFC, `miles_ext/node`, contract tests with no Modal in the interpreter | none |
| 1 | Storage interface; every commit and reload goes through it | Qwen3.5-9B, 2 steps |
| 2 | Split: the rollout role and the registry; the engine reads no Dict | split recipe, per-run approval |
| 3 | Trainer gang from env; worker release and shard tokens through rank 0's gang server | 2-node Qwen3.5-9B |
| 4 | Elastic pool on the engine provider hook; late joiners; load in the status file | second rollout node mid-run |
| 5 | `python -m miles_ext.node` for every role and job; wrappers read `manifest` | Qwen3.5-9B, 2 steps |
| 6 | Launching code leaves the image source: deferred, `modal_app` stays as the stand-in caller until the owner moves it | decided by the owner |
| 7 | Docs and skills: `fork/guide/image.md` now; the Modal how-tos move with phase 6 | none |
