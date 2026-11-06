"""What a caller gives a node and what a node gives back (RFC 0005): the single owner of these names."""

from miles_ext.recipes import recipe as R
from miles_ext.recipes.presets import PRESETS

# ---- gang: trainer nodes, as torchrun's RANK, WORLD_SIZE, MASTER_ADDR ----
NODE_RANK_ENV = "MILES_NODE_RANK"
NNODES_ENV = "MILES_NNODES"
HEAD_ADDR_ENV = "MILES_HEAD_ADDR"
NODE_IP_ENV = "MILES_NODE_IP"

STORAGE_ENV = "MILES_STORAGE"  # miles_ext.node.storage
STATE_ENV = "MILES_STATE"  # miles_ext.node.state
STATE_URL_ENV = "MILES_STATE_URL"  # MILES_STATE=http: where a node puts run state, as W&B's endpoint
STATE_TOKEN_ENV = "MILES_STATE_TOKEN"  # and its bearer token

# ---- optional, for the caller's own records ----
CALL_ID_ENV = "MILES_CALL_ID"  # names this segment in the exit record
SANDBOX_ENVIRONMENT_ENV = "MILES_SANDBOX_ENVIRONMENT"  # the sandbox provider's environment, for exit cleanup
FINAL_RETENTION_ENV = (
    "MILES_FINAL_RETENTION"  # "1": prune checkpoints as the recipe says; a caller that retains itself sets "0"
)

SESSION_URL_ENV = (
    "MILES_SESSION_PUBLIC_URL"  # in-sandbox harnesses: the public URL that reaches rank 0's session proxy
)
DEADLINE_ENV = "MILES_SEGMENT_DEADLINE"  # epoch seconds: save and exit before it (SIGTERM does the same)

# ---- rollout pool (split) ----
TRAINER_URL_ENV = "MILES_TRAINER_URL"  # where a rollout node reaches rank 0's registry
TRAINER_URL_FILE_ENV = "MILES_TRAINER_URL_FILE"  # or a file holding it, reread on each beat (a relaunch moves it)
POOL_TOKEN_ENV = "MILES_POOL_TOKEN"  # one per run, on both roles
ROLLOUT_URL_ENV = "MILES_ROLLOUT_PUBLIC_URL"  # the public URL that reaches this rollout node's proxy port

GPU_ENV = "MILES_GPU"  # a node whose GPU is not the recipe's says which, so it keeps its own compiled kernels

# ---- ports a node opens ----
SESSION_PROXY_PORT = R.PROXY_PORT  # rank 0, in-sandbox harnesses
REGISTRY_PORT = 30999  # rank 0, split
GANG_PORT = 30998  # rank 0 of more than one node, private to the gang's network like Ray's ports
ROLLOUT_PROXY_PORT = 32999  # a rollout node: its engines and sync agent, behind one token

COMPILE_CACHE_MOUNT = "/compile-cache"


def required_secrets(recipe: dict) -> list[str]:
    """The env vars holding credentials a run's nodes read. The image never reads a .env."""
    keys = ["WANDB_API_KEY"]
    if recipe["sandbox"]["provider"] == "sail":
        keys.append("SAIL_API_KEY")
    return keys


def node_secrets(recipe: dict) -> list[str]:
    """Everything a node off Modal needs; a Modal sandbox provider also needs a client token there."""
    keys = required_secrets(recipe)
    if recipe["sandbox"]["provider"] == "modal" and not R.is_single_turn(recipe):
        keys += ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"]
    return keys


def _draft_checkpoint(recipe: dict) -> dict | None:
    """The draft model's weights, when the run decodes speculatively with a separate checkpoint."""
    draft = PRESETS[recipe["model"]["preset"]].speculative_draft
    if not R.speculative_algorithm(recipe) or not draft:
        return None
    repo, revision = draft
    return {"path": str(R.weights_path(repo, revision)), "hf_repo": repo, "revision": revision}


def _inputs(recipe: dict, run_id: str) -> dict:
    """Where the run reads its inputs. Mount them there, or put prompts and tasks under ``fallback``."""
    tasks = R.tasks_volume(recipe)
    return {
        "draft_checkpoint": _draft_checkpoint(recipe),
        "model_dir": str(R.model_dir(recipe)),
        "prompts": str(R.PROMPTS),
        "tasks": (
            f"{R.DATASETS}/{tasks[1].lstrip('/')}" if tasks else (str(R.TASKS) if R.reads_task_dirs(recipe) else None)
        ),
        "fallback": f"{R.RESULTS}/{run_id}/inputs",
    }


def _role(shape: dict, nodes: int | None) -> dict:
    return {
        "gpu": shape["gpu"],
        "gpus_per_node": shape.get("gpus_per_node", R.GPUS_PER_NODE),
        "nodes": nodes,  # None: as many as the caller wants
        "cpu": shape["cpu"],
        "memory_gib": shape["memory_gib"],
        "disk_gib": shape["disk_gib"],
        "timeout_hours": shape["timeout_hours"],
    }


def manifest(recipe: dict, run_id: str) -> dict:
    """What the caller has to provide for this run. The caller never parses a recipe itself; the recipe's
    ``cluster`` keys reach it here as advice."""
    shape = R.shape(recipe)
    roles = {"trainer": {**_role(shape, shape["nodes"]), "ports": []}}
    if shape["nodes"] > 1:
        # Ray and NCCL pick ports at run time, so a gang's nodes reach each other on any port
        roles["trainer"].update(placement="one_cluster", interconnect="rdma", open_between_nodes="all ports")
    if R.needs_tunnel(recipe):
        roles["trainer"]["ports"].append(SESSION_PROXY_PORT)
    if R.is_split(recipe):
        roles["trainer"]["ports"].append(REGISTRY_PORT)
        engine = {**shape, **shape["engine"], "gpus_per_node": shape["engine"].get("gpus", R.GPUS_PER_NODE)}
        rollout = _role(engine, None)
        rollout["gpu_advice"] = rollout.pop(
            "gpu"
        )  # not binding: gpus_per_engine, precision and memory for the model are
        roles["rollout"] = {
            **rollout,
            "gpus_per_engine": int(recipe["rollout"]["gpus_per_engine"]),
            "precision": R.rollout_precision(recipe),
            "ports": [ROLLOUT_PROXY_PORT],
        }
    mounts = {str(R.MODELS): "ro", str(R.RESULTS): "rw", COMPILE_CACHE_MOUNT: "rw,optional"}
    if R.tasks_volume(recipe):
        mounts[str(R.DATASETS)] = "ro"
    return {
        "run_id": run_id,
        "roles": roles,
        "mounts": mounts,
        "inputs": _inputs(recipe, run_id),
        "secrets": node_secrets(recipe),
    }


def missing(recipe: dict, role: str, env) -> list[str]:
    """Return what this role needs from its caller and did not get."""
    need = []
    if role == "trainer":
        need += [NODE_RANK_ENV, NNODES_ENV, HEAD_ADDR_ENV, NODE_IP_ENV]
        if R.is_split(recipe):
            need.append(POOL_TOKEN_ENV)
        if R.needs_tunnel(recipe) and env.get(NODE_RANK_ENV, "0") == "0":
            need.append(SESSION_URL_ENV)
    elif role == "rollout":
        need += [POOL_TOKEN_ENV, ROLLOUT_URL_ENV]
        if not env.get(TRAINER_URL_FILE_ENV):
            need.append(TRAINER_URL_ENV)
    need += node_secrets(recipe)
    return [name for name in need if not env.get(name)]
