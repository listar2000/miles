"""What a caller gives a node and what a node gives back (RFC 0005): the single owner of these names."""

from miles_ext.recipes import recipe as R

STORAGE_ENV = "MILES_STORAGE"  # miles_ext.node.storage

# ---- ports a node opens ----
SESSION_PROXY_PORT = R.PROXY_PORT  # rank 0, in-sandbox harnesses
REGISTRY_PORT = 30999  # rank 0, split
ROLLOUT_PROXY_PORT = 32999  # a rollout node: its engines and sync agent, behind one token

COMPILE_CACHE_MOUNT = "/compile-cache"


def required_secrets(recipe: dict) -> list[str]:
    """The env vars holding credentials a run's nodes read. The image never reads a .env."""
    keys = ["WANDB_API_KEY"]
    if recipe["sandbox"]["provider"] == "sail":
        keys.append("SAIL_API_KEY")
    return keys


def _role(shape: dict, nodes: int | None) -> dict:
    return {
        "gpu": shape["gpu"],
        "gpus_per_node": R.GPUS_PER_NODE,
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
        roles["trainer"].update(placement="one_cluster", interconnect="rdma")
    if R.needs_tunnel(recipe):
        roles["trainer"]["ports"].append(SESSION_PROXY_PORT)
    if R.is_split(recipe):
        roles["trainer"]["ports"].append(REGISTRY_PORT)
        rollout = _role({**shape, **shape["engine"]}, None)
        rollout["gpu_advice"] = rollout.pop("gpu")  # any GPU that meets gpus_per_engine and precision will do
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
        "secrets": required_secrets(recipe),
    }
