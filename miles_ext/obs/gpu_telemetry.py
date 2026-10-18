"""W&B system metrics per GPU node, labelled with what its GPUs do: ``train``, ``rollout`` or ``colocated``.

Plugged in with ``--custom-placement-hook-path miles_ext.obs.gpu_telemetry.start`` (hook-002) and
``WANDB__DISABLE_STATS=true`` in every Miles process's env, so these observers are the only samplers. Keys carry the
label ``<role>-node<k>`` (k counts nodes in IP order), e.g. ``gpu.0.gpu/l:rollout-node2``; the run config's
``gpu_nodes/<label>`` gives that node's IP and GPU ids.
"""

import logging
import os
from typing import NamedTuple

import ray
import wandb
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from miles.utils.tracking_utils.wandb_utils import _is_offline_mode, _wandb_settings

logger = logging.getLogger(__name__)

_observers = []  # Ray stops an actor once no handle to it is left, so the driver keeps these for the whole run


class NodeRole(NamedTuple):
    label: str
    node_id: str
    node_ip: str
    gpu_ids: list[int]


def node_roles(
    gpus: list[tuple[str, int]], *, num_train_gpus: int, colocate: bool, node_ips: dict[str, str]
) -> list[NodeRole]:
    """Group ``(node_id, gpu_id)`` pairs, in Miles' logical GPU order, by node and role. The first
    ``num_train_gpus`` logical GPUs train and the rest serve rollouts, unless both share every GPU."""
    nodes = list(dict.fromkeys(node for node, _ in gpus))
    groups: dict[tuple[str, str], list[int]] = {}
    for index, (node, gpu) in enumerate(gpus):
        role = "colocated" if colocate else "train" if index < num_train_gpus else "rollout"
        groups.setdefault((role, node), []).append(gpu)
    return [
        NodeRole(f"{role}-node{nodes.index(node)}", node, node_ips[node], sorted(gpu_ids))
        for (role, node), gpu_ids in groups.items()
    ]


def start(args, pgs) -> None:
    """The ``--custom-placement-hook-path`` function: start one observer per node and role. Never fails the run."""
    try:
        _start(args, pgs)
    except Exception:
        logger.exception("W&B GPU metrics per node not started")


def _start(args, pgs) -> None:
    actor_pg, rollout_pg = pgs["actor"], pgs["rollout"]
    if getattr(args, "wandb_run_id", None) is None or actor_pg.pg is None:
        return
    bundle_nodes = ray.util.placement_group_table(actor_pg.pg)["bundles_to_node_id"]
    gpus = [
        (bundle_nodes[bundle], gpu)
        for bundle, gpu in zip(actor_pg.pg_reordered_bundle_indices, actor_pg.pg_reordered_gpu_ids, strict=True)
    ]
    num_train_gpus = len(gpus) - len(rollout_pg.pg_reordered_bundle_indices)
    node_ips = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
    for node in node_roles(gpus, num_train_gpus=num_train_gpus, colocate=args.colocate, node_ips=node_ips):
        logger.info(f"W&B system metrics for {node.node_ip} GPUs {node.gpu_ids} under label {node.label}")
        strategy = NodeAffinitySchedulingStrategy(node_id=node.node_id, soft=False)
        _observers.append(_GpuNodeObserver.options(scheduling_strategy=strategy).remote(args, node))


@ray.remote(num_cpus=0, num_gpus=0)
class _GpuNodeObserver:
    """Joins the run as a W&B secondary, like ``init_wandb_secondary``, and stays alive so W&B keeps sampling."""

    def __init__(self, args, node: NodeRole):
        # Ray may hide the GPUs from an actor that asked for none; W&B needs to see the whole node
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            self._init_wandb(args, node)
        except Exception:
            logger.exception(f"W&B system metrics for {node.label} not started")

    @staticmethod
    def _init_wandb(args, node: NodeRole) -> None:
        if args.wandb_mode:
            os.environ["WANDB_MODE"] = args.wandb_mode
        offline = _is_offline_mode(args)
        if not offline and args.wandb_key is not None:
            wandb.login(key=args.wandb_key, host=args.wandb_host)
        mode = dict(mode="offline") if offline else dict(mode="shared", x_primary=False, x_update_finish_state=False)
        if args.wandb_dir:
            os.makedirs(args.wandb_dir, exist_ok=True)
        wandb.init(
            id=args.wandb_run_id,
            entity=args.wandb_team,
            project=args.wandb_project,
            dir=args.wandb_dir,
            # explicit False wins over the WANDB__DISABLE_STATS=true that silences every other Miles process
            settings=_wandb_settings(
                **mode, x_disable_stats=False, x_label=node.label, x_stats_gpu_device_ids=node.gpu_ids
            ),
        )
        wandb.config.update({f"gpu_nodes/{node.label}": {"ip": node.node_ip, "gpus": node.gpu_ids}})
