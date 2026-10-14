from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_nodes


class Box:
    def __init__(self, inner):
        self._inner = inner

    @property
    def inner(self):
        return self._inner


def compute_ray_pin_head_options():
    head_node_id = _get_head_node_id()
    return {
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        )
    }


def _get_head_node_id() -> str:
    # [fork fix-003] begin: read the GCS, not the dashboard's state API, which a
    # process off the head node reaches at 127.0.0.1:8265 unless the dashboard
    # listens on 0.0.0.0 and RAY_API_SERVER_ADDRESS is set (see fork/README.md)
    import ray

    for node in ray.nodes():
        if node.get("Alive") and "node:__internal_head__" in node.get("Resources", {}):
            return node["NodeID"]
    # [fork fix-003] end
    for node in list_nodes():
        if node.is_head_node:
            return node.node_id
    raise RuntimeError("Could not find a head node in the Ray cluster")
