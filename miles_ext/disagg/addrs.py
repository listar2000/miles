"""Print the rpc address of one pool's first worker in the Ray cluster at $RAY_ADDRESS, as JSON, or nothing.

    python -m miles_ext.disagg.addrs trainer-controller-actor

Miles' RayWorkerManager is a named actor in its driver's (anonymous) namespace; the state API names that namespace.
"""

import json
import sys

import ray
from ray.util.state import list_actors

MANAGER = "ray_worker_manager"


def main(pool_id: str) -> None:
    ray.init(logging_level="ERROR", log_to_driver=False)
    found = [a for a in list_actors(filters=[("name", "=", MANAGER), ("state", "=", "ALIVE")])]
    if not found:
        return
    manager = ray.get_actor(MANAGER, namespace=found[0].ray_namespace)
    addrs = ray.get(manager.get_addrs.remote()).get(pool_id) or []
    if addrs and (rpc := addrs[0].get("rpc")):
        print(json.dumps({"host": rpc.host, "port": rpc.port}))


if __name__ == "__main__":
    main(sys.argv[1])
