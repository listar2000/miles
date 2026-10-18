"""Engine medic for split runs: restarts the inference deployment's engine cells, run against its Ray cluster.

    python -m miles_ext.disagg.medic

Miles tears down an engine cell whose worker died and does not start it again. The medic starts such cells again,
and on request (Dict key ``restart_engines/<tag>``, written by the on-demand driver after a trainer was lost during a
weight push, which leaves every engine in the push stuck in the broken NCCL group) stops and starts every engine
cell, then acknowledges with ``engines_restarted/<tag>``. Restarted engines wait for weights until the next push.
"""

import os
import time

import ray
from miles_ext.disagg.addrs import MANAGER
from miles_ext.disagg.late_start import DICT_ENV, mark
from ray.util.state import list_actors

ENGINE_POOL_PREFIX = "inference-engine"
RESTART_KEY, RESTARTED_KEY = "restart_engines/", "engines_restarted/"
POLL_S = 10


def engine_pools(addrs: dict) -> list[str]:
    return [p for p in addrs if p.startswith(ENGINE_POOL_PREFIX)]


def torn_down(infos: dict) -> list[str]:
    """Cells with no workers: stopped after a worker died (``start_cells`` skips cells that still have workers)."""
    return sorted(cell_id for cell_id, info in infos.items() if not info.worker_names)


def pending_requests(keys) -> list[str]:
    done = {k[len(RESTARTED_KEY) :] for k in keys if k.startswith(RESTARTED_KEY)}
    return sorted(t for k in keys if k.startswith(RESTART_KEY) and (t := k[len(RESTART_KEY) :]) not in done)


def _manager_and_pools():
    """The worker manager once its init has run (its methods fail until then) and its engine pools."""
    while True:
        try:
            if found := list_actors(filters=[("name", "=", MANAGER), ("state", "=", "ALIVE")]):
                manager = ray.get_actor(MANAGER, namespace=found[0].ray_namespace)
                if pools := engine_pools(ray.get(manager.get_addrs.remote())):
                    return manager, pools
        except Exception as e:  # not initialized yet
            print(f"medic: worker manager not ready ({type(e).__name__})", flush=True)
        time.sleep(POLL_S)


def _tick(d, manager, pools: list[str]) -> None:
    infos = ray.get(manager.get_cell_infos.remote(pool_ids=pools))
    for tag in pending_requests(list(d.keys())):
        print(f"medic: restarting all engines ({tag})", flush=True)
        ray.get(manager.stop_cells.remote(sorted(infos)))
        ray.get(manager.start_cells.remote(sorted(infos)))
        d[RESTARTED_KEY + tag] = time.time()
        mark(d, f"engines_restarted_{tag}", cells=sorted(infos))
        infos = ray.get(manager.get_cell_infos.remote(pool_ids=pools))
    if dead := torn_down(infos):
        print(f"medic: starting torn-down cells {dead}", flush=True)
        ray.get(manager.start_cells.remote(dead))
        mark(d, f"engines_started_{int(time.time())}", cells=dead)


def main() -> None:
    import modal

    d = modal.Dict.from_name(os.environ[DICT_ENV])
    ray.init(logging_level="ERROR", log_to_driver=False)
    manager, pools = _manager_and_pools()
    print(f"medic: watching {pools}", flush=True)
    while True:
        try:
            _tick(d, manager, pools)
        except Exception as e:  # keep watching; a failed restart is retried next tick
            print(f"medic: tick failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
