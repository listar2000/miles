"""Hold the Mooncake memory a split run's object store puts into, for the life of the rollout app.

In Miles only the trainer ranks contribute Mooncake segments. A trainer that joins after rollout 0 leaves that
batch nowhere to go, and a per-step trainer would take stored objects with it when it exits, so the rollout app
runs this process and trainers contribute none.

    python -m miles_ext.disagg.segment_host <master host:port> <local host> <size, e.g. 16gb>
"""

import sys
import time

from mooncake.store import MooncakeDistributedStore

from miles.utils.object_store_config import MOONCAKE_MASTER_ADDRESS_KEY, compute_mooncake_store_config

if __name__ == "__main__":
    master, host, size = sys.argv[1:4]
    kwargs = {"protocol": "tcp", MOONCAKE_MASTER_ADDRESS_KEY: master, "local_hostname": host}
    store = MooncakeDistributedStore()
    config = compute_mooncake_store_config({**kwargs, "global_segment_size": size}, contribute_segment=True)
    if err := store.setup(config):
        sys.exit(f"Mooncake setup failed: {err}")
    print(f"segment host: {size} mounted at {host}", flush=True)
    while True:
        time.sleep(3600)
