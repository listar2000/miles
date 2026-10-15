"""miles_ext.platform.shards with in-memory fakes: rank 0 commits only after every worker has.

    python -m pytest miles_ext/tests/test_shards.py
"""

import queue as queue_lib
import threading

from miles_ext.platform import shards


class FakeQueue:
    def __init__(self):
        self.parts = {}

    def _q(self, partition):
        return self.parts.setdefault(partition, queue_lib.Queue())

    def put(self, item, partition):
        self._q(partition).put(item)

    def get(self, partition, timeout):
        try:
            return self._q(partition).get(timeout=timeout)
        except queue_lib.Empty:
            return None


def _cluster(nodes, live_workers):
    d, q, order, stop = {}, FakeQueue(), [], threading.Event()
    lock = threading.Lock()

    def committer(name):
        def commit():
            with lock:
                order.append(name)
        return commit

    for rank in range(1, live_workers + 1):
        threading.Thread(target=shards.worker_commits, daemon=True,
                         args=("run1", rank, d, q, committer(f"w{rank}"), stop),
                         kwargs={"key": "commit:run1", "every": 3600, "poll": 0.01}).start()
    head = shards.head_commit("run1", nodes, d, q, committer("head"), key="commit:run1", timeout=0.5,
                              log=lambda *_: None)
    return head, order, stop, q


def test_head_commits_after_every_worker():
    head, order, stop, _ = _cluster(nodes=3, live_workers=2)
    head()
    stop.set()
    assert order[-1] == "head" and sorted(order[:-1]) == ["w1", "w2"]


def test_dead_worker_does_not_hold_the_head():
    head, order, stop, _ = _cluster(nodes=3, live_workers=1)
    head()  # waits the 0.5 s timeout for rank 2
    stop.set()
    assert order == ["w1", "head"]


def test_stale_ack_is_ignored():
    head, order, stop, q = _cluster(nodes=2, live_workers=0)
    q.put((1, "old-token"), partition=shards.ack_partition("run1"))
    head()
    stop.set()
    assert order == ["head"]  # committed after the timeout, not on the stale ack


def test_single_node_is_a_plain_commit():
    calls = []
    assert shards.head_commit("run1", 1, {}, None, lambda: calls.append(1), key="k") is not None
    shards.head_commit("run1", 1, {}, None, lambda: calls.append(1), key="k")()
    assert calls == [1]
