"""Rank 0's gang server, with ``shards`` on both ends as a multi-node save uses it."""

import queue
import socket
import threading
import time

import pytest

pytest.importorskip("aiohttp")

from miles_ext.node import gang  # noqa: E402
from miles_ext.node.gang import GangClient, GangServer  # noqa: E402
from miles_ext.platform import shards  # noqa: E402


_servers: list[GangServer] = []


@pytest.fixture(autouse=True)
def _stop_servers():
    yield
    while _servers:
        _servers.pop().stop()


def _server() -> tuple[GangServer, GangClient]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    _servers.append(GangServer().start(port))
    return _servers[-1], GangClient("127.0.0.1", port)


def test_workers_commit_before_rank_0_over_http():
    server, client = _server()
    order: list[str] = []
    stop = threading.Event()
    worker = threading.Thread(
        target=shards.worker_commits,
        args=("run1", 1, client.kv, client.queue, lambda: order.append("worker"), stop),
        kwargs={"key": "commit:run1", "every": 1000, "poll": 0.01},
        daemon=True,
    )
    worker.start()
    commit = shards.head_commit("run1", 2, server.kv, server.queue, lambda: order.append("head"), key="commit:run1")
    commit()
    stop.set()
    worker.join(5)
    assert order == ["worker", "head"]


def test_release_stays_up_until_every_worker_confirmed(monkeypatch):
    monkeypatch.setattr(gang, "DRAIN_SECS", 5)
    server, client = _server()
    release = threading.Thread(target=server.release, args=("run1", False, 3), daemon=True)
    release.start()
    time.sleep(0.3)
    assert release.is_alive()  # nobody has read it yet
    assert client.wait_release("run1", 1, timeout=1, poll=0.01) is False
    time.sleep(0.3)
    assert release.is_alive()  # one of two
    assert client.wait_release("run1", 2, timeout=1, poll=0.01) is False
    release.join(1)
    assert not release.is_alive()


def test_a_worker_whose_reply_was_lost_reads_the_result_again():
    server, client = _server()
    server.kv["release:run1"] = True
    assert client.wait_release("run1", 1, timeout=1, poll=0.01) is True
    assert client.wait_release("run1", 1, timeout=1, poll=0.01) is True


def test_a_worker_with_the_result_succeeds_though_its_confirmation_is_lost():
    server, client = _server()
    server.kv["release:run1"] = True

    class LostReply(type(client.kv)):
        def __setitem__(self, key, value):
            raise OSError("rank 0 exited")

    client.kv = LostReply(client.kv.base)
    assert client.wait_release("run1", 1, timeout=1, poll=0.01) is True


def test_release_gives_up_on_a_worker_that_never_reads(monkeypatch):
    monkeypatch.setattr(gang, "DRAIN_SECS", 0.05)
    server, _ = _server()
    release = threading.Thread(target=server.release, args=("run1", True, 2), daemon=True)
    release.start()
    release.join(2)
    assert not release.is_alive()


def test_a_queue_get_raises_empty_at_its_timeout_on_both_ends():
    server, client = _server()
    with pytest.raises(queue.Empty):
        server.queue.get(partition="p", timeout=0)
    with pytest.raises(queue.Empty):
        client.queue.get(partition="p", timeout=0)


def test_a_worker_fails_when_rank_0_stays_unreachable():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    client = GangClient("127.0.0.1", port)
    with pytest.raises(RuntimeError, match="unreachable"):
        client.wait_release("run1", 1, timeout=2, head_gone_secs=0.01, poll=0.01)
