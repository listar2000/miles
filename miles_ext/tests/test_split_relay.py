"""The split trainer's local relay and its heartbeat to the engine container."""

import threading
import time
from unittest import mock

import pytest


def test_relay_forwards_to_the_engine_port():
    pytest.importorskip("aiohttp")
    import asyncio

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from miles_ext.launch.split import relay

    async def run():
        seen = []

        async def upstream(request):
            seen.append((request.path, request.query_string, await request.text()))
            return web.json_response({"ok": True})

        up = web.Application()
        up.router.add_route("*", "/{tail:.*}", upstream)
        async with TestServer(up) as server:
            base = f"http://{server.host}:{server.port}/tok"
            async with TestClient(TestServer(relay.make_app(base, 32001))) as client:
                resp = await client.post("/generate?x=1", data="body")
                assert resp.status == 200 and (await resp.json()) == {"ok": True}
        return seen

    assert asyncio.run(run()) == [("/tok/32001/generate", "x=1", "body")]


def test_no_heartbeat_overwrites_release(monkeypatch):
    from miles_ext.launch import split
    from miles_ext.launch.split import relay

    monkeypatch.setattr(relay, "HEARTBEAT_SECS", 0.001)
    progress = {}
    relay.start_heartbeat("run1", progress, segment="s1", on_node_failure=lambda _note: None)
    relay.release("run1", progress)
    threading.Event().wait(0.05)  # many beat intervals
    assert progress[split.trainer_key("run1")]["done"] is True


def test_the_relay_opens_after_the_engines_serve_and_the_call_returns_before(monkeypatch):
    from miles_ext.launch import split
    from miles_ext.launch.split import relay

    monkeypatch.setattr(relay, "POLL_SECS", 0.001)
    proc = mock.Mock()
    connected = threading.Event()

    def fake_popen(*_args, **_kwargs):
        connected.set()
        return proc

    monkeypatch.setattr(relay.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(relay, "_wait_port", lambda *_, **__: None)
    progress = {split.engine_key("run1"): {"phase": "starting"}}
    failures = []
    monkeypatch.setattr(relay.R, "num_engines", lambda _recipe: 1)
    stop = relay.connect_in_background({}, "run1", progress, timeout=60, on_failure=failures.append)

    assert not connected.wait(0.05)
    progress[split.engine_key("run1")] = {
        "phase": "ready",
        "url": "https://t",
        "token": "tok",
        "engine_ports": [32000],
        "agent_port": 32001,
    }
    assert connected.wait(5)
    _wait_until(lambda: (stop(), proc.terminate.called)[1])
    assert failures == []


def _wait_until(condition, timeout: float = 5):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.001)


def test_one_stop_before_the_engines_serve_ends_the_connect_without_a_failure(monkeypatch):
    from miles_ext.launch import split
    from miles_ext.launch.split import relay

    monkeypatch.setattr(relay, "POLL_SECS", 0.001)
    popen = mock.Mock()
    monkeypatch.setattr(relay.subprocess, "Popen", popen)
    connect, returned = relay.connect, threading.Event()

    def tracked_connect(*args, **kwargs):
        try:
            return connect(*args, **kwargs)
        finally:
            returned.set()

    monkeypatch.setattr(relay, "connect", tracked_connect)
    progress = {split.engine_key("run1"): {"phase": "starting"}}
    failures = []
    stop = relay.connect_in_background({}, "run1", progress, timeout=60, on_failure=failures.append)
    try:
        stop()
        assert returned.wait(5)
        assert failures == [] and not popen.called
    finally:
        progress[split.engine_key("run1")] = {"phase": "ended"}  # ends the wait if cancellation regressed


READY = {"phase": "ready", "url": "https://t", "token": "tok", "engine_ports": [32000], "agent_port": 32001}


def _ready_engine_with_a_relay(monkeypatch, wait_port):
    from miles_ext.launch import split
    from miles_ext.launch.split import relay

    proc = mock.Mock()
    monkeypatch.setattr(relay.subprocess, "Popen", lambda *_args, **_kwargs: proc)
    monkeypatch.setattr(relay.R, "num_engines", lambda _recipe: 1)
    monkeypatch.setattr(relay, "_wait_port", wait_port)
    return proc, {split.engine_key("run1"): READY}


def test_a_stop_while_the_relay_ports_open_terminates_the_relay(monkeypatch):
    from miles_ext.launch.split import relay

    waiting = threading.Event()

    def wait_port(_port, _proc, *, stop):
        waiting.set()
        stop.wait(5)
        raise relay._Stopped

    proc, progress = _ready_engine_with_a_relay(monkeypatch, wait_port)
    failures = []
    stop = relay.connect_in_background({}, "run1", progress, timeout=60, on_failure=failures.append)
    assert waiting.wait(5)
    stop()
    _wait_until(lambda: proc.terminate.called)
    assert failures == []


def test_a_relay_whose_ports_never_open_is_terminated_and_reported(monkeypatch):
    from miles_ext.launch.split import relay

    def wait_port(_port, _proc, *, stop):
        raise RuntimeError("relay port 31000 did not open")

    proc, progress = _ready_engine_with_a_relay(monkeypatch, wait_port)
    notes = []
    relay.connect_in_background({}, "run1", progress, timeout=60, on_failure=notes.append)
    _wait_until(lambda: notes)
    assert proc.terminate.called and "did not open" in notes[0]


def test_a_relay_started_during_stop_is_terminated(monkeypatch):
    from miles_ext.launch.split import relay

    proc, entered, release = mock.Mock(), threading.Event(), threading.Event()

    def slow_connect(*_args, **_kwargs):
        entered.set()
        release.wait(5)
        return proc

    monkeypatch.setattr(relay, "connect", slow_connect)
    stop = relay.connect_in_background({}, "run1", {}, timeout=60, on_failure=lambda _note: None)
    assert entered.wait(5)
    stop()
    release.set()
    _wait_until(lambda: proc.terminate.called)


def test_rank_0_stops_the_run_for_this_segments_failed_node_only(monkeypatch):
    from miles_ext.launch.split import relay

    monkeypatch.setattr(relay, "HEARTBEAT_SECS", 0.001)
    monkeypatch.setattr(relay, "_RELEASED", threading.Event())
    progress, notes = {}, []
    old_segment = relay.publish_segment("run1", progress)
    segment = relay.publish_segment("run1", progress)
    assert relay.current_segment("run1", progress) == segment != old_segment
    relay.start_heartbeat("run1", progress, segment=segment, on_node_failure=notes.append)
    relay.report_node_failure("run1", progress, segment=old_segment, note="a late report from the old segment")
    time.sleep(0.05)
    assert notes == []
    relay.report_node_failure("run1", progress, segment=segment, note="node 1: relay port did not open")
    _wait_until(lambda: notes)
    relay._RELEASED.set()
    assert notes[0] == "node 1: relay port did not open"


def test_an_engine_that_ends_during_startup_fails_the_background_connect(monkeypatch):
    from miles_ext.launch import split
    from miles_ext.launch.split import relay

    monkeypatch.setattr(relay, "POLL_SECS", 0.001)
    progress = {split.engine_key("run1"): {"phase": "ended", "reason": "error: OOM"}}
    failed = threading.Event()
    notes = []

    def on_failure(note):
        notes.append(note)
        failed.set()

    relay.connect_in_background({}, "run1", progress, timeout=60, on_failure=on_failure)
    assert failed.wait(5) and "error: OOM" in notes[0]


@pytest.mark.parametrize(
    "state", [{"phase": "starting", "container_at": 1.0}, {"phase": "ready"}], ids=["container up", "a reused engine"]
)
def test_the_launcher_waits_only_for_the_engine_container(state):
    from miles_ext.launch import split
    from miles_ext.launch.split import relay

    relay.wait_container("run1", {split.engine_key("run1"): state}, timeout=0)


def test_the_launcher_refuses_an_engine_that_already_ended():
    from miles_ext.launch import split
    from miles_ext.launch.split import relay

    state = {"phase": "ended", "container_at": 1.0, "reason": "error: OOM"}
    with pytest.raises(RuntimeError, match="error: OOM"):
        relay.wait_container("run1", {split.engine_key("run1"): state}, timeout=60)


def test_the_launcher_keeps_waiting_while_the_engine_queues_for_gpus(monkeypatch):
    from miles_ext.launch import split
    from miles_ext.launch.split import relay

    monkeypatch.setattr(relay, "POLL_SECS", 0.001)
    with pytest.raises(TimeoutError, match="not running"):
        relay.wait_container("run1", {split.engine_key("run1"): {"phase": "starting"}}, timeout=0.01)
