"""Run state behind MILES_STATE, and the trainer role's env."""

import http.server
import subprocess
import sys
import threading
import time

import pytest

from miles_ext.node import contract as C
from miles_ext.node import state as S


def test_file_state_round_trip(tmp_path):
    state = S.FileState("run1", tmp_path)
    assert state.get("run:run1") is None
    state["run:run1"] = {"phase": "training"}
    assert state.get("run:run1") == {"phase": "training"}
    assert state.pop("run:run1") == {"phase": "training"}
    assert state.pop("run:run1", "gone") == "gone"


def test_http_state_round_trip_with_its_token(monkeypatch):
    store: dict = {}
    seen: list = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, code, body=b"null"):
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_PUT(self):
            seen.append(self.headers["Authorization"])
            store[self.path] = self.rfile.read(int(self.headers["Content-Length"]))
            self._reply(200)

        def do_GET(self):
            self._reply(200, store[self.path]) if self.path in store else self._reply(404)

        def do_DELETE(self):
            store.pop(self.path, None)
            self._reply(200)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv(C.STATE_ENV, S.HTTP)
        monkeypatch.setenv(C.STATE_URL_ENV, f"http://127.0.0.1:{server.server_address[1]}/ingest/")
        monkeypatch.setenv(C.STATE_TOKEN_ENV, "tok")
        state = S.for_run("run1")
        assert state.get("run:run1") is None
        state["run:run1"] = {"phase": "training"}
        assert seen == ["Bearer tok"] and "/ingest/run1/run%3Arun1" in store
        assert state.get("run:run1") == {"phase": "training"}
        assert state.pop("run:run1") == {"phase": "training"} and state.get("run:run1") is None
    finally:
        server.shutdown()
        server.server_close()


def test_http_state_raises_oserror_when_the_viewer_is_away(monkeypatch):
    monkeypatch.setenv(C.STATE_URL_ENV, "http://127.0.0.1:9")
    monkeypatch.setenv(C.STATE_TOKEN_ENV, "tok")
    state = S.HttpState("run1")
    with pytest.raises(OSError):
        state["run:run1"] = {}


def test_an_unknown_state_store_is_refused(monkeypatch):
    monkeypatch.setenv(C.STATE_ENV, "redis")
    with pytest.raises(ValueError, match="redis"):
        S.for_run("run1")


def test_the_trainer_role_imports_without_modal():
    code = (
        "import sys, types; sys.modules['modal'] = None; sys.modules.setdefault('ray', types.ModuleType('ray'))\n"
        "import miles_ext.node.trainer"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_gang_reads_the_contracts_env(monkeypatch):
    pytest.importorskip("aiohttp")
    from miles_ext.node.trainer import Gang

    for name, value in (
        (C.NODE_RANK_ENV, "1"),
        (C.NNODES_ENV, "2"),
        (C.HEAD_ADDR_ENV, "10.0.0.1"),
        (C.NODE_IP_ENV, "10.0.0.2"),
    ):
        monkeypatch.setenv(name, value)
    assert Gang.from_env() == Gang(1, 2, "10.0.0.1", "10.0.0.2")


def test_a_node_whose_gpu_differs_keeps_its_own_compile_cache(monkeypatch):
    runtime = pytest.importorskip("miles_ext.node.runtime")
    monkeypatch.setattr(runtime, "version", lambda name: "1")
    assert runtime.compile_cache_key("H200") == "h200/sglang-1"
    monkeypatch.setenv(C.GPU_ENV, "B300")
    assert runtime.compile_cache_key("H200") == "b300/sglang-1"


def test_tee_logs_copies_a_process_and_its_children(tmp_path):
    runtime = pytest.importorskip("miles_ext.node.runtime")
    code = (
        "import os, subprocess, sys\n"
        "from pathlib import Path\n"
        "from miles_ext.node import runtime\n"
        f"runtime.tee_logs(Path({str(tmp_path)!r}), 'run1', 'trainer-0')\n"
        "print('from me', flush=True)\n"
        "subprocess.run(['sh', '-c', 'echo from a child; echo to stderr 1>&2'])\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert runtime and "from me" in out.stdout
    for _ in range(50):
        log = (tmp_path / "run1" / "logs" / "trainer-0.log").read_text()
        if "to stderr" in log:
            break
        time.sleep(0.1)
    assert "from me" in log and "from a child" in log and "to stderr" in log


def _links(monkeypatch, wait_members):
    """Run the trainer's split links with a fake relay; returns what reached on_failure."""
    pytest.importorskip("aiohttp")
    pytest.importorskip("torch")
    import contextlib

    from miles_ext.launch.split import relay
    from miles_ext.node import registry, trainer

    monkeypatch.setenv(C.POOL_TOKEN_ENV, "t")
    monkeypatch.setattr(relay, "start", lambda url, token: subprocess.Popen(["sleep", "30"]))
    monkeypatch.setattr(relay, "wait_open", lambda members, proc: None)
    monkeypatch.setattr(relay, "check_members", lambda recipe, members: None)
    monkeypatch.setattr(registry, "wait_members", wait_members)
    monkeypatch.setattr(trainer.R, "shape", lambda recipe: {"engine": {"timeout_hours": 1}})
    failures: list[str] = []
    done = threading.Event()

    def on_failure(note):
        failures.append(note)
        done.set()

    env: dict = {}
    with contextlib.ExitStack() as stack:
        trainer._split_links({}, "/results/r1", env, stack, "http://registry", None, on_failure)
        assert "MILES_SPLIT_REGISTRY_URL" in env  # known at once: Ray starts without waiting for the engines
        done.wait(5)
    return failures


def test_a_rollout_node_that_never_comes_stops_the_run(monkeypatch):
    def wait_members(url, token, *, timeout, check):
        raise TimeoutError("no rollout node joined")

    assert "no rollout node joined" in _links(monkeypatch, wait_members)[0]


def test_ray_can_start_while_a_rollout_node_is_still_pending(monkeypatch):
    pending, returned_while_pending = threading.Event(), threading.Event()
    joined = threading.Event()

    def wait_members(url, token, *, timeout, check):
        pending.set()
        assert joined.wait(5)
        returned_while_pending.set()
        return []

    pytest.importorskip("aiohttp")
    pytest.importorskip("torch")
    import contextlib

    from miles_ext.launch.split import relay
    from miles_ext.node import registry, trainer

    monkeypatch.setenv(C.POOL_TOKEN_ENV, "t")
    monkeypatch.setattr(relay, "start", lambda url, token: subprocess.Popen(["sleep", "30"]))
    monkeypatch.setattr(relay, "wait_open", lambda members, proc: None)
    monkeypatch.setattr(relay, "check_members", lambda recipe, members: None)
    monkeypatch.setattr(registry, "wait_members", wait_members)
    monkeypatch.setattr(trainer.R, "shape", lambda recipe: {"engine": {"timeout_hours": 1}})
    failures: list[str] = []
    env: dict = {}
    try:
        with contextlib.ExitStack() as stack:
            trainer._split_links({}, "/results/r1", env, stack, "http://registry", None, failures.append)
            # _split_links is back, with the env Ray needs, and the node has not joined
            assert pending.wait(5) and not returned_while_pending.is_set()
            assert "MILES_SPLIT_REGISTRY_URL" in env
            joined.set()
            assert returned_while_pending.wait(5)
    finally:
        joined.set()
    assert not failures
