"""A rollout node and the trainer's registry, end to end on one host with fake engines (no GPU, no Modal)."""

import contextlib
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

pytest.importorskip("aiohttp")

from miles_ext.launch.split import relay  # noqa: E402
from miles_ext.node import contract as C  # noqa: E402
from miles_ext.node import registry as Reg  # noqa: E402
from miles_ext.node import rollout
from miles_ext.node import storage as S  # noqa: E402

FAKE_ENGINE = (
    "import http.server, sys\n"
    "class H(http.server.BaseHTTPRequestHandler):\n"
    "    def do_GET(self):\n"
    "        self.send_response(200); self.end_headers(); self.wfile.write(sys.argv[1].encode() + self.path.encode())\n"
    "    def log_message(self, *a): pass\n"
    "http.server.ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()\n"
)


def _free_port(span: int = 1) -> int:
    """The first of ``span`` free ports in a row."""
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            base = s.getsockname()[1]
        try:
            for port in range(base, base + span):
                with socket.socket() as s:
                    s.bind(("127.0.0.1", port))
            return base
        except OSError:
            continue


@pytest.fixture
def node(tmp_path, monkeypatch):
    """Patch the rollout role down to two fake engines on free ports; yields a starter for it."""
    script = tmp_path / "engine.py"
    script.write_text(FAKE_ENGINE)
    base = _free_port(span=3)  # two engines and the agent
    monkeypatch.setattr(rollout, "ENGINE_HTTP_PORT", base)
    monkeypatch.setattr(C, "ROLLOUT_PROXY_PORT", _free_port())
    monkeypatch.setattr(rollout, "POLL_SECS", 0.05)
    monkeypatch.setattr(Reg, "BEAT_SECS", 0.2)
    monkeypatch.setattr(Reg, "WAIT_POLL_SECS", 0.05)
    monkeypatch.setattr(rollout.stack, "MILES_SRC", tmp_path)
    monkeypatch.setattr(
        rollout,
        "commands",
        lambda recipe, run_id, env: {
            "commands": [f"{sys.executable} {script} {base + i}" for i in range(2)],
            "env": {},
            "gpus_per_engine": 1,
        },
    )
    started, threads = [], []
    popen = rollout.subprocess.Popen

    def tracking_popen(*args, **kwargs):
        started.append(popen(*args, **kwargs))
        return started[-1]

    monkeypatch.setattr(rollout.subprocess, "Popen", tracking_popen)
    url_file = tmp_path / "trainer-url"
    states: list[dict] = []
    result: dict = {}
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": ":".join(sys.path),
        C.TRAINER_URL_FILE_ENV: str(url_file),
        C.POOL_TOKEN_ENV: "pool",
        C.ROLLOUT_URL_ENV: f"http://127.0.0.1:{C.ROLLOUT_PROXY_PORT}",
    }

    def start(idle_minutes: float = 10) -> threading.Thread:
        recipe = {"cluster": {"engine": {"idle_minutes": idle_minutes}}}

        def run():
            with contextlib.suppress(RuntimeError):  # the fixture's teardown ended the engines
                result["state"] = rollout.serve(
                    recipe,
                    "run1",
                    env=env,
                    storage=S.PosixStorage(tmp_path),
                    on_state=lambda s: states.append(dict(s)),
                )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        threads.append(thread)
        return thread

    yield start, url_file, states, result
    for proc in started:  # a test that failed before its release leaves the engines up
        proc.kill()
        proc.wait(10)
    for thread in threads:
        thread.join(30)


_registries: list[Reg.Registry] = []


@pytest.fixture(autouse=True)
def _stop_registries():
    yield
    while _registries:
        _registries.pop().stop()


def _registry() -> tuple[Reg.Registry, str]:
    port = _free_port()
    _registries.append(Reg.Registry("pool").start(port))
    return _registries[-1], f"http://127.0.0.1:{port}"


def _join(thread: threading.Thread, timeout: float = 30) -> None:
    thread.join(timeout)
    assert not thread.is_alive()


def test_a_node_joins_serves_through_a_relay_and_leaves_on_release(node, monkeypatch):
    start, url_file, states, result = node
    registry, url = _registry()
    url_file.write_text(url)
    thread = start()

    (member,) = Reg.wait_members(url, "pool", timeout=30)
    assert states[-1]["phase"] == "ready"
    assert member["url"] == f"http://127.0.0.1:{C.ROLLOUT_PROXY_PORT}" and len(member["engine_ports"]) == 2

    local = _free_port(span=3)  # the relay's two engines and agent
    monkeypatch.setattr(relay.R, "num_engines", lambda recipe: 2)
    monkeypatch.setattr(relay.R, "SPLIT_ENGINE_PORT", local)
    proc, agent_url = relay.connect({}, member)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{local + 1}/generate", timeout=10) as r:
            assert r.read().decode() == f"{member['engine_ports'][1]}/generate"
        req = urllib.request.Request(f"{agent_url}/connect", data=json.dumps({"engines": 2}).encode())
        with urllib.request.urlopen(req, timeout=10) as r:
            assert json.loads(r.read())["ok"] is True
    finally:
        proc.terminate()

    registry.release()
    _join(thread)
    assert result["state"]["reason"] == "trainer done" and result["state"]["phase"] == "ended"


def test_a_relaunched_trainer_finds_the_live_node(node):
    start, url_file, _, result = node
    old, old_url = _registry()
    url_file.write_text(old_url)
    thread = start()
    (first,) = Reg.wait_members(old_url, "pool", timeout=30)

    new, new_url = _registry()
    url_file.write_text(new_url)
    (again,) = Reg.wait_members(new_url, "pool", timeout=10)
    assert again["node_id"] == first["node_id"]

    new.release()
    _join(thread)
    assert result["state"]["reason"] == "trainer done"


def test_a_node_reads_done_on_its_beat_when_the_push_was_missed(node):
    start, url_file, _, result = node
    registry, url = _registry()
    url_file.write_text(url)
    thread = start()
    Reg.wait_members(url, "pool", timeout=30)
    registry.done = True
    _join(thread)
    assert result["state"]["reason"] == "trainer done"


def test_a_node_stays_until_its_idle_limit_with_no_trainer(monkeypatch):
    monkeypatch.setattr(rollout, "POLL_SECS", 0.01)
    agent = SimpleNamespace(released=threading.Event())
    done: list[str] = []
    args = ({"node_id": "a"}, {}, agent, [], "pool", 1)
    thread = threading.Thread(target=lambda: done.append(rollout._serve_until_released(*args)), daemon=True)
    thread.start()
    time.sleep(0.5)
    assert not done  # still within the limit
    thread.join(5)
    assert done == ["no trainer for 0 min"]


def test_the_fixed_url_form_needs_no_file():
    assert rollout.trainer_url({C.TRAINER_URL_ENV: "http://t", C.TRAINER_URL_FILE_ENV: "/nope"}) == "http://t"


def test_the_registry_answers_only_the_pool_token():
    _, url = _registry()
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(f"{url}/members", timeout=5)
    assert e.value.code == 401


def test_members_drop_out_when_they_stop_beating(monkeypatch):
    registry = Reg.Registry("pool")
    registry.beat({"node_id": "a", "url": "http://a"})
    assert [m["node_id"] for m in registry.members()] == ["a"]
    monkeypatch.setattr(Reg, "MEMBER_TTL_SECS", 0)
    time.sleep(0.01)
    assert registry.members() == []


def test_the_rollout_role_imports_without_modal():
    import subprocess

    code = "import sys; sys.modules['modal'] = None\nimport miles_ext.node.rollout, miles_ext.node.registry"
    subprocess.run([sys.executable, "-c", code], check=True)
