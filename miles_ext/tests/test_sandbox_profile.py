"""ext-040's timing hooks: off they change nothing, on they wrap the provider and the proxy:

    python -m pytest miles_ext/tests/test_sandbox_profile.py
"""

import asyncio
import json
from types import SimpleNamespace

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from miles_ext.harbor import agent_function as af
from miles_ext.harbor.session_proxy import make_app
from miles_ext.obs import sandbox_profile as sp
from miles_ext.recipes import recipe as R
from miles_ext.sandboxes import modal_env


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_off_leaves_the_environment_config_alone(monkeypatch):
    monkeypatch.delenv(sp.ENV, raising=False)
    monkeypatch.delenv("HARBOR_ENV_IMPORT_PATH", raising=False)
    monkeypatch.setenv("HARBOR_ENV_TYPE", "modal")
    monkeypatch.setenv("HARBOR_ENV_KWARGS", '{"app_name": "miles-mobius-x"}')
    config = af._environment_config()
    assert config.type is None and config.import_path == modal_env.IMPORT_PATH


def test_on_wraps_the_same_provider_and_keeps_every_field(monkeypatch):
    monkeypatch.delenv("HARBOR_ENV_IMPORT_PATH", raising=False)
    monkeypatch.setenv("HARBOR_ENV_TYPE", "modal")
    monkeypatch.setenv("HARBOR_ENV_KWARGS", '{"app_name": "miles-mobius-x"}')
    monkeypatch.setenv("HARBOR_CPU_ENFORCEMENT_POLICY", "limit")
    monkeypatch.delenv(sp.ENV, raising=False)
    plain = af._environment_config()
    monkeypatch.setenv(sp.ENV, "1")
    wrapped = af._environment_config()
    assert wrapped.import_path == sp.IMPORT_PATH and wrapped.type is None
    assert wrapped.model_dump(exclude={"type", "import_path"}) == plain.model_dump(exclude={"type", "import_path"})
    assert sp.inner_import_path() == modal_env.IMPORT_PATH


def test_import_path_provider_is_the_inner_class(monkeypatch):
    monkeypatch.setenv("HARBOR_ENV_IMPORT_PATH", R.SAIL_IMPORT_PATH)
    assert sp.inner_import_path() == R.SAIL_IMPORT_PATH


class _Fake:
    """A provider: its start execs once (a nested span), its exec fails on 'boom'."""

    def __init__(self, trial_dir):
        self.trial_paths = SimpleNamespace(trial_dir=trial_dir)

    async def start(self, force_build):
        await self.exec("mkdir -p /app")

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if command == "boom":
            raise RuntimeError("exec failed")
        return SimpleNamespace(return_code=3 if command == "false" else 0)

    async def stop(self, delete):
        pass

    async def upload_file(self, source_path, target_path):
        pass

    async def upload_dir(self, source_dir, target_dir):
        pass


def test_spans_time_each_call_and_mark_nested_ones(tmp_path, monkeypatch):
    monkeypatch.setenv(sp.PROBES_ENV, "2")
    env = sp.profiled(_Fake)(tmp_path)

    async def go():
        await env.start(False)
        await env.exec("false", timeout_sec=60)
        try:
            await env.exec("boom")
        except RuntimeError:
            pass
        await env.stop(True)

    asyncio.run(go())
    spans = _lines(tmp_path / sp.TRIAL_FILE)
    ops = [(s["op"], s["depth"]) for s in spans]
    # the nested exec ends (and is written) before the start that made it
    assert ops == [
        ("exec", 1),
        ("start", 0),
        ("probe_exec", 0),
        ("probe_exec", 0),
        ("exec", 0),
        ("exec", 0),
        ("stop", 0),
    ]
    assert spans[4]["rc"] == 3 and spans[4]["timeout_sec"] == 60 and spans[4]["cmd"] == "false"
    assert spans[5]["ok"] is False and spans[5]["error"] == "RuntimeError"
    assert all(s["dt"] >= 0 for s in spans)


def test_transfers_by_keyword_are_timed(tmp_path):
    """Harbor's artifact handler passes source_path= / source_dir= by keyword."""
    env = sp.profiled(_Fake)(tmp_path)

    async def go():
        await env.upload_file(source_path="/a/b", target_path="/c")
        await env.upload_dir(source_dir="/d", target_dir="/e")
        await env.upload_file("/f", "/g")
        await env.stop(True)

    asyncio.run(go())
    spans = _lines(tmp_path / sp.TRIAL_FILE)
    assert [(s["op"], s["src"], s["dst"]) for s in spans[:3]] == [
        ("upload_file", "/a/b", "/c"),
        ("upload_dir", "/d", "/e"),
        ("upload_file", "/f", "/g"),
    ]
    assert sp._depth.get() == 0


def test_spans_reach_the_file_only_when_the_sandbox_stops(tmp_path):
    env = sp.profiled(_Fake)(tmp_path)

    async def go():
        await env.exec("true")
        assert not (tmp_path / sp.TRIAL_FILE).exists()
        await env.stop(True)

    asyncio.run(go())
    assert [s["op"] for s in _lines(tmp_path / sp.TRIAL_FILE)] == ["exec", "stop"]


def test_module_attribute_wraps_the_configured_provider(monkeypatch):
    monkeypatch.setenv("HARBOR_ENV_IMPORT_PATH", f"{__name__}:_Fake")
    cls = sp.ProfiledEnvironment
    assert issubclass(cls, _Fake) and cls is sp.profiled(_Fake)


def test_note_session_keys_the_trial_by_port_and_session(tmp_path):
    sp.note_session(tmp_path, "http://10.0.0.1:30003/sessions/abc")
    assert _lines(tmp_path / sp.TRIAL_FILE)[0]["path"] == "/30003/sessions/abc"


def test_proxy_logs_each_forwarded_request_without_the_token(tmp_path):
    log = tmp_path / "proxy" / "proxy.jsonl"

    async def go():
        upstream = web.Application()

        async def ok(request):
            return web.json_response({})

        upstream.router.add_post("/{tail:.*}", ok)
        async with TestServer(upstream) as up:
            port = up.port
            app = make_app("127.0.0.1", range(up.port, up.port + 1), "secret-tok", str(log))
            async with TestServer(app) as proxy, ClientSession() as client:
                url = proxy.make_url(f"/secret-tok/{up.port}/sessions/s1/v1/chat/completions")
                async with client.post(url, json={}) as r:
                    assert r.status == 200
                async with client.get(proxy.make_url("/healthz")) as r:
                    assert r.status == 200
        return port

    port = asyncio.run(go())
    (line,) = _lines(log)  # healthz is not a forwarded request
    assert line["path"] == f"/{port}/sessions/s1/v1/chat/completions" and line["status"] == 200
    assert "secret-tok" not in log.read_text()


def test_proxy_without_a_log_has_no_middleware():
    app = make_app("127.0.0.1", range(30000, 30001), "tok")
    assert not app.middlewares


def test_recipe_switch_sets_the_env_only_when_on():
    base = R.from_dict({"taskset": {"harness": R.HARNESS_SANDBOX}})
    assert sp.ENV not in R.harbor_env(base, "smoke-x")
    on = R.from_dict({"taskset": {"harness": R.HARNESS_SANDBOX}, "obs": {"sandbox_profile": True}})
    assert R.harbor_env(on, "smoke-x")[sp.ENV] == "1"
