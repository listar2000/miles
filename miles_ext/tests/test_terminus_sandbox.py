"""terminus-2-sandbox (ext-045): Harbor's Terminus-2 inside the task sandbox, with absolute-threshold compaction:

    python -m pytest miles_ext/tests/test_terminus_sandbox.py
"""

import asyncio
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.llms.base import SingleTurnMaxSeqLenExceededError

from miles_ext.harbor import agent_function as af
from miles_ext.harnesses.terminus import runner as terminus_runner
from miles_ext.harnesses.terminus import sandbox as terminus_sandbox
from miles_ext.harnesses.terminus.sandbox import SandboxTerminus2
from miles_ext.recipes import recipe as R

SESSION = "http://10.0.0.1:30000/sessions/s/v1"
PUBLIC = "https://abc123.r9.modal.host/tok"


BASE = {"model": {"preset": "qwen3.5-9b"}, "tracking": {"wandb_project": "miles-test"}}


MEGATRON = {"preset": "qwen3.5-9b", "backend": "megatron"}


def _recipe(**agent):
    model = MEGATRON if agent else BASE["model"]
    return R.from_dict({**BASE, "model": model, "taskset": {"harness": R.HARNESS_TERMINUS_SANDBOX}, "agent": agent})


def test_compaction_switches_to_session_server_v2_and_sets_the_threshold():
    recipe = _recipe(compact_at_tokens=24576)
    R.validate(recipe, local_inputs=False)
    argv = R.miles_args(recipe, "run1", resume=False)
    assert argv[argv.index("--use-session-server") + 1] == "v2"
    env = R.harbor_env(recipe, "run1")
    assert env["MILES_EXT_COMPACT_AT_TOKENS"] == "24576"
    assert env["AGENT_MAX_INPUT_TOKENS"] == str(recipe["algorithm"]["max_seq_len"])
    assert env["HARBOR_TERMINUS_2_LINEAR_HISTORY"] == "true"
    assert R.needs_tunnel(recipe)


def test_without_compaction_the_run_keeps_session_server_v1():
    recipe = _recipe()
    argv = R.miles_args(recipe, "run1", resume=False)
    assert argv[argv.index("--use-session-server") + 1].startswith("--")
    assert "MILES_EXT_COMPACT_AT_TOKENS" not in R.harbor_env(recipe, "run1")


@pytest.mark.parametrize(
    "spec, error",
    [
        ({"agent": {"compact_at_tokens": 1000}, "taskset": {"harness": R.HARNESS_SANDBOX}}, "needs taskset.harness"),
        ({"agent": {"compact_at_tokens": 24576}, "taskset": {"harness": R.HARNESS_TERMINUS_SANDBOX}}, "megatron"),
        (
            {
                "model": MEGATRON,
                "agent": {"compact_at_tokens": 30000},
                "taskset": {"harness": R.HARNESS_TERMINUS_SANDBOX},
            },
            "must fit",
        ),
    ],
)
def test_compaction_is_refused_where_it_cannot_work(spec, error):
    recipe = R.from_dict({**BASE, **spec})
    with pytest.raises(ValueError, match=error):
        R.validate(recipe, local_inputs=False)


def _agent(tmp_path, **kwargs):
    return SandboxTerminus2(
        logs_dir=tmp_path,
        model_name="openai/model",
        terminus={"parser_name": "xml"},
        model_info={"max_input_tokens": 262144},
        **kwargs,
    )


def _runner_kwargs(agent):
    """The Terminus2 kwargs ``run`` hands the in-sandbox runner."""
    sandbox = _Sandbox(json.dumps({"context": {}}))
    asyncio.run(agent.run("solve it", sandbox, _context()))
    return sandbox.uploads[terminus_sandbox.RUN_CONFIG]["kwargs"]


def test_compact_at_tokens_becomes_terminus_free_token_threshold(tmp_path):
    kwargs = _runner_kwargs(_agent(tmp_path, compact_at_tokens=131072))
    assert kwargs["enable_summarize"] is True
    assert kwargs["proactive_summarization_threshold"] == 262144 - 131072
    assert kwargs["model_info"]["max_input_tokens"] == 262144
    assert "enable_summarize" not in _runner_kwargs(_agent(tmp_path))
    with pytest.raises(ValueError, match="below the context limit"):
        _agent(tmp_path, compact_at_tokens=262144)


def test_terminus_kwargs_point_at_the_tunnel(monkeypatch):
    monkeypatch.setenv("MILES_SESSION_PUBLIC_URL", PUBLIC)
    monkeypatch.setenv(af.COMPACT_ENV, "131072")
    config = af.hf.HARNESS_BINDINGS[af.TERMINUS_SANDBOX_HARNESS].kwargs(
        af.public_session_url(SESSION), "dummy", {"temperature": 1.0}, "openai/model", 4096
    )
    terminus = config["terminus"]
    assert terminus["api_base"] == f"{PUBLIC}/30000/sessions/s/v1"
    assert terminus["response_length_exceeded_policy"] == "abort"  # a truncated reply must not desync TITO
    assert terminus["llm_call_kwargs"]["timeout"] == af.hf._trial_timeout_s()
    assert config["compact_at_tokens"] == 131072


def test_agent_phase_allows_the_session_tunnel_host(monkeypatch):
    def upstream_config(metadata, session_url, request_kwargs):
        return SimpleNamespace(
            agent=SimpleNamespace(
                name=metadata["agent_name"],
                import_path=None,
                override_setup_timeout_sec=None,
                extra_allowed_hosts=[],
                kwargs={"max_seq_len": 4096},
                env={},
                api_base=session_url,
            )
        )

    monkeypatch.setattr(af.hf, "build_trial_config", upstream_config)
    monkeypatch.setenv("MILES_EXT_HARNESS", af.TERMINUS_SANDBOX_HARNESS)
    monkeypatch.setenv("MILES_SESSION_PUBLIC_URL", PUBLIC)
    config = af.build_trial_config({"instance_id": "x"}, SESSION, {})
    assert config.agent.extra_allowed_hosts == ["abc123.r9.modal.host"]
    assert config.agent.import_path == "miles_ext.harnesses.terminus.sandbox:SandboxTerminus2"
    assert config.agent.env == {}  # mini-swe-agent's ContextExitAgent window is not set


def test_bundle_carries_harbor_with_its_metadata_and_the_runner():
    with tarfile.open(fileobj=io.BytesIO(terminus_sandbox.bundle())) as tar:
        names = tar.getnames()
    assert "harbor/agents/terminus_2/terminus_2.py" in names
    assert any(n.endswith(".dist-info/METADATA") for n in names)
    assert Path(terminus_sandbox.RUNNER).name in names


class _Sandbox:
    session_id = "trial__env"

    def __init__(self, result):
        self.result = result
        self.commands = []
        self.uploads = {}

    async def upload_file(self, source_path, target_path):
        if target_path.endswith(".json"):
            self.uploads[target_path] = json.loads(open(source_path).read())

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        if command.startswith("cat "):
            return SimpleNamespace(return_code=0 if self.result else 1, stdout=self.result, stderr="")
        return SimpleNamespace(return_code=0, stdout="", stderr="")


def _context():
    from harbor.models.agent.context import AgentContext

    return AgentContext()


def _run(tmp_path, result, context=None):
    context = context or _context()
    asyncio.run(_agent(tmp_path).run("solve it", _Sandbox(json.dumps(result) if result else ""), context))
    return context


def test_run_reports_the_sandbox_context(tmp_path):
    context = _run(tmp_path, {"context": {"n_input_tokens": 10, "metadata": {"summarization_count": 2}}})
    assert context.n_input_tokens == 10 and context.metadata["summarization_count"] == 2


def test_run_raises_the_sandbox_exception_under_its_own_name(tmp_path):
    context = _context()
    with pytest.raises(SingleTurnMaxSeqLenExceededError):  # mapped to SequenceLengthLimitExceeded upstream
        _run(
            tmp_path,
            {
                "context": {"n_output_tokens": 5},
                "exception": {"type": "SingleTurnMaxSeqLenExceededError", "message": "too long"},
            },
            context,
        )
    assert context.n_output_tokens == 5
    with pytest.raises(RuntimeError) as e:
        _run(tmp_path, {"context": {}, "exception": {"type": "ContextWindowFull", "message": "x"}})
    assert type(e.value).__name__ == "ContextWindowFull"


def test_a_runner_that_died_fails_the_agent(tmp_path):
    with pytest.raises(RuntimeError, match="without a result"):
        _run(tmp_path, None)


def test_local_environment_runs_commands_with_env_and_timeout():
    env = terminus_runner.LocalEnvironment("s")
    ok = asyncio.run(env.exec("echo $X; exit 3", env={"X": "hi"}))
    assert (ok.stdout, ok.return_code) == ("hi\n", 3)
    assert asyncio.run(env.exec("sleep 5", timeout_sec=0.2)).return_code == 124
