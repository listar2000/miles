"""SandboxMiniSweAgent selects ContextExitAgent and reports its exit status to Miles."""

import asyncio
import json
from types import SimpleNamespace

from harbor.models.agent.context import AgentContext

from miles_ext.harbor import agent_function as af
from miles_ext.harnesses.mini_swe_agent import context_exit
from miles_ext.harnesses.mini_swe_agent.sandbox import SandboxMiniSweAgent


def _agent(tmp_path):
    return SandboxMiniSweAgent(logs_dir=tmp_path, model_name="openai/m")


def test_cli_selects_the_context_exit_class(tmp_path):
    assert f"--agent-class={context_exit.AGENT_CLASS}" in _agent(tmp_path).build_cli_flags().split()


def test_exit_status_reaches_agent_metadata(tmp_path):
    (tmp_path / "mini-swe-agent.trajectory.json").write_text(
        json.dumps({"info": {"exit_status": context_exit.EXIT_STATUS}, "messages": []})
    )
    context = AgentContext()
    _agent(tmp_path).populate_context_post_run(context)
    assert context.metadata["mswea_exit_status"] == context_exit.EXIT_STATUS


def _run(monkeypatch, tmp_path, agent_status, key="mswea_exit_status"):
    """af.run_trial on a stand-in trial the verifier scored 1.0 and whose agent ended with agent_status."""

    class _Trial:
        paths = SimpleNamespace(trial_dir=tmp_path)

        @classmethod
        async def create(cls, config):
            return cls()

        async def run(self):
            return "result"

    monkeypatch.setattr("harbor.trial.trial.Trial", _Trial)
    monkeypatch.setattr(af, "build_trial_config", lambda *a: SimpleNamespace(task=SimpleNamespace(path=None)))
    monkeypatch.setattr(af.hf, "_ensure_provider_key", lambda: None)
    monkeypatch.setattr(
        af.hf,
        "trial_result_to_metadata",
        lambda r: {
            "reward": 1.0,
            "exit_status": "Submitted",
            "eval_report": {},
            "agent_metrics": {key: agent_status},
        },
    )
    return asyncio.run(af.run_trial("http://h:1/sessions/s", "p", {}, {"instance_id": "t"}))


def test_context_exhausted_overrides_submitted(monkeypatch, tmp_path):
    for agent_status in (context_exit.EXIT_STATUS, "Submitted"):
        out = _run(monkeypatch, tmp_path, agent_status)
        assert out["exit_status"] == agent_status and out["reward"] == 1.0


def test_context_exhausted_scores_zero_when_switched_on(monkeypatch, tmp_path):
    monkeypatch.setenv("MILES_EXT_CONTEXT_EXHAUSTED_SCORES_ZERO", "1")
    assert _run(monkeypatch, tmp_path, context_exit.EXIT_STATUS)["reward"] == 0.0
    assert _run(monkeypatch, tmp_path, "Submitted")["reward"] == 1.0


def test_deepseek_sandbox_reports_context_exhausted_too(monkeypatch, tmp_path):
    out = _run(monkeypatch, tmp_path, context_exit.EXIT_STATUS, key=af.deepseek_sandbox.AGENT_EXIT_STATUS)
    assert out["exit_status"] == context_exit.EXIT_STATUS
