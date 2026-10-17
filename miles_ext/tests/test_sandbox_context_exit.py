"""SandboxMiniSweAgent selects ContextExitAgent and reports its exit status to Miles."""

import asyncio
import json
from types import SimpleNamespace

from harbor.models.agent.context import AgentContext

from miles_ext.harbor import agent_function as af
from miles_ext.harbor import context_exit
from miles_ext.harbor.sandbox_agent import SandboxMiniSweAgent


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


def test_context_exhausted_overrides_submitted(monkeypatch, tmp_path):
    class _Trial:
        paths = SimpleNamespace(trial_dir=tmp_path)

        @classmethod
        async def create(cls, config):
            return cls()

        async def run(self):
            return "result"

    monkeypatch.setattr("harbor.trial.trial.Trial", _Trial)
    monkeypatch.setattr(af, "build_trial_config", lambda *a: None)
    monkeypatch.setattr(af.hf, "_ensure_provider_key", lambda: None)
    for agent_status, expected in [(context_exit.EXIT_STATUS, context_exit.EXIT_STATUS), ("Submitted", "Submitted")]:
        monkeypatch.setattr(
            af.hf,
            "trial_result_to_metadata",
            lambda r, s=agent_status: {
                "reward": 0.0,
                "exit_status": "Submitted",
                "eval_report": {},
                "agent_metrics": {"mswea_exit_status": s},
            },
        )
        out = asyncio.run(af.run("http://h:1/sessions/s", "p", {}, {"instance_id": "t"}))
        assert out["exit_status"] == expected
