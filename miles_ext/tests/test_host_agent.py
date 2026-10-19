"""HostMiniSweAgent matches the in-sandbox harness: window exit, output cap, step timings, start directory, uname."""

import asyncio
import json
from types import SimpleNamespace

import minisweagent.models.litellm_model as litellm_model
from harbor.agents.installed import mini_swe_agent as stock
from harbor.models.agent.context import AgentContext
from minisweagent.models.test_models import DeterministicModel, make_output

from miles_ext.harbor import context_exit
from miles_ext.harbor.host_agent import HostMiniSweAgent, cap_output


class _Sandbox:
    def __init__(self):
        self.calls = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None):
        self.calls.append((command, cwd))
        if command.startswith("uname"):
            return SimpleNamespace(stdout="Linux\n6.1.0-test\n#1 SMP\nx86_64\n", stderr="", return_code=0)
        out = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n" if "COMPLETE_TASK" in command else "ok\n"
        return SimpleNamespace(stdout=out, stderr="", return_code=0)


def _reply(command, prompt_tokens=10):
    out = make_output("thinking", [{"command": command}])
    out["extra"]["response"] = {"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 5}}
    return out


def _run(monkeypatch, tmp_path, replies, **kwargs):
    model = DeterministicModel(outputs=replies)
    monkeypatch.setattr(litellm_model, "LitellmModel", lambda **kw: model)
    agent = HostMiniSweAgent(logs_dir=tmp_path, model_name="openai/m", api_base="http://h/v1", **kwargs)
    sandbox, context = _Sandbox(), AgentContext()
    asyncio.run(agent.run("fix it", sandbox, context))
    return sandbox, context


def test_submit_records_timings_and_starts_where_the_sandbox_starts(monkeypatch, tmp_path):
    shell = {"cwd": "", "env": {}, "timeout": 60}
    sandbox, context = _run(
        monkeypatch, tmp_path, [_reply("ls"), _reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")], shell=shell
    )
    assert [cwd for command, cwd in sandbox.calls if not command.startswith("uname")] == [None, None]
    trajectory = json.loads((tmp_path / "mini-swe-agent.trajectory.json").read_text())
    assert "Linux 6.1.0-test #1 SMP x86_64" in trajectory["messages"][1]["content"]
    assert context.metadata["mswea_exit_status"] == "Submitted"
    assert len(context.metadata["api_request_times_msec"]) == 2
    assert len(context.metadata["command_exec_times_msec"]) == 1


def test_full_window_ends_context_exhausted(monkeypatch, tmp_path):
    _, context = _run(monkeypatch, tmp_path, [_reply("ls", prompt_tokens=9000), _reply("ls")], max_seq_len=10000)
    assert context.metadata["mswea_exit_status"] == context_exit.EXIT_STATUS


def test_output_cap_matches_harbor():
    cap = stock.MAX_COMMAND_OUTPUT_BYTES
    text = "a" * cap + "b" * 10 + "z" * cap
    capped = cap_output(text)
    assert capped.startswith("a" * (cap // 2)) and capped.endswith("z" * (cap // 2))
    assert f"[harbor: elided {len(text) - cap} chars]" in capped
    assert cap_output("short") == "short"
