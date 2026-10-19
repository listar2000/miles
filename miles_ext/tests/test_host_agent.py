"""HostMiniSweAgent matches the in-sandbox harness: window exit, output cap, step timings, start directory, uname."""

import asyncio
import json
from types import SimpleNamespace

import httpx
from harbor.models.agent.context import AgentContext
from miles_ext.harbor import context_exit, host_agent
from miles_ext.harbor.host_agent import HostMiniSweAgent, cap_output
from minisweagent.models.test_models import DeterministicModel, make_output


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
    monkeypatch.setattr(host_agent, "_openai_client_model_class", lambda: lambda **kw: model)
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


def test_output_cap_keeps_head_and_tail():
    assert cap_output("abcdefgh", cap=8) == "abcdefgh"
    assert cap_output("abcdefghij", cap=8) == "abcd\n...[harbor: elided 2 chars]...\nghij"


def test_openai_client_model_sends_what_litellm_sends(monkeypatch):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        call = {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
        message = {"role": "assistant", "content": "look", "tool_calls": [call]}
        return httpx.Response(
            200,
            json={
                "id": "r",
                "object": "chat.completion",
                "created": 0,
                "model": "model",
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    monkeypatch.setattr(host_agent, "_http_client", httpx.Client(transport=httpx.MockTransport(handler)))
    model = host_agent._openai_client_model_class()(
        model_name="openai/model",
        cost_tracking="ignore_errors",
        model_kwargs={
            "api_base": "http://h/sessions/s/v1",
            "api_key": "k",
            "drop_params": True,
            "temperature": 1.0,
            "top_k": 20,
            "parallel_tool_calls": False,
        },
    )
    message = model.query([{"role": "user", "content": "fix it"}])
    assert message["extra"]["actions"] == [{"command": "ls", "tool_call_id": "c1"}]
    assert message["extra"]["response"]["usage"]["prompt_tokens"] == 3
    # litellm 1.106 (openai/ provider, drop_params) sends top_k as a top-level body field too
    body = sent[0]
    assert body["model"] == "model" and body["temperature"] == 1.0 and body["parallel_tool_calls"] is False
    assert body["top_k"] == 20 and "drop_params" not in body
