"""deepseek-sandbox (ext-046): DeepSeek's own agent (dsh) inside the task sandbox:

    python -m pytest miles_ext/tests/test_deepseek_sandbox.py
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.llms.base import SingleTurnMaxSeqLenExceededError

from miles_ext.harbor import agent_function as af
from miles_ext.harbor import context_exit
from miles_ext.harbor import deepseek_sandbox as ds
from miles_ext.recipes import recipe as R

SESSION = "http://10.0.0.1:30000/sessions/s/v1"
PUBLIC = "https://abc123.r9.modal.host/tok"
BASE = {"model": {"preset": "qwen3.5-9b", "backend": "megatron"}, "tracking": {"wandb_project": "miles-test"}}


def _recipe(**agent):
    return R.from_dict({**BASE, "taskset": {"harness": R.HARNESS_DEEPSEEK_SANDBOX}, "agent": agent})


def test_recipe_runs_inline_through_the_tunnel_with_its_window_and_compaction():
    recipe = _recipe(compact_at_tokens=24576)
    R.validate(recipe, local_inputs=False)
    argv = R.miles_args(recipe, "run1", resume=False)
    assert argv[argv.index("--use-session-server") + 1] == "v2"
    assert argv[argv.index("--custom-agent-function-mode") + 1] == "inline"
    env = R.harbor_env(recipe, "run1")
    assert env["MILES_EXT_COMPACT_AT_TOKENS"] == "24576"
    assert env["AGENT_MAX_INPUT_TOKENS"] == str(recipe["algorithm"]["max_seq_len"])
    assert env["AGENT_MAX_OUTPUT_TOKENS"] == str(recipe["algorithm"]["max_response_len"])
    assert "HARBOR_TERMINUS_2_LINEAR_HISTORY" not in env
    assert R.needs_tunnel(recipe)


def test_compaction_below_what_dsh_keeps_is_refused():
    with pytest.raises(ValueError, match="keeps verbatim"):
        R.validate(_recipe(compact_at_tokens=4000), local_inputs=False)


def test_compact_at_tokens_becomes_dsh_trigger():
    config = ds.compaction(262144, 32768, 131072)
    # trigger = min(W * thresholdRatio, W - O - headroomTokens)
    assert min(262144 * config["thresholdRatio"], 262144 - 32768 - config["headroomTokens"]) == 131072
    assert config["maxTokens"] == 32768
    with pytest.raises(ValueError, match="must be above"):
        ds.compaction(262144, 32768, 30000)


def _rows(patch):
    rows = {row["id"]: row for row in patch if "id" in row}
    inserted = {row["id"]: row for row in next(row["insert"] for row in patch if "insert" in row)}
    return rows, inserted


def test_patch_turns_off_what_miles_cannot_serve():
    rows, inserted = _rows(ds.patch("model", 32768, 4096, None))
    for front_door_or_upload in ("sdk-app-startup", "sdk-jsonrpc-server", "session-log-deepseek"):
        assert rows[front_door_or_upload]["disabled"] is True
    assert rows["tool-fs"]["config"]["readImage"] is False  # no image blocks on Miles' Messages route
    assert rows["compaction-basic"]["config"] == {"auto": False}  # v1 session server is append-only
    llm = rows["llm-deepseek"]["config"]
    assert (llm["defaultContextWindow"], llm["maxTokens"], llm["apiKeyEnv"]) == (32768, 4096, "DEEPSEEK_API_KEY")
    assert inserted["headless-runner"]["config"] == {"task": "-", "json": True}  # task on stdin
    assert inserted["agent-default-model"]["config"]["model"] == "model"
    rows, _ = _rows(ds.patch("model", 32768, 4096, 24576))
    assert rows["compaction-basic"]["config"]["headroomTokens"] == 32768 - 4096 - 24576


def test_binding_points_dsh_at_the_tunnel_session_root(monkeypatch):
    monkeypatch.setenv("MILES_SESSION_PUBLIC_URL", PUBLIC)
    monkeypatch.setenv(af.COMPACT_ENV, "131072")
    binding = af.hf.HARNESS_BINDINGS[af.DEEPSEEK_SANDBOX_HARNESS]
    url = af.public_session_url(SESSION)
    assert binding.env(url, "dummy") == {
        "DEEPSEEK_BASE_URL": f"{PUBLIC}/30000/sessions/s",  # dsh appends /v1/messages
        "DEEPSEEK_API_KEY": "dummy",
    }
    assert binding.kwargs(url, "dummy", {}, "openai/model", 4096) == {"compact_at_tokens": 131072}


def test_agent_phase_allows_the_session_tunnel_host(monkeypatch):
    def upstream_config(metadata, session_url, request_kwargs):
        agent = SimpleNamespace(
            name=metadata["agent_name"], import_path=None, extra_allowed_hosts=[], kwargs={}, env={}
        )
        return SimpleNamespace(agent=agent)

    monkeypatch.setattr(af.hf, "build_trial_config", upstream_config)
    monkeypatch.setenv("MILES_EXT_HARNESS", af.DEEPSEEK_SANDBOX_HARNESS)
    monkeypatch.setenv("MILES_SESSION_PUBLIC_URL", PUBLIC)
    config = af.build_trial_config({"instance_id": "x"}, SESSION, {})
    assert config.agent.extra_allowed_hosts == ["abc123.r9.modal.host"]
    assert config.agent.import_path == "miles_ext.harbor.deepseek_sandbox:SandboxDeepSeek"


def _events(*events):
    return "\n".join(json.dumps(e) for e in events)


def _turn_end(reason):
    return {"type": "status", "phase": "turn_end", "turn": 1, "reason": reason}


STEP = {"type": "status", "phase": "step_end", "usage": {"inputTokens": 10, "outputTokens": 3}}
CAP = 4096


def _step(output):
    return {"type": "status", "phase": "step_end", "usage": {"inputTokens": 261712, "outputTokens": output}}


@pytest.mark.parametrize(
    "steps, reason, expected",
    [
        ([STEP], {"kind": "completed"}, None),
        # the server cut the last reply to the room left in the window: grade the workspace
        ([STEP, _step(430)], {"kind": "max-tokens"}, context_exit.EXIT_STATUS),
        ([STEP], {"kind": "error", "error": {"code": "CONTEXT_WINDOW_EXCEEDED"}}, context_exit.EXIT_STATUS),
        # one reply ran to the recipe's cap: SequenceLengthLimitExceeded, as for terminus
        ([STEP, _step(CAP)], {"kind": "max-tokens"}, SingleTurnMaxSeqLenExceededError),
        ([STEP], {"kind": "error", "error": {"code": "INVALID_REQUEST", "message": "bad"}}, NonZeroAgentExitCodeError),
    ],
)
def test_turn_end_maps_to_an_exit(steps, reason, expected):
    result = ds.outcome(_events(*steps, _turn_end(reason)), CAP)
    assert result == expected if expected is None or isinstance(expected, str) else isinstance(result, expected)


def test_a_run_without_events_fails():
    assert isinstance(ds.outcome("dsh: crashed\n", CAP), NonZeroAgentExitCodeError)


class _Sandbox:
    def __init__(self, events):
        self.events, self.commands, self.uploads = events, [], {}

    async def upload_file(self, source_path, target_path):
        self.uploads[target_path] = open(source_path).read()

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        stdout = self.events if command.startswith("cat ") else ""
        return SimpleNamespace(return_code=0, stdout=stdout, stderr="")


def _agent(tmp_path):
    return ds.SandboxDeepSeek(
        logs_dir=tmp_path, model_name="openai/model", model_info={"max_input_tokens": 32768, "max_output_tokens": CAP}
    )


def test_run_feeds_the_task_on_stdin_and_reports_usage(tmp_path):
    from harbor.models.agent.context import AgentContext

    agent = _agent(tmp_path)
    sandbox, context = _Sandbox(_events(STEP, STEP, _turn_end({"kind": "completed"}))), AgentContext()
    asyncio.run(agent.run("pkill me", sandbox, context))
    assert sandbox.uploads[f"{ds.FILES}/task.md"] == "pkill me"
    launch = next(c for c in sandbox.commands if ds.BIN in c)
    assert "pkill me" not in launch and f"< {ds.FILES}/task.md" in launch
    assert json.loads(sandbox.uploads[f"{ds.FILES}/patch.json"]) == agent._patch
    assert (context.n_input_tokens, context.n_output_tokens) == (20, 6)


def test_a_full_window_ends_the_run_for_the_verifier(tmp_path):
    from harbor.models.agent.context import AgentContext

    context = AgentContext()
    asyncio.run(_agent(tmp_path).run("x", _Sandbox(_events(_step(430), _turn_end({"kind": "max-tokens"}))), context))
    assert context.metadata[ds.AGENT_EXIT_STATUS] == context_exit.EXIT_STATUS  # no raise: the verifier runs
