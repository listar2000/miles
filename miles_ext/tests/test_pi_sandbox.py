"""pi-sandbox (ext-055): Pi inside the task sandbox, configured as the Mobius platform runs it."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.llms.base import SingleTurnMaxSeqLenExceededError
from harbor.models.agent.context import AgentContext

from miles_ext.harbor import agent_function as af
from miles_ext.harnesses.mini_swe_agent import context_exit
from miles_ext.harnesses.pi import sandbox as pi
from miles_ext.recipes import recipe as R

SESSION = "http://10.0.0.1:30000/sessions/s/v1"
PUBLIC = "https://abc123.r9.modal.host/tok"
CAP = 4096


def test_recipe_runs_inline_through_the_tunnel_with_its_window():
    recipe = R.from_dict(
        {
            "model": {"preset": "qwen3.5-9b", "backend": "megatron"},
            "tracking": {"wandb_project": "miles-test"},
            "taskset": {"harness": R.HARNESS_PI_SANDBOX},
        }
    )
    R.validate(recipe, local_inputs=False)
    argv = R.miles_args(recipe, "run1", resume=False)
    assert argv[argv.index("--custom-agent-function-mode") + 1] == "inline"
    env = R.harbor_env(recipe, "run1")
    assert env["AGENT_MAX_INPUT_TOKENS"] == str(recipe["algorithm"]["max_seq_len"])
    assert env["AGENT_MAX_OUTPUT_TOKENS"] == str(recipe["algorithm"]["max_response_len"])
    assert R.needs_tunnel(recipe)
    with pytest.raises(ValueError, match="compact_at_tokens"):  # Pi never compacts here
        R.validate(
            R.from_dict({**recipe, "agent": {**recipe["agent"], "compact_at_tokens": 65536}}), local_inputs=False
        )


def test_trial_config_loads_pi_and_points_it_at_the_tunnel(monkeypatch):
    def upstream_config(metadata, session_url, request_kwargs):
        binding = af.hf.HARNESS_BINDINGS[metadata["agent_name"]]
        agent = SimpleNamespace(
            name=metadata["agent_name"],
            import_path=None,
            extra_allowed_hosts=[],
            kwargs=binding.kwargs(session_url, "dummy", {}, "openai/model", None),
            env=binding.env(session_url, "dummy"),
        )
        return SimpleNamespace(agent=agent)

    monkeypatch.setattr(af.hf, "build_trial_config", upstream_config)
    monkeypatch.setenv("MILES_EXT_HARNESS", af.PI_SANDBOX_HARNESS)
    monkeypatch.setenv("MILES_SESSION_PUBLIC_URL", PUBLIC)
    config = af.build_trial_config({"instance_id": "x"}, SESSION, {})
    assert config.agent.import_path == "miles_ext.harnesses.pi.sandbox:SandboxPi"
    assert config.agent.extra_allowed_hosts == ["abc123.r9.modal.host"]
    # Pi appends /chat/completions
    assert config.agent.kwargs == {"base_url": f"{PUBLIC}/30000/sessions/s/v1", "strict_tools": False}


def _end(stop_reason, output=10, error=None):
    message = {
        "role": "assistant",
        "stopReason": stop_reason,
        "usage": {"input": 100, "cacheRead": 0, "output": output},
    }
    if error:
        message["errorMessage"] = error
    return {"type": "message_end", "message": message}


def _events(*events):
    return "\n".join(json.dumps(e) for e in events)


USER = {"type": "message_end", "message": {"role": "user", "content": []}}


@pytest.mark.parametrize(
    "last, expected",
    [
        (_end("stop"), None),
        # the server cut the reply to the room left in the window: grade the workspace
        (_end("length", output=430), context_exit.EXIT_STATUS),
        (
            _end(
                "error",
                error="400 The input (130000 tokens) is longer than the model's context length (131072 tokens).",
            ),
            context_exit.EXIT_STATUS,
        ),
        # one reply ran to the recipe's cap: SequenceLengthLimitExceeded, as for the other harnesses
        (_end("length", output=CAP), SingleTurnMaxSeqLenExceededError),
        (_end("error", error="500 upstream"), NonZeroAgentExitCodeError),
        (_end("aborted"), NonZeroAgentExitCodeError),
    ],
)
def test_last_reply_maps_to_an_exit(last, expected):
    result = pi.outcome(_events(USER, _end("toolUse"), last), CAP)
    assert result == expected if expected is None or isinstance(expected, str) else isinstance(result, expected)


def test_a_run_without_a_reply_fails():
    assert isinstance(pi.outcome("node: not found\n", CAP), NonZeroAgentExitCodeError)


class _Sandbox:
    def __init__(self, events):
        self.events, self.commands, self.uploads = events, [], {}

    async def upload_file(self, source_path, target_path):
        self.uploads[target_path] = open(source_path).read()

    async def exec(self, command, env=None, **kwargs):
        self.commands.append((command, env))
        stdout = self.events if command.startswith("cat ") else ""
        return SimpleNamespace(return_code=0, stdout=stdout, stderr="")


def _agent(tmp_path):
    return pi.SandboxPi(
        logs_dir=tmp_path,
        model_name="openai/model",
        base_url=f"{PUBLIC}/30000/sessions/s/v1",
        model_info={"max_input_tokens": 131072, "max_output_tokens": CAP},
    )


def test_run_is_a_plain_loop_without_compaction_on_the_session_server(tmp_path):
    sandbox, context = _Sandbox(_events(_end("toolUse"), _end("stop"))), AgentContext()
    asyncio.run(_agent(tmp_path).run("- solve it\n", sandbox, context))
    assert json.loads(sandbox.uploads[f"{pi.CONFIG_DIR}/settings.json"]) == {"compaction": {"enabled": False}}
    provider = json.loads(sandbox.uploads[f"{pi.CONFIG_DIR}/models.json"])["providers"][pi.PROVIDER]
    assert provider["baseUrl"] == f"{PUBLIC}/30000/sessions/s/v1"
    # Pi's defaults for an OpenAI-style server: plain tool schemas, no thinking parameters
    assert "compat" not in provider["models"][0] and "reasoning" not in provider["models"][0]
    assert (provider["models"][0]["contextWindow"], provider["models"][0]["maxTokens"]) == (131072, CAP)
    launch, env = next((c, e) for c, e in sandbox.commands if "pi --print" in c)
    for flag in pi.PLAIN_FLAGS.split():
        assert flag in launch
    assert "--tools read,write,edit,bash,grep,find,ls" in launch
    # argv as Mobius passes it: Pi trims a prompt read from stdin
    assert "-- '- solve it\n'" in launch  # a task starting with "-" stays the prompt
    assert env["PI_CODING_AGENT_DIR"] == pi.CONFIG_DIR and env["PI_OFFLINE"] == "1"
    assert (context.n_input_tokens, context.n_output_tokens) == (200, 20)


def test_a_full_window_ends_the_run_for_the_verifier(tmp_path):
    context = AgentContext()
    asyncio.run(_agent(tmp_path).run("x", _Sandbox(_events(_end("length", output=430))), context))
    assert context.metadata[pi.AGENT_EXIT_STATUS] == context_exit.EXIT_STATUS  # no raise: the verifier runs


def test_strict_tool_schemas_reach_pis_provider_compat(monkeypatch):
    base = {
        "model": {"preset": "qwen3.5-9b", "backend": "megatron"},
        "tracking": {"wandb_project": "miles-test"},
        "agent": {"strict_tool_schemas": True},
    }
    with pytest.raises(ValueError, match="strict_tool_schemas"):
        R.validate(R.from_dict({**base, "taskset": {"harness": R.HARNESS_SANDBOX}}), local_inputs=False)
    recipe = R.from_dict({**base, "taskset": {"harness": R.HARNESS_PI_SANDBOX}})
    R.validate(recipe, local_inputs=False)
    for key, value in R.harbor_env(recipe, "run1").items():
        monkeypatch.setenv(key, value)
    kwargs = af.hf.HARNESS_BINDINGS[af.PI_SANDBOX_HARNESS].kwargs(SESSION, "dummy", {}, "openai/model", None)
    models = pi.models_json(kwargs["base_url"], "model", 131072, CAP, kwargs["strict_tools"])
    assert models["providers"][pi.PROVIDER]["models"][0]["compat"] == {"supportsStrictMode": True}


def test_install_pins_pi_and_the_node_of_the_traces(tmp_path):
    sandbox = _Sandbox("")
    asyncio.run(_agent(tmp_path).install(sandbox))
    script = next(c for c, _ in sandbox.commands if "npm install" in c)
    assert f"{pi.PI_PACKAGE}@{pi.PI_VERSION}" in script and f"nvm install {pi.NODE_VERSION}" in script
