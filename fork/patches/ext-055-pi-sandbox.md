# ext-055: Pi inside the task sandbox, as the Mobius platform runs it

## Why
SFT data comes from GLM-5.3-Flash traces the Mobius platform recorded with the Pi agent, and the students are
evaluated (and later trained with RL) in Miles. That needs Pi in Miles with the same setup. The pinned Harbor
(0.20) has a Pi agent, but it installs Pi under its old npm name (`@mariozechner/pi-coding-agent`, renamed at
0.74, so no 1.5.0), only reaches Pi's built-in providers (no custom base URL for the session server), and has
no control over tools or compaction. Harbor 0.24's Pi agent, which Mobius builds on, imports modules 0.20 lacks,
so it cannot be vendored verbatim.

## Decision
- **`taskset.harness: pi-sandbox`** (`miles_ext/harnesses/pi/sandbox.py`) runs Pi
  `@earendil-works/pi-coding-agent@1.5.0`, unpatched, with Mobius' launch (mobius-platform `staging`,
  `programs/harbor-eval/patches/harness/pi.py`, PR #3190): `--no-extensions --no-skills --no-prompt-templates
  --no-context-files --no-mcp --no-approve` (nothing from the task, image or home: no AGENTS.md, skills, MCP),
  `--tools read,write,edit,bash,grep,find,ls`, `settings.json` `{"compaction": {"enabled": false}}` in
  `PI_CODING_AGENT_DIR` (stops threshold and overflow compaction), `PI_OFFLINE=1`, `PI_TELEMETRY=0`. The
  session is still saved under the trial's agent logs; no turn cap (Mobius sets none); `agent.timeout` applies.
- **The model** is a custom `openai-completions` provider in Pi's `models.json` pointing at the session's
  OpenAI root through the tunnel (ext-009), with the recipe's window and reply cap. Pi's own compat defaults for
  such a server stay: no thinking parameters (the chat template decides reasoning) and plain tool schemas, unless:
  - **`agent.strict_tool_schemas`** (default false) turns on Pi's `compat.supportsStrictMode`, which Pi's
    Z.ai entry sets: every tool argument becomes required (optional ones nullable) and `strict: true` is sent.
    An eval of a model trained on traces recorded that way (the GLM-5.3-Flash Pi traces) sets it, so the tool
    definitions match the training data; the harness default stays provider-neutral.
- **Same prompt text as the traces**: Pi's system prompt names its install path, so Node is pinned to the
  22.23.4 the traces ran on and Pi is installed globally under nvm, as Harbor's Pi does. The task goes on argv
  as Harbor and Mobius pass it, since Pi trims a prompt read from stdin; a task command `pkill -f <task text>`
  can therefore match the agent, as on Mobius.
- Not carried from Mobius: its extension (a Z.ai UTF-8 repair and a request-context log for its ATIF) and its
  ATIF conversion; Miles records tokens in the session server.
- **Exit status** from Pi's last reply (`--mode json` events): `stop` completes; a `length` stop shorter than
  the reply cap, or a 400 naming the context length, is ContextExhausted (the verifier grades the workspace,
  `agent.context_exhausted_scores_zero` applies); a reply at the cap raises `SingleTurnMaxSeqLenExceededError`;
  anything else is AgentError. Shared with deepseek-sandbox: `harnesses.common` now holds the Node bootstrap,
  the install retries and the exit-status key.

## How it is applied
`miles_ext/harnesses/pi/sandbox.py`; `agent_function.py` (binding: `base_url` = the tunnelled session URL);
`recipe.py` (`HARNESS_PI_SANDBOX`, `agent.strict_tool_schemas`, `WINDOWED_HARNESSES` for `AGENT_MAX_INPUT_TOKENS`/`AGENT_MAX_OUTPUT_TOKENS`);
tests in `miles_ext/tests/test_pi_sandbox.py`. Each trial leaves `pi.jsonl` (events without `message_update`),
`pi.stderr.txt` and `pi/sessions/` in its agent log dir. Verify with a smoke run: trials reach `Submitted`, the
first request's system prompt and tools equal a Mobius Pi trace's apart from the cwd. On a Pi bump: recheck the
flags against Mobius' `MobiusPi`, and the system prompt against new traces.

## Remove when
The pinned Harbor's Pi agent takes a custom endpoint and Pi >= 0.74 (Harbor 0.24 does), and Mobius' launch
can be expressed as its kwargs; upstream status: not filed.
