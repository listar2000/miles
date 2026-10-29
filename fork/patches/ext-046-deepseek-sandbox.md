# ext-046: DeepSeek's own agent (dsh) inside the task sandbox

## Why
DeepSeek-V4.1 drops Terminus-2's opening `<response>` tag in about a third of turns, so
terminus-2-sandbox fails on it, and mini-swe-agent filled the 262k context on almost every
ProgramBench trial. Model-specific patches to Terminus-2 are out of scope for the fork; the
reference for a DeepSeek harness is prime-rl's `verifiers/v1/harnesses/deepseek`
(`PrimeIntellect-ai/verifiers@60290d43`), which runs dsh in the sandbox.

## Decision
- **`taskset.harness: deepseek-sandbox`** (`miles_ext/harbor/deepseek_sandbox.py`) runs the pinned
  DeepSeek Harness CLI, `@deepseek-ai/dsh@0.2.1-rc.3` (the release verifiers pins), unchanged.
  Like the other harnesses, only its own version is pinned; its npm dependencies resolve at install.
  - Rejected: a checked-in lockfile (exact installs, but ~10k lines to regenerate on every bump, and
    no other harness here or in verifiers locks its dependencies).
  Like the other in-sandbox harnesses, only model calls leave the sandbox (session tunnel,
  ext-009) and setup gets public network (`sandbox_agent.PublicSetup`).
- **The same composition as verifiers**: dsh's `sdk-minimal` profile (persistent bash, a
  working-directory tool, JSONL session, `compaction-basic`), with its SDK front door and
  DeepSeek's telemetry rows disabled. Verifiers puts its ACP bridge in the front door's place;
  we put dsh's own one-shot runner there (the `headless-runner` row of dsh's `headless`
  profile, task on stdin, `--json` events). Everything goes through one `--patch` file.
  - Rejected: ACP like verifiers (needs an ACP client and its Python deps in every sandbox for a
    one-prompt run); dsh's shipped `headless` profile (it sits on `dsh-base`: web, subagent and
    plugin rows we would then have to turn off one by one).
- **Off, because Miles cannot serve it**: `read_image` (the Messages route takes no images).
  A small `--import` shim, the counterpart of verifiers' `usage.mjs`, drops the request fields
  Miles' Anthropic adapter rejects: `thinking` and `output_config` (dsh always sends them; the
  policy's chat template decides reasoning) and `tool_result.is_error`. Sampling comes from the
  session server's defaults (dsh sends none), as for claude-code.
  The shim also sends those requests with `node:http(s)`, untimed: the session server answers only
  once the whole reply is generated, and fetch's 300 s headers timeout cut long turns under load
  (`TRANSPORT`; dsh's 5 retries redo the whole turn). The trial's agent timeout bounds the wait.
- **Window and reply cap from the recipe**: `AGENT_MAX_INPUT_TOKENS = max_seq_len` is dsh's
  context window, `AGENT_MAX_OUTPUT_TOKENS = max_response_len` its `max_tokens`.
- **`agent.compact_at_tokens`** (default None) sets `compaction-basic`'s trigger
  `min(W * thresholdRatio, W - O - headroomTokens)` to the value (`thresholdRatio 1`,
  `headroomTokens W - O - compact_at`), and moves the run to session server v2 as for ext-045.
  Unset, dsh's automatic compaction is off (`auto: false`): its default trigger is 80% of the
  window, and v1 rejects the rewritten history. dsh counts with Miles' reported usage where
  the request matches, else a heuristic. Validation adds: above the 16% of `W - O` it keeps verbatim.
- **Exit status**: from the last `turn_end` in the events. A full context window (the server cut the
  last reply short of `max_tokens` to the room left, or refused the request) ends the run like
  mini-swe-agent's ContextExhausted: no exception, so the verifier grades the workspace, and the
  agent function keeps `ContextExhausted` and applies `agent.context_exhausted_scores_zero`. A reply that hits `max_tokens` itself
  (dsh ends the run, nothing is re-sent) raises `SingleTurnMaxSeqLenExceededError`
  (SequenceLengthLimitExceeded); anything else is AgentError.
  dsh has no step cap; `agent.max_steps` does not apply, `agent.timeout` does.

## How it is applied
`miles_ext/harbor/deepseek_sandbox.py`, `agent_function.py` (binding: `DEEPSEEK_BASE_URL` = the
session root, dsh appends `/v1/messages`), `miles_ext/recipes/recipe.py` (harness,
`COMPACTING_HARNESSES`, validation); recipe `qwen35-9b-swegym8-deepseek.yaml` and its golden;
tests in `miles_ext/tests/test_deepseek_sandbox.py`. Install: curl as root, then Node 22 from
Harbor's nvm snippet (unless the image has Node >= 22.19) and, as the agent user,
`npm install --ignore-scripts` of the pinned version into `~/.miles-deepseek` (about 75 s per sandbox
in the GPU smoke). A `.ready-<version>` marker lets an install retry skip a finished install, and
another pin installs afresh. Each trial leaves `deepseek-sandbox.jsonl` (events) and
`deepseek-sandbox.stderr.txt` in its agent log dir.
Verify with a smoke run: trials reach `Submitted`, the events end in `turn_end` `completed`, and
with compaction W&B shows `rollout/num_training_samples` above `rollout_batch_size * n_samples`.
On a dsh bump (`DSH_VERSION`): a patched row's `config` replaces the bundle's whole config, so recheck the
restated fields of `llm-deepseek` and `tool-fs` against `dsh-sdk-minimal/cordis.patch.yml`, and
the request fields the shim drops against Miles' adapter.

## Remove when
Harbor ships a DeepSeek installed agent that fits the session server; upstream status: not filed.
