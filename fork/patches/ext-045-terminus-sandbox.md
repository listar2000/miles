# ext-045: Terminus-2 inside the task sandbox, with compaction at a token threshold

## Why
Harbor's Terminus-2 is the harness Terminal-Bench style tasks are tuned for, and it has built-in context
compaction (summarization). At our Harbor pin (`917b3c1b`, `harbor-miles-v0.20.0`) it only runs on the
host: the loop lives in the rollout worker and drives a tmux session in the sandbox with several
`environment.exec` calls per turn (send keys, wait, capture pane). On Modal and Sail each exec is a
network round trip, and the fork's recipes allowed only the two mini-swe-agent harnesses and single-turn.

Compaction is also the planned answer to long trajectories hitting the context wall (most of a
DeepSeek-V4.1 ProgramBench eval did). Terminus-2's own trigger is relative ("fewer than
`proactive_summarization_threshold` = 8k tokens free"); with a 256k window we want an absolute one
(compact at e.g. 128k) and room to spare.

## Decision
- **`taskset.harness: terminus-2-sandbox`** (`miles_ext/harbor/terminus_sandbox.py`). Like
  mini-swe-agent-sandbox, the whole agent runs in the sandbox and only model calls leave it, through the
  session tunnel and the agent-phase allowlist (ext-009). Setup runs with public network
  (`sandbox_agent.PublicSetup`, shared with mini-swe-agent-sandbox), so it works on no-network tasks.
- **The unchanged `Terminus2` class runs there.** `terminus_runner.py` gives it `LocalEnvironment`, which
  runs its commands as local processes. The rollout image's own Harbor (`harbor` package and dist-info)
  is copied in as a tarball, so both sides run the same commit, and a uv venv at `/opt/miles-terminus` gets
  the image's Python minor version and Terminus-2's import-time dependencies pinned to the image's versions
  (`RUNTIME_DEPS`). The venv install is skipped when it already imports them (bake it into a task image to
  save the download). tmux comes from Terminus-2's own installer (`TmuxSession._attempt_tmux_installation`).
  - Rejected: installing Harbor from git in each sandbox (needs git, and pulls Harbor's full dependency
    set: supabase, polars, boto3, fastapi...); writing our own Terminus clone (drifts from Harbor's prompts,
    parser and compaction).
- **Same kwargs as host terminus-2.** The binding is upstream's `terminus-2` kwargs (xml parser, abort on a
  truncated reply, `HARBOR_TERMINUS_2_*`), plus the model-call timeout the mini-swe-agent harnesses use and
  no asciinema recording (it needs pip at run time).
- **Errors keep their class name.** The runner writes the run's `AgentContext` and exception to
  `/logs/agent/terminus-2-sandbox.result.json`; the host raises an exception of the same class name
  (Harbor's own class when it has one), so upstream's mapping (`SingleTurnMaxSeqLenExceededError` ->
  `SequenceLengthLimitExceeded`, the rest -> `AgentError`) works as for host terminus-2.
- **Cut-off replies.** At our pin, Terminus-2 returns a reply salvaged from one cut off at `max_tokens` as
  a bare `str` where its caller reads `.content`, so the trial ends as `AgentError`; the runner wraps it in
  an `LLMResponse`. `agent.response_length_policy` (default None: the binding's `abort`) sets
  `HARBOR_RESPONSE_LENGTH_POLICY`; `regenerate` re-asks for a shorter reply twice before aborting. A
  DeepSeek-V4.1 ProgramBench run (thinking on, 32k replies) lost 23 of 200 trials in 20 minutes to these
  two paths.
- **`agent.compact_at_tokens`** (default None: off). Set, Terminus-2 summarizes once its context holds
  that many tokens (`proactive_summarization_threshold = max_input_tokens - compact_at_tokens`), with
  `AGENT_MAX_INPUT_TOKENS = algorithm.max_seq_len`. One key derives the rest: summarization on, linear
  history on, and `--use-session-server v2`, which records a compacted episode as several linear
  segments and trains each as its own sample with the episode's reward (upstream
  `examples/experimental/terminus-compaction`). The v1 server is append-only and would reject the
  post-summary history. Validation: only with this harness, only with `model.backend: megatron`, and
  `compact_at_tokens + max_response_len <= max_seq_len`. Megatron because a compacted step carries more
  samples than `rollout_batch_size * n_samples`, which only its precomputed DP schedule takes; the FSDP
  trainer asserts at `rollout.py`'s micro-batch check on the first step (smoke run smoke-t2-pr119-2).

## How it is applied
`miles_ext/harbor/terminus_sandbox.py`, `terminus_runner.py`, `agent_function.py` (binding, tunnel),
`sandbox_agent.py` (`PublicSetup`), `miles_ext/recipes/recipe.py` (harness, key, env, v2 flag); recipe
`qwen35-9b-swegym8-terminus.yaml` and its golden; tests in `miles_ext/tests/test_terminus_sandbox.py`.
Verify with that recipe's smoke run: trials reach `Submitted`, `terminus-2-sandbox.result.json` is in
each trial's agent dir, and with compaction W&B shows `rollout/num_training_samples` above
`rollout_batch_size * n_samples` once episodes compact.
Terminus-2 counts tokens with litellm's tokenizer for `openai/model`, not the policy's, so the threshold
is approximate; leave a margin.
On a Harbor bump: recheck `RUNTIME_DEPS` against `harbor.agents.terminus_2`'s imports (setup fails
loudly if one is missing), the `TmuxSession` and `Terminus2` names the runner uses, and upstream's
`terminus-2` binding.

## Remove when
Harbor ships an in-sandbox Terminus-2 (an installed-agent variant); upstream status: not filed.
