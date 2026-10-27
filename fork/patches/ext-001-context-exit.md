# ext-001: in-sandbox agent ends "ContextExhausted" at the context window

## Why
A mini-swe-agent run that fills `max_seq_len` keeps going. The engine truncates the next turn (and the prompt) to fit, the
turn ends `finish_reason=length` with no tool call, the agent is nagged "reached the output token limit", and after three
such turns (usually 0 tokens) it exits `RepeatedFormatError`. In dsv41-tb4-overfit16-4 that was 55% of trajectories (26%
in -3), with reward ~0.04, and Miles saw them as ordinary `Submitted` samples.

## Decision
A mini-swe-agent agent class, `miles_ext/harbor/context_exit.py:ContextExitAgent` (a subclass of `InteractiveAgent`, the
class `mini` uses by default), selected with mini-swe-agent's own `--agent-class` option. The alternatives lost:
- Rewriting mini-swe-agent's `default.py` at install time, as Harbor's own output-cap and step-timing patches do. That
  edits installed dependency code, which fork/README.md keeps out of this repo.
- Refusing the turn in the Miles session server: an upstream file change (a Fix), and the agent would still see an
  error and nag, not exit.
- A lower `step_limit`: doesn't target the cause.

The exit is `LimitsExceeded`, so Harbor still runs the verifier and the sample is scored normally (not aborted; algo
decided to keep the one cut-off turn the Miles session records, as on-policy data). `agent_function.run` reports
`ContextExhausted` instead of Harbor's generic `Submitted`, so the sample filters can tell it apart. A recipe can score
such a sample 0 whatever its verifier says (`agent.context_exhausted_scores_zero`, on by default since 2026-10-27, env
`MILES_EXT_CONTEXT_EXHAUSTED_SCORES_ZERO=1`): running out of a 256k window counts as failure, while 17 of 82 such
trials in the 1017 eval passed their verifier. The sample stays in the batch with reward 0, not masked.

## How it is applied
- `SandboxMiniSweAgent.install` copies `context_exit.py` (self-contained: it imports only mini-swe-agent) into the
  sandbox's mini-swe-agent environment as `miles_mswea_context.py` and imports it once, after Harbor's patches;
  `build_cli_flags` adds `--agent-class=miles_mswea_context.ContextExitAgent`.
- `build_trial_config` sets the agent env `MSWEA_CONTEXT_LIMIT` = the agent's `max_seq_len` (Harbor applies agent env to
  every agent exec; the name avoids the words Harbor treats as secret, whose values it replaces with `[REDACTED]` in
  every trial file). Unset, the class behaves like `InteractiveAgent`. `MSWEA_CONTEXT_MIN_REPLY` (default 4096) is the
  reply reserve.
- `populate_context_post_run` records mini-swe-agent's exit status as agent metadata `mswea_exit_status` (as the host
  harness does).
- Sizing (overfit16-4): calls that hit the window ended 2-6 tokens short of it (the 64-token threshold); observation
  tokens per char/3 p50 1.26, p90 1.67 (hence chars / 2); replies p50 980, p90 4750 tokens.
- Verify: `python -m pytest miles_ext/tests/test_context_exit.py miles_ext/tests/test_sandbox_context_exit.py`.
- On a mini-swe-agent upgrade: check `InteractiveAgent.query` still raises `FormatError` before appending the reply, and
  that `--agent-class` still exists.

## Remove when
mini-swe-agent or Harbor ends a run at the context window by itself; upstream status: not filed.
