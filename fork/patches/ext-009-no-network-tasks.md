# ext-009: the in-sandbox harness runs on no-network tasks

## Why
Every task in the in-house TB-4.0-style taskset (`tb4-targeted-model-evaluated-20261016`,
1330 tasks) sets `network_mode = "no-network"` for the environment, agent and verifier.
At our Harbor pin (`917b3c1b`) two things then fail for `mini-swe-agent-sandbox`:

1. `Trial._setup_agent` runs `agent.setup` under the environment's baseline policy, so
   `MiniSweAgent.install` (`uv tool install mini-swe-agent`, apt) cannot download anything.
2. The agent phase keeps the task policy, so the harness cannot reach the session server
   through the Modal tunnel (`MILES_SESSION_PUBLIC_URL`).

Run 4's TB4 tasks had network, so neither showed up before.

## Decision
- `SandboxMiniSweAgent.setup` switches the sandbox to public network for the install and
  restores the task's policy in a `finally`, before the agent phase starts. This is what the
  platform's evals do (mobius-platform `programs/harbor-eval/patches/harness/setup.py`,
  `ManagedSetupAgent`). A task that is already public is set up unchanged.
- `build_trial_config` adds the tunnel host to `AgentConfig.extra_allowed_hosts`, Harbor's
  own run-time allowlist for the agent phase. A no-network task becomes an allowlist with
  that one host; Harbor ignores the entry on a public task. The verifier phase is untouched.

Rejected: allowlisting install hosts (PyPI, astral.sh, apt mirrors) in the environment
baseline, which would also leave them open to the agent and a shared verifier.

## How it is applied
`miles_ext/harbor/sandbox_agent.py`, `miles_ext/harbor/agent_function.py`; tests in
`miles_ext/tests/test_no_network_tasks.py`. Needs a provider with
`dynamic_network_policy` and hostname allowlists (Sail has both, ext-003). The host harness
(`mini-swe-agent-host`) needs neither and is unchanged.

## Remove when
Harbor gives agent setup its own network phase. Upstream status: not filed.
