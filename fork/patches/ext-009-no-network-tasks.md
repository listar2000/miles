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
- `SandboxMiniSweAgent.install` retries Harbor's stock install on `NonZeroAgentExitCodeError`, up to
  `INSTALL_ATTEMPTS` (4) attempts, 15/30/45 s apart (`INSTALL_BACKOFF_SECS`). The install downloads from
  PyPI, astral.sh and the distro mirrors; a dropped connection otherwise aborts the trial before its first
  model call and drops the sample's whole group. Each stock install step skips what is already in place,
  so a repeat is safe.
- `SandboxMiniSweAgent.exec_as_agent` keeps the task out of mini-swe-agent's argv. Stock Harbor
  starts `mini-swe-agent --yolo --model=<m> --task=<quoted task>`, so the task text is in
  `/proc/<pid>/cmdline`; a ProgramBench agent running `pkill -f /reference/program` (a path the
  task names) killed mini-swe-agent itself: `AgentError`, empty `/app`, reward 0 (8 of the first
  ~100 trials of a DeepSeek-V4.1 ProgramBench eval on Sail). The subclass writes the task to
  `/tmp/mswea-task.md` (as base64 through `echo | base64 -d`) and replaces the `mini-swe-agent ...
  --task=...` prefix with a `python -c` entry point that appends `--task <file contents>` to
  `sys.argv`; every other flag, `-c` config, `tee` and env stay as Harbor built them.
  - The match is an exact substring of Harbor's command, built from `render_instruction(instruction)`
    (Harbor's `run` renders the prompt template first). If it misses (MCP servers append to the
    task; a Harbor change to the command), the stock command runs, the bug comes back, and a
    warning "mini-swe-agent command not recognised" is logged.
  - Rejected: forking Harbor's `_run_inner` (a large private method) or patching mini-swe-agent.

Rejected: allowlisting install hosts (PyPI, astral.sh, apt mirrors) in the environment
baseline, which would also leave them open to the agent and a shared verifier.

## How it is applied
`miles_ext/harnesses/mini_swe_agent/sandbox.py`, `miles_ext/harbor/agent_function.py`; tests in
`miles_ext/tests/test_no_network_tasks.py`. Needs a provider with
`dynamic_network_policy` and hostname allowlists (Sail has both, ext-003). The host harness
(`mini-swe-agent-host`) needs neither and is unchanged.
On a Harbor or mini-swe-agent bump, recheck: the command string in `MiniSweAgent._run_inner`
(`mini_swe_agent.py:1008` at `917b3c1b`), that `minisweagent.run.mini:app` is still the CLI entry
and takes `--task`, and that the tool venv is still `$(uv tool dir)/mini-swe-agent`. A warning in
trial logs means the match missed.
When "recipe harness = any Harbor agent" lands, move this into a wrapper over every
in-sandbox agent instead of this one subclass.

## Remove when
Harbor gives agent setup its own network phase (the argv rewrite: Harbor stops passing the task
as `--task=<text>`). Upstream status: not filed.
