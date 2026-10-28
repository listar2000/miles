# ext-033: trials off the rollout loop, and a watchdog for what blocks it

## Why

In fully async recipes every Harbor trial ran `inline` on the rollout process's one asyncio loop
(`--custom-agent-function-mode inline`, hard-coded in `recipe.py`). In a 3-node DeepSeek-V4.1-Flash run with
256 trials in flight, that loop went silent for 90-265 s at a time, over and over. Sandbox steps, verifiers and
new trial starts all waited on it (agent end to verifier start went from a 0.1 min median to 20-60 min), 18
sample collects timed out at 1,800 s, and 11 groups of 16 were dropped. The engines were never the limit: KV use
stayed at or below 0.33. Nothing in the logs said what held the loop.

## Decision

- **Trials run in subprocesses by default** (`agent.function_mode: subproc`, Miles' own default and the mode it
  documents for this case): each trial gets a child process, so one trial's blocking step stalls only that
  trial. `inline` stays available. The cost is one Python process per running trial (Harbor + Sail SDK imports,
  a few hundred MB each) and a cold `task_hash` memo per child (ext-001); the hash then runs in that child, not
  on the shared loop.
- **A watchdog names the blocker.** `miles_ext.obs.loop_watchdog.generate` wraps Miles' agentic generate
  function and starts one thread per loop. The thread posts a heartbeat every 0.5 s. When none has run for 5 s,
  it logs every thread's stack (another thread holding the GIL can be the cause) and re-logs the loop thread's
  stack every 30 s while the stall lasts. When the loop recovers, it logs the stall's length.
  Between stalls the watcher only wakes and posts a heartbeat; stacks are formatted and logged only during a
  stall. It records stalls that subprocesses do not remove (the executor's own batch conversion, debug rollout
  save and checkpoint save).
- Not chosen: moving each known blocking call (Sail downloads, trial-dir writes) into a thread. Nobody has yet
  shown which call blocks, and the list would only grow.

## How it is applied

- `miles_ext/recipes/recipe.py`: `agent.function_mode` (default `subproc`) feeds
  `--custom-agent-function-mode`, and `--custom-generate-function-path` points at the wrapper.
- `miles_ext/obs/loop_watchdog.py`: `watch(loop, thread_id)` plus the `generate` wrapper. The wrapper re-exports
  `add_arguments` so Miles still registers the agentic flags.
- Verify: `python -m pytest miles_ext/tests/test_loop_watchdog.py`. In a run, look for
  `rollout event loop stalled` in the rollout executor's log.
- On a sync: the wrapper calls `agentic_tool_call.generate`. If upstream renames it, the import fails loudly.

## Remove when

Upstream Miles monitors its rollout loop itself, and its recipes default to subprocess agents.

## Since 2026-10-27

`agent.function_mode` is gone: no recipe ran `inline`, so every run passes `--custom-agent-function-mode subproc`.

## Since 2026-10-28

Harbor trials still get a process each, but forked from a preloaded server (ext-042), not through Miles'
subproc mode: Miles runs the agent function `inline` and `agent_function.run` forks the trial. Single-turn
recipes keep `subproc`.
