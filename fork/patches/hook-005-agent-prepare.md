# hook-005: the agent plugin's optional `prepare(args, data_source)`, called once at rollout setup

## Why
The rollout executor builds the rollout function about 18 minutes before the first weight sync
and then sits idle. In overfit-1018-a2 (a 2026-10-18 DeepSeek-V4.1 Harbor RL run on 8 tasks,
checking that training can overfit them; W&B mobiuslab/miles-dsv41 run phldkdx1) it was built at
19:45:47 UTC and the sync finished at 20:03:28. The agent module is first imported by the first
sample's `generate`, so everything an agent integration does once happens in the first wave, on
the rollout loop's process, while 128 trials start. For Harbor that is the first `Trial.create`
imports (Harbor's trial, the harness and the Sail provider; Sail is the sandbox provider the
trials run in) and the task-dir hashes (ext-001). In that run the loop logged nothing for 7 min
13 s after the worker started, then all 128 trials started within 8 s. Nothing in Miles lets an
agent plugin start work at setup.

## Decision
Upstream already looks up an optional sibling `abort` in the module of
`--custom-agent-function-path` (`call_agent_abort_hook`, `miles/utils/misc.py`). The hook adds the
same lookup for a sibling `prepare(args, data_source)`, called once from
`RolloutExecutor.__init__` right after the rollout function is built. A module without
`prepare` (every upstream agent) changes nothing. A failing `prepare` logs a warning and the
executor starts as before. It must return quickly; the first user starts a thread.
First user: ext-001, `miles_ext.harbor.agent_function.prepare` (imports, then
`task_hash.prewarm` over the prompt set). Rejected: a new `--custom-*-path` flag (the abort
lookup already defines where an agent's hooks live, and a flag would need a launcher change too).
Rejected: starting the work at module import (it is first imported in the first wave).

## How it is applied
`call_agent_prepare_hook` in `miles/utils/misc.py`, next to `call_agent_abort_hook`; one call in
`RolloutExecutor.__init__` (`miles/ray/rollout/rollout_executor.py`) after the
"import ... as generate_rollout function" log, plus its import; test
`tests/fast/utils/test_agent_prepare_hook.py`. All spots are marked. A module that is not found
skips `prepare` silently, as the abort lookup does. Not called
when `--load-debug-rollout-data` replaces the rollout function. On a sync, re-insert the call if
upstream reshapes the executor's init.

Acceptance on the next Harbor run: in launch.log, the time from "Started fully-async rollout
worker" to the first "Selected strategy" (overfit-1018-a2: 7 min 43 s), and a "task_hash prewarm: N/N task
dirs hashed" line before the first weight sync.

## Remove when
Upstream gives agent plugins a setup hook. Status: not filed.
