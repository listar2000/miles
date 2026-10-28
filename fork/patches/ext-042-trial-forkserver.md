# ext-042: Harbor trials forked from a preloaded server

## Why
ext-033 runs each Harbor trial in its own process through Miles' `--custom-agent-function-mode subproc`,
which starts a fresh interpreter per trial. Each one imports Miles, Harbor and litellm again: about 6 CPU-s
alone, 16 to 36 CPU-s each when 512 start at once. A CPU probe of 512 trials on 32 cores
(hashing off) put the first sandbox call 261 s after rollout start and
the last at 312 s; 64 cores was no faster. The same 512 trials inline took 6 s, and from a forkserver 42 s.
In the V4.1 LoRA split run at 512 in flight, this start storm is what held the first wave.

## Decision
- **Trials are children of one forkserver.** `miles_ext/harbor/trial_procs.py` keeps a `multiprocessing`
  forkserver that imports the trial modules once (Miles' agent function, the fork's agent function, Harbor's
  trial classes, the run's harness and environment classes). Each trial is a child forked from it, about
  1 CPU-s. Isolation is unchanged from ext-033: one process per trial, so a trial that blocks stalls only itself.
- **Miles' subproc contract, reused.** The child runs Miles' own `_run_and_report` and the caller reads with
  `_read_results`, so reports, outcomes, error tracebacks, SIGTERM cancel with the grace periods and the
  SIGKILL backstop behave as under subproc. The names used are checked at import, so a rename upstream fails
  the import, not a rollout. Children die with the server (PDEATHSIG), and the server exits with the rollout
  process. The child takes the caller's environment and rollout-limit scope at call time.
- **Miles runs the agent function inline** for Harbor harnesses (`recipe.AGENT_FUNCTION_MODES`), and
  `agent_function.run` hands each trial to `trial_procs.call`. Single-turn stays `subproc` (no Harbor trial).
- **Starts leave the loop.** `Process.start` waits for the server to fork and send the pid, and the server forks
  one at a time (about 18 a second under a 512 burst), so on the loop it held the rollout loop 28 s. Starts run in
  order on one thread of their own, and a started child leaves `multiprocessing`'s process table (`_watch`
  reaps it), so later starts do not poll it from that thread while the loop does.
- **Image venv ships compiled bytecode** (`Fork-Infra: deps-bytecode`, same PR): `uv pip install
  --compile-bytecode`, and the deps layer adds the `.pyc` files, so the server and Miles' own subprocesses do
  not compile Harbor and litellm on first import.
- Rejected: a pool of long-lived workers running many trials each (12 s in the probe, but trials share a
  loop again and a stuck trial holds its worker); inline (6 s, but this is the stall ext-033 removed).

## Measured (CPU probe, 512 trials, sandbox start stubbed)
| | last trial start | max rollout loop stall |
| --- | --- | --- |
| subproc (before) | 312 s | - |
| this PR, hashing off | 29 s (32 or 64 cores) | 0.01 s |
| this PR, prewarm (as in a run) | 28-30 s | 0.03 s |
| this PR, no prewarm, 32 cores | 53 s | 0.03 s |

Child CPU per trial 0.6-1 s (2 s max), RSS ~300 MB. The rest of the 29 s is the server's serial fork rate.

## How it is applied
- New `miles_ext/harbor/trial_procs.py` (`start`, `call`, `in_agent_process`). `agent_function.prepare`
  starts the server at rollout setup, before the hash prewarm; `call` starts it on first use otherwise.
- `agent_function.run` calls `run_trial` directly inside an agent process, else through `trial_procs.call`.
- Check: `python -m pytest miles_ext/tests/test_trial_procs.py miles_ext/tests/test_task_hash.py`.
- In a run: `trial forkserver up, preloaded [...]` in the rollout log; trial logs say `agent_function pid=`.
- On a Miles sync: the names in `trial_procs._AF_API`. On a Python bump: `multiprocessing.forkserver`.

## Remove when
Upstream Miles runs agent calls from a preloaded forkserver or pool itself. Upstream status: not filed.
