# ext-040: where a rollout's sandbox time goes, and a CPU-only benchmark of Sail and Modal sandboxes

## Why
The a6 env-time study (10-21; listar2000/miles-ops `notes/rollout-env/RESULT-2026-10-21.md`) found 46% of trajectory time outside the model: 37% tool commands, 6% verifier,
2% start and install, read from mini-swe-agent's step timestamps. It could not say whether Sail runs the same
command slower than another provider, or how much of each model round trip is the tunnel and proxy. Measuring
that with real inference costs GPUs and mixes in model speed.

## Decision
- Timing hooks in `miles_ext/obs/sandbox_profile.py`, switched by recipe `obs.sandbox_profile` (default off).
  Off, the trial path is unchanged: Harbor gets the same provider config and the proxy app has no middleware.
  On, `ProfiledEnvironment` subclasses the run's provider (Sail's class or Harbor's Modal class, resolved from
  `HARBOR_ENV_IMPORT_PATH` / `HARBOR_ENV_TYPE`) and writes `<trial_dir>/sandbox_profile.jsonl`; the proxy's
  `--profile-log` writes `<run_dir>/sandbox_profile/proxy.jsonl`. The in-sandbox side reuses the step timings
  Harbor's mini-swe-agent patch already records. A wrapper by import path, not a patch to the providers.
  Nothing is written on an event loop: a trial's spans are written once, in a thread, when its sandbox stops;
  the proxy queues its lines to one writer thread.
- The proxy's time is receipt to response, so it includes the upstream: near zero with the benchmark's scripted
  model, the session server and SGLang in training (there it reads "model + proxy"; tunnel = in-sandbox wait
  minus that, in both). A call that does not match its proxy line one to one (a retry) keeps its wait whole as
  `model_unsplit`, so every trial's pieces still sum to its wall time.
- Benchmark in `miles_ext/obs/sandbox_bench/`: a CPU Modal function runs the production agent function
  (`agent_function.run` through Miles' agent-function child processes) against a scripted OpenAI-compatible model
  behind the real `session_proxy` and `modal.forward` tunnel. Workloads: six synthetic scripts on one image pinned
  by digest (noop, read, write, test, fs-scan, install) and replays of a past run's commands. Rejected: a fake
  SGLang engine behind Miles' session server (times the session server, which is model-side; much more setup).
- Efficient by default: `analyze` breaks a past run's recorded command time down by command class and duration
  with no sandbox, and writes a short replay of a few distinct commands per class (median and p90 of their time;
  timed-out commands are counted, not replayed). `main` defaults to 3 trials per synthetic workload at
  concurrency 1, 16 at concurrency 16, 2 repeats, and every trace of the given extract at each concurrency.
- Benchmark tasks run with public network (synthetic ones as written, replayed ones rewritten on copy), so no
  network policy switches during a trial: the pinned Harbor's Modal sandboxes fail such a switch (ext-041).
- No data in the repo: traces are extracted to `<results>/sbx-bench/traces/`, results go to
  `<results>/sbx-bench/<bench id>/`, synthetic task dirs and the test repo are generated at run time.

## How it is applied
- `miles_ext/harbor/agent_function.py`: `_environment_config` wraps the provider when on; `run` records the
  trial's session path. `session_proxy.py`: optional `--profile-log`. `recipe.py`: `obs.sandbox_profile` sets
  `MILES_EXT_SANDBOX_PROFILE=1` in `harbor_env`. `launch/modal_app.py` passes the proxy log path when on.
- Run the benchmark: the usage in `miles_ext/obs/sandbox_bench/app.py`. The breakdown's pieces (sandbox start,
  install, proxy, tunnel, tools by command class, agent rest, verifier, teardown) sum to each trial's wall time.
- Check: `python -m pytest miles_ext/tests/test_sandbox_profile.py miles_ext/tests/test_sandbox_bench.py`.
- On a Harbor bump: the provider method signatures (`start`, `stop`, `exec`, uploads, downloads) and the
  `harbor_llm_wait_msec` / `harbor_tool_exec_msec` trajectory fields are what this reads.

## Remove when
Not an upstream gap; stays while we run Harbor sandboxes. Upstream status: not filed.
