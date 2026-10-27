# ext-039: read-only run monitor

## Why

Someone training with the fork needs to see a run while it happens: startup stages, trial progress, rewards per
step, engine throughput, GPUs held and what Modal bills. W&B shows only what the trainer logs, after each step;
the Modal dashboard shows apps, not runs. The monitor existed on one maintainer machine only, with that
project's names hard-coded and a wrapper script to register each launch.

## Decision

- **In the fork, under `miles_ext/monitor/`**, so maintainers and users run one copy. A stdlib HTTP server and a
  plain JS page: no build step, no service to deploy.
- **Read-only.** It lists, reads and tails; it never starts, stops or deletes anything, so the page is safe to
  share. Token-shaped strings and `.env` values are scrubbed from every response.
- **Plain files in one data dir** (`MILES_MONITOR_DIR`): manifests, state snapshot, JSONL history, cache. A
  database server was the alternative; nothing needs queries yet.
- **Registration by the launcher** when `MILES_MONITOR_DIR` is set, instead of a wrapper: a hand launch shows up
  with no extra step. It catches its own errors, so the monitor can never fail a launch.
- **Long-poll, not an event stream**: the cloudflared quick tunnel holds streamed responses back for tens of
  seconds.
- **Local by default.** `start.sh --tunnel` adds a quick tunnel and basic auth; `--no-auth` is an explicit
  choice for a public read-only page.
- **GPU inventory through Modal's internal client API** (the container list the CLI uses), in a subprocess
  (`gpus_probe.py`), so an API change breaks that tab only.

## How it is applied

`config.py` loads `<data dir>/monitor.toml` over defaults taken from the launcher's names; `manifest.py` writes
one file per launch attempt (`register_launch` from `miles_ext/launch/modal_app.py`, `register.py` by hand);
`server.py` collects and serves. Usage and config keys: `miles_ext/monitor/README.md`.

Verify with `python -m pytest miles_ext/tests/test_monitor.py -q`. On an upstream sync, check that Miles' per-step
log lines (`rollout N:`, `step N:`, `perf N:`) and the checkpoint and weight-sync messages the server parses still
match; after a Modal client upgrade, check the GPUs tab.

## Remove when

A supported hosted monitor covers live runs, attempts and trial progress.
