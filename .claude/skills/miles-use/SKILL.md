---
name: miles-use
description: "How to run RL training with miles-mobius (listar2000/miles) on Modal from your own machine: launch and resume runs, write recipes, plug in algorithms, harnesses, rewards or task data, monitor and debug runs, and report problems. Use when training with this repo rather than changing the fork itself (that is the miles-fork skill)."
---

# Training with miles-mobius

You are a user of this repo: you run training and plug code in at the extension
points. You do not change upstream Miles files, SGLang patches, kernels or the
image. The pages under `fork/guide/` own the facts. This skill is the order to
read them in and the rules that keep a run cheap and reproducible.

## First run

1. Read `fork/guide/quickstart.md` and follow it in order: credentials, your own
   Modal names, a resolved trainer version (set `MILES_IMAGE` and check out its
   commit), weights, tasks.
2. Start from the closest recipe in `miles_ext/recipes/configs/`. Copy it into your
   own project, and look up every field you change in `DEFAULTS`
   (`miles_ext/recipes/recipe.py`).
3. Always run `--check-only` before a real launch, and fix everything it reports.
4. Begin with a short run (`--num-rollout 2 --save-interval 1`, run id `smoke-...`),
   then launch the same run id with `--num-rollout 3` to check resume, before a long
   or large run. A relaunch of a finished run id returns at once without training.

## Monitor your runs

The run monitor (`miles_ext/monitor/`, details in its README) is how you and the
person you work for watch runs. It only reads, so it is safe to leave running and share.

1. Pick one data dir and use it everywhere: `export MILES_MONITOR_DIR=<dir>` in the
   shell that launches and in the one that starts the monitor.
2. `miles_ext/monitor/start.sh` (Python 3.11+) serves http://127.0.0.1:8787. To share it, add `--tunnel`
   (a cloudflared quick tunnel; needs `cloudflared` on `PATH`). That prints a
   `trycloudflare.com` URL in `<dir>/url.txt`, and the login is in `<dir>/auth.txt`.
   Give out both. A restart keeps the URL while the tunnel lives. `stop.sh` stops
   it, and the next tunnel gets a new URL.
3. Launch with `MILES_MONITOR_DIR` set and the output redirected
   (`modal run --detach ... > <run_id>.log 2>&1`). The launcher writes
   `<dir>/manifests/<app_id>.json` itself, and the run appears within seconds with
   startup stages, step rewards and error signals read from that log. A launch made
   without it can be added afterwards with `python -m miles_ext.monitor.register <run_id> <log> <recipe>`.
   The launcher finds the log through `/proc`, which exists only on Linux. On macOS,
   run that command after each launch to attach the log.
4. In a shared Modal workspace, set `app_prefixes`, `launcher_apps` and
   `results_volumes` in `<dir>/monitor.toml` to your own names, so the page shows
   your runs only.
5. Group runs: `<dir>/groups.json` holds rules that map a run-id regex to a group
   and a type (train, eval, bench, smoke). Add one rule per campaign. To compare
   two runs of a group, tick both on the Runs tab. The link
   (`#tab=compare&a=<run>&b=<run>`) is the one to send.
6. When you report a run, link its monitor page and its W&B run. Scripts can read
   `/api/state` (the README lists the fields that stay stable).

## Rules

- Never commit credentials. They live in the `.env` the launcher reads (`MILES_DOTENV_DIR`).
- Use your own Modal names (`MILES_APP_NAME`, `MILES_RESULTS_VOLUME`,
  `MILES_WEIGHTS_VOLUME`) in a shared workspace. Stop apps by id, and only your own.
- Pin the trainer: one image per run, so a resume never changes build.
- Keep data, task sets and run-specific recipes in your own project, not in this repo.
- New code goes under `miles_ext/` on a branch, following `fork/guide/extending.md`.
  An edit to anything outside `miles_ext/` is a maintainer change.

## When something breaks

1. `fork/guide/running.md`, "When something fails": find the first error, not the last line.
2. `fork/guide/limits.md`: it may be a known limit.
3. If the cause is below the recipe (kernels, SGLang, memory, weight sync, the
   launcher), stop the run and open an issue with the run-failure form. Its fields
   say what evidence to attach (`fork/guide/help.md`). Don't patch around it.
