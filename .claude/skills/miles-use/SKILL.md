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
4. Begin with a short run (`--num-rollout 2`, run id `smoke-...`), then launch the
   same run id again to check resume, before a long or large run.

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
