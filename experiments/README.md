# Experiments

One folder per experiment (or series of runs): the code, scripts, configs and write-up that only that
experiment uses. Everything a second experiment could use as is lives in `miles_ext/` instead
(trainers, connectors, harnesses, presets).

| Rule | Why |
| --- | --- |
| `experiments/<name>/` is self-contained; deleting it breaks nothing else | an experiment's filters, paths and one-off tools don't become an API others must keep working |
| experiments import `miles_ext`, never the reverse | `miles_ext` stays dataset- and experiment-agnostic, and syncs with upstream only care about it |
| `<name>` is snake_case; its commits carry `Fork-Exp: <name>` and touch only that folder | `fork/tools/ledger.py check` enforces it, so an experiment commit can't edit shared code |
| moving shared code into `miles_ext` is decided case by case (the owner's call, an agent may recommend it) | no fixed "second user" threshold for this |
| `README.md` in each folder: goal, data, decisions, how to reproduce, results | the folder is the record of the experiment |

Experiments run on a published image with this checkout's `miles_ext/` and `experiments/` laid over it
(`miles_ext.sft.modal_common`), so a change here needs no image rebuild.

| Experiment | What |
| --- | --- |
| [glm53_pi_sft](glm53_pi_sft/README.md) | SFT of Qwen3-30B-A3B-Thinking-2507 on GLM-5.3-Flash Pi traces from Mobius TB-2 evals |
