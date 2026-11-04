# Experiments

One folder per experiment (or series of runs): the code, scripts, configs and write-up that only that
experiment uses. Code meant to be reused (trainers, connectors, harnesses, presets) lives in `miles_ext/`.

| Rule | Why |
| --- | --- |
| `experiments/<name>/` is self-contained; deleting it breaks nothing else | an experiment's filters, paths and one-off tools don't become an API others must keep working |
| experiments import `miles_ext`, never the reverse | `miles_ext` stays dataset- and experiment-agnostic, and syncs with upstream only care about it |
| `<name>` is snake_case; its commits carry `Fork-Exp: <name>` and touch only that folder | `fork/tools/ledger.py check` enforces it, so an experiment commit can't edit shared code |
| moving shared code into `miles_ext` is decided case by case (the owner's call, an agent may recommend it) | no fixed "second user" threshold for this |
| `README.md` in each folder: goal, data, decisions, how to reproduce, results; it is the experiment's patch page, so its commits say `Patch page: experiments/<name>/README.md` | the folder is the record of the experiment, and design notes live next to the code they explain |
