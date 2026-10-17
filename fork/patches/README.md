# Patch pages

One row per change important enough for its own page (rules: fork/README.md,
"Patch pages"). The full list of fork changes is
`python fork/tools/ledger.py markdown`; this index is the subset worth reading.

| Id | Kind | What | Status | Page |
| --- | --- | --- | --- | --- |
| fix-004 | Fix | `--async-single-pass`: the fully-async producer hands each prompt out once and the step waits for every group (evals) | active | [fix-004-async-single-pass.md](fix-004-async-single-pass.md) |
| ext-003 | Ext (vendored) | Sail sandboxes: Harbor main's Sail provider carried verbatim, loaded by Harbor `import_path`, plus lifetime/idle caps | active; remove when the pinned harbor-miles-* branch has `environments/sail.py` | [ext-003-harbor-sail.md](ext-003-harbor-sail.md) |
| ext-004 | Ext | Megatron trainer in recipes: `model.backend`, `parallel`, HF weights via mbridge, trainers spanning nodes, layout validation | active | [ext-004-megatron-recipes.md](ext-004-megatron-recipes.md) |
| ext-006 | Ext | Platform API `miles_ext.platform`: target guard, validate (CPU/1 GPU), teardown adapters, per-run secret, weights cache, checkpoints and retention | active | [ext-006-platform-api.md](ext-006-platform-api.md) |
| ext-009 | Ext | In-sandbox harness on no-network tasks: public network for the install only, session tunnel host allowlisted for the agent phase | active; remove when Harbor gives agent setup its own network phase | [ext-009-no-network-tasks.md](ext-009-no-network-tasks.md) |
