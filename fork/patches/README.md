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
| ext-007 | Ext | DeepSeek V4.1 rollout in its released FP8 dense + FP4 expert format: weight sync writes each tensor as the rollout checkpoint stores it | active | [ext-007-lowprec-rollout.md](ext-007-lowprec-rollout.md) |
| hook-002 | Hook | `--custom-placement-hook-path`: the driver hands the placement groups to an extension once they exist (first user: ext-015, W&B GPU metrics labelled train / rollout per node) | active | [hook-002-placement-hook.md](hook-002-placement-hook.md) |
| hook-001 | Hook | `--custom-weight-quantizer-path`: a rollout checkpoint layout brings its own weight-sync quantizer | active | [hook-001-custom-weight-quantizer.md](hook-001-custom-weight-quantizer.md) |
| fix-006 | Fix (port) | `--mxfp4-qat-routed-experts` and the EVEN E8M0 scale rule from upstream branch `jiajun/mimo26-mxfp4-qat` | active; remove when upstream main has them | [fix-006-mxfp4-qat.md](fix-006-mxfp4-qat.md) |
| ext-012 | Ext | DeepSeek V4.1-Flash training recipe: model flags in the preset, 8-GPU engines, FP8 + FP4 rollout, CP/EP, disk-streamed Adam, R3, expert QAT as recipe fields | active | [ext-012-dsv41-recipe.md](ext-012-dsv41-recipe.md) |
| ext-008 | Ext | Compiled-kernel cache (`SGLANG_CACHE_DIR`: FlashInfer autotune, CUDA, Triton, TileLang, DeepGEMM) kept on a Volume between runs; DeepSeek-V4.1 engine start 16 min to 4 min | active | [ext-008-compile-cache.md](ext-008-compile-cache.md) |
| ext-013 | Ext | `taskset.tasks_volume`: task dirs read from a Modal volume mounted read-only, so the launching machine keeps only the prompt file; the deep check matches prompt ids to task dirs | active | [ext-013-tasks-volume.md](ext-013-tasks-volume.md) |
| ext-001 | Ext | Harbor's per-trial task-dir hashes computed once per task in a worker thread (patched `Packager.compute_content_hash` / `Task.checksum`) | active; remove when Harbor hashes task dirs off the caller's loop or caches them | [ext-001-task-hash.md](ext-001-task-hash.md) |
| ext-001 | Ext | In-sandbox mini-swe-agent ends `ContextExhausted` at the context window (agent class via `--agent-class`), reported to Miles as that exit status | active | [ext-001-context-exit.md](ext-001-context-exit.md) |
| fix-010 | Fix | NVMe-streamed fp32 mains written once, by the checkpoint load, not first from the init params (~10 min less trainer init for DeepSeek-V4.1) | active; remove when upstream stops initialising the mains before the load | [fix-010-nvme-mains-once.md](fix-010-nvme-mains-once.md) |
| fix-007 | Fix | DeepSeek-V4.1 indexer candidates carried as block ids (uint16, up to 524,280 compressed keys), not a dense `[q, n_kv]` mask; the carry stops after the last consumer | active; remove when upstream stops building the dense mask | [fix-007-dsv41-candidate-block-ids.md](fix-007-dsv41-candidate-block-ids.md) |
