# Known limits

Read these before planning a run. Each line links to where the limit is owned.
When one blocks you, open an issue ([help.md](help.md)). Don't work around it in
a recipe.

## Models and layouts

- Presets exist for Qwen3.5-9B, Qwen3-8B and DeepSeek V4.1 Flash
  (`miles_ext/recipes/presets.py`). Each preset's `shapes` lists the clusters it
  has actually run on. Other shapes are untested.
- DeepSeek V4.1 Flash runs on the Megatron backend on B300s. Layouts that have
  run are in the `dsv41-flash-*` recipes and
  [ext-012](../patches/ext-012-dsv41-recipe.md). That page also lists the layouts
  the recipe refuses.
- The trainer needs a BF16 copy of the V4.1 weights on the weights volume. The tool
  that makes it is not in this repo yet (see `docs/models/deepseek/deepseek-v4-1-flash.md`, §3.2).
- RL on V4.1 is text only: the engines skip the vision tower (preset comment).

## LoRA

- LoRA adapters sync to separate engines only with pipeline parallel size 1
  ([hook-011](../patches/hook-011-native-lora-disaggregated.md)).
- LoRA keeps the FP8/FP4 base on the engines and applies a BF16 adapter
  ([hook-010](../patches/hook-010-dsv4-lora-serving.md)).

## Algorithms

- `algorithm.loss: score_centering` accepts DSpark speculative decoding and refuses
  every other speculative mode ([fix-058](../patches/fix-058-score-centering-dspark.md)).
- `advantage_estimator: ppo` is not available: recipes build no critic (`ADVANTAGE_ESTIMATORS` in `recipe.py`).
- Fully async mode rejects pause `abort` and partial rollout (same place).

## Rollout and sandboxes

- Harnesses: mini-swe-agent inside the sandbox, mini-swe-agent on the host,
  Terminus-2 and DeepSeek's dsh inside the sandbox, and single-turn (`HARNESSES` in `recipe.py`).
  Harbor installs the harness in each trial's sandbox. Task images stay the
  benchmark's own.
- Context compaction (`agent.compact_at_tokens`) is Terminus-2 and dsh only, Megatron only, and moves
  the run to session server v2 ([ext-045](../patches/ext-045-terminus-sandbox.md),
  [ext-046](../patches/ext-046-deepseek-sandbox.md)).
- deepseek-sandbox: no step cap (`agent.max_steps` is ignored), no images, glibc task images or
  Alpine's own Node; each sandbox installs Node and dsh (about a minute)
  ([ext-046](../patches/ext-046-deepseek-sandbox.md)).
- Sandboxes are Modal or Sail. Each has its own capacity, and a provider that is
  slow to start shows up as setup timeouts ([running.md](running.md)).

## Infrastructure

- GPU capacity on Modal (B300 in particular) can queue. A large cluster may wait
  before it starts.
- Launches go through `modal run` from your machine. Keep the launching shell or
  use `--detach`.
- The repo runs no CI on PRs. Run the checks yourself ([extending.md](extending.md)).

The full list of fork changes, with the upstream bugs they work around, is in
[fork/patches/README.md](../patches/README.md) and [fork/UPSTREAM.md](../UPSTREAM.md).
