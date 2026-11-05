# ext-067: GLM-5.3-Flash with the DFlash2 draft, top-p keys and early refusal of sampling conflicts

## Why
The target setup is GLM-5.3-Flash LoRA with MIS, score centering, top-p replay and a speculative draft. The preset's
EAGLE draft cannot do that: SGLang refuses `return_sampling_mask` outside the DFlash family (`spec_sampling_mask.py`,
pin and main), and Miles refuses score centering with it (fix-058). The DSpark draft is DeepSeek's. A DFlash2 draft for
GLM-5.3-Flash exists, `incoai/GLM-5.3-Flash-DFlash2`, and SGLang's GLM cookbook lists its flags.

## Decision
- A preset `glm-5.3-flash-dflash2` built on `glm-5.3-flash`, with the draft's repo and revision. EAGLE stays on the
  existing preset, since `glm53-flash-split-gpud.yaml` uses it.
- Recipe keys `algorithm.top_p` and `algorithm.top_k`. They replace `--rollout-top-p` in `extra_args`, which still works.
- `lora_exclude_modules` is a preset field, not a recipe key: no run wants a different value, and native LoRA rejects a
  selection that keeps routed experts or `kv_b_proj`.
- Recipes refuse at load: top-p or top-k with a draft outside the DFlash family, score centering with a draft outside
  the DFlash family, and top-p without top-k. Otherwise these fail after the engines start.
- Score centering with DFLASH comes from fix-085, with its own sampling check.
- Rejected: `rollout.speculative` naming a draft. Each preset has one draft today, and the flags differ by draft anyway.

License: the draft is CC BY-NC-ND 4.0, for research and evaluation, with commercial use by arrangement with inco.ai.

## How it is applied
`miles_ext/recipes/presets.py`, `miles_ext/recipes/recipe.py`, `miles_ext/recipes/configs/glm53-flash-lora-dflash2.yaml`
and its golden argv. Weights: `prepare_weights("incoai/GLM-5.3-Flash-DFlash2",
"bf582e4eacc1810f76656d1811693ff6c6737d2a", ...)`. Verify: `python -m pytest miles_ext/tests/test_glm53_draft.py
miles_ext/tests/test_placement_golden.py`, then the recipe's run. On an upstream sync, diff the flags against SGLang's
GLM cookbook.

## Remove when
Upstream Miles ships a recipe-level draft selection; upstream status: not applicable (recipes are ours).
