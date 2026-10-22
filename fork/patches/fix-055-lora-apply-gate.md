# fix-055: a model's `should_apply_lora` also gates which modules SGLang wraps (sgl-project/sglang#39798)

## Why
At our pin `LoRAManager.init_lora_modules` reads a model's `should_apply_lora` only for the memory pool, not for
the wrap. Any `FusedMoE` is wrapped as soon as the targets include both `gate_up_proj` and `down_proj`, which a
shared-expert adapter does. On DeepSeek-V4.1 that wraps the routed experts, which no adapter trains, and with the
release's FP4 experts the wrap fails at engine start: `'Mxfp4FlashinferTrtllmMoEMethod' object has no attribute
'get_moe_quant_info'` (lora-unmerged-slice-1021-a1, adapter cell on the FP8/FP4 slice). hook-010's
`should_apply_lora` already excludes the routed experts; the wrap just never asked it.

## Decision
- Backport the gate from sgl-project/sglang#39798 (on SGLang main since 2026-10-13 and on `sglang-miles`
  `3a938213c1`): skip a module when the model defines `should_apply_lora` and it returns false. 4 lines, the
  same code and place as upstream. The rest of #39798 (NemotronH VL support) is left out, and so is its move of
  the Intern-S2-Mobius `model.meta_mlp` ValueError above the gate: with the gate alone, that model's
  `should_apply_lora` skips its routed banks silently instead of raising. We do not serve Intern-S2-Mobius.
- Models without `should_apply_lora` are unchanged.

## How it is applied
`fork/deps/sglang/fix-055-lora-apply-gate.patch`, applied by `fork/deps/sglang/apply.py` to SGLang `7b6993a`:
`lora/lora_manager.py`, with a `[fork fix-055]` comment.

## Remove when
SGLANG_COMMIT includes sgl-project/sglang#39798 (SGLang main `2a8e9db3b0`).
