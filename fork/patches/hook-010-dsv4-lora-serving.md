# hook-010: SGLang serves LoRA adapters on DeepSeek-V4/V4.1 (sgl-project/sglang#45231)

## Why
With merged sync (hook-006) the trainer folds the adapter into the weights and re-quantizes them to the
release's FP8 blocks before each sync. A LoRA step is about 1/100 of an FP8 ulp, so most of it rounds away:
for the final adapter of a 16-step V4.1 LoRA overfit run (rank 32, lr 3e-5) only 2-13% of the update
reached the engines, by CPU emulation of the trainer's own FP8 quantizer on two layers' weights
([RESULT.md](https://github.com/listar2000/miles-ops/blob/main/notes/algo/fp8lora/RESULT.md), private ops repo;
the slice smoke in the PR measures it on GPUs). Serving the adapter unmerged keeps the base frozen in FP8/FP4 and adds `B @ A x` in
bf16 per token, so the update reaches the engine in full. SGLang at our pin has no LoRA path for V4.

## Decision
- Carry sgl-project/sglang#45231 (RadixArk; open against `sglang-miles`, head `34033fe9`), the SGLang half
  of radixark/miles#4205. It makes `wq_a`, `wkv` (replicated), `wq_b`, `wo_a`, `wo_b` (attention TP) and the
  shared expert's `w1`/`w3`/`w2` LoRA targets, adds grouped `wo_a` LoRA to the Triton backend, keeps
  `wq_a`/`wkv` unfused and `wo_a` off the fused inverse RoPE when LoRA is on, and skips 4D expert buffers
  for experts the model never wraps. Routed experts, the indexer, compressor and engram are not adapted.
- It copies SGLang's existing LoRA pattern: `ColumnParallelLinearWithLoRA` / `RowParallelLinearWithLoRA`
  wrappers selected by `should_apply_lora`, `get_hidden_dim` and `get_lora_target_module_name` on the model
  class (as other models in `models/` define them), and the existing Triton segment GEMMs, run per head group
  for `wo_a`. The grouped `wo_a` semantics and the `wq_b` disambiguation follow sgl-project/sglang#36579 and
  #36578 (open on SGLang main).
- Above the SGLang patch size rule (about 30 lines; this is 170) and it adds model support: carried on the
  user's OK (DECISIONS.md, 2026-10-21).
- Rejected: a Miles-side merge that keeps the base in BF16 on the engine (the engine is FP8/FP4 by choice);
  dithered or scaled merges (they still round each element to the FP8 grid).

## How it is applied
`fork/deps/sglang/hook-010-dsv4-lora-serving.patch`, applied with fix-053 by `fork/deps/sglang/apply.py` to
SGLang `7b6993a`, one `[fork hook-010]` comment per file: `lora/backend/base_backend.py`,
`lora/backend/triton_backend.py`, `lora/layers.py`, `lora/mem_pool.py`, `lora/utils.py`, `models/deepseek_v4.py`.
The PR is written for a newer `sglang-miles`; at our pin four `deepseek_v4.py` hunks were placed by hand:
three differ only in context (no `_is_hip` early return in `_normalize_q_lora`, no `_is_gfx95_supported or`
in the multi-stream check, no `inv_rope` kwarg), and one is structural: our pin has no `_project_wo_b` and
no `wo_b.use_decode_attn_tp` read, so the grouped `wo_a` adapter call goes right before the inline `wo_b`,
and the CP-decode part of that hunk has nothing to patch. Instead `should_apply_lora` refuses CP decode attention TP,
which flips flags on the linears a LoRA wrapper holds (5 lines beyond the PR).
Used when a recipe sets `lora.serving: adapter`: Miles then starts the engines with `--enable-lora` and
pushes the bf16 adapter at each sync (upstream's adapter path; hook-011 for disaggregated engines).
LoRA with DSpark needs SGLang's static ragged verify (the default), no `--speculative-adaptive`, no
`experimental_sgl_trtllm` MoE runner; the draft runs without the adapter.

## Remove when
SGLANG_COMMIT moves past #45231's merge into `sglang-miles`. Upstream status 2026-10-21: #45231 open, CI red;
SGLang main `aa10927271` has no LoRA path for V4 (no `should_apply_lora` in `models/deepseek_v4.py`;
#36579 and #36578 open).
