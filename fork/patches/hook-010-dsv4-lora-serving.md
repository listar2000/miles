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
SGLang `3a938213c1` (v0.1.3's `sglang-miles`), one `[fork hook-010]` comment per file: `lora/backend/base_backend.py`,
`lora/backend/triton_backend.py`, `lora/layers.py`, `lora/mem_pool.py`, `lora/utils.py`, `models/deepseek_v4.py`.
It is #45231 at head `34033fe9` placed on our pin by hand, plus two guards beyond the PR:
- `should_apply_lora` refuses CP decode attention TP, which flips flags on the linears a LoRA wrapper holds
  (5 lines);
- `_project_wo_b` reads `wo_b.use_decode_attn_tp` with `getattr(..., False)`: a LoRA wrapper has no such
  flag, and with the guard above it is always false.
Redone in sync-v0.1.3 (from `7b6993a`; `sglang-miles` was rebuilt on SGLang main on 10-13, so the old patch's
context was gone): the `utils.py` name lists sit next to the GLM-5.3 KDA `f_*`/`g_*` projections, and the
grouped `wo_a` adapter call is the last step of the `wo_a` branch, before `_project_wo_b`.
Not taken from #45231's later commits (head `1fb53a23`, 10-23): `wo_a`'s output kept unquantized under LoRA
(only matters when `wo_b` is not a target; our V4.1 LoRA spec always adapts `wo_b`, so the fused MXFP8 epilogue is off),
the TP-global LoRA layout for MLP rows gathered across attention-DP ranks (we run no DP attention), and CP decode
support in place of our guard. Re-read them at the next `sglang-miles` bump.
Used when a recipe sets `lora.serving: adapter`: Miles then starts the engines with `--enable-lora` and
pushes the bf16 adapter at each sync (upstream's adapter path; hook-011 for disaggregated engines).
LoRA with DSpark needs SGLang's static ragged verify (the default), no `--speculative-adaptive`, no
`experimental_sgl_trtllm` MoE runner; the draft runs without the adapter.

## Remove when
SGLANG_COMMIT moves past #45231's merge into `sglang-miles`. Upstream status 2026-10-25: #45231 open (head
`1fb53a23`, rebased on `sglang-miles` past `3a938213c1`). 2026-10-21:
SGLang main `aa10927271` has no LoRA path for V4 (no `should_apply_lora` in `models/deepseek_v4.py`;
#36579 and #36578 open).
