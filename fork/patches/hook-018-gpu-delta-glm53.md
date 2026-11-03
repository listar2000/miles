# hook-018: GPU-delta receiver mapping for GLM-5.3-Flash (glm5_next)

## Why
Full-parameter GLM-5.3-Flash runs in split placement need GPU-delta weight sync (port-001, port-002), but the
receiver's model mapping covers only DeepSeek V2/V3/V3.2 and GLM-5/5.2 (`GlmMoeDsaForCausalLM`); for
`Glm5NextForConditionalGeneration` it raises "unsupported GPU delta model mapping". GLM-5.3-Flash adds KDA linear
attention, mHC, and serves its experts on the triton runner (R3 needs its top-k ids), which the PR does not admit.

## Decision
- A `Glm5NextMapping` beside the PR's `DeepSeekMlaMapping`, same contract (`bind`, `finish`, `aliases`). Every
  binding is a byte view of a live parameter: fused KDA projections and conv1d by their loader's shard offsets,
  the DSA q/kv a-projection fusion, MLA `w_kc`/`w_vc` as the PR's derived images, triton expert `w13`/`w2` rows and
  FP32 block scales (fused shared expert included), dense block-FP8 weights and FP32 scales.
- Engines run with `SGLANG_ENABLE_JIT_DEEPGEMM=0` so dense FP8 layers keep the checkpoint's FP32 block scales.
  With DeepGEMM on, SGLang requantizes them to UE8M0 at load and packs the scales into int32 for DeepGEMM; that
  would need a packed-scale binding plus a requantized trainer baseline. Rejected for now: dense layers are a
  small share of GLM decode, and the trainer's FP8 quantizer already emits FP32 block scales without DeepGEMM.
- Tensors the loader casts BF16 -> FP32 (mHC `hc_*_fn`, indexer `weights_proj`, `k_norm`, kpool ape, conv1d)
  travel as complete values (`raw_bytes`), as the PR does for UE8M0 scales. KDA `A_log`/`dt_bias` have a second
  copy in the attention backend; both receive the value.
- The vision tower and the NextN draft stay static (excluded); KV-cache scales are not checkpoint tensors.
- Size: about 230 added lines, past the ~30-line patch guideline; allowed as a hook on top of a port that the
  port's feature needs (fork/README.md, "SGLang patches"), with the user's OK (2026-11-03).

## How it is applied
`fork/deps/sglang/hook-018-gpu-delta-glm53.patch` edits port-001's `weight_sync/gpu_delta/models.py`; `apply.py`
applies ports first. The mapping refuses engines whose dense FP8 scales were requantized to UE8M0 (DeepGEMM on),
expert runners other than triton, expert TP other than 1, presharded checkpoints, non-trivial or elastic expert
placement, and attention TP other than 1 for the conv1d casts.

Benchmark tools (ext-052, `miles_ext/gpu_delta/`): `make_glm_fp8_slice.py` builds a 4-layer FP8 slice of the
release (layers 0, 3, 4, 7, bytes unchanged) at `miles-rl-weights:/gpudelta/GLM-5.3-Flash-fp8-4layer` (Modal app
`miles-rl-gpudelta-slice`); `glm_bench.py` runs Miles' `tests/manual/gpu_delta/bench_gpu_delta.py` with GLM
serving settings (triton MoE, tilelang DSA, BF16 KV, no MTP, `SGLANG_ENABLE_JIT_DEEPGEMM=0`) and also mutates FP8
weights without making a NaN; `run_glm_bench.py` runs it on one B300:8 node (app `miles-rl-gpudelta-glmbench`),
with results under `miles-rl-results:/_probes/glm-bench-<tag>/`:

    modal run miles_ext/gpu_delta/run_glm_bench.py --tag <tag>

Result (2026-11-03, one EP8 engine with data-parallel attention, snappy-zstd, hashing off, 3 versions,
`--validate-final-target`): PASS. The final text, token ids and full input/output logprobs equal a freshly
loaded target checkpoint on every route. Pause per update 3850.8 ms (first, kernel compilation), 58.1 ms,
55.7 ms; 0.26 GiB payload per version at the benchmark's synthetic mutation rate. The image's build log lists
`hook-018-gpu-delta-glm53.patch` among the applied SGLang patches.

## Remove when
SGLang's GPU-delta receiver maps glm5_next itself; upstream status: not filed, to be offered on
sgl-project/sglang#42210 (not on SGLang main `4709a3a24`).
