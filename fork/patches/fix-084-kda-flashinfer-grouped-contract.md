# fix-084: FlashInfer KDA prefill honours the grouped kernel's contract (SGLang, sgl-project/sglang#45018)

## Why
On SM100/SM103 a KDA model with a gate lower bound (GLM-5.3-Flash) defaults to FlashInfer prefill. FlashInfer's
`recurrent_kda` uses its grouped kernel when no sequence in the packed batch extends more than one token, which happens
with the radix cache on. That kernel binds `dt_bias` as flat `[H*K]` and reads the state pool as bf16; our pin passes
`[H, 128]` and an fp32 pool. The scheduler dies:
`RuntimeError: Expected strides[leading_dim] == 1, but got 128` (full GLM-5.3-Flash LoRA on one B300:8, rollout 2, 33 s
after the first adapter sync; `flashinfer/kda_kernels/recurrent_kda.py:1009`). The same run with this patch finished all
5 rollouts.
It has nothing to do with LoRA. Upstream's GLM launchers turn the radix cache off, and ext-051 keeps it on for the
session server, so any of our GLM recipes can hit it. n_samples of one prompt make the fully cached prefix common.

## Decision
Copy #45018's two source changes: flatten `dt_bias`, and route batches with `num_tokens <= num_seqs` to Triton. Its unit
test stays upstream, since `apply.py` patches the image's source only; it passes with this patch and fails without
(evidence page).
- Rejected: `--sglang-linear-attn-prefill-backend triton` in the preset, which slows every prefill.
- Rejected: `--sglang-disable-radix-cache`, which the session server needs off (ext-051).

## How it is applied
`fork/deps/sglang/fix-084-kda-flashinfer-grouped-contract.patch`, applied by `fork/deps/sglang/apply.py` to SGLang
`3a938213c1`: `layers/attention/linear/kda_backend.py` and `kernels/kda_flashinfer_prefill.py`, marked `[fork fix-084]`.
Evidence: [evidence/fix-084](evidence/fix-084/README.md). Verify: a GLM-5.3-Flash run with the radix cache on and n_samples > 1 passes its second rollout.

## Remove when
SGLANG_COMMIT includes sgl-project/sglang#45018; upstream status: merged on main `aaa9388c91`, not in our pin.
