# fix-085 evidence: DFlash2's sampling path at SGLang `3a938213c1`

Read from the source; no test exercises this path directly apart from the calibration in the patch page.

| Step | Code | What it does |
| --- | --- | --- |
| The checkpoint has a selector | [`DFlash2DraftModel.__init__`](https://github.com/sgl-project/sglang/blob/3a938213c1b349e1a8df0b5fdc57ef36a41c743b/python/sglang/srt/models/dflash.py#L1205-L1221) | Refuses a config without `selector_rank`, and always builds `candidate_selector` |
| Each non-greedy step records the proposal | [`dflash_worker_v2.py#L2695-L2709`](https://github.com/sgl-project/sglang/blob/3a938213c1b349e1a8df0b5fdc57ef36a41c743b/python/sglang/srt/speculative/dflash_worker_v2.py#L2695-L2709), [`L1460-L1468`](https://github.com/sgl-project/sglang/blob/3a938213c1b349e1a8df0b5fdc57ef36a41c743b/python/sglang/srt/speculative/dflash_worker_v2.py#L1460-L1468) | Stores the proposed tokens' candidate ids and draft probabilities `q` in `_selector_sample`, in the graph path and the eager one |
| Acceptance takes the selector path | [`_accept_block`](https://github.com/sgl-project/sglang/blob/3a938213c1b349e1a8df0b5fdc57ef36a41c743b/python/sglang/srt/speculative/dflash_worker_v2.py#L2138-L2162) | With `_selector_sample` set, it calls `_selector_sampling_accept`. The other branches run only without it |
| Accepted and rejected tokens | [`_selector_sampling_accept`](https://github.com/sgl-project/sglang/blob/3a938213c1b349e1a8df0b5fdc57ef36a41c743b/python/sglang/srt/speculative/dflash_worker_v2.py#L1471-L1507) | Spreads `q` over the vocabulary and calls DSpark's `accept_sampling` with the target's verify logits |
| The rejection rule and bonus token | [`_accept_sampling_core` and `accept_sampling`](https://github.com/sgl-project/sglang/blob/3a938213c1b349e1a8df0b5fdc57ef36a41c743b/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py#L92-L198) | Builds the target probabilities with temperature, top-k and top-p, draws one uniform per position, and takes the bonus from `predicts` at the first rejected position. This is the path fix-058 checked for DSpark |
| Support reported to Miles | [`SpeculativeSamplingMaskCapture`](https://github.com/sgl-project/sglang/blob/3a938213c1b349e1a8df0b5fdc57ef36a41c743b/python/sglang/srt/speculative/spec_sampling_mask.py#L58-L100), [`dflash_worker_v2.py#L2866-L2878`](https://github.com/sgl-project/sglang/blob/3a938213c1b349e1a8df0b5fdc57ef36a41c743b/python/sglang/srt/speculative/dflash_worker_v2.py#L2866-L2878) | Rebuilds the same target probabilities from the verify logits for the committed tokens |
| Selected logprob and top-k | [`compute_spec_logprobs`](https://github.com/sgl-project/sglang/blob/3a938213c1b349e1a8df0b5fdc57ef36a41c743b/python/sglang/srt/layers/logprob_processor.py#L386), [`dflash_worker_v2.py#L2879-L2885`](https://github.com/sgl-project/sglang/blob/3a938213c1b349e1a8df0b5fdc57ef36a41c743b/python/sglang/srt/speculative/dflash_worker_v2.py#L2879-L2885) | One call over the committed tokens, on the verify logits, as for DSpark |

Miles' check: a draft whose `architectures` holds `DFlash2DraftModel` has the selector, thresholds at 1.0 are the only ones
`spec_sampling_mask.py` accepts, and `SGLANG_SIMULATE_ACC_LEN` stays refused.

Limits of this trace: at the pin, the DFlash family's target builder applies top-k and then top-p (fixed on SGLang main by
sgl-project/sglang#41916). Replay records the realized support, so training scores the same set, but "top-p 0.95" means
slightly less than it says. A draft sampler that runs eager uses the same acceptance and is only slower.
