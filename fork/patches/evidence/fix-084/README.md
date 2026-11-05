# fix-084 evidence: one B300:8, full GLM-5.3-Flash, native LoRA

Configs are side by side in this folder. All three runs: rank 16, lr 2e-3, 5 rollouts of 4 STAK prompts x 8 samples,
radix cache on, `reasoning_effort: low`, EAGLE off.

| Run | Image | Config difference from the first run | Result |
| --- | --- | --- | --- |
| Without the patch | `main@fabdf41722`, `starlimobius/miles-mobius@sha256:1838926fd4a8...` | none | 4 scheduler exceptions in rollout 2, run stopped by hand |
| With the patch | built from `c48ff087b2` (`fabdf41722` plus the patch and its page) | `timeout_hours` 1.4 to 1.0 | 5 rollouts, no scheduler exception |
| With the patch, zero staleness | same | `timeout_hours` 0.7, `max_weight_staleness` 0 | 5 rollouts, no scheduler exception |

The patch's comments were shortened after these runs; the code lines are unchanged. Excerpts below are from the launcher
logs on the box that launched them. The runs are one sample of each arm, and whether the failing batch recurs depends on
the prefill batches a run happens to produce.

## Without the patch
```
[2026-11-05 08:39:00 TP0 EP0] Prefill batch, #new-seq: 1, #new-token: 257, #cached-token: 256, full token usage: 0.02, mamba usage: 0.08, #running-req: 45, #queue-req: 0,
[2026-11-05 08:39:01 TP2 EP2] RuntimeError: Expected strides[leading_dim] == 1, but got 128.
  File "/opt/sglang/lib/python3.12/site-packages/flashinfer/kda_kernels/recurrent_kda.py", line 1009, in _get_grouped_compiled
[2026-11-05 08:39:01 TP0 EP0] Scheduler hit an exception: Traceback (most recent call last
```

## With the patch: installed markers and the end of the run
```
HEAD is now at c48ff087b2 infra(patches): patch page for fix-084, FlashInfer KDA grouped kernel contract
fix-084: /sgl-workspace/sglang/python/sglang/srt/layers/attention/linear/kda_backend.py
fix-084: /sgl-workspace/sglang/python/sglang/srt/layers/attention/linear/kernels/kda_flashinfer_prefill.py
[2026-11-05 09:06:40.722 rollout_executor] fully_async_rollout.py:295 - fully-async rollout worker ended
✓ App completed. View run at 
rollout 0: rollout 1: rollout 2: rollout 3: rollout 4: 
```

## With the patch, zero staleness (image already built from `c48ff087b2`, so no build lines)
```
[2026-11-05 09:37:51.180 rollout_executor] fully_async_rollout.py:295 - fully-async rollout worker ended
✓ App completed. View run at 
rollout 0: rollout 1: rollout 2: rollout 3: rollout 4: 
```
