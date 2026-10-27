# fix-048: regenerate only a group's aborted tries

## Why

A fully async group trains only if none of its tries is aborted (`apply_aborted_filter` in the data buffer).
One aborted try therefore discards all of its finished siblings, and the prompt needs a whole new group. In a
3-node DeepSeek-V4.1-Flash run with 16 tries per prompt, 18 sample-collect timeouts and 3 cut-off replies
dropped 11 groups (176 tries, 1.4 batches) in about 90 minutes. Each of those groups had only a few aborted
tries.

## Decision

`--async-retry-aborted-tries N` (default 1 since fix-059, which also sets it to 0 under `--group-rm` or
deterministic inference; upstream's tests run with 0): with it on, each try of a group runs as a group of one. When a try
is aborted, it is regenerated at once from a fresh copy of its prompt sample, stamped with the current KV-cache
namespace, up to N times, while its siblings keep running. The group keeps its finished tries. A try still
aborted after the last retry leaves the whole group to `--async-unused-samples-handler`, as before. Recipes
set N = 1.

- Not chosen: training the group without its aborted tries. Groups would then be smaller than
  `--n-samples-per-prompt`, and the batch smaller than `--global-batch-size`. That needs
  `--use-dynamic-global-batch-size`, which Miles allows only with `--use-dynamic-batch-size`. The DeepSeek-V4.1
  recipes use static micro-batches of 1 (bshd).
- Not chosen: retrying only after the whole group finishes. The group would then wait for its slowest try plus
  a full retry, which makes it more likely to go stale before the trainer takes it.
- Running tries one by one means no `--group-rm` (a reward over the whole group) and no per-try seed under
  deterministic inference. With either, the retry count is forced to 0 (since fix-059; a value above
  the default logs a warning) and aborted groups are handled whole.
- A retried try samples with newer weights. The group's staleness is its oldest try's, so the staleness check
  is unchanged. Retries are counted in flight (`on_submit`) and release their slots as they finish. They may
  briefly exceed `--async-max-concurrent-samples`.
- Only `ABORTED` tries are retried: failed sample collects and transport errors. A length-limit cut-off is
  `TRUNCATED` and a full context window ends `ContextExhausted`; both stay scored and are never redrawn.
- Metrics: `rollout/fully_async/retried_tries` (cumulative), plus `first_try_reward_mean` and
  `retried_try_reward_mean` over the tries finished since the last step.

## How it is applied

`miles/rollout/fully_async_rollout.py` (`_generate_group`, `_generate_tries`, `_generate_try`, `_try_metrics`) and `miles/utils/fork_arguments.py` (the
flag, hook-014), marked `[fork fix-048]`. Default 1 since fix-059, forced to 0 with a group reward or deterministic inference; recipes no longer set it. Verify:
`pytest tests/fast/rollout/test_fully_async_rollout.py -k only_aborted_tries`. In a run, look for
`Regenerating aborted try` in the rollout log and `retried_tries` in W&B.

## Remove when

Upstream fully async mode retries or drops tries individually.
