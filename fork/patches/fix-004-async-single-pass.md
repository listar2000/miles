# fix-004: `--async-single-pass`, an eval hands each prompt out once and waits for all of it

## Why
An eval run as one fully-async rollout (`--num-rollout 1`, `--rollout-batch-size` = number of
tasks, `--async-max-concurrent-samples` below tasks x samples) hands tasks out more than once.
`RolloutDataSource.get_samples` starts epoch 2 once every prompt is out, the producer keeps every
slot full, and the step ends after `rollout_batch_size` groups of any prompt. Run 4
(dsv41-tb4-pass4-2, 52 TB4 tasks x 4, 128 in flight): 37 tasks got a second group of 4, and the
run would have ended at 52 finished groups with 15-16 slow tasks still on their first attempts.

## Decision
A launch flag on the fully-async producer, off by default:
- the producer stops taking prompts when the data source's epoch is used up (it never wraps);
  recycled groups (`--async-unused-samples-handler retry`) are still resubmitted;
- once every handed-out group has finished, a step that cannot fill `--rollout-batch-size` takes
  what is left (a group the buffer filtered out is not waited for), and a step past the pass fails
  with a clear error;
- the concurrency cap is unchanged.

Rejected: Miles' native eval path (`--eval-prompt-data`, `run_eval_datasets`) runs each prompt
exactly n times but has no concurrency cap, so every sandbox and agent timer would start at once.

## How it is applied
`miles/utils/arguments.py` (flag, `--fully-async` check), `miles/rollout/data_source.py`
(`DataSource.epoch_handed_out`), `miles/rollout/fully_async_rollout.py`
(producer stop, `_get_from_pass` drain), `miles/rollout/fully_async_data_buffer.py`
(`DataBuffer.num_ready`; a custom buffer needs it only with this flag). Recipes: `async.single_pass`.
Verify: `tests/fast/rollout/test_fully_async_single_pass.py`. On a sync, re-apply if upstream
reshapes `_worker_loop` or `_drain`.

## Remove when
Upstream's fully-async producer can stop after one pass, or its eval path gets a concurrency cap;
upstream status: not filed.
