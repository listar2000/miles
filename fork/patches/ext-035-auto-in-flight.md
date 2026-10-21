# ext-035: auto in-flight size for fully async rollout (with hook-009)

## Why
At a fixed `--async-max-concurrent-samples` the producer makes groups at (groups in flight) / (group time),
whatever the trainer consumes. overfit-1020-a6 (LoRA DeepSeek V4.1, 128 tries = 16 groups in flight, 8 groups
a step) made 16 groups per ~46 min while a step took ~32 min. Finished groups waited ~3 steps in the FIFO
buffer (post-generation staleness ~3) and 5, 5, 2 and 1 groups were dropped as stale at steps 5-8.

A lower fixed number fits one trainer speed only. A discrete-event model of the producer and the FIFO buffer,
replaying a6's 93 complete groups (8 groups a batch, max staleness 4, 200 seeds; at 128 in flight it gives mean
staleness 2.9 against a6's measured ~3.3):

| step | in flight | trainer idle/step | stale tries/step |
| --- | --- | --- | --- |
| 32 min | 128 | 0 | 10.9 |
| 32 min | 96 | 0.2 min | 0.8 |
| 22 min | 96 | 8.0 min | 0 |
| 22 / 32 / 42 min | auto | 3.8 / 0.3 / 0 min | 1.5 / 2.4 / 2.0 |

At 22 min the fixed 128 itself leaves the trainer 3.5 min idle: there the cap, not the controller, binds.

## Decision
hook-009 adds `--async-in-flight-controller-path` (default unset: the budget is fixed). The class is built with
`args`; the rollout calls `on_submit(task)` for each group's task and `on_drain()` after each batch is taken,
reads `budget(cap)` wherever it reads the in-flight group budget, and merges `metrics()` into the step's
metrics. The fixed budget stays the upper bound.

ext-035 `miles_ext.rollout.in_flight.AutoInFlight` sets, at each drain,
`budget = ceil(1.15 x rollout_batch_size x mean group time / mean drain period)`, between one batch and the
cap, over the last 4 batches of finished groups and the last 3 drain periods. A group is timed by a done
callback on its task, so a producer blocked on a full buffer does not stretch it. The drain period includes
the trainer's own waits, so while production is short the target exceeds the budget it was measured under and
the budget climbs back to the cap. Nothing is checkpointed: a resume starts at the cap.
Metrics: `rollout/fully_async/in_flight_budget_groups`, `in_flight_group_time_s`, `in_flight_drain_period_s`.

Recipes: `async.in_flight` `fixed` (default) or `auto`; `async.max_concurrent_samples` stays the cap.

## How it is applied
`miles/utils/arguments.py` (flag) and `miles/rollout/fully_async_rollout.py` (construction, budget, submit and
drain events, metrics) carry `[fork hook-009]` markers. Verify with
`tests/fast/rollout/test_fully_async_rollout.py::test_an_in_flight_controller_sets_the_budget`,
`miles_ext/tests/test_in_flight.py` and `miles_ext/tests/test_dsv41_recipe.py::test_in_flight_modes`.

## Remove when
Upstream sizes fully async in-flight work from consumption. Upstream status: not filed.
