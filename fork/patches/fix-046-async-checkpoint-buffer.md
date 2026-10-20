# fix-046: fully async saves keep buffered groups as prompts

## Why
A fully async rollout save pickles every finished group waiting in the data buffer (tokens,
log probabilities, expert-routing records) to the checkpoint directory, on the rollout event
loop. On a DeepSeek V4.1 agentic run with a 256k-token context and routing replay, that file was
about 50 GB: the first save blocked the loop for 22 min (no new trials started), and a later
save was followed seven minutes in by `ActorUnavailableError: keepalive watchdog timeout` on
`RolloutExecutor.save`, which ended the run.

## Decision
A launch flag `--async-checkpoint-buffer {groups,prompts}`:
- `groups` (Miles' default) keeps upstream behaviour: a resume trains on the saved groups.
  Upstream's restore and sample-ownership tests rely on it, so the default stays.
- `prompts` saves the buffered groups as reset prompts in the existing `running` list, so a
  resume regenerates them, the same way it already treats groups that were running. The cost is
  extra rollout after a resume (about 75 min for 16 buffered agentic groups on that run).
- Retry copies are reset on a shallow copy before the deep copy, so a save no longer copies
  generated outputs it is about to drop.
- `prompts` needs the default data buffers (a list of `DataBufferInput` as state); a custom
  buffer is refused at startup.

Our recipes always pass `prompts` (one route; ext-012). Not done here: moving the save off the
event loop. Checkpoint size and save time after the change are not yet measured on a GPU run.

## How it is applied
`miles/utils/arguments.py` (flag), `miles/rollout/fully_async_rollout.py` (`__init__` guard,
`save`, `_copy_and_reset_for_retry`), `miles_ext/recipes/recipe.py` (argv). Verify:
`tests/fast/rollout/test_fully_async_rollout_checkpoint.py::test_prompts_mode_regenerates_buffered_groups_after_restore`
(fails on the base code). On a sync, re-apply if upstream reshapes `FullyAsyncRolloutFn.save`
or the checkpoint state.

## Remove when
Upstream checkpoints the fully async buffer without the generated payloads, or off the rollout
loop. Upstream status: not filed.
