# hook-008: `--async-checkpoint-buffer log`, a custom buffer stores and saves the buffered groups

## Why
The write-ahead group log (ext-034) and fix-046's prompts mode handle the buffered groups in different ways.
The group log keeps every group, written once as it enters the buffer, and a resume trains on them. Prompts mode
keeps only the prompts, and a resume regenerates them. Running the group log under the flag value `groups` hid
that difference: the value claims the save pickles whole groups, which the log never does. With prompts mode
kept as a fallback, the user asked for three modes (2026-10-20).

## Decision
`--async-checkpoint-buffer` has a third value, `log`. It declares that the `--custom-async-data-buffer-path`
buffer stores each group itself, and that its `state_dict` is the saved buffer state. Miles refuses `log`
without a custom buffer, and the group log refuses any other value. fix-046's guard now rejects a custom
buffer only under `prompts`. A save made under `log` holds the custom buffer's own state, so a resume under
another mode is refused with a clear error; a resume under `log` reads `groups` and `prompts` saves. Recipes pick the mode with `async.checkpoint_buffer` (default `log`; `prompts`
is the fallback, `groups` is upstream's default).

## How it is applied
`miles/utils/arguments.py` adds the choice and its help text. `miles/rollout/fully_async_rollout.py`
holds the two `__init__` guards and the `load` guard. Both carry `[fork hook-008]` markers. Verify with
`miles_ext/tests/test_dsv41_recipe.py::test_checkpoint_buffer_modes` and `miles_ext/tests/test_group_log.py`.
On a sync, keep the choice next to fix-046's flag.

## Remove when
ext-034 and fix-046 are both removed. Upstream status: not filed.
