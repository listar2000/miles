# ext-060: single-turn graders by dotted path

## Why
STAK's grader sat in `miles_ext/tasks/` behind a `VERIFIERS` table, but experiments may import `miles_ext` and not the
reverse, so an experiment could not own its grader.

## Decision
`metadata.task` in a prompt row is the dotted path of `grade(reply, label) -> (reward, kind)`. No built-in table. STAK
moved to `experiments/dsv41_stak_overfit/`. Old rows (`"task": "stackvm"`) fail `check_row` with a clear message.
Rejected: a recipe key, since the rows already name their verifier.

## How it is applied
`miles_ext/tasks/single_turn.py`. Verify: `python -m pytest miles_ext/tests/test_single_turn.py
experiments/dsv41_stak_overfit -q`. Not yet run on a GPU; the image has `/root/miles` on `PYTHONPATH`, which is what makes
`experiments.*` importable there. Recipes that use STAK prompts need the rows from that folder
(`MILES_DATA=experiments/dsv41_stak_overfit`).

## Remove when
Miles ships a prompt-only harness (ext-038's condition); upstream status: not filed.
