<!--
Title: `type(scope): what`, the same as the commit subject (e.g. `ext(harbor): ...`, `fix(ray): ... [fork fix-NNN]`).
Stacked on another PR? Say so on the first line: "Stacked on #N."
Write for someone who has not seen the chat: no "requested by" header, no generated-by footer,
and no links to claude.ai artifacts or other internal pages. Link GitHub, W&B or Linear when evidence lives there.
Keep Outcome, Contract and Done; delete Side-effects or Verification when they would only say "none".
-->

## Outcome

<!-- The end state after merge, understandable without reading the diff. -->

## Contract

<!--
Every new or changed thing callers must follow: the miles_ext.platform API, recipe or run-spec
fields and their defaults, image refs, Modal names, teardown, config keys, persisted layouts.
Write "No caller-facing changes." when there are none.
-->

## Side-effects

<!--
Deletions, renames, behaviour differences, and open PRs this affects (check the open PR list).
Upstream files touched, with their [fork fix-NNN] / [fork hook-NNN] markers.
-->

## Done

<!-- The implementation changes, grouped by area. -->

## Verification

<!--
The exact commands and their results, e.g. `python fork/tools/ledger.py check`,
`python -m pytest miles_ext/tests -q` (N passed), `--check-only` dry checks, GPU run ids.
-->
