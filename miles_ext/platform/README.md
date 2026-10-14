# miles_ext/platform (reserved)

Platform-facing extensions go here: the run-spec schema and its JSON Schema
export, `validate(spec)` (resolve to argv, run Miles' `parse_args`, return
structured errors; it runs in a CPU-only container of the fork image), an
importable launcher library with no env reads at import, deterministic Modal
naming and idempotent `teardown(run_id)`, the rank-0 supervisor (signals,
lease, deadline save via `--save-trigger-sentinel`, progress from MetricEvents)
and `observe(run_id)`, structured exit records, and per-run secrets.

Requirements come from the "Platform to trainer interface" design. All of
them are extensions; anything that would need a change inside Miles' own code
becomes a Hook commit (fork/README.md).
