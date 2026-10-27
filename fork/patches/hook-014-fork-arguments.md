# hook-014: every fork flag is registered in `miles/utils/fork_arguments.py`

## Why
Fork flags sat in six of upstream's argument groups in `miles/utils/arguments.py`, each a block an upstream
sync could conflict with, and nothing showed at a glance which flags were ours. The flag prune (fix-059,
fix-060) left nine; the rule in `fork/README.md` "Flags, recipe keys and env vars" says where new ones go.

## Decision
- One module with `add_fork_arguments(parser)`, called once in `add_miles_arguments` before the
  user-provided function arguments read the partial parse. Each flag keeps its own `[fork <id>]` block.
- Validation that reads a fork flag stays in `arguments.py` next to the upstream check it extends
  (fix-004, fix-006, fix-054, fix-059): moving it would put the same upstream condition in two places.
- hook-006's `--lora-merged-sync` and `--custom-lora-spec-path` stay in `miles/utils/lora/arguments.py`:
  upstream tests build a LoRA parser with `add_lora_arguments` alone and pass them.
- Rejected: keeping the blocks where they were (every sync meets them one by one).

## How it is applied
`miles/utils/fork_arguments.py` (new) and two lines in `miles/utils/arguments.py` (the import and the
call), marked `[fork hook-014]`. On a sync, keep the call after upstream's own argument groups.

## Remove when
The fork carries no Miles flags.
