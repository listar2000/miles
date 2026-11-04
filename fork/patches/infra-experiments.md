# experiments: `experiments/<folder>` and the `Fork-Exp` commit kind

## Why
Experiment code kept landing in `miles_ext/`: the tmax SFT runs put their driver, data paths and eval configs
under `miles_ext/sft/tmax/`, so the next dataset would have trained through `tmax.train`. Code that one
experiment uses then reads as an API others must keep working, and syncs must carry it.

## Decision
One folder per experiment under `experiments/`, with its own commit kind so `ledger.py check` can keep an
experiment commit inside its folder. Rules: [experiments/README.md](../../experiments/README.md); the kind:
[fork/README.md](../README.md) "Kinds of change". Rejected: `miles_ext/experiments/` (it would sit inside the
package whose point is reuse), and Ext ids for experiments (they number capabilities, and an Ext commit may touch
all of `miles_ext/`).

## How it is applied
`OWN_PATHS` includes `experiments/`; `Fork-Exp: <folder>` commits touching anything else fail `ledger.py check`.
Launches that run experiment code lay `experiments/` over the image (ext-054). Nothing to do on a sync.

## Remove when
never (fork layout); upstream status: not applicable
