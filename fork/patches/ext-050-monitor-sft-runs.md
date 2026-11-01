# ext-050: SFT runs on the monitor, and the evals of their checkpoints

## Why
An SFT run (trainer only: no engines, no trials) registered with the monitor (ext-039) showed as
an empty RL run: its loss was kept off the curves, learning rate, throughput and held-out NLL
were not read, and nothing tied an eval of a checkpoint back to the run that produced it.

## Decision
- A manifest's `kind: "sft"` makes the run type `sft`. This is the one case where `groups.py`
  reads the registry's kind (RL benches were registered `train`, so kind is not trusted
  otherwise); a `groups.json` rule still wins.
- The eval owns the link: an eval run's manifest or `runs.json` entry says
  `"evaluates": {"run", "step", "label"}`, and the server derives each training run's list of
  evals from that. The training run stores nothing, so there is no second copy to keep in sync.
- Step numbering: `step` in `evaluates` and N in `validation at step N` count optimizer
  steps done (0 = base model / before training). Miles' `step N:` lines count from 0, so
  validation after N steps belongs to step line N - 1, and the curve point sits at that
  line's time, which is where the step axis puts it.
- pass@k is the unbiased estimator over tasks with at least k finished trials, with k = the
  eval's `n_samples` (else the most trials of any task); reward 1 counts as solved.
- Rejected: listing evals in the SFT run's manifest (two owners of one fact); a separate
  step axis for validation points (one axis keeps loss and val NLL on one chart).

## How it is applied
`miles_ext/monitor/`: `server.py` (log keys, `VAL_LINE`, `HF_LINE`, SFT stages and signals,
`pass_at`, `link_evals`), `groups.py`, `metrics.py`, `register.py` (`--kind`, `--evaluates`,
recipe optional), the page (`app.js`, `pages.js`, `curves.js`, `index.html`) and the README's
"SFT runs" section. Tests: `python -m pytest miles_ext/tests/test_monitor.py`.

## Remove when
The monitor is retired, or SFT runs report through a channel other than the launch log.
