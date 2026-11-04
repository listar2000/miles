# hook-019: `--custom-trainer-actor-class` names the trainer Ray actor class

## Why
The SFT trainer (ext-054) runs held-out validation as a forward-only step inside the trainer actor, which needs a
`MegatronTrainRayActor` subclass. Miles picks the class from a private table in `miles/ray/specs/train.py`
(`_TRAINER_ACTOR_CLASSES`) with no flag, so the extension had to assign into that table.

## Decision
A fork flag (`miles/utils/fork_arguments.py`) holding an import path, used in place of the backend's default
class. Multi-LoRA keeps its own class. Rejected: assigning into `_TRAINER_ACTOR_CLASSES` (a private name, and
only effective in the driver process before the actors are created).

## How it is applied
Two marked spots: the flag in `fork_arguments.py`, and the `worker_class` choice in `miles/ray/specs/train.py`.
Verify: `miles_ext/tests/test_sft_config.py` checks the SFT argv names the class; an SFT run logs `val/nll`.
On a sync: re-apply in `_compute_spec_trainer`'s `worker_class` if upstream changes that expression.

## Remove when
upstream Miles lets a run choose its trainer actor class; upstream status: not filed
