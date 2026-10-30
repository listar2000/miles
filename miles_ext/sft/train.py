"""SFT driver: Miles' train.py loop for ``--debug-train-only`` runs, plus held-out NLL.

No inference engines start. Each rollout is one batch of pre-rendered trajectories
(``miles_ext.sft.rollout``) and one optimizer step. A validation pass
(``miles_ext.sft.val``) runs before the first step and every ``MILES_SFT_VAL_INTERVAL``
steps (default 100), and after the last one.

    python -m miles_ext.sft.train <Miles args>
"""

import asyncio
import logging
import os

from miles.ray.placement_group import create_rollout_components, create_training_models
from miles.utils.arguments import parse_args
from miles.utils.async_utils import Disposer, with_disposer
from miles.utils.data import remove_rollout_data_refs
from miles.utils.misc import should_run_periodic_action
from miles.utils.orchestration_utils import init_orchestration_script
from miles.ray.specs import train as train_specs
from miles_ext.sft.ids import ACTOR_CLASS, VAL_BASE

logger = logging.getLogger(__name__)


async def train(args, *, disposer: Disposer):
    assert args.debug_train_only, "the SFT driver runs without inference engines (--debug-train-only)"
    assert args.train_backend == "megatron"
    train_specs._TRAINER_ACTOR_CLASSES["megatron"] = ACTOR_CLASS  # the trainer that answers validation ids
    init_orchestration_script(args, disposer=disposer)
    inference_controller, rollout_executor, num_rollout_per_epoch = await create_rollout_components(args)
    disposer.add(inference_controller, rollout_executor)
    actor_model, critic_model = await create_training_models(args, rollout_executor)
    disposer.add(critic_model, actor_model)
    val_interval = int(os.environ.get("MILES_SFT_VAL_INTERVAL", "100"))

    async def validate(step: int):
        pack = await rollout_executor.get(VAL_BASE + step)
        await actor_model.train(VAL_BASE + step, pack)
        remove_rollout_data_refs(args, pack)

    async def save(rollout_id: int, force_sync: bool = False):
        await rollout_executor.save(rollout_id)
        await actor_model.save_model(rollout_id, force_sync=force_sync or rollout_id == args.num_rollout - 1)

    if args.start_rollout_id == 0 and val_interval > 0:
        await validate(0)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        pack = await rollout_executor.get(rollout_id)
        await actor_model.train(rollout_id, pack)
        remove_rollout_data_refs(args, pack)
        step = rollout_id + 1
        external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
        if external_save or should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch,
                                                       args.num_rollout):
            await save(rollout_id, force_sync=external_save)
            if external_save:
                os.remove(args.save_trigger_sentinel)
        if val_interval > 0 and (step % val_interval == 0 or step == args.num_rollout):
            await validate(step)


if __name__ == "__main__":
    asyncio.run(with_disposer(train, parse_args()))
