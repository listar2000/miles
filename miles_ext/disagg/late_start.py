"""Pre-trainer hook (--custom-pre-trainer-hook-path, hook-003): start rollout 0 before the split trainer exists.

On a fresh run the engines already serve --hf-checkpoint, the weights the trainer starts from, so rollout 0 needs no
trainer: this admits the engines to the router, publishes weight version 0 (the engines are launched with
--sglang-weight-version 0) and the FSDP data split the trainer will report, starts rollout 0, then waits for the
trainer's address (env ``MILES_DISAGG_DICT``, key ``trainer_addr``, written by the trainer app) and hands it over.
"""

import asyncio
import logging
import os
import time

from miles.backends.training_utils.checkpoint.tracker import read_checkpoint_tracker_iteration
from miles.utils.async_utils import eager_create_task

logger = logging.getLogger(__name__)

DICT_ENV = "MILES_DISAGG_DICT"
TRAINER_ADDR_KEY = "trainer_addr"
EVENTS_KEY = "events"
TRAINER_ID = "actor"


def fsdp_parallel_config(args) -> dict:
    assert args.train_backend == "fsdp", "the late-start hook knows the data split of FSDP trainers only"
    return {"dp_size": args.actor_num_nodes * args.actor_num_gpus_per_node}


async def admit_engines(inference_controller) -> None:
    """Engines join the router only when a weight update ends; they already serve the starting weights, so end
    an update that pushes nothing."""
    info = await inference_controller.start_update_weights()
    await inference_controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)


def mark(d, event: str, **extra) -> None:
    """Append a timestamped event to the run's Dict (read by the conductor and the benchmark)."""
    d[f"{EVENTS_KEY}/{event}"] = {"t": time.time(), **extra}


async def start_rollout_before_trainer(args, inference_controller, rollout_executor):
    import modal

    if read_checkpoint_tracker_iteration(args.requested_load) is not None:
        logger.info("late-start hook: resuming from a checkpoint, keeping the default order")
        return None
    d = modal.Dict.from_name(os.environ[DICT_ENV])
    await admit_engines(inference_controller)
    await rollout_executor.set_train_parallel_config(fsdp_parallel_config(args))
    await rollout_executor.set_weight_version(0)

    async def rollout_0():
        await inference_controller.prepare_rollout(0)
        return await rollout_executor.get(0)

    task = await eager_create_task(rollout_0())
    mark(d, "rollout0_started")
    logger.info("late-start hook: rollout 0 started; waiting for the trainer's address")
    while (addr := await d.get.aio(TRAINER_ADDR_KEY)) is None:
        if task.done() and task.exception() is not None:
            raise task.exception()
        await asyncio.sleep(2)
    args.trainer_controller_addrs = [f"{TRAINER_ID}={addr}"]
    mark(d, "trainer_addr_seen", addr=addr)
    logger.info(f"late-start hook: trainer at {addr}")
    return task
