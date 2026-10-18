"""Per-step trainer driver (the primary): rollout stays up while a fresh trainer deployment serves each step.

    python -m miles_ext.disagg.ondemand <miles args> --deploy-component primary ...

Per step k: wait for batch k, ask the conductor for a trainer (Dict ``want/<k>``), take it over (it loads model and
Adam from --load), train, save, have the trainer app commit the volume, push weights, release it. Saving before the
push makes a trainer that dies mid-step restart from the last saved step and retrain the same batch: an update is
applied once. Timings per step go to the Dict (``steps/<k>``) and stdout.
"""

import asyncio
import logging
import os
import time

import modal
from miles_ext.disagg.late_start import DICT_ENV, TRAINER_ID, admit_engines, fsdp_parallel_config, mark
from miles_ext.disagg.medic import RESTART_KEY, RESTARTED_KEY

from miles.backends.megatron_utils.megatron_config import compute_trainer_args
from miles.ray.placement_group import (
    create_rollout_components,
    create_trainer_handles,
    create_training_model,
    take_over_trainers,
)
from miles.ray.specs.train import compute_trainer_configs
from miles.utils.arguments import parse_args, validate_async_off_policy_correction
from miles.utils.async_utils import Disposer, eager_create_task, with_disposer
from miles.utils.data import remove_rollout_data_refs
from miles.utils.orchestration_utils import init_orchestration_script

logger = logging.getLogger(__name__)

TRAINER_ADDR_KEY = "trainer_addr/{step}"  # written by the trainer app serving step <step>
WANT_KEY, COMMIT_KEY, COMMITTED_KEY = "want/{}", "commit/{}", "committed/{}"
RELEASE_KEY, STEP_KEY, LOST_KEY = "release/{}", "steps/{}", "lost/{}"
ADDR_TIMEOUT_S = 1200
ENGINE_RESTART_TIMEOUT_S = 900
MAX_ATTEMPTS = 3


class TrainerLost(Exception):
    pass


def superseded(want: dict | None, attempt: int) -> bool:
    """The driver has moved on from ``attempt`` to a later attempt of the same step."""
    return want is not None and want["attempt"] > attempt


async def _await_key(d, key: str, *, lost: str | None = None, timeout: float | None = None, poll: float = 2.0):
    deadline = None if timeout is None else time.monotonic() + timeout
    while (value := await d.get.aio(key)) is None:
        if lost is not None and (why := await d.get.aio(lost)) is not None:
            raise TrainerLost(why)
        if deadline is not None and time.monotonic() > deadline:
            raise TrainerLost(f"no {key} after {timeout:.0f}s")
        await asyncio.sleep(poll)
    return value


async def _or_lost(d, lost: str, coro, poll: float = 2.0):
    """Await ``coro`` unless the conductor reports this attempt's trainer lost first (an rpc to a dead trainer can
    hang)."""
    task = asyncio.ensure_future(coro)
    while not task.done():
        if (why := await d.get.aio(lost)) is not None:
            task.cancel()
            raise TrainerLost(why)
        await asyncio.wait({task}, timeout=poll)
    return task.result()


async def _push(handle, rollout_executor, inference_controller, *, step: int) -> None:
    """placement_group.update_weights, except for the version: each step's trainer is a new process that numbers its
    pushes from 1, so the run's version (engines and rollout executor, which judge staleness by it) is set to
    ``step + 1``, the version an always-on trainer publishes after training step ``step``."""
    version = step + 1
    info = await inference_controller.start_update_weights()
    try:
        await handle.update_weights(info=info, rollout_id=step)
        await asyncio.gather(*(e.update_weight_version(weight_version=str(version)) for e in info.rollout_engines))
    except BaseException:
        await inference_controller.abort_update_weights()
        raise
    await inference_controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)
    await rollout_executor.set_weight_version(version)


async def _restart_engines(d, inference_controller, *, tag: str) -> None:
    """A push cut off midway leaves every engine in it waiting on a broken NCCL group: have the medic restart
    them, then wait until they are back (they serve again after the retried step's push)."""
    logger.warning(f"weight push {tag} interrupted; restarting the engines")
    await d.put.aio(RESTART_KEY + tag, time.time())
    await _await_key(d, RESTARTED_KEY + tag, timeout=ENGINE_RESTART_TIMEOUT_S)
    await inference_controller.wait_expected_num_cells()


def _load_from_own_checkpoints(args) -> None:
    """parse_args resolved the load once, before any checkpoint existed: --load fell back to the HF weights with no
    optimizer or RNG state. From step 1 the trainer must restore this run's own checkpoints from --save."""
    args.load = args.requested_load
    args.no_load_optim = args.no_load_rng = args.finetune = False


async def _train_step(args, d, *, step: int, data, rollout_executor, inference_controller, attempt: int) -> dict:
    t: dict = {"step": step, "attempt": attempt, "asked": time.time()}
    await d.put.aio(WANT_KEY.format(step), {"attempt": attempt, "t": t["asked"]})
    lost = LOST_KEY.format(step) + f"/{attempt}"
    addr = await _await_key(d, TRAINER_ADDR_KEY.format(step=step) + f"/{attempt}", lost=lost, timeout=ADDR_TIMEOUT_S)
    t["addr_seen"] = time.time()
    args.trainer_controller_addrs = [f"{TRAINER_ID}={addr}"]
    args.start_rollout_id = None
    if step > 0:
        _load_from_own_checkpoints(args)
    [config] = compute_trainer_configs(args)
    handles = create_trainer_handles(args, trainer_configs=[config])
    resumed = await _or_lost(d, lost, take_over_trainers(args, handles=handles))
    handle = handles[config.trainer_id]
    info = await _or_lost(
        d,
        lost,
        create_training_model(
            compute_trainer_args(args, config), handle=handle, trainer_id=config.trainer_id, resumed=resumed
        ),
    )
    t["loaded"] = time.time()
    t["restored_rollout_id"] = info.restored_rollout_id
    await rollout_executor.set_train_parallel_config(await handle.get_train_parallel_config())
    if info.restored_rollout_id == step + 1:  # a lost attempt already saved this step: push it, don't retrain
        t["resumed_after_save"] = True
    else:
        assert info.restored_rollout_id == step, (
            f"the trainer for step {step} restored rollout {info.restored_rollout_id}: the checkpoint and the step "
            f"counter disagree, so training on this batch would skip or repeat an update"
        )
        await _or_lost(d, lost, handle.train(step, data))
        t["trained"] = time.time()
        await _or_lost(d, lost, handle.save_model(step, force_sync=True))
        await rollout_executor.save(step)
        t["saved"] = time.time()
        await d.put.aio(COMMIT_KEY.format(step), attempt)
        await _await_key(d, COMMITTED_KEY.format(step), lost=lost)
        t["committed"] = time.time()
    try:
        await _or_lost(d, lost, _push(handle, rollout_executor, inference_controller, step=step))
    except Exception:
        await _restart_engines(d, inference_controller, tag=f"s{step}a{attempt}")
        raise
    t["synced"] = time.time()
    await d.put.aio(RELEASE_KEY.format(step), True)
    return t


async def run(args, *, disposer: Disposer) -> None:
    validate_async_off_policy_correction(args)
    _worker_manager = init_orchestration_script(args, disposer=disposer)
    d = modal.Dict.from_name(os.environ[DICT_ENV])
    inference_controller, rollout_executor, _ = await create_rollout_components(args)
    disposer.add(inference_controller, rollout_executor)
    await admit_engines(inference_controller)
    await rollout_executor.set_train_parallel_config(fsdp_parallel_config(args))
    await rollout_executor.set_weight_version(0)

    async def generate(rollout_id: int):
        await inference_controller.prepare_rollout(rollout_id)
        return await rollout_executor.get(rollout_id)

    next_batch = await eager_create_task(generate(0))
    mark(d, "rollout0_started")
    for step in range(args.num_rollout):
        t_wait = time.time()
        data = await next_batch
        rollout_s = time.time() - t_wait
        for attempt in range(MAX_ATTEMPTS):
            try:
                t = await _train_step(
                    args,
                    d,
                    step=step,
                    data=data,
                    rollout_executor=rollout_executor,
                    inference_controller=inference_controller,
                    attempt=attempt,
                )
                break
            except Exception as e:  # a lost trainer is retried from the last saved step
                logger.warning(f"step {step} attempt {attempt}: trainer lost ({type(e).__name__}: {e}); retrying")
                mark(d, f"step{step}_attempt{attempt}_lost", error=f"{type(e).__name__}: {e}"[:500])
        else:
            raise RuntimeError(f"step {step}: no trainer finished it in {MAX_ATTEMPTS} attempts")
        t["rollout_wait_s"] = rollout_s
        await d.put.aio(STEP_KEY.format(step), t)
        logger.info(f"on-demand step {step}: {t}")
        remove_rollout_data_refs(args, data)
        if step + 1 < args.num_rollout:
            next_batch = await eager_create_task(generate(step + 1))


if __name__ == "__main__":
    asyncio.run(with_disposer(run, parse_args()))
