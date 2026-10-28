import asyncio
import logging
import os

from miles.ray.placement_group import (
    connect_training_models,  # [fork fix-022]
    create_rollout_components,
    create_training_models,
    init_training_models,  # [fork fix-022]
    maybe_start_api_server,
    publish_initial_weights,  # [fork fix-063]
    rollout_can_start_before_trainer,  # [fork fix-063]
    update_weights,
)
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.utils.arguments import parse_args, validate_async_off_policy_correction
from miles.utils.async_utils import Disposer, eager_create_task, with_disposer
from miles.utils.data import remove_rollout_data_refs, remove_train_output_refs
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.misc import should_run_periodic_action
from miles.utils.orchestration_utils import init_orchestration_script

logger = logging.getLogger(__name__)


# The framework supports other asynchronous approaches such as fully async (see miles/rollout/fully_async_rollout.py).
async def train(args, *, disposer: Disposer):
    assert not args.colocate or args.fully_async, "Colocation is only supported for async training with --fully-async."
    validate_async_off_policy_correction(args)
    _worker_manager = init_orchestration_script(args, disposer=disposer)

    # [fork fix-022] begin
    # [fork fix-059] always for fully async runs on separate GPUs with --num-rollout: the trainer needs nothing
    # from the rollout side until it connects. Colocated engines size their memory from what is free at start,
    # and with only --num-epoch the trainer waits for the rollout side to count the dataset.
    first_rollout = None  # [fork fix-063]
    if args.fully_async and not args.colocate and args.num_rollout is not None:
        rollout_starts_early = rollout_can_start_before_trainer(args)  # [fork fix-063]

        # each side registers its handles as it finishes, so a failure on the other side still disposes them
        async def start_rollout_side():
            components = await create_rollout_components(args)
            disposer.add(components[0], components[1])
            # [fork fix-063] a fresh run's engines already hold the starting policy: begin rollout 0 now
            logger.info(f"rollout 0 starts {'at engine ready' if rollout_starts_early else 'after the first sync'}")
            try:
                published = rollout_starts_early and await publish_initial_weights(args, components[0], components[1])
            except BaseException:
                # the gather below raises it only once the trainer side has finished loading
                logger.exception("publishing the engines' loaded weights failed")
                raise
            if published:

                async def generate_first():
                    await components[0].prepare_rollout(0)
                    return await components[1].get(0)

                return components, await eager_create_task(generate_first())
            return components, None

        async def start_trainer_side():
            models = await init_training_models(args)
            disposer.add(models[1], models[0])
            return models

        rollout_side, trainer_side = await asyncio.gather(
            start_rollout_side(), start_trainer_side(), return_exceptions=True
        )
        for result in (rollout_side, trainer_side):
            if isinstance(result, BaseException):
                if not isinstance(rollout_side, BaseException) and rollout_side[1] is not None:
                    rollout_side[1].cancel()  # [fork fix-063]
                raise result
        (inference_controller, rollout_executor, num_rollout_per_epoch), first_rollout = rollout_side  # [fork fix-063]
        actor_model, critic_model = trainer_side
        try:
            # [fork fix-063] a take-over may restore a later rollout yet keep start_rollout_id 0
            assert first_rollout is None or args.start_rollout_id == args.trainer_restored_rollout_id == 0, (
                f"rollout 0 started on the engines' loaded weights, but the trainer restored rollout "
                f"{args.trainer_restored_rollout_id} and starts at {args.start_rollout_id}"
            )
            await connect_training_models(args, actor_model, rollout_executor)
        except BaseException:
            if first_rollout is not None:
                first_rollout.cancel()  # [fork fix-063]
            raise
    else:
        # create the rollout manager, with sglang engines inside.
        # need to initialize rollout manager first to calculate num_rollout
        inference_controller, rollout_executor, num_rollout_per_epoch = await create_rollout_components(args)
        disposer.add(inference_controller, rollout_executor)

        # create the actor and critic models
        actor_model, critic_model = await create_training_models(args, rollout_executor)
        disposer.add(critic_model, actor_model)
    # [fork fix-022] end

    disposer.add(
        maybe_start_api_server(args, trainer_models={"actor": actor_model}, inference_controller=inference_controller)
    )
    disposer.add(rollout_executor)
    maybe_start_mini_ft_controller(args)

    # always update weight first so that sglang has the loaded weights from training.
    await update_weights(args, actor_model, rollout_executor, inference_controller)
    if first_rollout is not None:
        await inference_controller.prepare_rollout(0)  # [fork fix-063] the sync paused health monitoring

    if args.check_weight_update_equal:
        await inference_controller.check_weights(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    if args.offload_rollout:
        await inference_controller.onload_kv()

    eval_dispatcher = EvalDispatcher(args, actor_model, rollout_executor)
    disposer.add(eval_dispatcher.drain)

    if args.eval_interval is not None and args.start_rollout_id == 0 and not args.skip_eval_before_train:
        await inference_controller.prepare_eval()
        await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)

    async def save_training_model(model, rollout_id, force_sync):
        if args.use_critic and args.offload_train:
            await model.onload()
        await model.save_model(rollout_id, force_sync=force_sync)
        if args.use_critic and args.offload_train:
            await model.offload()

    async def offload_train():
        if args.use_critic:
            return
        if args.offload_train:
            await actor_model.offload()
        else:
            await actor_model.clear_memory()

    async def prepare_and_generate(rollout_id):
        await inference_controller.prepare_rollout(rollout_id)
        return await rollout_executor.get(rollout_id)

    # async train loop.
    rollout_data_next_future = first_rollout  # [fork fix-063]
    if rollout_data_next_future is None:
        rollout_data_next_future = await eager_create_task(prepare_and_generate(args.start_rollout_id))
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = await rollout_data_next_future

        has_next_rollout = rollout_id + 1 < args.num_rollout
        weight_update_due = (rollout_id + 1) % args.update_weights_interval == 0

        # A fully-async producer keeps generating without a pending get(). When
        # weights will change, defer the next drain so it uses the new version.
        defer_next_drain = args.fully_async and has_next_rollout and weight_update_due
        if args.colocate:
            await inference_controller.pause_generation(mode=args.pause_generation_mode)
            await inference_controller.flush_cache()
            if args.offload_rollout:
                await inference_controller.offload_kv()
                await inference_controller.offload_weights()
        else:
            # Start the next rollout early.
            if has_next_rollout and not defer_next_drain:
                rollout_data_next_future = await eager_create_task(prepare_and_generate(rollout_id + 1))

        if args.use_critic:
            values = await critic_model.train(rollout_id, rollout_data_curr_ref)
            if args.offload_train:
                await critic_model.offload()
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_curr_ref, external_data=values)
                if args.offload_train:
                    await actor_model.offload()
            remove_train_output_refs(values)
        else:
            await actor_model.train(rollout_id, rollout_data_curr_ref)
        remove_rollout_data_refs(args, rollout_data_curr_ref)

        external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
        if external_save or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            force_sync = external_save or rollout_id == args.num_rollout - 1
            await save_training_model(actor_model, rollout_id, force_sync)
            if args.use_critic:
                await save_training_model(critic_model, rollout_id, force_sync)
            await rollout_executor.save(rollout_id)
            if external_save:
                os.remove(args.save_trigger_sentinel)

        if args.colocate:
            await offload_train()
            if args.offload_rollout:
                await inference_controller.onload_weights()
        if weight_update_due:
            if not args.fully_async:
                # sync generate before update weights to prevent update weight in the middle of generation
                rollout_data_curr_ref = (await x) if (x := rollout_data_next_future) is not None else None
                rollout_data_next_future = None
            await update_weights(args, actor_model, rollout_executor, inference_controller, rollout_id=rollout_id)
            if defer_next_drain and not args.colocate:
                rollout_data_next_future = await eager_create_task(prepare_and_generate(rollout_id + 1))
        if args.colocate:
            if args.offload_rollout:
                await inference_controller.onload_kv()
            await inference_controller.continue_generation()
            if rollout_id + 1 < args.num_rollout:
                rollout_data_next_future = await eager_create_task(prepare_and_generate(rollout_id + 1))

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch, args.num_rollout):
            await inference_controller.prepare_eval()
            await eval_dispatcher.dispatch(rollout_id, force=rollout_id == args.num_rollout - 1)

        if (
            args.debug_exit_after_rollout is not None
            and (rollout_id - args.start_rollout_id + 1) >= args.debug_exit_after_rollout
        ):
            logger.info(
                "debug_exit_after_rollout=%d reached at rollout_id=%d, exiting",
                args.debug_exit_after_rollout,
                rollout_id,
            )
            break


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(with_disposer(train, args))
