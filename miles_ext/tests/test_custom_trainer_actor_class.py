from tests.fast.ray.specs.test_train import _make_args

from miles.ray.specs.train import specs_trainer


def test_a_custom_trainer_actor_class_replaces_the_default_except_for_multi_lora():
    (default,) = specs_trainer(_make_args())
    actor, critic = specs_trainer(_make_args(custom_trainer_actor_class="ext.Actor", use_critic=True))
    (multi_lora,) = specs_trainer(_make_args(custom_trainer_actor_class="ext.Actor", multi_lora=True))

    assert actor.worker_class == critic.worker_class == "ext.Actor"
    assert default.worker_class == "miles.backends.megatron_utils.actor.MegatronTrainRayActor"
    assert multi_lora.worker_class == "miles.backends.megatron_utils.lora.actor.MultiLoRATrainRayActor"
