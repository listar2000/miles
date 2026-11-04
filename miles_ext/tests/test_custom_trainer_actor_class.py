from tests.fast.ray.specs.test_train import _make_args

from miles.ray.specs.train import specs_trainer


def test_a_custom_trainer_actor_class_replaces_the_backend_default():
    actor_spec, critic_spec = specs_trainer(_make_args(custom_trainer_actor_class="ext.Actor", use_critic=True))

    assert actor_spec.worker_class == critic_spec.worker_class == "ext.Actor"


def test_multi_lora_keeps_its_actor_class_over_a_custom_one():
    (actor_spec,) = specs_trainer(_make_args(custom_trainer_actor_class="ext.Actor", multi_lora=True))

    assert actor_spec.worker_class == "miles.backends.megatron_utils.lora.actor.MultiLoRATrainRayActor"


def test_without_one_the_backend_default_stays():
    (actor_spec,) = specs_trainer(_make_args())

    assert actor_spec.worker_class == "miles.backends.megatron_utils.actor.MegatronTrainRayActor"
