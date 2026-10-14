"""Reward hook (``--custom-rm-path miles_ext.harbor.reward.reward_func``).

Harbor's verifier already scored the episode inside the agent function, which
stored the verdict in ``sample.metadata["reward"]``.
"""


async def reward_func(args, samples, **kwargs):
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)
