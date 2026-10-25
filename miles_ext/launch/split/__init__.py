"""Split placement (``cluster.placement: split``, ext-036): the trainer and the rollout engines run as two Modal
functions; see fork/patches/ext-036-split-placement.md."""

ENGINE_KEY = "split-engine:{run_id}"  # the engine's address and state, in the run's progress dict
TRAINER_KEY = "split-trainer:{run_id}"  # the trainer's heartbeat and "done", read by the engine
ENGINE_HTTP_PORT = 32000  # engine i serves HTTP on ENGINE_HTTP_PORT + i inside the engine container
PROXY_PORT = 32999  # the session proxy in front of the engines and the sync agent; the tunnel's port


def engine_key(run_id: str) -> str:
    return ENGINE_KEY.format(run_id=run_id)


def trainer_key(run_id: str) -> str:
    return TRAINER_KEY.format(run_id=run_id)
