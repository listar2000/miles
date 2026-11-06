"""Split placement (``cluster.placement: split``, ext-036): the trainer and the rollout engines run as two Modal
functions; see fork/patches/ext-036-split-placement.md."""

ENGINE_KEY = "split-engine:{run_id}"  # the engine's address and state, in the run's progress dict
TRAINER_KEY = "split-trainer:{run_id}"  # the trainer's heartbeat and "done", read by the engine
ENGINE_HTTP_PORT = 32000  # engine i serves HTTP on ENGINE_HTTP_PORT + i inside the engine container
SEGMENT_KEY = "split-segment:{run_id}"  # the token of the run's current segment, written by rank 0
RELAY_FAILURE_KEY = "split-relay-failure:{run_id}:{segment}"  # a trainer node's failed relay, read by rank 0
PROXY_PORT = 32999  # the session proxy in front of the engines and the sync agent; the tunnel's port


def engine_key(run_id: str) -> str:
    return ENGINE_KEY.format(run_id=run_id)


def trainer_key(run_id: str) -> str:
    return TRAINER_KEY.format(run_id=run_id)


def segment_key(run_id: str) -> str:
    return SEGMENT_KEY.format(run_id=run_id)


def relay_failure_key(run_id: str, segment: str) -> str:
    return RELAY_FAILURE_KEY.format(run_id=run_id, segment=segment)
