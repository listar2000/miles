"""Modal functions for split (disaggregated) Miles runs: argv check, one-container split, always-on baseline.

    modal run -m miles_ext.disagg.app --mode check
    modal run --detach -m miles_ext.disagg.app --mode single --run-id smoke-split-1
"""

import json
import os
import time
import uuid

import modal

from miles_ext import stack
from miles_ext.disagg import argv as A

# main-23fdb78c74b3, the main build this branch starts from
IMAGE = f"{stack.IMAGE_REPO}@sha256:19173e7534381f1a5f1678bba33c94d5ce797ba292a57c3a62bca26512837fd0"
# i6pn only links containers in one region, so cross-app runs pin a cloud and region; DISAGG_PIN=0 lifts it.
_PIN = (os.environ.get("DISAGG_CLOUD", "aws"), os.environ.get("DISAGG_REGION", "us-west"))
CLOUD, REGION = _PIN if os.environ.get("DISAGG_PIN", "1") == "1" else (None, None)
STATE = "/state"
HOURS = 3600
TRAINER_POOL = f"trainer-controller-{A.TRAINER_ID}"
INFERENCE_CONTROLLER_POOL = "inference-controller"
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPSTREAM_OVERLAY = (
    "train_async.py",
    "miles/utils/arguments.py",
    "miles/backends/training_utils/weight_update/protocols/broadcast.py",
    "miles/utils/workers/addr_allocator.py",
)
# GPUs per app are bound when Modal defines the functions; the container reads them back from the image env
SHAPE_ENV = "MILES_DISAGG_SHAPE"
SHAPE = json.loads(os.environ.get(SHAPE_ENV) or '{"rollout_gpus": 1, "trainer_gpus": 1}')

app = modal.App(os.environ.get("DISAGG_APP", "miles-disagg"))
models = modal.Volume.from_name("miles-disagg-models", version=2)
state = modal.Volume.from_name("miles-disagg-state", create_if_missing=True, version=2)
secrets = [modal.Secret.from_dotenv(os.environ.get("MILES_DOTENV_DIR", "/data/home/listar2000/modal-miles-launch"))]
base_image = (
    modal.Image.from_registry(IMAGE)
    .entrypoint([])
    .env({"PYTHONUNBUFFERED": "1", "NCCL_NVLS_ENABLE": "0", "PYTHONPATH": "/root/miles", SHAPE_ENV: json.dumps(SHAPE)})
)


def with_code(img: modal.Image) -> modal.Image:
    """This branch's miles_ext and its hook/fix files over the image's copy of the same base commit (added last)."""
    img = img.add_local_dir(f"{_ROOT}/miles_ext", "/root/miles/miles_ext", ignore=["**/__pycache__", "tests"])
    for f in UPSTREAM_OVERLAY:
        img = img.add_local_file(f"{_ROOT}/{f}", f"/root/miles/{f}")
    return img


image = with_code(base_image)
volumes = {A.MODELS: models, STATE: state}


@app.function(image=image, secrets=secrets, volumes=volumes, cpu=4, memory=16384, timeout=1800)
def check(cfg: dict) -> dict:
    """Parse every deployment's argv with Miles' own parser (CPU): validators run, nothing starts."""
    import sys

    from miles.utils.arguments import parse_args

    cfg = A.RunConfig.of(cfg)
    out = {}
    for component, kwargs in {
        A.TRAINER: {"mooncake_master": "[fd00::1]:50051"},
        A.PRIMARY: {"mooncake_master": "[fd00::1]:50051", "trainer_addr": "[fd00::2]:20000"},
        A.INFERENCE: {"mooncake_master": "[fd00::1]:50051", "inference_controller_addr": "[fd00::1]:20001"},
    }.items():
        sys.argv = ["miles", *A.build(cfg, component, **kwargs)]
        try:
            parse_args()
            out[component] = "ok"
        except (Exception, SystemExit) as e:  # an argparse exit is a result here
            out[component] = f"{type(e).__name__}: {e}"[:2000]
    return out


@app.function(
    image=image,
    secrets=secrets,
    volumes=volumes,
    gpu="H100:2",
    cpu=16,
    memory=131072,
    cloud=CLOUD,
    region=REGION,
    timeout=2 * HOURS,
)
def single(cfg: dict) -> dict:
    """Primary, inference and trainer deployments as three Ray clusters in this one container."""
    from miles_ext.disagg import procs as P

    cfg = A.RunConfig.of(cfg)
    wb = A.wandb_env(cfg, "single")
    t0 = time.time()
    marks = {}
    host = "127.0.0.1"
    clusters = [
        P.RayCluster("trainer", 0, host, gpus=(0,)),
        P.RayCluster("primary", 1, host),
        P.RayCluster("inference", 2, host, gpus=(1,)),
    ]
    drivers = []
    moon = P.start_mooncake_master()
    try:
        for c in clusters:
            c.start()
        mooncake = A.host_port(host, P.MOONCAKE_PORT)
        entry = ["python", "-m", "miles.utils.workers.deployment_entrypoint"]
        trainer = P.Driver(
            "trainer", entry + A.build(cfg, A.TRAINER, mooncake_master=mooncake), {**clusters[0].env(), **wb}
        ).start()
        drivers.append(trainer)
        trainer_addr = A.host_port(*clusters[0].pool_addr(TRAINER_POOL, driver=trainer))
        marks["trainer_addr"] = time.time() - t0
        primary = P.Driver(
            "primary",
            ["python", "-m", "miles_ext.disagg.run_exit", "train_async.py"]
            + A.build(cfg, A.PRIMARY, mooncake_master=mooncake, trainer_addr=trainer_addr),
            {**clusters[1].env(), **wb},
        ).start()
        drivers.append(primary)
        ic_addr = A.host_port(*clusters[1].pool_addr(INFERENCE_CONTROLLER_POOL, driver=primary))
        marks["inference_controller_addr"] = time.time() - t0
        drivers.append(
            P.Driver(
                "inference",
                entry + A.build(cfg, A.INFERENCE, mooncake_master=mooncake, inference_controller_addr=ic_addr),
                {**clusters[2].env(), **wb},
            ).start()
        )
        rc = primary.proc.wait()
        marks["primary_exit"] = time.time() - t0
        return {"returncode": rc, "marks": marks}
    finally:
        for d in reversed(drivers):
            d.stop()
        for c in clusters:
            c.stop()
        moon.terminate()
        state.commit()


@app.function(
    image=image,
    secrets=secrets,
    volumes=volumes,
    gpu=f"H100:{SHAPE['rollout_gpus'] + SHAPE['trainer_gpus']}",
    cpu=16,
    memory=262144,
    cloud=CLOUD,
    region=REGION,
    timeout=2 * HOURS,
)
def baseline(cfg: dict) -> dict:
    """The always-on run, in one app (train_async on one Ray cluster holding engines and trainer)."""
    from miles_ext.disagg import procs as P

    cfg = A.RunConfig.of(cfg)
    n = SHAPE["rollout_gpus"] + SHAPE["trainer_gpus"]
    c = P.RayCluster("all", 0, "127.0.0.1", gpus=tuple(range(n)))
    t0 = time.time()
    try:
        c.start()
        drv = P.Driver(
            "all",
            ["python", "-m", "miles_ext.disagg.run_exit", "train_async.py"] + A.build(cfg, A.ALL),
            {**c.env(), **A.wandb_env(cfg, "always-on")},
        ).start()
        rc = drv.proc.wait()
        return {"returncode": rc, "wall_s": time.time() - t0}
    finally:
        c.stop()
        state.commit()


@app.local_entrypoint()
def main(mode: str = "check", run_id: str = "", num_rollout: int = 3, bench: bool = False, replay: str = ""):
    run_id = run_id or f"smoke-split-{time.strftime('%m%d-%H%M%S')}"
    if replay:
        cfg = A.replay(run_id, uuid.uuid4().hex[:16], replay).as_dict()
    elif bench:
        cfg = A.bench(run_id, uuid.uuid4().hex[:16]).as_dict()
    else:
        cfg = A.RunConfig(run_id=run_id, run_uuid=uuid.uuid4().hex[:16], num_rollout=num_rollout).as_dict()
    fn = {
        "check": lambda: check.remote(cfg),
        "single": lambda: single.remote(cfg),
        "baseline": lambda: baseline.remote(cfg),
    }[mode]
    print(json.dumps(fn(), indent=1, default=str))
