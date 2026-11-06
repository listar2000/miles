"""``python -m miles_ext.node <role> ...``: RFC 0005's entry commands.

spec --recipe <yaml> --run-id I [--num-rollout N]  write the spec for a recipe file to stdout
trainer|rollout|check|manifest --spec <file>       the spec is {"run_id": ..., "recipe": {...}}
prepare-weights --hf-repo R --revision C --path P  base weights from the HF Hub
convert-fsdp --step N --spec <file>                FSDP's save to HF weights
sft-prepare --config <yaml> [--tests]              SFT records to token shards
sft-train --config <yaml> --run-id I --num-rollout N --save-interval M
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

from miles_ext.node import contract as C


def load_spec(path: str) -> tuple[dict, str]:
    from miles_ext.recipes import recipe as R

    spec = json.loads(Path(path).read_text())
    return R.from_dict(spec["recipe"]), spec["run_id"]


def _require(recipe: dict, role: str) -> None:
    if missing := C.missing(recipe, role, os.environ):
        sys.exit(
            f"{role}: the caller did not provide: {', '.join(missing)} (fork/rfcs/0005-provider-agnostic-image.md)"
        )
    from miles_ext.node import state as node_state
    from miles_ext.node import storage as node_storage
    from miles_ext.recipes import recipe as R

    if (os.environ.get(C.STATE_ENV) or node_state.FILE) == node_state.FILE and not node_storage.for_mount(
        R.RESULTS
    ).live:
        sys.exit(f"{role}: {C.STATE_ENV}={node_state.FILE} needs live {R.RESULTS} storage: a caller could not read it")


def _log(run_id: str, name: str):
    from miles_ext.node import runtime
    from miles_ext.node import storage as node_storage
    from miles_ext.recipes import recipe as R

    return runtime.node_log(R.RESULTS, run_id, name, node_storage.for_mount(R.RESULTS))


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m miles_ext.node", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    roles = ap.add_subparsers(dest="role", required=True)
    spec = roles.add_parser("spec")
    spec.add_argument("--recipe", required=True)
    spec.add_argument("--run-id", required=True)
    spec.add_argument("--num-rollout", type=int, default=0)
    for role in ("trainer", "rollout", "manifest"):
        roles.add_parser(role).add_argument("--spec", required=True)
    check = roles.add_parser("check")
    check.add_argument("--spec", required=True)
    check.add_argument("--no-gpu", action="store_true", help="check on a node with no GPU, with a stub libcuda")
    weights = roles.add_parser("prepare-weights")
    weights.add_argument("--hf-repo", required=True)
    weights.add_argument("--revision", required=True)
    weights.add_argument("--path", required=True)
    convert = roles.add_parser("convert-fsdp")
    convert.add_argument("--spec", required=True)
    convert.add_argument("--step", type=int, required=True)
    prep = roles.add_parser("sft-prepare")
    prep.add_argument("--config", required=True)
    prep.add_argument("--tests", action="store_true")
    train = roles.add_parser("sft-train")
    train.add_argument("--config", required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--num-rollout", type=int, required=True)
    train.add_argument("--save-interval", type=int, required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    # a worker or a rollout node commits on its way out; rank 0's supervisor installs its own handler
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))

    if a.role == "spec":
        from miles_ext.recipes import recipe as R

        overrides = {"algorithm": {"num_rollout": a.num_rollout}} if a.num_rollout else None
        print(json.dumps({"run_id": a.run_id, "recipe": R.load(a.recipe, overrides)}, indent=1))
    elif a.role == "manifest":
        recipe, run_id = load_spec(a.spec)
        print(json.dumps(C.manifest(recipe, run_id), indent=1))
    elif a.role == "trainer":
        from miles_ext.node import trainer

        recipe, run_id = load_spec(a.spec)
        _require(recipe, a.role)
        with _log(run_id, f"trainer-{os.environ[C.NODE_RANK_ENV]}"):
            trainer.run(recipe, run_id)
    elif a.role == "rollout":
        from miles_ext.node import rollout

        recipe, run_id = load_spec(a.spec)
        _require(recipe, a.role)
        with _log(run_id, f"rollout-{socket.gethostname()}"):
            rollout.run(recipe, run_id)
    elif a.role == "check":
        from miles_ext.platform.validate import run_deep_check

        recipe, run_id = load_spec(a.spec)
        result = run_deep_check(recipe, run_id, libcuda_stub=a.no_gpu)
        print(json.dumps(result, indent=1, default=str))
        return 1 if result["errors"] else 0
    elif a.role == "prepare-weights":
        from miles_ext.node import storage as node_storage
        from miles_ext.platform.weights_job import download
        from miles_ext.recipes import recipe as R

        print(json.dumps(download(a.hf_repo, a.revision, a.path)))
        node_storage.for_mount(R.MODELS).commit()
    elif a.role == "convert-fsdp":
        from miles_ext.node import storage as node_storage
        from miles_ext.platform.checkpoints import finalize_hf_command
        from miles_ext.recipes import recipe as R

        recipe, run_id = load_spec(a.spec)
        results = node_storage.for_mount(R.RESULTS)
        node_storage.for_mount(R.MODELS).refresh()
        results.refresh()
        subprocess.run(finalize_hf_command(run_id, a.step, str(R.model_dir(recipe))), check=True)
        results.commit()
    elif a.role == "sft-prepare":
        from miles_ext.node import storage as node_storage
        from miles_ext.sft import config as sft_config
        from miles_ext.sft import job

        node_storage.for_mount(sft_config.DATA_MOUNT).refresh()
        job.prepare(Path(a.config).read_text(), a.tests, node_storage.for_mount(sft_config.DATA_MOUNT).commit)
    elif a.role == "sft-train":
        from miles_ext.node import storage as node_storage
        from miles_ext.sft import config as sft_config
        from miles_ext.sft import job

        node_storage.for_mount(sft_config.DATA_MOUNT).refresh()
        node_storage.for_mount(sft_config.RESULTS_MOUNT).refresh()
        job.train(
            Path(a.config).read_text(),
            a.run_id,
            a.num_rollout,
            a.save_interval,
            node_storage.for_mount(sft_config.RESULTS_MOUNT).commit,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
