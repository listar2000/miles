"""The image's node entry (RFC 0005) with no provider SDK in the interpreter."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles_ext.node import __main__ as node_main
from miles_ext.node import contract
from miles_ext.node import storage as node_storage
from miles_ext.recipes import recipe as R

CONFIGS = Path(__file__).resolve().parents[1] / "recipes" / "configs"

NO_MODAL = "import sys; sys.modules['modal'] = None\n"


def _recipe(name: str, monkeypatch) -> dict:
    monkeypatch.setenv("MILES_DATA", "/data")
    return R.load(CONFIGS / f"{name}.yaml")


def _spec(tmp_path, recipe: dict) -> Path:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"run_id": "run1", "recipe": recipe}))
    return path


def test_manifest_runs_without_modal(tmp_path, monkeypatch):
    spec = _spec(tmp_path, _recipe("qwen35-9b-swegym8-sail", monkeypatch))
    code = (
        NO_MODAL + f"from miles_ext.node.__main__ import main; sys.exit(main(['manifest', '--spec', {str(spec)!r}]))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    manifest = json.loads(out)
    assert manifest["run_id"] == "run1"
    assert manifest["roles"]["trainer"]["nodes"] == 1 and "rollout" not in manifest["roles"]
    assert manifest["secrets"] == ["WANDB_API_KEY", "SAIL_API_KEY"]
    assert manifest["mounts"]["/results"] == "rw"


def test_split_manifest_has_a_rollout_role_with_no_node_count(monkeypatch):
    recipe = _recipe("dsv41-flash-lora-split", monkeypatch)
    roles = contract.manifest(recipe, "run1")["roles"]
    assert roles["rollout"]["nodes"] is None
    assert roles["rollout"]["ports"] == [contract.ROLLOUT_PROXY_PORT]
    assert contract.REGISTRY_PORT in roles["trainer"]["ports"]
    assert "gpu" not in roles["rollout"] and roles["rollout"]["gpu_advice"] == "B300"
    assert roles["rollout"]["gpus_per_engine"] == 4 and roles["rollout"]["precision"] == "fp8_fp4"


def test_a_part_node_split_says_so_for_each_role(monkeypatch):
    recipe = _recipe("dsv41-flash-lora-split", monkeypatch)
    recipe["cluster"]["train_gpus"] = 4
    recipe["cluster"]["engine"]["gpus"] = 4
    roles = contract.manifest(recipe, "run1")["roles"]
    assert roles["trainer"]["gpus_per_node"] == 4 and roles["rollout"]["gpus_per_node"] == 4


def test_the_manifest_names_a_draft_checkpoint_only_when_the_run_uses_one(monkeypatch):
    recipe = _recipe("glm53-flash-lora-dflash2", monkeypatch)
    draft = contract.manifest(recipe, "r1")["inputs"]["draft_checkpoint"]
    assert draft["hf_repo"] == "incoai/GLM-5.3-Flash-DFlash2" and draft["path"].startswith("/models/")
    recipe["rollout"]["speculative"] = False
    assert contract.manifest(recipe, "r1")["inputs"]["draft_checkpoint"] is None
    single = _recipe("qwen35-9b-split-elastic", monkeypatch)
    assert contract.manifest(single, "r1")["inputs"]["draft_checkpoint"] is None


def test_the_first_node_timeout_comes_from_env():
    assert contract.first_node_timeout_secs({}) == contract.DEFAULT_FIRST_NODE_TIMEOUT_SECS
    assert contract.first_node_timeout_secs({contract.FIRST_NODE_TIMEOUT_ENV: "90"}) == 90


def test_a_trainer_of_several_nodes_is_one_cluster_with_rdma(monkeypatch):
    several = contract.manifest(_recipe("qwen35-9b-swegym8-megatron-2node", monkeypatch), "run1")["roles"]["trainer"]
    assert several["placement"] == "one_cluster" and several["interconnect"] == "rdma"
    one = contract.manifest(_recipe("qwen35-9b-swegym8-sail", monkeypatch), "run1")["roles"]["trainer"]
    assert "placement" not in one and "interconnect" not in one


@pytest.mark.parametrize("name", sorted(p.stem for p in CONFIGS.glob("*.yaml")))
def test_manifest_covers_every_recipes_shape(name, monkeypatch):
    recipe = _recipe(name, monkeypatch)
    trainer = contract.manifest(recipe, "run1")["roles"]["trainer"]
    shape = R.shape(recipe)
    assert {k: trainer[k] for k in ("gpu", "nodes", "cpu", "memory_gib", "disk_gib")} == {
        k: shape[k] for k in ("gpu", "nodes", "cpu", "memory_gib", "disk_gib")
    }


def test_a_node_names_what_its_caller_forgot(tmp_path, monkeypatch):
    spec = _spec(tmp_path, _recipe("qwen35-9b-swegym8-sail", monkeypatch))
    code = NO_MODAL + f"from miles_ext.node.__main__ import main; sys.exit(main(['trainer', '--spec', {str(spec)!r}]))"
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": ":".join(sys.path)}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode != 0
    for name in (contract.NODE_RANK_ENV, contract.SESSION_URL_ENV, "WANDB_API_KEY", "SAIL_API_KEY"):
        assert name in proc.stderr


def test_the_spec_command_gives_a_spec_the_roles_read(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MILES_DATA", str(tmp_path))
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "stak32.jsonl").write_text("")
    node_main.main(
        ["spec", "--recipe", str(CONFIGS / "qwen35-9b-split-elastic.yaml"), "--run-id", "r1", "--num-rollout", "2"]
    )
    path = tmp_path / "spec.json"
    path.write_text(capsys.readouterr().out)
    recipe, run_id = node_main.load_spec(str(path))
    assert run_id == "r1" and recipe["algorithm"]["num_rollout"] == 2


def test_the_manifest_says_where_inputs_go(monkeypatch):
    inputs = contract.manifest(_recipe("qwen35-9b-split-elastic", monkeypatch), "r1")["inputs"]
    assert inputs["prompts"] == "/inputs/prompts.jsonl" and inputs["tasks"] is None
    assert inputs["model_dir"] == "/models/Qwen3.5-9B" and inputs["fallback"] == "/results/r1/inputs"
    swegym = contract.manifest(_recipe("qwen35-9b-swegym8-sail", monkeypatch), "r1")["inputs"]
    assert swegym["tasks"] == "/inputs/tasks"
    volume = contract.manifest(_recipe("dsv41-flash-lora-split", monkeypatch), "r1")["inputs"]
    assert volume["tasks"] == "/datasets/tb4t-20261016/tasks"


def test_a_modal_sandbox_off_modal_needs_a_token(monkeypatch):
    recipe = _recipe("qwen35-9b-swegym8-sandbox", monkeypatch)
    assert recipe["sandbox"]["provider"] == "modal"
    assert "MODAL_TOKEN_ID" in contract.manifest(recipe, "r1")["secrets"]
    assert "MODAL_TOKEN_ID" not in contract.required_secrets(recipe)
    single_turn = _recipe("qwen35-9b-split-elastic", monkeypatch)
    assert "MODAL_TOKEN_ID" not in contract.manifest(single_turn, "r1")["secrets"]


def test_a_rollout_node_needs_the_trainer_url_in_either_form(monkeypatch):
    recipe = _recipe("dsv41-flash-lora-split", monkeypatch)
    env = {
        contract.POOL_TOKEN_ENV: "p",
        contract.ROLLOUT_URL_ENV: "http://x",
        "WANDB_API_KEY": "k",
        "SAIL_API_KEY": "s",
    }
    assert contract.missing(recipe, "rollout", env) == [contract.TRAINER_URL_ENV]
    assert contract.missing(recipe, "rollout", {**env, contract.TRAINER_URL_FILE_ENV: "/tmp/u"}) == []


def test_file_state_needs_live_results(monkeypatch):
    monkeypatch.setattr(contract, "missing", lambda *a: [])
    monkeypatch.setattr(node_storage, "for_mount", lambda mount: SimpleNamespace(live=False))
    monkeypatch.delenv(contract.STATE_ENV, raising=False)
    with pytest.raises(SystemExit, match="live"):
        node_main._require({}, "trainer")
    monkeypatch.setenv(contract.STATE_ENV, "http")
    node_main._require({}, "trainer")


def _job(monkeypatch):
    """Record each storage call with the mount it went to."""
    calls = []

    def for_mount(mount):
        return SimpleNamespace(
            live=True,
            refresh=lambda: calls.append(("refresh", str(mount))),
            commit=lambda: calls.append(("commit", str(mount))),
        )

    monkeypatch.setattr(node_storage, "for_mount", for_mount)
    monkeypatch.setattr(node_main, "load_spec", lambda path: ({}, "run1"))
    return calls


def test_prepare_weights_commits_what_it_downloaded_to_the_models_mount(monkeypatch, tmp_path):
    from miles_ext.platform import weights_job

    calls = _job(monkeypatch)
    monkeypatch.setattr(weights_job, "download", lambda *a: calls.append("download") or {"bytes": 1})
    node_main.main(["prepare-weights", "--hf-repo", "r", "--revision", "c", "--path", str(tmp_path)])
    assert calls == ["download", ("commit", str(R.MODELS))]


def test_convert_fsdp_reads_committed_inputs_and_commits_the_export_to_results(monkeypatch):
    from miles_ext.platform import checkpoints

    calls = _job(monkeypatch)
    monkeypatch.setattr(checkpoints, "finalize_hf_command", lambda *a: ["convert"])
    monkeypatch.setattr(R, "model_dir", lambda recipe: "/models/m")
    monkeypatch.setattr(node_main.subprocess, "run", lambda cmd, check: calls.append(cmd[0]))
    node_main.main(["convert-fsdp", "--spec", "s", "--step", "1"])
    convert = calls.index("convert")
    assert {("refresh", str(R.MODELS)), ("refresh", str(R.RESULTS))} == set(calls[:convert])
    assert calls[convert + 1 :] == [("commit", str(R.RESULTS))]
