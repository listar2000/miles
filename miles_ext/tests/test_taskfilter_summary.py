from miles_ext.taskfilter.summary import ROW_KEYS, reasons, seconds, summarize


def _t(start, end):
    return {"started_at": f"2026-10-17T01:{start}Z", "finished_at": f"2026-10-17T01:{end}Z"}


RESULT = {
    "started_at": "2026-10-17T01:00:00Z", "finished_at": "2026-10-17T01:20:00.5Z",
    "environment_setup": _t("00:00", "01:30"),
    "agent_setup": _t("01:30", "03:00"),
    "agent_execution": _t("03:00", "04:00"),
    "verifier": _t("04:00", "19:00"),
    "verifier_result": {"rewards": {"reward": 1.0}},
    "exception_info": None,
}
PROBE = {"task_network": "no-network", "install_ok": True, "install_s": 90.2, "smoke_ok": True, "error": None}


def test_seconds():
    assert seconds(_t("00:00", "01:30")) == 90.0
    assert seconds({"started_at": "2026-10-17T01:00:00Z", "finished_at": None}) is None
    assert seconds(None) is None


def test_pass_row():
    row = summarize("t1", "abc", RESULT, PROBE, verifier_cap_s=1800)
    assert tuple(row) == ROW_KEYS
    assert row["env_build_s"] == 90.0 and row["verifier_s"] == 900.0 and row["total_s"] == 1200.5
    assert row["oracle_reward"] == 1.0 and row["oracle_exception"] is None and not row["verifier_timed_out"]
    assert row["install_ok"] and row["smoke_ok"] and row["install_s"] == 90.2 and row["install_error"] is None
    assert reasons(row, 1800) == []


def test_verifier_timeout_by_duration_and_by_exception():
    capped = {**RESULT, "verifier": _t("04:00", "34:00"), "verifier_result": None}
    assert summarize("t", "v", capped, PROBE, verifier_cap_s=1800)["verifier_timed_out"]
    by_exc = {**RESULT, "verifier": {"started_at": "2026-10-17T01:04:00Z", "finished_at": None},
              "verifier_result": None,
              "exception_info": {"exception_type": "VerifierTimeoutError", "exception_message": "x"}}
    row = summarize("t", "v", by_exc, PROBE, verifier_cap_s=1800)
    assert row["verifier_timed_out"] and row["verifier_s"] is None
    assert row["oracle_exception"].startswith("VerifierTimeoutError")
    assert reasons(row, 1800) == ["verifier timeout (>=1800 s)"]


def test_failed_install_and_build_error():
    probe = {"install_ok": False, "install_s": 12.0, "smoke_ok": False, "error": "RuntimeError: apt-get"}
    build = {"environment_setup": _t("00:00", "05:00"), "exception_info": {
        "exception_type": "ImageBuildError", "exception_message": "COPY failed"}}
    row = summarize("t", "v", build, probe, verifier_cap_s=1800)
    assert row["oracle_reward"] is None and row["oracle_exception"] == "ImageBuildError: COPY failed"
    assert row["install_error"] == "RuntimeError: apt-get"
    assert reasons(row, 1800) == ["image build failed (ImageBuildError)"]


def test_install_ok_but_smoke_failed():
    probe = {"install_ok": True, "install_s": 50.0, "smoke_ok": False, "error": "smoke: RuntimeError: not found"}
    row = summarize("t", "v", RESULT, probe, verifier_cap_s=1800)
    assert reasons(row, 1800) == ["mini-swe-agent install fail"] and row["install_error"].startswith("smoke")


def test_runner_error_without_result():
    row = summarize("t", "v", None, None, verifier_cap_s=1800, error="runner: HTTPStatusError: 404")
    assert row["oracle_exception"] == "runner: HTTPStatusError: 404"
    assert row["install_ok"] is None and row["smoke_ok"] is None and row["env_build_s"] is None and not row["verifier_timed_out"]


def _row(exc=None, reward=1.0, install=True, **kw):
    return {"oracle_exception": exc, "oracle_reward": reward, "install_ok": install, "smoke_ok": install,
            "verifier_timed_out": False, **kw}


def test_reasons_sandbox_never_ran_gives_one_reason():
    assert reasons(_row("SailboxExecutionError: mcr.microsoft.com", None, None), 1800) == [
        "image spec rejected by Sail (registry or Dockerfile syntax)"]
    assert reasons(_row("ValueError: cpus=16 but the largest Sailbox size provides 8", None, None), 1800) == [
        "resources exceed largest Sailbox"]
    assert reasons(_row("InvalidArgumentError: context has 10000 entries", None, None), 1800) == [
        "build context >10000 files"]
    assert reasons(_row("EnvironmentStartTimeoutError: 600 s", None, None), 1800) == ["image build timeout"]
    assert reasons(None, 1800) == ["not run"]


def test_reasons_oracle_and_install_failures_add_up():
    assert reasons(_row(reward=0.0, install=False), 1800) == ["oracle fail", "mini-swe-agent install fail"]
    assert reasons(_row("AgentTimeoutError: 3600", None), 1800) == ["oracle over its time limit"]
    assert reasons(_row("RewardFileNotFoundError: x", None), 900) == ["oracle fail (RewardFileNotFoundError)"]
