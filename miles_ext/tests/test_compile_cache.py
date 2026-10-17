"""miles_ext.platform.compile_cache on local directories (no Modal).

    python -m pytest miles_ext/tests/test_compile_cache.py
"""

import os
import tarfile
import time

from miles_ext.platform import compile_cache as CC

KEY = CC.key("B300", "0.5.22.dev46+g7b6993a")


def _write(path, data=b"x", age=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    t = time.time() - age
    os.utime(path, (t, t))


def test_key_and_env(tmp_path):
    assert KEY == "b300/sglang-0.5.22.dev46+g7b6993a"
    assert CC.archive(KEY, tmp_path, node=2) == tmp_path / "b300/sglang-0.5.22.dev46+g7b6993a/node-2.tar"
    env = CC.env(tmp_path)
    assert env["SGLANG_CACHE_DIR"] == str(tmp_path) and env["TRITON_CACHE_DIR"] == str(tmp_path / "triton")
    assert env["CUDA_CACHE_PATH"] == str(tmp_path / "nv")
    assert env["SGLANG_DG_CACHE_DIR_PER_PROCESS"] == "0"  # SGLang reads "1"/"true"/"yes" as on


def test_save_then_restore_round_trip(tmp_path):
    local, mount, other = tmp_path / "local", tmp_path / "vol", tmp_path / "next-container"
    tuned = local / "flashinfer/autotune/0.6.18/sm103/abc/rank_tp0_pp0_dp0.json"
    _write(tuned, b'{"t": 1}', age=120)
    _write(local / "triton/k/kernel.cubin", b"cubin", age=120)
    _write(local / "deep_gemm/locks/l", age=120)  # lock files stay local
    _write(local / "kernels.lock", age=120)
    commits = []
    state = CC.save(KEY, lambda: commits.append(1), mount=mount, local=local)
    assert state and commits == [1]
    with tarfile.open(CC.archive(KEY, mount)) as tar:
        assert sorted(tar.getnames()) == ["flashinfer/autotune/0.6.18/sm103/abc/rank_tp0_pp0_dp0.json",
                                          "triton/k/kernel.cubin"]
    assert not list(mount.rglob("*.tmp"))
    CC.restore(KEY, mount=mount, local=other)
    assert (other / tuned.relative_to(local)).read_bytes() == b'{"t": 1}'
    # unchanged since the last save: no write, no commit
    assert CC.save(KEY, lambda: commits.append(1), mount=mount, local=local, last=state) == state
    assert commits == [1]


def test_save_waits_for_files_to_settle(tmp_path):
    local, mount = tmp_path / "local", tmp_path / "vol"
    _write(local / "flashinfer/autotune/a.json", age=120)
    _write(local / "flashinfer/autotune/b.json", age=0)  # another rank's file still being written
    assert CC.save(KEY, lambda: None, mount=mount, local=local) is None
    assert not CC.archive(KEY, mount).exists()
    assert CC.save(KEY, lambda: None, mount=mount, local=local, min_age=0)


def test_changed_file_is_saved_again(tmp_path):
    local, mount, other = tmp_path / "local", tmp_path / "vol", tmp_path / "other"
    f = local / "flashinfer/autotune/a.json"
    _write(f, b"old", age=300)
    state = CC.save(KEY, lambda: None, mount=mount, local=local)
    _write(f, b"newer", age=120)
    assert CC.save(KEY, lambda: None, mount=mount, local=local, last=state) != state
    CC.restore(KEY, mount=mount, local=other)
    assert (other / "flashinfer/autotune/a.json").read_bytes() == b"newer"


def test_failures_never_raise(tmp_path):
    logs = []
    CC.restore(KEY, mount=tmp_path / "missing", local=tmp_path / "l", log=logs.append)  # empty volume
    assert "nothing saved" in logs[-1]
    CC.archive(KEY, tmp_path / "v").parent.mkdir(parents=True)
    CC.archive(KEY, tmp_path / "v").write_bytes(b"not a tar")
    CC.restore(KEY, mount=tmp_path / "v", local=tmp_path / "l", log=logs.append)
    assert "restore failed" in logs[-1]

    def boom():
        raise OSError("volume gone")

    _write(tmp_path / "l/triton/k", age=120)
    assert CC.save(KEY, boom, mount=tmp_path / "v2", local=tmp_path / "l", log=logs.append) is None
    assert "save failed" in logs[-1]


def test_restore_refuses_paths_outside_the_cache(tmp_path):
    evil = CC.archive(KEY, tmp_path / "vol")
    evil.parent.mkdir(parents=True)
    (tmp_path / "payload").write_bytes(b"x")
    with tarfile.open(evil, "w") as tar:
        tar.add(tmp_path / "payload", arcname="../../escaped")
    logs = []
    CC.restore(KEY, mount=tmp_path / "vol", local=tmp_path / "deep/local", log=logs.append)
    assert not (tmp_path / "escaped").exists() and "restore failed" in logs[-1]


def test_saver_skips_what_was_restored_and_saves_fresh_files_on_stop(tmp_path):
    local, mount, commits = tmp_path / "local", tmp_path / "vol", []
    _write(local / "triton/k/restored.cubin", age=600)
    saver = CC.Saver(KEY, lambda: commits.append(1), every=0.05, mount=mount, local=local).start()
    time.sleep(0.3)
    assert commits == []  # only restored files so far
    _write(local / "triton/k/fresh.cubin", age=0)
    saver.stop()
    with tarfile.open(CC.archive(KEY, mount)) as tar:
        assert "triton/k/fresh.cubin" in tar.getnames()
    assert commits == [1]


def test_each_node_saves_its_own_archive_and_restore_unpacks_all(tmp_path):
    vol, nodes = tmp_path / "vol", [tmp_path / "node0", tmp_path / "node1"]
    _write(nodes[0] / "flashinfer/tune/rank0.json", b"head", age=600)
    _write(nodes[1] / "flashinfer/tune/rank0.json", b"worker", age=600)
    _write(nodes[1] / "nv/worker-only.bin", age=600)
    for rank, local in reversed(list(enumerate(nodes))):  # node 0 saves last, so its copy is the newest
        CC.save(KEY, lambda: None, mount=vol, local=local, node=rank)
        t = time.time() - 100 * rank
        os.utime(CC.archive(KEY, vol, rank), (t, t))
    assert sorted(p.name for p in (vol / KEY).iterdir()) == ["node-0.tar", "node-1.tar"]
    fresh = tmp_path / "next-container"
    CC.restore(KEY, mount=vol, local=fresh)
    assert (fresh / "nv/worker-only.bin").exists()
    assert (fresh / "flashinfer/tune/rank0.json").read_bytes() == b"head"  # whole archives, newest last
