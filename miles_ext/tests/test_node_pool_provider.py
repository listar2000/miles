"""Rollout nodes as Miles sees them: cells from the registry, through upstream's reconcile loop."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("torch")

from miles_ext.launch.split import volume_sync  # noqa: E402
from miles_ext.node import contract as C  # noqa: E402
from miles_ext.node import pool_provider as PP  # noqa: E402
from miles_ext.node import registry as Reg  # noqa: E402


def _member(node_id: str, slots: list[int], agent_slot: int) -> dict:
    return {
        "node_id": node_id,
        "url": f"http://{node_id}",
        "token": "t",
        "slots": slots,
        "agent_slot": agent_slot,
        "engine_ports": [32000 + i for i in range(len(slots))],
        "agent_port": 32000 + len(slots),
        "gpus_per_engine": 4,
    }


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv(volume_sync.REGISTRY_ENV, "http://registry")
    monkeypatch.setenv(C.POOL_TOKEN_ENV, "pool")
    return PP.RegistryEngineProvider(SimpleNamespace(sglang_api_key=None))


def test_each_engine_is_a_cell_at_its_slot(provider, monkeypatch):
    base = PP.SPLIT_ENGINE_PORT
    monkeypatch.setattr(Reg, "members", lambda url, token: [_member("a", [base, base + 1], base + 2)])
    seen: dict = {}

    async def reconcile(cell_id, info):
        seen[cell_id] = info

    async def watch():
        await provider.init()
        await (await provider.watch_cells(reconcile))()

    asyncio.run(watch())
    assert provider.expected_num_cells(group_id="default") == 2
    cell_id, cell = sorted(seen.items())[0]
    assert cell.workers_hash == "a:32000" and cell.meta["gpu_offset"] == 0
    addrs = asyncio.run(provider.get_addrs(cell.worker_names[0]))
    assert addrs["primary"].port == base
    ((info,),) = provider.get_worker_infos(cell_ids=[cell_id])
    assert info.name == cell.worker_names[0]


def test_cells_appear_and_go_as_nodes_do(provider, monkeypatch):
    base = PP.SPLIT_ENGINE_PORT
    node_a, node_b = _member("a", [base], base + 1), _member("b", [base + 2], base + 3)
    seen: dict = {}

    async def reconcile(cell_id, info):
        seen[cell_id] = info

    async def run():
        members = [node_a]
        monkeypatch.setattr(Reg, "members", lambda url, token: list(members))
        monkeypatch.setattr(PP, "POLL_SECS", 0.01)
        stop = await provider.watch_cells(reconcile)
        assert sum(v is not None for v in seen.values()) == 1
        members.append(node_b)
        await asyncio.sleep(0.05)
        assert sum(v is not None for v in seen.values()) == 2
        members.remove(node_a)
        await asyncio.sleep(0.05)
        assert sum(v is not None for v in seen.values()) == 1
        await stop()

    asyncio.run(run())


def test_a_sync_reaches_only_the_nodes_in_mileses_snapshot(monkeypatch):
    base = PP.SPLIT_ENGINE_PORT
    joined_later = _member("b", [base + 2], base + 3)
    monkeypatch.setenv(volume_sync.REGISTRY_ENV, "http://registry")
    monkeypatch.setenv(C.POOL_TOKEN_ENV, "pool")
    monkeypatch.setattr(Reg, "members", lambda url, token: [_member("a", [base], base + 1), joined_later])
    nodes = volume_sync._serving_nodes([SimpleNamespace(server_url=f"http://127.0.0.1:{base}")])
    assert [n["node_id"] for n in nodes] == ["a"]


def test_a_node_that_fails_a_sync_is_evicted_and_the_sync_stands(monkeypatch):
    base = PP.SPLIT_ENGINE_PORT
    good, bad = _member("a", [base], base + 1), _member("b", [base + 2], base + 3)
    monkeypatch.setenv(volume_sync.REGISTRY_ENV, "http://registry")
    monkeypatch.setenv(C.POOL_TOKEN_ENV, "pool")
    evicted = []
    monkeypatch.setattr(Reg, "evict", lambda url, token, node_id, why: evicted.append(node_id))

    def call(url, body, timeout):
        if str(bad["agent_slot"]) in url:
            raise OSError("gone")
        return {"ok": True}

    monkeypatch.setattr(volume_sync, "_call", call)
    assert volume_sync._on_every_node("/push", {}, [good, bad], 1) == [{"ok": True}]
    assert evicted == ["b"]
    with pytest.raises(RuntimeError, match="no rollout node"):
        volume_sync._on_every_node("/push", {}, [bad], 1)
