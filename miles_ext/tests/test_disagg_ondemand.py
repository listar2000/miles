import asyncio
from types import SimpleNamespace

import pytest
from miles_ext.disagg.medic import engine_pools, pending_requests, torn_down
from miles_ext.disagg.ondemand import TrainerLost, _or_lost, _restart_engines, superseded


class _Dict:
    def __init__(self, data):
        self.data = data

        async def aio(key):
            return self.data.get(key)

        self.get = type("G", (), {"aio": staticmethod(aio)})


async def _slow(value, s):
    await asyncio.sleep(s)
    return value


def test_or_lost_returns_result_while_trainer_alive():
    assert asyncio.run(_or_lost(_Dict({}), "lost/0/0", _slow(7, 0.05), poll=0.01)) == 7


def test_or_lost_raises_when_trainer_lost_mid_call():
    d = _Dict({})

    async def go():
        asyncio.get_running_loop().call_later(0.05, d.data.__setitem__, "lost/0/0", "killed")
        await _or_lost(d, "lost/0/0", _slow(7, 10), poll=0.01)

    with pytest.raises(TrainerLost, match="killed"):
        asyncio.run(go())


def test_superseded_only_by_a_later_attempt():
    assert not superseded(None, 0)
    assert not superseded({"attempt": 1}, 1)
    assert superseded({"attempt": 2}, 1)


def test_restart_engines_waits_for_the_medic_then_for_the_cells():
    calls = []

    class _Controller:
        async def wait_expected_num_cells(self):
            calls.append("cells")

    d = _Dict({})

    async def put(key, value):
        d.data[key] = value
        asyncio.get_running_loop().call_later(0.02, d.data.__setitem__, "engines_restarted/s1a0", 1.0)

    d.put = type("P", (), {"aio": staticmethod(put)})
    asyncio.run(_restart_engines(d, _Controller(), tag="s1a0"))
    assert "restart_engines/s1a0" in d.data and calls == ["cells"]


def test_medic_finds_torn_down_cells_and_unanswered_requests():
    assert engine_pools({"inference-engine-e0": [], "router": []}) == ["inference-engine-e0"]
    infos = {"c1": SimpleNamespace(worker_names=["w"]), "c0": SimpleNamespace(worker_names=[])}
    assert torn_down(infos) == ["c0"]
    keys = ["restart_engines/s1a0", "engines_restarted/s1a0", "restart_engines/s3a1"]
    assert pending_requests(keys) == ["s3a1"]
