"""[fork fix-078] owner payloads pack straight into the pinned host slab, in the device packer's layout."""

import torch

from miles.utils.gpu_delta import encoder


def test_host_pack_matches_the_device_layout(monkeypatch):
    zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *a, pin_memory=False, **k: zeros(*a, **k))  # no CUDA host pinning here
    generator = torch.Generator().manual_seed(0)
    groups = [
        [torch.randint(0, 256, (n,), dtype=torch.uint8, generator=generator) for n in sizes]
        for sizes in ([5, 17], [], [16], [1, 31, 3], [])
    ]
    device, device_offsets, slab = encoder._pack_device_arenas(groups, torch.zeros(15, dtype=torch.uint8))
    transfer = {}
    host, host_offsets = encoder._pack_host_arenas(groups, transfer)
    assert host_offsets == device_offsets
    assert len(transfer["device"]) == sum(map(len, groups))
    for arena, location in zip(device, host, strict=True):
        if arena is None:
            assert location is None
            continue
        start, size = location
        assert start == arena.data_ptr() - slab.data_ptr() and size == arena.numel()
        assert torch.equal(transfer["host"][start : start + size], arena)
