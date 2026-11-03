"""Build GLM-5.3-Flash's trainer-format FP8 base: release bytes XOR the step-0 GPU-delta publication.

The trainer's FP8 re-quantization differs from the release in ~95% of bytes, so the first sync is
~240 GB and the receiver cannot pin host arenas that large under Modal. Serving this base makes later
deltas small. Each shard is rebuilt in its own CPU container; every XOR'd tensor is checked against the
publication's changed_bytes count.
"""

import json
import os
import shutil
from collections import defaultdict

import modal

SRC = "/models/zai-org--GLM-5.3-Flash/46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
DST = "/models/zai-org--GLM-5.3-Flash/29523a33f41b2cea7f85f9288433a50fb506eb39"
PUB = "/results/glm53-gpud-full-1103-a7/gpu_delta/30405180334b45129ea20eae4084c980/weight_v000001"

app = modal.App("miles-rl-rebase-glm53")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "lz4")
weights = modal.Volume.from_name("miles-rl-weights")
results = modal.Volume.from_name("miles-rl-results")
VOLS = {"/models": weights, "/results": results}


def _header(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return 8 + n, json.loads(f.read(n))


@app.function(image=image, volumes=VOLS, cpu=4, memory=32768, timeout=3600)
def rebuild(shard: str, entries: list[dict]) -> dict:
    import lz4.block
    import numpy as np

    src, dst = f"{SRC}/{shard}", f"{DST}/{shard}"
    base, header = _header(src)
    data = bytearray(open(src, "rb").read())
    handles, stats = {}, {"tensors": 0, "changed": 0, "raw": 0}
    for e in entries:
        meta = header[e["name"]]
        lo, hi = meta["data_offsets"]
        view = np.frombuffer(memoryview(data)[base + lo : base + hi], dtype=np.uint8)
        assert hi - lo == e["nbytes"], (e["name"], hi - lo, e["nbytes"])
        if e["encoding"] == "raw_bytes":
            if "raw" in e:
                r = e["raw"]
                f = handles.setdefault(r["file"], open(f"{PUB}/{r['file']}", "rb"))
                f.seek(r["encoded_offset"])
                view[:] = np.frombuffer(f.read(r["encoded_bytes"]), dtype=np.uint8)
                stats["raw"] += 1
            continue
        if not e.get("frames"):
            continue
        outer = e["outer"]
        f = handles.setdefault(outer["file"], open(f"{PUB}/{outer['file']}", "rb"))
        nonzero = 0
        for fr in e["frames"]:
            f.seek(outer["encoded_offset"] + fr["encoded_offset"])
            decoded = lz4.block.decompress(f.read(fr["encoded_bytes"]), uncompressed_size=fr["decoded_bytes"])
            delta = np.frombuffer(decoded, dtype=np.uint8)
            nonzero += int(np.count_nonzero(delta))
            seg = view[fr["decoded_offset"] : fr["decoded_offset"] + fr["decoded_bytes"]]
            np.bitwise_xor(seg, delta, out=seg)
        if nonzero != e["changed_bytes"]:
            raise ValueError(f"{e['name']}: decoded {nonzero} changed bytes, manifest {e['changed_bytes']}")
        stats["tensors"] += 1
        stats["changed"] += nonzero
    os.makedirs(DST, exist_ok=True)
    with open(dst + ".tmp", "wb") as out:
        out.write(data)
    os.replace(dst + ".tmp", dst)
    weights.commit()
    return {"shard": shard, **stats}


@app.function(image=image, volumes=VOLS, cpu=2, memory=16384, timeout=7200)
def main():
    manifest = json.load(open(f"{PUB}/manifest.json"))
    assert manifest["codec"] == "lz4", manifest["codec"]
    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    groups = defaultdict(list)
    for t in manifest["tensors"]:
        groups[index[t["name"]]].append(t)
    os.makedirs(DST, exist_ok=True)
    for name in os.listdir(SRC):
        if not name.endswith(".safetensors") and os.path.isfile(f"{SRC}/{name}") and name != ".ready":
            shutil.copy2(f"{SRC}/{name}", f"{DST}/{name}")
    untouched = [s for s in sorted(set(index.values())) if s not in groups]
    for s in untouched:  # shards with only excluded tensors (vision tower, MTP)
        shutil.copy2(f"{SRC}/{s}", f"{DST}/{s}")
    weights.commit()
    totals = defaultdict(int)
    for r in rebuild.starmap(sorted(groups.items())):
        print(r, flush=True)
        for k in ("tensors", "changed", "raw"):
            totals[k] += r[k]
    weights.reload()
    with open(f"{DST}/.ready", "w") as f:
        f.write(json.dumps({"from": SRC, "publication": PUB, **totals}))
    weights.commit()
    print("DONE", dict(totals), "untouched shards", len(untouched), flush=True)
