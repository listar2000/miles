"""Build a four-layer FP8 checkpoint fixture for GLM-5.3-Flash GPU-delta tests."""

import json
import os
import re

import modal

app = modal.App("miles-rl-gpudelta-slice")
vol = modal.Volume.from_name("miles-rl-weights")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1", "safetensors==0.4.5", "numpy<2", extra_index_url="https://download.pytorch.org/whl/cpu"
)
SRC = "/w/zai-org--GLM-5.3-Flash/46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
SLICE_CFG = "/w/CharyZeng--GLM-5.3-Flash-4layer/7fb7883207fe73ea04430ac392f6ffe2b9f936c4/config.json"
DST = "/w/gpudelta/GLM-5.3-Flash-fp8-4layer"
KEEP = {0: 0, 3: 1, 4: 2, 7: 3}
LAYER = re.compile(r"(\.layers\.)(\d+)(\.)")


def remap(name):
    m = LAYER.search(name)
    if not m or "visual" in name or "vision" in name:
        return name
    old = int(m.group(2))
    if old not in KEEP:
        return None
    return name[: m.start(2)] + str(KEEP[old]) + name[m.end(2) :]


@app.function(image=image, volumes={"/w": vol}, cpu=16, memory=131072, timeout=7200)
def build():
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(DST, exist_ok=True)
    idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    groups = {}
    for name, f in idx.items():
        new = remap(name)
        if new is None:
            continue
        m = LAYER.search(new)
        g = f"layer{m.group(2)}" if m and "visual" not in new and "vision" not in new else "rest"
        groups.setdefault(g, []).append((name, new, f))
    weight_map = {}
    for gi, (g, items) in enumerate(sorted(groups.items())):
        out, shard = {}, f"model-{gi + 1:05d}-of-{len(groups):05d}.safetensors"
        by_file = {}
        for name, new, f in items:
            by_file.setdefault(f, []).append((name, new))
        for f, pairs in by_file.items():
            with safe_open(f"{SRC}/{f}", framework="pt") as h:
                for name, new in pairs:
                    out[new] = h.get_tensor(name)
        save_file(out, f"{DST}/{shard}", metadata={"format": "pt"})
        weight_map.update({k: shard for k in out})
        print(g, len(out), shard, flush=True)
    json.dump({"metadata": {}, "weight_map": weight_map}, open(f"{DST}/model.safetensors.index.json", "w"), indent=1)

    full = json.load(open(f"{SRC}/config.json"))
    sl = json.load(open(SLICE_CFG))
    cfg = json.loads(json.dumps(full))
    tc, stc = cfg.get("text_config", cfg), sl.get("text_config", sl)
    for k in (
        "num_hidden_layers",
        "layer_types",
        "mlp_layer_types",
        "linear_attn_config",
        "first_k_dense_replace",
        "indexer_types",
        "num_nextn_predict_layers",
    ):
        tc[k] = stc[k]
    q = cfg.get("quantization_config") or tc.get("quantization_config")
    mods = []
    for m in q.get("modules_to_not_convert", []):
        new = remap(m) if LAYER.search(m) else m
        if new is not None:
            mods.append(new)
    q["modules_to_not_convert"] = mods
    json.dump(cfg, open(f"{DST}/config.json", "w"), indent=2)
    for f in os.listdir(SRC):
        if f.endswith((".json", ".jinja", ".txt", ".model", ".py")) and f not in (
            "config.json",
            "model.safetensors.index.json",
        ):
            shutil.copy(f"{SRC}/{f}", f"{DST}/{f}")
    open(f"{DST}/.ready", "w").write("release 46a050be layers 0,3,4,7 -> 0-3\n")
    vol.commit()
    return sorted(os.listdir(DST))


@app.local_entrypoint()
def main():
    print(build.remote())
