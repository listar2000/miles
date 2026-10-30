"""Build the strict-thinking SFT set from nvidia/Nemotron-SFT-SWE-v3 on Modal CPUs.

    modal run -m miles_ext.sft.prepare::weights     # Qwen3-30B-A3B-Thinking-2507 onto the weights volume
    modal run -m miles_ext.sft.prepare::build       # map the 96 shards, then reduce to train/val

Map (one container per source shard): validate and normalize every row (``render.normalize_row``:
roles, string fields, tool-call arguments that decode to JSON objects, reasoning on every assistant
turn, ends with an assistant turn), drop SWE-bench Verified contamination, render with the TITO
renderer (``render.TrajectoryRenderer``) and write the token ids and loss masks of the rows that fit
in MAX_TOKENS. Reduce: one trajectory per (task, harness), content dedup, task-level 2% validation
split (sha1(task) % 100 < 2), then re-read every kept row from the shard files and check its digest.
"""

import hashlib
import json
import os
import re
from pathlib import Path

import modal

from miles_ext.sft import common as C

app = modal.App("miles-rl-sft-prepare")
image = C.cpu_image()
data_volume = modal.Volume.from_name(C.DATA_VOLUME, create_if_missing=True, version=2)
weights_volume = modal.Volume.from_name(C.WEIGHTS_VOLUME, create_if_missing=True, version=2)
secrets = [modal.Secret.from_dotenv(os.environ.get("MILES_DOTENV_DIR") or os.getcwd())]

MAX_TOKENS = 65536
VAL_PERCENT = 2
NUM_SHARDS = 96
OUT = Path(C.DATA_MOUNT) / C.DATASET_DIR


@app.function(image=image, volumes={C.WEIGHTS_MOUNT: weights_volume}, secrets=secrets, cpu=8, memory=16384,
              timeout=3 * 3600)
def weights():
    from huggingface_hub import snapshot_download

    dst = Path(C.WEIGHTS_MOUNT) / C.BASE_DIR
    snapshot_download(C.BASE_REPO, revision=C.BASE_REVISION, local_dir=dst, max_workers=16)
    (dst / ".revision").write_text(C.BASE_REVISION)
    weights_volume.commit()
    print(sorted(p.name for p in dst.iterdir()))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


_ISSUE_TAGS = ("issue_description", "pr_description", "issue_details", "issue")


def task_key(user: str) -> str:
    """The task a trajectory solves, independent of the harness that wraps it: the issue text."""
    body = user
    for tag in _ISSUE_TAGS:
        m = re.search(rf"<{tag}>(.*?)(</{tag}>|$)", user, re.S)
        if m:
            body = m.group(1)
            break
    body = re.sub(r"\[/?ISSUE\]", "", body)
    return hashlib.sha1(re.sub(r"[^a-z0-9]+", "", body.lower())[:600].encode()).hexdigest()


def verified_probes() -> tuple[list[str], list[str]]:
    """SWE-bench Verified problem-statement windows (normalized) and instance ids."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("princeton-nlp/SWE-bench_Verified", "data/test-00000-of-00001.parquet",
                           repo_type="dataset")
    rows = pq.read_table(path, columns=["instance_id", "problem_statement"]).to_pylist()
    windows = []
    for r in rows:
        text = _norm(r["problem_statement"])
        if len(text) < 160:
            windows.append(text)
            continue
        for frac in (0.0, 0.3, 0.6):
            start = int(len(text) * frac)
            windows.append(text[start:start + 120])
    return windows, [r["instance_id"] for r in rows]


@app.function(image=image, volumes={C.DATA_MOUNT: data_volume, C.WEIGHTS_MOUNT: weights_volume.read_only()},
              secrets=secrets, cpu=2, memory=12288, timeout=4 * 3600, max_containers=96, retries=1)
def map_shard(shard: int, windows: list[str], ids: list[str], limit: int = 0) -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from miles_ext.sft.render import TrajectoryRenderer, load_tito, normalize_row

    path = hf_hub_download(C.DATASET_REPO, f"data/train-{shard:05d}-of-{NUM_SHARDS:05d}.parquet",
                           repo_type="dataset", revision=C.DATASET_REVISION)
    rows = pq.read_table(path).to_pylist()
    rows = rows[:limit] if limit else rows
    renderer = TrajectoryRenderer(load_tito(str(Path(C.WEIGHTS_MOUNT) / C.BASE_DIR)))
    stats = {"shard": shard, "rows": len(rows), "invalid": {}, "contaminated": 0, "too_long": 0, "kept": 0,
             "full_render_equal": 0, "full_render_text_equal": 0}
    toks, masks, meta = [], [], []
    offset = 0
    for idx, row in enumerate(rows):
        try:
            messages = normalize_row(row["messages"])
        except (ValueError, json.JSONDecodeError, AttributeError, TypeError) as e:
            key = str(e).split(":")[0][:60]
            stats["invalid"][key] = stats["invalid"].get(key, 0) + 1
            continue
        users = [m["content"] for m in messages if m["role"] == "user"]
        if not users:
            stats["invalid"]["no user message"] = stats["invalid"].get("no user message", 0) + 1
            continue
        hay = _norm(" ".join(users))
        if any(w in hay for w in windows) or any(i in hay for i in ids):
            stats["contaminated"] += 1
            continue
        try:
            r = renderer.render(messages)
        except Exception as e:  # a row the template cannot render is invalid data, not a crash
            key = f"render {type(e).__name__}"
            stats["invalid"][key] = stats["invalid"].get(key, 0) + 1
            continue
        if len(r.token_ids) > MAX_TOKENS:
            stats["too_long"] += 1
            continue
        n = len(r.token_ids)
        toks.append(np.asarray(r.token_ids, dtype=np.int32))
        masks.append(np.asarray(r.loss_mask, dtype=np.uint8))
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        meta.append({
            "uuid": row["uuid"], "shard": shard, "row": idx, "offset": offset, "length": n,
            "train_tokens": int(sum(r.loss_mask)), "turns": r.turns,
            "task": task_key(users[0]), "harness": hashlib.sha1(system[:200].encode()).hexdigest()[:12],
            "content": hashlib.sha1(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
            "digest": hashlib.sha1(np.asarray(r.token_ids, dtype=np.int32).tobytes()
                                   + np.asarray(r.loss_mask, dtype=np.uint8).tobytes()).hexdigest(),
            "full_render_equal": r.full_render_equal, "full_render_text_equal": r.full_render_text_equal,
            "multi_user": len(users) > 1,
        })
        stats["full_render_equal"] += r.full_render_equal
        stats["full_render_text_equal"] += r.full_render_text_equal
        offset += n
    stats["kept"] = len(meta)
    shard_dir = OUT / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.save(shard_dir / f"{shard:05d}.tokens.npy", np.concatenate(toks) if toks else np.zeros(0, np.int32))
    np.save(shard_dir / f"{shard:05d}.mask.npy", np.concatenate(masks) if masks else np.zeros(0, np.uint8))
    with open(shard_dir / f"{shard:05d}.meta.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    (shard_dir / f"{shard:05d}.stats.json").write_text(json.dumps(stats))
    data_volume.commit()
    print(json.dumps(stats))
    return stats


@app.function(image=image, volumes={C.DATA_MOUNT: data_volume}, cpu=4, memory=32768, timeout=3600)
def reduce(stats: list[dict]) -> dict:
    import numpy as np

    data_volume.reload()
    shard_dir = OUT / "shards"
    rows = []
    for s in sorted(stats, key=lambda s: s["shard"]):
        rows += [json.loads(line) for line in open(shard_dir / f"{s['shard']:05d}.meta.jsonl")]
    candidates = len(rows)
    # one trajectory per (task, harness), then content dedup; deterministic: the first in shard order
    seen_th, seen_content, kept = set(), set(), []
    for r in rows:
        th = (r["task"], r["harness"])
        if th in seen_th or r["content"] in seen_content:
            continue
        seen_th.add(th)
        seen_content.add(r["content"])
        kept.append(r)
    for i, r in enumerate(kept):
        r["id"] = i
        r["split"] = "val" if int(hashlib.sha1(r["task"].encode()).hexdigest(), 16) % 100 < VAL_PERCENT else "train"
    # verify: every kept row re-read from the serialized shards matches its digest; ids unique; no task overlap
    cache = {}
    for r in kept:
        if r["shard"] not in cache:
            cache = {r["shard"]: (np.load(shard_dir / f"{r['shard']:05d}.tokens.npy", mmap_mode="r"),
                                  np.load(shard_dir / f"{r['shard']:05d}.mask.npy", mmap_mode="r"))}
        t, m = cache[r["shard"]]
        a, b = r["offset"], r["offset"] + r["length"]
        digest = hashlib.sha1(np.asarray(t[a:b]).tobytes() + np.asarray(m[a:b]).tobytes()).hexdigest()
        assert digest == r["digest"], r
    train_tasks = {r["task"] for r in kept if r["split"] == "train"}
    val_tasks = {r["task"] for r in kept if r["split"] == "val"}
    assert not train_tasks & val_tasks
    assert len({r["id"] for r in kept}) == len(kept)
    summary = {"source": f"{C.DATASET_REPO}@{C.DATASET_REVISION}", "tokenizer": f"{C.BASE_REPO}@{C.BASE_REVISION}",
               "renderer": "miles_ext.sft.render (tito qwennext)", "max_tokens": MAX_TOKENS,
               "source_rows": sum(s["rows"] for s in stats),
               "invalid": {k: sum(s["invalid"].get(k, 0) for s in stats) for s in stats for k in s["invalid"]},
               "contaminated": sum(s["contaminated"] for s in stats),
               "too_long": sum(s["too_long"] for s in stats), "rendered_candidates": candidates,
               "kept": len(kept)}
    for split in ("train", "val"):
        part = [r for r in kept if r["split"] == split]
        summary[split] = {"rows": len(part), "tasks": len({r["task"] for r in part}),
                          "tokens": sum(r["length"] for r in part),
                          "train_tokens": sum(r["train_tokens"] for r in part),
                          "full_render_equal": sum(r["full_render_equal"] for r in part),
                          "full_render_text_equal": sum(r["full_render_text_equal"] for r in part)}
        with open(OUT / f"{split}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
        # the prompt file Miles' data source shuffles: one row id per line (miles_ext.sft.rollout)
        with open(OUT / f"{split}_prompts.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps({"text": str(r["id"])}) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=1))
    data_volume.commit()
    print(json.dumps(summary, indent=1))
    return summary


@app.local_entrypoint()
def probe(shard: int = 0, limit: int = 200):
    """Render the first rows of one shard and print the stats (writes that shard's files)."""
    import time

    windows, ids = verify_probes.remote()
    t = time.time()
    print(json.dumps(map_shard.remote(shard, windows, ids, limit), indent=1), f"{time.time() - t:.0f}s")


@app.local_entrypoint()
def build(shards: str = ""):
    picked = [int(s) for s in shards.split(",")] if shards else list(range(NUM_SHARDS))
    windows, ids = verify_probes.remote()
    stats = list(map_shard.map(picked, kwargs={"windows": windows, "ids": ids}, return_exceptions=True))
    bad = [s for s in stats if isinstance(s, Exception)]
    print(f"{len(bad)} shards failed: {bad[:3]}")
    print(json.dumps(reduce.remote([s for s in stats if not isinstance(s, Exception)]), indent=1))


@app.function(image=image, secrets=secrets, cpu=1, memory=4096, timeout=900)
def verify_probes():
    return verified_probes()
