"""Parity gate: real rollouts vs the SFT renderer.

For each trial of an eval run, re-render the harness's message history (mini-swe-agent's
trajectory, with the tool schema it sent) through ``render.TrajectoryRenderer`` and compare the
token ids and loss mask with the Sample the session server recorded in ``rollout_data/<id>.pt``.

    modal run -m miles_ext.sft.parity --run-id swebv100-qwen3-base-r2
"""

import json
from pathlib import Path

import modal

from miles_ext.sft import common as C

app = modal.App("miles-rl-sft-parity")
image = C.cpu_image()
results_volume = modal.Volume.from_name(C.RESULTS_VOLUME, version=2)
weights_volume = modal.Volume.from_name(C.WEIGHTS_VOLUME, version=2)


def _messages(traj: dict) -> list[dict]:
    """mini-swe-agent v2 messages as the OpenAI-style list it sent (its 'exit' bookkeeping message dropped)."""
    import ast

    out = []
    for m in traj["messages"]:
        if m.get("role") not in ("system", "user", "assistant", "tool"):
            continue
        msg = {"role": m["role"], "content": m.get("content") or ""}
        if m["role"] == "assistant":
            if m.get("reasoning_content") is not None:
                msg["reasoning_content"] = m["reasoning_content"]
            calls = m.get("tool_calls")
            if isinstance(calls, str):
                calls = ast.literal_eval(calls)
            if calls:
                msg["tool_calls"] = [{"id": c.get("id"), "type": "function",
                                      "function": {"name": c["function"]["name"],
                                                   "arguments": c["function"]["arguments"]}} for c in calls]
        if m["role"] == "tool" and m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        out.append(msg)
    return out


def _norm(text: str) -> str:
    """The one boundary the template cannot reproduce: the model writes a blank line before <tool_call>; the
    template, rendering the parsed message, writes one newline."""
    return text.replace("\n\n<tool_call>", "\n<tool_call>")


@app.function(image=image, volumes={C.RESULTS_MOUNT: results_volume.read_only(),
                                    C.WEIGHTS_MOUNT: weights_volume.read_only()}, cpu=8, memory=65536, timeout=3600)
def check(run_id: str, inspect_only: bool = False) -> dict:
    import torch

    from miles_ext.sft.render import TrajectoryRenderer, load_tito

    run = Path(C.RESULTS_MOUNT) / run_id
    data = torch.load(run / "rollout_data" / "0.pt", weights_only=False)
    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
    if inspect_only:
        s = samples[0]
        fields = s if isinstance(s, dict) else vars(s)
        return {"type": str(type(data)), "keys": list(data.keys()) if isinstance(data, dict) else None,
                "n": len(samples), "sample": {k: (str(v)[:300]) for k, v in fields.items()}}
    from minisweagent.models.utils.actions_toolcall import BASH_TOOL  # the schema mini-swe-agent v2 sends

    tito = load_tito(str(Path(C.WEIGHTS_MOUNT) / C.BASE_DIR))
    renderer = TrajectoryRenderer(tito, tools=[BASH_TOOL])
    report = {"samples": 0, "matched_trials": 0, "token_equal": 0, "text_equal_modulo_tool_call_newline": 0,
              "mask_equal": 0, "examples": []}
    by_instance: dict[str, list[Path]] = {}
    for d in sorted((run / "trials").iterdir()):
        by_instance.setdefault(d.name.split("__20")[0], []).append(d)
    for s in samples:
        s = s if isinstance(s, dict) else vars(s)
        report["samples"] += 1
        tokens = list(s["tokens"])
        mask = [0] * (len(tokens) - s["response_length"]) + list(s["loss_mask"])
        best = None
        # two attempts per instance; the Sample does not name its trial, so take the attempt that matches best
        for d in by_instance.get((s.get("metadata") or {}).get("instance_id"), []):
            traj_path = d / "agent" / "mini-swe-agent.trajectory.json"
            if not traj_path.exists():
                continue
            msgs = _messages(json.loads(traj_path.read_text()))
            while msgs and msgs[-1]["role"] != "assistant":  # the Sample ends at the last generated turn
                msgs.pop()
            try:
                r = renderer.render(msgs)
            except Exception as e:
                best = best or {"trial": d.name, "error": f"{type(e).__name__}: {e}"[:300]}
                continue
            norm_eq = _norm(tito.tokenizer.decode(r.token_ids)) == _norm(tito.tokenizer.decode(tokens))
            score = (r.token_ids == tokens, norm_eq, r.loss_mask == mask)
            if best is None or "error" in best or score > best["score"]:
                best = {"trial": d.name, "score": score, "r": r}
        if best is None:
            continue
        report["matched_trials"] += 1
        if "error" in best:
            report["examples"].append(best)
            continue
        r, name = best["r"], best["trial"]
        teq, neq, meq = best["score"]
        report["token_equal"] += teq
        report["text_equal_modulo_tool_call_newline"] += neq
        report["mask_equal"] += meq
        if not (neq and meq) and len(report["examples"]) < 6:
            first = next((i for i, (a, b) in enumerate(zip(r.token_ids, tokens)) if a != b),
                         min(len(r.token_ids), len(tokens)))
            report["examples"].append({
                "trial": name, "len_render": len(r.token_ids), "len_rollout": len(tokens), "first_diff": first,
                "render": tito.tokenizer.decode(r.token_ids[max(0, first - 20):first + 20]),
                "rollout": tito.tokenizer.decode(tokens[max(0, first - 20):first + 20]),
                "mask_diff": sum(a != b for a, b in zip(r.loss_mask, mask))})
    return report


@app.local_entrypoint()
def main(run_id: str, inspect_only: bool = False):
    print(json.dumps(check.remote(run_id, inspect_only), indent=1))


@app.function(image=image, volumes={C.RESULTS_MOUNT: results_volume.read_only()}, cpu=4, memory=16384, timeout=1800)
def trial_stats(run_id: str) -> dict:
    """Per-run shape of the mini-swe-agent histories: turns, reasoning and observation sizes, format errors."""
    import statistics

    rows = []
    for d in sorted((Path(C.RESULTS_MOUNT) / run_id / "trials").iterdir()):
        p = d / "agent" / "mini-swe-agent.trajectory.json"
        if not p.exists():
            continue
        traj = json.loads(p.read_text())
        msgs = traj["messages"]
        asst = [m for m in msgs if m.get("role") == "assistant"]
        tools = [m for m in msgs if m.get("role") == "tool"]
        users = [m for m in msgs if m.get("role") == "user"][1:]
        rows.append({
            "turns": len(asst),
            "reasoning": statistics.mean(len(m.get("reasoning_content") or "") for m in asst) if asst else 0,
            "content": statistics.mean(len(m.get("content") or "") for m in asst) if asst else 0,
            "obs": statistics.mean(len(m.get("content") or "") for m in tools) if tools else 0,
            "obs_max": max((len(m.get("content") or "") for m in tools), default=0),
            "format_errors": sum("Tool call error" in (m.get("content") or "") or "format" in (m.get("content") or "")[:80].lower() for m in users),
            "exit": traj.get("info", {}).get("exit_status"),
        })
    def med(k):
        return statistics.median(r[k] for r in rows)
    by_exit = {}
    for r in rows:
        by_exit.setdefault(r["exit"], []).append(r)
    return {"trials": len(rows), **{f"median_{k}": med(k) for k in ("turns", "reasoning", "content", "obs", "obs_max", "format_errors")},
            "by_exit": {k: {"n": len(v), "median_turns": statistics.median(x["turns"] for x in v),
                            "median_reasoning": statistics.median(x["reasoning"] for x in v),
                            "median_obs_max": statistics.median(x["obs_max"] for x in v),
                            "median_format_errors": statistics.median(x["format_errors"] for x in v)} for k, v in by_exit.items()}}


@app.local_entrypoint()
def stats(runs: str):
    for run_id in runs.split(","):
        print(run_id, json.dumps(trial_stats.remote(run_id), indent=1))


@app.function(image=image, volumes={C.RESULTS_MOUNT: results_volume.read_only()}, cpu=4, memory=65536, timeout=1800)
def time_split(run_id: str) -> dict:
    """Where a trial's wall time goes: model calls (session-server request to response) vs everything between them."""
    import statistics

    import torch

    data = torch.load(Path(C.RESULTS_MOUNT) / run_id / "rollout_data" / "0.pt", weights_only=False)
    rows = []
    for s in data["samples"]:
        s = s if isinstance(s, dict) else vars(s)
        lc = [t for t in ((s.get("metadata") or {}).get("lifecycle") or []) if isinstance(t, dict) and "t1" in t]
        if not lc:
            continue
        gen = sum(t["t1"] - t["t0"] for t in lc)
        queue = sum(t["t0"] - t["req_ts"] for t in lc if t.get("req_ts"))
        span = lc[-1]["t1"] - lc[0]["req_ts"]
        rows.append({"turns": len(lc), "gen": gen, "queue": queue, "span": span, "tokens": len(s["tokens"]),
                     "out_per_turn": sum(s["loss_mask"]) / len(lc)})
    q = lambda k, p=0.5: statistics.quantiles([r[k] for r in rows], n=20)[int(p * 20) - 1]
    return {"samples": len(rows), **{f"{k}_p50": q(k) for k in ("turns", "gen", "queue", "span", "tokens")},
            **{f"{k}_p90": q(k, 0.9) for k in ("gen", "span")},
            "gen_share": sum(r["gen"] for r in rows) / sum(r["span"] for r in rows),
            "per_turn_gen_p50": statistics.median(r["gen"] / r["turns"] for r in rows),
            "out_tokens_per_turn_p50": statistics.median(r["out_per_turn"] for r in rows),
            "out_tok_per_s_p50": statistics.median(r["out_per_turn"] * r["turns"] / max(r["gen"], 1e-6) for r in rows)}


@app.local_entrypoint()
def times(runs: str):
    for run_id in runs.split(","):
        print(run_id, json.dumps(time_split.remote(run_id), indent=1))
