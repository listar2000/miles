"""Made-up stack language tasks: generator, interpreter and verifier.

Every task is one program in a small invented language; the model reads the spec, runs the program in its head and
answers with the final stack. Reward is 1 for an exact match of the stack, 0 otherwise (also 0 for a missing or
repeated answer tag). Tasks come from a seed, so a set is reproducible.

  python stackvm.py make --out tasks.jsonl --per-level 32 --seed 1026 --levels 4,5
  python stackvm.py selftest
"""

import argparse
import json
import random
import re

MOD = 97
MAX_DEPTH = 8  # stack items, so the answer stays short


def spec(mod):
    return f"""You will run a program written in STAK, a small stack language. The stack holds whole numbers from 0 to {mod - 1};
every result is taken mod {mod} (so it is always in 0..{mod - 1}). The program starts with an empty stack and runs left to
right. Below, "top" is the most recently pushed item and "second" is the one under it.

PUSH n   push n
POP      remove top
DUP      push a copy of top
SWAP     exchange top and second
OVER     push a copy of second
ROT      take the third item from the top out and push it on top (bottom..top: x y z -> y z x)
SINK     take top out and put it at the bottom of the stack
FLIP     reverse the whole stack
ADD      pop a (top), pop b, push a + b
SUB      pop a (top), pop b, push a - b   (note the order: top minus second)
MUL      pop a, pop b, push a * b
NEG      replace top t with -t (that is, {mod} - t, and 0 stays 0)
HALF     replace top t with t / 2 if t is even, otherwise with 3 * t + 1
CMP      pop a (top), pop b, push 1 if a > b, otherwise 0
REP k {{ ... }}   run the block inside the braces k times (blocks can be nested)

Every program is valid: no instruction ever runs on a stack that is too short."""


INSTRUCTION = """Give the final stack from top to bottom, numbers separated by single spaces, inside one answer tag, as the last
thing you write. For example, if the stack ends as bottom..top 5 0 12, answer <answer>12 0 5</answer>."""

# op -> (items it needs, net change in stack size)
OPS = {
    "PUSH": (0, 1),
    "POP": (1, -1),
    "DUP": (1, 1),
    "SWAP": (2, 0),
    "OVER": (2, 1),
    "ROT": (3, 0),
    "SINK": (1, 0),
    "FLIP": (1, 0),
    "ADD": (2, -1),
    "SUB": (2, -1),
    "MUL": (2, -1),
    "NEG": (1, 0),
    "HALF": (1, 0),
    "CMP": (2, -1),
}

# level -> straight-line instructions, REP blocks, max nesting, executed-instruction range, modulus, max REP count
LEVELS = {
    1: dict(n=6, reps=0, nest=0, exec=(0, 6), mod=97, kmax=3),
    2: dict(n=9, reps=1, nest=1, exec=(0, 24), mod=97, kmax=3),
    3: dict(n=12, reps=2, nest=1, exec=(0, 48), mod=97, kmax=3),
    4: dict(n=16, reps=2, nest=2, exec=(0, 96), mod=97, kmax=3),
    5: dict(n=20, reps=3, nest=2, exec=(60, 120), mod=97, kmax=4),
    6: dict(n=24, reps=3, nest=3, exec=(140, 280), mod=97, kmax=5),
    7: dict(n=16, reps=2, nest=2, exec=(30, 96), mod=9973, kmax=3),
    8: dict(n=20, reps=3, nest=2, exec=(60, 120), mod=9973, kmax=4),
}


def step(op, arg, s, mod=MOD):
    """One instruction on stack s (a list, bottom first); returns the new stack or raises IndexError."""
    need = OPS[op][0]
    if len(s) < need:
        raise IndexError(op)
    s = list(s)
    if op == "PUSH":
        s.append(arg % mod)
    elif op == "POP":
        s.pop()
    elif op == "DUP":
        s.append(s[-1])
    elif op == "SWAP":
        s[-1], s[-2] = s[-2], s[-1]
    elif op == "OVER":
        s.append(s[-2])
    elif op == "ROT":
        s.append(s.pop(-3))
    elif op == "SINK":
        s.insert(0, s.pop())
    elif op == "FLIP":
        s.reverse()
    elif op in ("ADD", "SUB", "MUL", "CMP"):
        a, b = s.pop(), s.pop()
        s.append({"ADD": (a + b) % mod, "SUB": (a - b) % mod, "MUL": (a * b) % mod, "CMP": int(a > b)}[op])
    elif op == "NEG":
        s[-1] = (-s[-1]) % mod
    elif op == "HALF":
        t = s[-1]
        s[-1] = t // 2 if t % 2 == 0 else (3 * t + 1) % mod
    return s


def run(prog, s=None, mod=MOD):
    """prog: list of ("OP", arg) or ("REP", k, body). Returns (stack bottom first, executed instruction count)."""
    s, n = list(s or []), 0
    for ins in prog:
        if ins[0] == "REP":
            for _ in range(ins[1]):
                s, m = run(ins[2], s, mod)
                n += m
        else:
            s = step(ins[0], ins[1], s, mod)
            n += 1
            if len(s) > MAX_DEPTH:
                raise IndexError("deep")
    return s, n


def render(prog):
    out = []
    for ins in prog:
        if ins[0] == "REP":
            out.append(f"REP {ins[1]} {{ {render(ins[2])} }}")
        elif ins[0] == "PUSH":
            out.append(f"PUSH {ins[1]}")
        else:
            out.append(ins[0])
    return "  ".join(out)


def _rand_ins(rng, depth, mod):
    """A random instruction that is valid at stack size depth (PUSH more likely when the stack is short)."""
    if depth < 2 or (depth < MAX_DEPTH - 1 and rng.random() < 0.25):
        return ("PUSH", rng.randrange(mod))
    ok = [op for op, (need, d) in OPS.items() if op != "PUSH" and need <= depth and depth + d <= MAX_DEPTH]
    return (rng.choice(ok), None)


def _block(rng, n, depth, reps, nest, mod, kmax):
    """n instructions with `reps` REP blocks spliced in; tracks stack size only (validity is checked by running)."""
    prog = []
    rep_at = set(rng.sample(range(n), reps)) if reps else set()
    for i in range(n):
        if i in rep_at:
            body_n = rng.randint(2, 4)
            inner = 1 if nest > 1 and rng.random() < 0.5 else 0
            body, _ = _block(rng, body_n, depth, inner, nest - 1, mod, kmax)
            prog.append(("REP", rng.randint(2, kmax), body))
        ins = _rand_ins(rng, depth, mod)
        prog.append(ins)
        depth += OPS[ins[0]][1]
    return prog, depth


def make_task(rng, level):
    lv = LEVELS[level]
    lo, hi = lv["exec"]
    for _ in range(200000):
        prog, _ = _block(rng, lv["n"], 0, lv["reps"], lv["nest"], lv["mod"], lv["kmax"])
        if lv["reps"] and not any(i[0] == "REP" for i in prog):
            continue
        try:
            stack, executed = run(prog, mod=lv["mod"])
        except IndexError:
            continue
        if not stack or not lo <= executed <= hi:
            continue
        if max(stack) < 2:  # an all-0/1 stack is too easy to guess
            continue
        # no trivial endings: the answer should not be just the last pushed constant
        if prog[-1][0] == "PUSH":
            continue
        return prog, stack, executed
    raise RuntimeError(f"no valid program at level {level}")


def prompt_text(prog, mod):
    return f"{spec(mod)}\n\nProgram:\n{render(prog)}\n\n{INSTRUCTION}"


def answer_of(stack):
    return " ".join(str(x) for x in reversed(stack))


TASK = "experiments.dsv41_stak_overfit.stackvm.grade"  # metadata.task: how miles_ext.tasks.single_turn finds grade
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


def grade(response, label):
    """(reward, kind): kind is ok, wrong, no_answer, multi_answer or bad_format. Only text after the reasoning counts."""
    text = response.split("</think>")[-1]
    found = ANSWER.findall(text)
    if not found:
        return 0, "no_answer"
    if len(found) > 1:
        return 0, "multi_answer"
    got = found[0].strip().split()
    if not got or not all(re.fullmatch(r"\d+", t) for t in got):
        return 0, "bad_format"
    return (1, "ok") if " ".join(str(int(t)) for t in got) == label else (0, "wrong")


def make(out, per_level, seed, levels):
    rng = random.Random(seed)
    rows = []
    for level in levels:
        for i in range(per_level):
            prog, stack, executed = make_task(rng, level)
            rows.append(
                {
                    "prompt": [{"role": "user", "content": prompt_text(prog, LEVELS[level]["mod"])}],
                    "metadata": {
                        "task": TASK,
                        "label": answer_of(stack),
                        "level": level,
                        "id": f"L{level}-{i:03d}",
                        "seed": seed,
                        "executed": executed,
                        "mod": LEVELS[level]["mod"],
                        "program": render(prog),
                    },
                }
            )
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def selftest():
    assert run([("PUSH", 5), ("PUSH", 3), ("SUB", None)])[0] == [95]  # 3 - 5 mod 97
    assert run([("PUSH", 1), ("PUSH", 2), ("PUSH", 3), ("ROT", None)])[0] == [2, 3, 1]
    assert run([("PUSH", 1), ("PUSH", 2), ("PUSH", 3), ("SINK", None)])[0] == [3, 1, 2]
    assert run([("PUSH", 7), ("HALF", None), ("PUSH", 10), ("HALF", None)])[0] == [22, 5]
    assert run([("PUSH", 0), ("NEG", None), ("PUSH", 4), ("NEG", None)])[0] == [0, 93]
    assert run([("PUSH", 2), ("REP", 3, [("DUP", None), ("ADD", None)])])[0] == [16]
    assert run([("PUSH", 4), ("PUSH", 9), ("CMP", None)])[0] == [1]
    assert answer_of([5, 0, 12]) == "12 0 5"
    for resp, want in [
        ("<think>maybe <answer>1</answer></think>\nSo <answer>12 0 5</answer>", (1, "ok")),
        ("<answer>12 00 5</answer>", (1, "ok")),
        ("<answer> 12  0\n5 </answer>", (1, "ok")),
        ("<answer>12 0</answer>", (0, "wrong")),
        ("<answer>12 0 5 0</answer>", (0, "wrong")),
        ("12 0 5", (0, "no_answer")),
        ("<answer>12 0 5</answer> or <answer>5 0 12</answer>", (0, "multi_answer")),
        ("<answer>12, 0, 5</answer>", (0, "bad_format")),
        ("<answer>-3</answer>", (0, "bad_format")),
        ("<answer></answer>", (0, "bad_format")),
        ("<think>long reasoning cut off <answer>12 0 5", (0, "no_answer")),
    ]:
        assert grade(resp, "12 0 5") == want, (resp, grade(resp, "12 0 5"))
    rows = make("/dev/null", 32, 1026, list(LEVELS))
    for r in rows:
        lab = r["metadata"]["label"]
        assert grade(f"<answer>{lab}</answer>", lab) == (1, "ok")
    assert len({r["metadata"]["program"] for r in rows}) == len(rows)
    print(
        "selftest ok;",
        {lv: sorted(r["metadata"]["executed"] for r in rows if r["metadata"]["level"] == lv)[16] for lv in LEVELS},
        "median executed per level",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--out", required=True)
    m.add_argument("--per-level", type=int, default=32)
    m.add_argument("--seed", type=int, default=1026)
    m.add_argument("--levels", default="1,2,3,4")
    sub.add_parser("selftest")
    a = p.parse_args()
    if a.cmd == "make":
        make(a.out, a.per_level, a.seed, [int(x) for x in a.levels.split(",")])
    else:
        selftest()
