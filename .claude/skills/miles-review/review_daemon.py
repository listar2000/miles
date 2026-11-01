#!/usr/bin/env python3
"""Start review.py on each listar2000/miles PR that needs a review, and land PRs on `/merge`, as SKILL.md
"Review daemon" says."""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import review as R

HERE = Path(__file__).resolve().parent
MARKER = re.compile(r"<!-- miles-review head=(\w+) mode=(\w+) verdict=([\w-]+)(?: round=\d+)? -->")
COMMAND = re.compile(r"/review( deep)?")
MERGE = re.compile(r"/merge")
MERGE_MARKER = re.compile(r"<!-- miles-merge result=(\w+) -->")
TRAINER = re.compile(r"^(miles/|miles_plugins/|train_async\.py$)")
DEEP_LABEL = "deep-review"
BODY_LIST = {"context": re.compile(r"^Review context:(.*)$", re.M), "evidence": re.compile(r"^Evidence:(.*)$", re.M)}
MAX_FAILURES = 2
PACIFIC = ZoneInfo("America/Los_Angeles")
KEEP_DAYS = 14  # review work dirs older than this are pruned


def gh(*args):
    return json.loads(R.sh(["gh", *args], timeout=120).stdout)


def open_prs():
    fields = "number,headRefOid,headRefName,baseRefName,isDraft,author,labels,isCrossRepository,files,body"
    return gh("pr", "list", "-R", R.REPO_SLUG, "--state", "open", "--limit", "100", "--json", fields)


def comments(n):
    return [c for page in gh("api", "--paginate", "--slurp", f"repos/{R.REPO_SLUG}/issues/{n}/comments") for c in page]


def decide(pr, thread, allowed, done, now, first_seen, settle):
    """What to review on this PR now, as (mode, command comment or None), or None.

    `thread` is the PR's comments, oldest first; only comments by `allowed` users count, so nobody else can
    trigger a review or forge a review marker. A command counts if it is newer than the last edit of a review
    (the summary is edited in place each round). `done` holds "<head>:<mode>" reviews and "cmd:<id>" commands
    already run."""
    if pr["author"]["login"] not in allowed or pr["isCrossRepository"]:
        return None
    head, ours = pr["headRefOid"], [c for c in thread if c["user"]["login"] in allowed]
    last = max((c["updated_at"] for c in ours if MARKER.match(c["body"])), default="")
    for c in [c for c in ours if c["created_at"] > last][::-1]:
        if (m := COMMAND.fullmatch(c["body"].strip())) and f"cmd:{c['id']}" not in done:
            return ("deep" if m.group(1) or auto_deep(pr) else "normal"), c
    if pr["isDraft"] or now - first_seen.setdefault(head, now) < settle:
        return None
    mode = "deep" if auto_deep(pr) else "normal"
    reviewed = done | {f"{m.group(1)}:{m.group(2)}" for c in ours if (m := MARKER.match(c["body"]))}
    return None if {f"{head}:{mode}", f"{head}:deep"} & reviewed else (mode, None)


def latest_first(thread, allowed):
    """Allowed users' comments, most recently edited first."""
    return sorted((c for c in thread if c["user"]["login"] in allowed), key=lambda c: c["updated_at"], reverse=True)


def auto_deep(pr):
    return any(lb["name"] == DEEP_LABEL for lb in pr["labels"]) or any(TRAINER.match(f["path"]) for f in pr["files"])


def merge_request(thread, allowed, done):
    """The newest `/merge` comment by an allowed user that no merge result answers yet, or None."""
    ours = [c for c in thread if c["user"]["login"] in allowed]
    last = max((i for i, c in enumerate(ours) if MERGE_MARKER.match(c["body"])), default=-1)
    asks = [c for c in ours[last + 1 :] if MERGE.fullmatch(c["body"].strip()) and f"cmd:{c['id']}" not in done]
    return asks[-1] if asks else None


def merge_refusal(pr, thread, allowed):
    """Why `/merge` may not land this PR, or None: it must be a ready single PR whose latest review, at the
    current head, says "ready" (a deep review when the PR needs one)."""
    if pr["author"]["login"] not in allowed or pr["isCrossRepository"]:
        return "only PRs by allowed authors from this repo merge here"
    if pr["isDraft"]:
        return "the PR is a draft"
    if pr["baseRefName"] != "main":
        return f"the PR is stacked on `{pr['baseRefName']}`; stacks merge through the merge desk"
    reviews = [m for c in latest_first(thread, allowed) if (m := MARKER.match(c["body"]))][::-1]
    if not reviews or reviews[-1].group(1) != pr["headRefOid"]:
        return f"no Codex review at the current head {pr['headRefOid'][:8]}; comment `/review` first"
    if reviews[-1].group(3) != "ready":
        return f"the latest Codex verdict is \"{reviews[-1].group(3).replace('-', ' ')}\", not \"ready\""
    if auto_deep(pr) and reviews[-1].group(2) != "deep":
        return "this PR needs a deep review; comment `/review deep`"
    return None


def land(args, pr, by):
    """Fold the PR onto main, check it, push the branch and then main, as fork/README.md "Landing" says.
    `by` (the `/merge` commenter) is the folded commits' committer. Returns (merged, note); nothing is pushed
    unless every check passes."""
    head, branch, target = pr["headRefOid"], pr["headRefName"], pr["baseRefName"]  # target is main (merge_refusal)
    work = args.out / f"merge{pr['number']}-{head[:8]}"
    git = lambda *a, **kw: R.sh(["git", *a], cwd=work, check=False, timeout=600, **kw)  # noqa: E731
    R.sh(["git", "fetch", "-q", "--tags", "origin", target, branch], cwd=args.repo, timeout=600)
    shutil.rmtree(work, ignore_errors=True)
    R.sh(["git", "worktree", "prune"], cwd=args.repo)
    R.sh(["git", "worktree", "add", "-q", "--detach", str(work), head], cwd=args.repo)
    try:
        base = git("rev-parse", f"origin/{target}").stdout.strip()
        email = f"{by['id']}+{by['login']}@users.noreply.github.com"
        env = {"GIT_SEQUENCE_EDITOR": ":", "GIT_COMMITTER_NAME": by["login"], "GIT_COMMITTER_EMAIL": email}
        if (rebase := git("rebase", "-q", "-i", "--autosquash", base, env=env)).returncode:
            git("rebase", "--abort")
            return False, f"the branch does not fold onto main:\n```\n{R.tail(rebase, 10)}\n```"
        folded = git("rev-parse", "HEAD").stdout.strip()
        main_moved = git("merge-base", "--is-ancestor", base, head).returncode
        if not main_moved and git("diff", "--quiet", head, folded).returncode:
            return False, "folding changed the code relative to the reviewed head"
        if R.FIXUP.search(git("log", "--format=%s", f"{base}..{folded}").stdout):
            return False, "a fixup! commit names no commit of this PR, so it does not fold"
        runner = (
            R.ModalRunner(work, (folded,), work) if args.checks_on == "modal" else R.LocalRunner(work, args.python)
        )
        try:
            ledger = runner.run([runner.python, "fork/tools/ledger.py", "check"], timeout=600)
        finally:
            runner.close()
        if ledger.returncode:
            return False, f"`ledger.py check` fails on the folded branch:\n```\n{R.tail(ledger, 15)}\n```"
        if args.dry_run:
            return False, f"dry run: would land {folded[:8]} on main {base[:8]}"
        # GitHub must see the folded PR head before main reaches it to recognize the merge.
        n = str(pr["number"])
        push = git(
            "push", "-q", f"--force-with-lease=refs/heads/{branch}:{head}", "origin", f"{folded}:refs/heads/{branch}"
        )
        if push.returncode:
            return False, f"the branch moved; nothing was pushed:\n```\n{R.tail(push, 10)}\n```"
        for _ in range(30):
            if gh("pr", "view", n, "-R", R.REPO_SLUG, "--json", "headRefOid")["headRefOid"] == folded:
                break
            time.sleep(5)
        else:
            return False, "GitHub did not show the folded head on the PR; the branch holds it, main is unchanged"
        if (push := git("push", "-q", "origin", f"{folded}:refs/heads/{target}")).returncode:
            return False, f"main moved; the branch now holds the folded commits:\n```\n{R.tail(push, 10)}\n```"
        for _ in range(30):
            if gh("pr", "view", n, "-R", R.REPO_SLUG, "--json", "state")["state"] == "MERGED":
                git("push", "-q", "origin", "--delete", branch)
                break
            time.sleep(10)
        return True, f"landed on main as {folded[:8]} (folded from {head[:8]}, ledger ok)"
    finally:
        R.sh(["git", "worktree", "remove", "--force", str(work)], cwd=args.repo, check=False)


def body_list(pr, key):
    """Paths from a PR-body line; these reach review.py's argv, so no option-like token (`--checks-on local`)
    and no evidence but our volumes'."""
    items = [x for line in BODY_LIST[key].findall(pr["body"] or "") for x in line.split() if not x.startswith("-")]
    return [x for x in items if x.startswith("modal://")] if key == "evidence" else items


def review_argv(args, pr, mode, prior):
    argv = [args.python, str(HERE / "review.py"), str(pr["number"]), "--python", args.python, "--out", str(args.out)]
    argv += ["--checks-on", args.checks_on, "--repo", str(args.repo)]
    argv += [] if args.dry_run else ["--post"]
    argv += ["--board", str(args.board)] if args.board else []
    argv += ["--prior", str(prior)] if prior else []
    if mode == "deep":
        argv.append("--deep")
        for key in ("context", "evidence"):
            if items := body_list(pr, key):
                argv += [f"--{key}", *items]
    return argv


class Review:
    """One review.py run in the background, in its own process group with its own log."""

    def __init__(self, args, pr, mode, command, prior):
        self.pr, self.mode, self.since = pr, mode, time.time()
        self.key = f"{pr['headRefOid']}:{mode}"
        self.settled = [self.key] + ([f"cmd:{command['id']}"] if command else [])
        logs = args.out / "logs"
        logs.mkdir(exist_ok=True)
        with (logs / f"pr{pr['number']}-{pr['headRefOid'][:8]}-{mode}.log").open("w") as f:
            argv = review_argv(args, pr, mode, prior)
            self.p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, cwd=args.repo, start_new_session=True)
        note(args.out, f"=== {self.since:.0f} PR #{pr['number']} {pr['headRefOid'][:8]} {mode}")

    def stop(self):
        if self.p.poll() is None:  # the agent and review.py's sandbox go with it (review.py closes it on SIGTERM)
            os.killpg(self.p.pid, signal.SIGTERM)
            self.p.wait()

    def info(self):
        return {"pr": self.pr["number"], "head": self.pr["headRefOid"], "mode": self.mode, "since": self.since}


def note(out, line):
    with (out / "daemon.log").open("a") as f:
        f.write(line + "\n")


def finish(args, state, review):
    """Record a finished review: done, or a failure (an error comment after MAX_FAILURES)."""
    pr, key = review.pr, review.key
    ok = review.p.returncode == 0
    note(args.out, f"PR #{pr['number']} {pr['headRefOid'][:8]} {review.mode}: {'ok' if ok else 'failed'}")
    if ok:
        state["done"] += review.settled
    elif (fails := state["failures"].get(key, 0) + 1) < MAX_FAILURES:
        state["failures"][key] = fails
    else:
        state["done"] += review.settled
        if not args.dry_run:
            marker = R.MARKER.format(head=pr["headRefOid"], mode=review.mode, verdict="error", round=0)
            body = f"{marker}\nThe review daemon could not review this head; comment `/review` to retry."
            R.sh(["gh", "pr", "comment", str(pr["number"]), "-R", R.REPO_SLUG, "--body", body], check=False)


def merge(args, pr, thread, ask):
    """Answer one `/merge` comment with a result comment, marked so it is answered only once."""
    n = pr["number"]
    if not args.dry_run:
        R.sh(["gh", "api", "-X", "POST", f"repos/{R.REPO_SLUG}/issues/comments/{ask['id']}/reactions",
              "-f", "content=eyes"], check=False)  # fmt: skip
    if why := merge_refusal(pr, thread, set(args.allow)):
        merged, result = False, f"Not merged: {why}."
    else:
        try:
            merged, result = land(args, pr, ask["user"])
        except SystemExit as e:  # a git, gh or sandbox step failed
            merged, result = False, f"the merge stopped: {e}"
        result = result if merged or result.startswith("dry run") else f"Not merged: {result}"
    note(args.out, f"=== {time.time():.0f} PR #{n} {pr['headRefOid'][:8]} merge\n{result}")
    if not args.dry_run:
        body = f"<!-- miles-merge result={'merged' if merged else 'refused'} -->\n{result}"
        R.sh(["gh", "pr", "comment", str(n), "-R", R.REPO_SLUG, "--body", body], check=False)


def beat(out, **fields):
    (out / "heartbeat.json").write_text(json.dumps({"time": time.time(), **fields}))


def pt(t):
    """A Unix time as the person on the box reads times: Pacific."""
    return datetime.fromtimestamp(t, PACIFIC).strftime("%b %d %I:%M %p PT")


def status(out):
    """What the daemon is doing, for a person on the box: `review_daemon.py --status`."""
    hb = json.loads((out / "heartbeat.json").read_text()) if (out / "heartbeat.json").exists() else {}
    age = f"{pt(hb['time'])} ({time.time() - hb['time']:.0f} s ago)" if hb else "never"
    print(f"heartbeat: {age} (a pass every --poll seconds)")
    for run in hb.get("running", []):
        print(f"reviewing: PR #{run['pr']} {run['head'][:8]} {run['mode']}, since {pt(run['since'])}")
    print(f"queue: {', '.join(f'#{n}' for n in hb.get('queue', [])) or 'empty'}")
    lines = (out / "daemon.log").read_text().splitlines() if (out / "daemon.log").exists() else []
    print("recent reviews:")
    for line in [ln for ln in lines if ln.startswith(("===", "PR #", "Not merged", "landed", "dry run"))][-12:]:
        if m := re.match(r"=== (\d+) (.*)", line):
            line = f"{pt(int(m.group(1)))} {m.group(2)}"
        print(f"  {line}")
    state = json.loads((out / "daemon-state.json").read_text()) if (out / "daemon-state.json").exists() else {}
    if fails := state.get("failures"):
        print(f"failed attempts: {fails}")


def update_checkout(repo):
    """Fast-forward the reviewer's checkout to main, so reviews use main's protocol; True if this file changed."""
    if R.sh(["git", "branch", "--show-current"], cwd=repo).stdout.strip() != "main":
        return False  # a test checkout on a topic branch stays as it is
    before = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    R.sh(["git", "fetch", "-q", "origin", "main"], cwd=repo, timeout=300)
    R.sh(["git", "merge", "-q", "--ff-only", "origin/main"], cwd=repo, timeout=300)
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != before


class Stopped(BaseException):
    """A signal to stop; not a SystemExit, so the `except SystemExit` around gh() and land() cannot swallow it."""


def stop(*_):
    raise Stopped


def main():
    for sig in (signal.SIGTERM, signal.SIGHUP):  # a stopped daemon runs main's finally, ending its reviews
        signal.signal(sig, stop)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=HERE.parents[2], help="the reviewer's own main checkout")
    ap.add_argument("--out", type=Path, default=Path("~/miles-review").expanduser())
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--board", type=Path, help="coordination board with fork id reservations")
    ap.add_argument("--allow", nargs="+", default=["listar2000"], help="whose PRs and /review comments count")
    ap.add_argument("--checks-on", choices=("local", "modal"), default="modal")
    ap.add_argument("--poll", type=int, default=120, help="seconds between GitHub polls")
    ap.add_argument("--parallel", type=int, default=5, help="reviews at once, each its own Codex session")
    ap.add_argument("--settle", type=int, default=120, help="seconds a new head waits, so quick pushes coalesce")
    ap.add_argument("--dry-run", action="store_true", help="review without posting; reviewed heads kept locally")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--status", action="store_true", help="print what the daemon is doing, then exit")
    args = ap.parse_args()
    if args.status:
        return status(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    state_file = args.out / "daemon-state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"done": [], "failures": {}}
    first_seen, running, restart = {}, {}, False
    try:
        while True:
            restart = restart or update_checkout(args.repo)
            if restart and not running:
                sys.exit("review_daemon.py changed on main; exiting for the supervisor to restart it")
            if args.board:  # fork id reservations
                R.sh(["git", "pull", "-q", "--ff-only"], cwd=args.board.resolve().parent, timeout=300, check=False)
            for old in args.out.glob("pr*-*/"):
                if time.time() - old.stat().st_mtime > KEEP_DAYS * 86400:
                    shutil.rmtree(old, ignore_errors=True)
            prs = {pr["number"]: pr for pr in open_prs()}
            for n, review in list(running.items()):
                if review.p.poll() is not None:
                    finish(args, state, running.pop(n))
                elif prs.get(n, {}).get("headRefOid") != review.pr["headRefOid"]:  # new head, or closed
                    running.pop(n).stop()
                    note(args.out, f"PR #{n} {review.pr['headRefOid'][:8]} {review.mode}: stopped, the PR moved on")
            jobs = []
            for n, pr in prs.items():
                thread = comments(n)
                if n not in running and (ask := merge_request(thread, set(args.allow), set(state["done"]))):
                    state["done"].append(f"cmd:{ask['id']}")
                    state_file.write_text(json.dumps(state))
                    merge(args, pr, thread, ask)  # one at a time: merges move main
                    continue
                if n in running or restart:
                    continue
                if want := decide(
                    pr, thread, set(args.allow), set(state["done"]), time.time(), first_seen, args.settle
                ):
                    jobs.append((want[1] is None, n, pr, thread, *want))
            for _, n, pr, thread, mode, command in sorted(jobs, key=lambda j: j[:2])[: args.parallel - len(running)]:
                if command and not args.dry_run:
                    R.sh(["gh", "api", "-X", "POST", f"repos/{R.REPO_SLUG}/issues/comments/{command['id']}/reactions",
                          "-f", "content=eyes"], check=False)  # fmt: skip
                reviews = [c for c in latest_first(thread, args.allow) if MARKER.match(c["body"])]
                prior = None
                if reviews:
                    prior = args.out / f"pr{n}-prior.md"
                    prior.write_text(reviews[0]["body"])
                running[n] = Review(args, pr, mode, command, prior)
            state_file.write_text(json.dumps(state))
            beat(
                args.out,
                running=[r.info() for r in running.values()],
                queue=[j[1] for j in jobs if j[1] not in running],
            )
            if args.once and not running:
                return
            time.sleep(args.poll)
    finally:  # a stop, a crash or a restart ends the reviews this daemon started
        for review in running.values():
            review.stop()


if __name__ == "__main__":
    try:
        main()
    except Stopped:
        sys.exit("stopped by a signal")
