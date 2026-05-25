"""
Batch task runner for data collection.

Reads tasks from suite.json (or a custom file), fires each one at the
/task endpoint, and writes results to a timestamped JSON file.

Usage:
    python -m tasks.runner                          # all tasks, both agents
    python -m tasks.runner --agents dom             # DOM-only baseline
    python -m tasks.runner --agents vision          # vision-only baseline
    python -m tasks.runner --filter form            # only tasks in the 'form' category
    python -m tasks.runner --filter herokuapp       # name substring match
    python -m tasks.runner --repeat 3               # run each task 3 times
    python -m tasks.runner --host http://remote:8000
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

DEFAULT_SUITE = Path(__file__).parent / "suite.json"
DEFAULT_HOST = "http://localhost:8000"
TIMEOUT = 120  # seconds per task — sessions can take a while


def _post_task(host: str, task: dict, agents: str) -> dict:
    payload = json.dumps({
        "task": task["task"],
        "url": task["url"],
        "agents": agents,
    }).encode()
    req = Request(
        f"{host}/task",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        body = e.read().decode()
        return {"error": f"HTTP {e.code}", "detail": body}
    except URLError as e:
        return {"error": "connection_failed", "detail": str(e)}
    except Exception as e:
        return {"error": "unknown", "detail": str(e)}


def _matches_filter(task: dict, filt: str) -> bool:
    filt = filt.lower()
    return (
        filt in task["name"].lower()
        or filt in task["category"].lower()
        or filt in task.get("difficulty", "").lower()
    )


def run(
    suite_path: Path,
    host: str,
    agents: str,
    filt: str | None,
    repeat: int,
) -> None:
    with open(suite_path) as f:
        all_tasks = json.load(f)

    if filt:
        all_tasks = [t for t in all_tasks if _matches_filter(t, filt)]

    if not all_tasks:
        print("No tasks matched the filter.")
        sys.exit(1)

    total = len(all_tasks) * repeat
    print(f"Running {len(all_tasks)} task(s) x{repeat} = {total} session(s)  agents={agents}")
    print(f"Host: {host}")
    print(f"{'─' * 80}")

    results = []
    successes = 0
    failures = 0

    for iteration in range(repeat):
        for i, task in enumerate(all_tasks):
            run_num = iteration * len(all_tasks) + i + 1
            label = f"[{run_num}/{total}]"
            print(f"{label} {task['name']:<35} ", end="", flush=True)

            t0 = time.time()
            resp = _post_task(host, task, agents)
            elapsed = time.time() - t0

            result = {
                "task_name": task["name"],
                "category": task["category"],
                "difficulty": task["difficulty"],
                "agents": agents,
                "iteration": iteration,
                "elapsed_s": round(elapsed, 1),
                "response": resp,
            }
            results.append(result)

            if "error" in resp:
                print(f"ERROR  {resp['error']}  ({elapsed:.1f}s)")
                failures += 1
            else:
                status = resp.get("status", "?")
                reason = resp.get("termination_reason", "?")
                steps = resp.get("step_count", 0)
                tag = "OK" if status == "complete" else "FAIL"
                print(f"{tag:<5}  {reason:<25} steps={steps:<3} ({elapsed:.1f}s)")
                if status == "complete":
                    successes += 1
                else:
                    failures += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 80}")
    print(f"  RESULTS  —  {successes} passed, {failures} failed, {total} total")
    print(f"{'═' * 80}")

    # Breakdown by category
    by_cat: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"ok": 0, "fail": 0}
        status = r["response"].get("status", "")
        if status == "complete":
            by_cat[cat]["ok"] += 1
        else:
            by_cat[cat]["fail"] += 1

    print(f"\n  {'category':<15} {'pass':>6} {'fail':>6} {'rate':>8}")
    for cat, counts in sorted(by_cat.items()):
        total_cat = counts["ok"] + counts["fail"]
        rate = counts["ok"] / total_cat * 100 if total_cat else 0
        print(f"  {cat:<15} {counts['ok']:>6} {counts['fail']:>6} {rate:>7.0f}%")

    # Breakdown by termination reason
    by_reason: dict[str, int] = {}
    for r in results:
        reason = r["response"].get("termination_reason", r["response"].get("error", "unknown"))
        by_reason[reason] = by_reason.get(reason, 0) + 1

    print(f"\n  {'termination':<25} {'count':>6}")
    for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
        print(f"  {reason:<25} {count:>6}")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent / f"results_{agents}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decido batch task runner")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--agents", default="both", choices=["both", "dom", "vision"])
    parser.add_argument("--filter", dest="filt", default=None, help="Filter by name/category/difficulty substring")
    parser.add_argument("--repeat", type=int, default=1, help="Run each task N times")
    args = parser.parse_args()
    run(args.suite, args.host, args.agents, args.filt, args.repeat)
