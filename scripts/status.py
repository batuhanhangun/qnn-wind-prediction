"""Task status: done, failed, running, and pending counts per group.

Usage::

    python scripts/status.py --config configs/experiment.yaml                 # all task files
    python scripts/status.py --config configs/experiment.yaml --group qnn --failed
    python scripts/status.py --config configs/experiment.yaml --calibration   # choose W

"running" means a worker holds the task's lock and its heartbeat is recent. A task whose lock
went stale (its job was killed, e.g. by the walltime) counts as pending and is rerun by the
next launcher. ``--calibration`` summarizes the calibration groups: time per circuit for
each W, the slowdown relative to W = 1, and the chosen W (written to
``<results>/calibration/summary.json``).
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qnnwind.io import Config, load_config, read_json, write_json  # noqa: E402
from qnnwind.tasks import (  # noqa: E402
    FINAL_GROUPS,
    log_path,
    read_task_file,
    task_output,
    task_status,
)
from runs_dir import add_runs_argument, use_runs_dir  # noqa: E402

STATES = ("done", "failed", "running", "pending")


def group_names(config: Config, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return sorted(p.stem for p in config.path("tasks").glob("*.json"))


def print_status(
    config: Config, groups: list[str], show_failed: bool, lenient: bool = False
) -> int:
    """Print the status table; return the number of failed tasks.

    With ``lenient`` (reporting all groups found on disk), a task file written under another
    config hash is skipped with a warning, unless it is a final group (tuning, qnn,
    classical), which always keeps the strict check.
    """
    print(f"{'group':<18}" + "".join(f"{s:>9}" for s in STATES) + f"{'total':>9}")
    failed_total = 0
    for group in groups:
        try:
            tasks = read_task_file(config, group)
        except RuntimeError as exc:
            if not lenient or group in FINAL_GROUPS:
                raise
            print(f"WARNING: skipping {group}: {exc}", file=sys.stderr)
            continue
        by_state: dict[str, list] = {s: [] for s in STATES}
        for task in tasks:
            by_state[task_status(config, task)].append(task)
        counts = {s: len(v) for s, v in by_state.items()}
        failed_total += counts["failed"]
        print(f"{group:<18}" + "".join(f"{counts[s]:>9}" for s in STATES) + f"{len(tasks):>9}")
        if show_failed:
            for task in by_state["failed"]:
                print(f"    FAILED {task.id}: {log_path(config, task)}")
    return failed_total


def calibration_summary(config: Config) -> dict:
    """Time per circuit for each W and the chosen W."""
    cal = config["calibration"]
    rows = []
    for w in cal["workers"]:
        group = f"calibration_W{w:03d}"
        per_copy = []
        for task in read_task_file(config, group):
            result_file = task_output(config, task) / "result.json"
            if result_file.is_file():
                s = read_json(result_file)["model_summary"]
                per_copy.append(s["optimizer_time"] / s["circuits_evaluated"])
        rows.append(
            {
                "W": w,
                "copies_completed": len(per_copy),
                "copies_expected": w,
                "ms_per_circuit_mean": 1e3 * statistics.fmean(per_copy) if per_copy else None,
                "ms_per_circuit_max": 1e3 * max(per_copy) if per_copy else None,
            }
        )
    base = rows[0]["ms_per_circuit_mean"]
    for row in rows:
        mean = row["ms_per_circuit_mean"]
        row["slowdown"] = mean / base - 1 if (mean is not None and base) else None
    eligible = [
        r["W"]
        for r in rows
        if r["slowdown"] is not None
        and r["copies_completed"] == r["copies_expected"]
        and r["slowdown"] <= cal["max_slowdown"]
    ]
    complete = all(r["copies_completed"] == r["copies_expected"] for r in rows)
    return {
        "rows": rows,
        "max_slowdown": cal["max_slowdown"],
        "complete": complete,
        "chosen_W": max(eligible) if (eligible and complete) else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--group", action="append", help="repeatable (default: all groups)")
    parser.add_argument("--failed", action="store_true", help="list failed tasks and logs")
    parser.add_argument("--calibration", action="store_true", help="summarize the calibration")
    add_runs_argument(parser)
    args = parser.parse_args(argv)
    use_runs_dir(args.runs)
    config = load_config(args.config)

    if args.calibration:
        summary = calibration_summary(config)
        out = config.path("results") / "calibration" / "summary.json"
        write_json(out, summary)
        print(f"{'W':>5}{'copies':>10}{'ms/circuit':>13}{'max':>9}{'slowdown':>11}")
        for r in summary["rows"]:
            mean = f"{r['ms_per_circuit_mean']:.3f}" if r["ms_per_circuit_mean"] else "-"
            peak = f"{r['ms_per_circuit_max']:.3f}" if r["ms_per_circuit_max"] else "-"
            slow = f"{100 * r['slowdown']:+.1f}%" if r["slowdown"] is not None else "-"
            copies = f"{r['copies_completed']}/{r['copies_expected']}"
            print(f"{r['W']:>5}{copies:>10}{mean:>13}{peak:>9}{slow:>11}")
        limit = 100 * summary["max_slowdown"]
        print(f"chosen W (largest with slowdown <= {limit:.0f}%): {summary['chosen_W']}")
        print(f"wrote {out}")
        return 0 if summary["chosen_W"] is not None else 1

    groups = group_names(config, args.group)
    if not groups:
        print(f"no task files in {config.path('tasks')}; run scripts/make_tasks.py first")
        return 1
    failed = print_status(config, groups, args.failed, lenient=not args.group)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
