"""Run the whole grid of one configuration on this PC, then analyze it.

Steps: write the task lists (scripts/make_tasks.py); run the QNN tasks and, at the same time,
the tuning studies followed by the classical runs, which need the tuned hyperparameters;
then write the tables, figures, and summary (scripts/analyze.py). QNN runs take far longer
than the rest, so they get W - 1 workers (at most one per QNN task) and the tuning and
classical tasks share the remaining worker(s). The two launchers claim tasks through the
same lock directories as on a cluster, so a task never runs twice. Tasks whose results
already exist are skipped, so an interrupted run resumes where it stopped.

Usage::

    python scripts/run_grid.py --config configs/quick.yaml [--workers W] [--runs DIR]

Results go to the run directory (default ``runs/``); the tables and figures go to
``<run directory>/outputs/<config name>`` unless ``--outputs`` is given. On a cluster, use
the Slurm templates in ``slurm/`` instead.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aggregate import require_dataset  # noqa: E402
from launcher import local_workers  # noqa: E402
from qnnwind.io import load_config  # noqa: E402
from qnnwind.tasks import read_task_file, task_is_done  # noqa: E402
from runs_dir import add_runs_argument, use_runs_dir  # noqa: E402

SCRIPTS = ROOT / "scripts"


def script(name: str, *args: str) -> list[str]:
    return [sys.executable, str(SCRIPTS / name), *args]


def minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} min"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workers", type=int, help="W (default: physical cores minus one)")
    parser.add_argument("--outputs", type=Path, help="tables and figures (see above)")
    parser.add_argument("--no-analyze", action="store_true", help="skip the analysis step")
    add_runs_argument(parser)
    args = parser.parse_args(argv)
    runs = use_runs_dir(args.runs)
    config_arg = str(args.config)
    config = load_config(args.config)
    require_dataset(config)
    workers = args.workers or local_workers()
    common = ["--config", config_arg, "--runs", str(runs), "--no-pin"]

    start = time.perf_counter()
    subprocess.run(script("make_tasks.py", "--config", config_arg), check=True)
    qnn_todo = [t for t in read_task_file(config, "qnn") if not task_is_done(config, t)]
    other_groups = [g for g in ("tuning", "classical") if read_task_file(config, g)]
    qnn_workers = min(len(qnn_todo), max(1, workers - 1)) if other_groups else workers
    other_workers = max(1, workers - qnn_workers)
    print(
        f"run_grid: {config.source.name}, W = {workers}: {len(qnn_todo)} QNN task(s) on "
        f"{qnn_workers} worker(s), {' then '.join(other_groups) or 'nothing else'} on "
        f"{other_workers} worker(s); run directory {runs}",
        flush=True,
    )
    procs = []
    if qnn_todo:
        procs.append(
            subprocess.Popen(
                script("launcher.py", *common, "--group", "qnn", "--workers", str(qnn_workers))
            )
        )
    if other_groups:
        groups = [a for g in other_groups for a in ("--group", g)]
        procs.append(
            subprocess.Popen(
                script("launcher.py", *common, *groups, "--workers", str(other_workers))
            )
        )
    failed = sum(p.wait() != 0 for p in procs)
    run_time = time.perf_counter() - start
    print(f"run_grid: tasks finished in {minutes(run_time)} ({failed} launcher(s) failed)")
    if failed:
        print("run_grid: see scripts/status.py --failed; rerun this command to resume")
        return 1
    if args.no_analyze:
        return 0

    outputs = args.outputs or runs / "outputs" / args.config.stem
    analyze = ["--config", config_arg, "--runs", str(runs), "--outputs", str(outputs)]
    subprocess.run(script("analyze.py", *analyze), check=True)
    total = time.perf_counter() - start
    print(f"run_grid: done in {minutes(total)} (W = {workers}); tables and figures in {outputs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
