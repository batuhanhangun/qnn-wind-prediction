"""Write the task files for the task-farm launcher.

Usage::

    python scripts/make_tasks.py --config CONFIG                  # tuning, qnn, classical
    python scripts/make_tasks.py --config CONFIG --calibration    # calibration_W*

Task IDs and file contents are deterministic: the same config always gives the same files.
Existing task files are overwritten; results are never touched.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qnnwind.io import load_config  # noqa: E402
from qnnwind.tasks import build_tasks, calibration_tasks, write_task_file  # noqa: E402
from runs_dir import add_runs_argument, use_runs_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--calibration", action="store_true", help="write the calibration groups instead"
    )
    add_runs_argument(parser)
    args = parser.parse_args(argv)
    use_runs_dir(args.runs)
    config = load_config(args.config)

    groups = calibration_tasks(config) if args.calibration else build_tasks(config)
    print(f"config {config.source.name} (hash {config.hash[:12]})")
    for group, tasks in groups.items():
        path = write_task_file(config, group, tasks)
        print(f"  {group:<16} {len(tasks):>5} tasks -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
