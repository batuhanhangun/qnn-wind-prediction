"""The run directory: results, logs, and task files of every configuration.

The configurations place these under ``${QNNWIND_SCRATCH:-.}``. The scripts choose the run
directory as follows: the ``--runs`` argument if given, else the ``QNNWIND_SCRATCH``
environment variable, else ``runs/`` inside the repository (ignored by git). The choice is
passed on through ``QNNWIND_SCRATCH``, so worker processes use the same directory. The
configuration files themselves are not changed, so their hashes stay the same.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "runs"
ENV = "QNNWIND_SCRATCH"


def use_runs_dir(runs: Path | str | None = None) -> Path:
    """Set ``QNNWIND_SCRATCH`` for this process and its children; return the run directory."""
    if runs is not None:
        os.environ[ENV] = str(Path(runs).resolve())
    elif not os.environ.get(ENV):
        os.environ[ENV] = str(DEFAULT_RUNS)
    return Path(os.environ[ENV])


def add_runs_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runs",
        type=Path,
        help=f"run directory (default: ${ENV} if set, else runs/ in the repository)",
    )
