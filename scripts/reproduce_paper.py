"""Level 1: regenerate every table and figure of the paper from the archived results.

The three result batches of the paper (blocked protocol with the target scaled to [-1, 1];
blocked protocol, QNN-1u to QNN-6u with the target scaled to [0, 1]; random K-fold protocol)
are read from the results package (``results/``, see results/README.md), analyzed with
scripts/analyze.py into a fresh directory, and every table and figure is compared with its
reference copy in ``paper/`` (scripts/compare_outputs.py): tables must be identical; figures
must be identical or, across platforms, equivalent (same pixels or same drawing content).

T1, T1b (descriptive statistics and fold blocks of the dataset) and A1 (actual vs predicted
power) are computed from the dataset itself. Without ``data/total_dataset.csv`` they are not
regenerated: their reference copies are placed in the output directory instead, and this is
reported. Everything else is regenerated from the package alone.

Usage::

    python scripts/reproduce_paper.py [--outputs DIR] [--reference DIR] [--archive DIR]

``--source NAME CONFIG RUNS`` (repeatable, as for scripts/analyze.py) replaces the batches of
the package. Exits non-zero if any regenerated table or figure differs from its reference.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze  # noqa: E402
import compare_outputs  # noqa: E402
from aggregate import dataset_available  # noqa: E402
from qnnwind.io import load_config  # noqa: E402
from runs_dir import DEFAULT_RUNS  # noqa: E402

ARCHIVE = ROOT / "results"
REFERENCE = ROOT / "paper"
# Outputs computed from the dataset itself.
DATA_OUTPUTS = (
    "tables/t1_descriptive.tex",
    "tables/t1b_folds.tex",
    "figures/A1_actual_vs_predicted.pdf",
    "figures/A1_actual_vs_predicted.png",
)


def package_sources(archive: Path) -> list[list[str]]:
    """``--source`` arguments for the batches listed in the package manifest."""
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    return [
        [b["name"], str(ROOT / b["config"]), str(archive / b["directory"])]
        for b in manifest["batches"]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outputs", type=Path, default=DEFAULT_RUNS / "paper")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--archive", type=Path, default=ARCHIVE, help="results package")
    parser.add_argument(
        "--source", nargs=3, action="append", metavar=("NAME", "CONFIG", "RUNS"), default=[]
    )
    args = parser.parse_args(argv)
    outputs, reference = args.outputs.resolve(), args.reference.resolve()
    if outputs.exists() and any(outputs.iterdir()):
        print(f"{outputs} is not empty: choose another --outputs or remove it", file=sys.stderr)
        return 2
    sources = args.source or package_sources(args.archive.resolve())

    start = time.perf_counter()
    copied: list[str] = []
    if not dataset_available(load_config(sources[0][1])):
        for rel in DATA_OUTPUTS:
            (outputs / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(reference / rel, outputs / rel)
            copied.append(rel)
    analyze_args = [a for s in sources for a in ("--source", *s)]
    code = analyze.main([*analyze_args, "--outputs", str(outputs)])
    if code:
        return code
    if copied:
        print(
            "reproduce_paper: no dataset: T1, T1b, and A1 are the reference copies "
            f"({len(copied)} files), not regenerated"
        )
    identical, differences, equivalent = compare_outputs.compare(outputs, reference)
    print(compare_outputs.report(identical, differences, args.reference, equivalent))
    print(f"reproduce_paper: {time.perf_counter() - start:.0f} s; outputs in {outputs}")
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
