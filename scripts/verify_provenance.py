"""Check that the configurations and the dataset match the archived results.

Every archived run records the SHA-256 hash of its configuration and the SHA-256 checksum of
the dataset. For each batch of the results package (``results/manifest.json``) this script
checks that:

* the configuration file in ``configs/`` has the hash recorded in the manifest and in every
  run of the batch;
* the dataset checksum recorded in every run equals the one expected by the configuration;
* every file of the package has the SHA-256 listed in the manifest;
* when ``data/total_dataset.csv`` is present, its SHA-256 equals the recorded checksum.

Usage: ``python scripts/verify_provenance.py [--archive DIR]``. Exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qnnwind.io import load_config  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, default=ROOT / "results", help="results package")
    args = parser.parse_args(argv)
    archive = args.archive.resolve()
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    failures = 0

    def report(ok: bool, text: str) -> None:
        nonlocal failures
        failures += not ok
        print(f"[{' OK ' if ok else 'FAIL'}] {text}")

    expected_datasets = set()
    for batch in manifest["batches"]:
        config = load_config(ROOT / batch["config"])
        runs = pd.read_csv(archive / batch["directory"] / "runs.csv.gz")
        recorded = set(runs["config_hash"])
        report(
            config.hash == batch["config_hash"] and recorded == {config.hash},
            f"{batch['name']}: {batch['config']} has hash {config.hash[:12]}, recorded in all "
            f"{len(runs)} runs" + ("" if recorded == {config.hash} else f" (runs: {recorded})"),
        )
        datasets = set(runs["dataset_sha256"])
        report(
            datasets == {config["data"]["sha256"]},
            f"{batch['name']}: dataset checksum {config['data']['sha256'][:12]} recorded in all "
            f"{len(runs)} runs" + ("" if len(datasets) == 1 else f" (runs: {datasets})"),
        )
        expected_datasets |= datasets

    bad = [f for f, info in manifest["files"].items() if sha256(archive / f) != info["sha256"]]
    report(
        not bad,
        f"package files: {len(manifest['files'])} checksums" + (f"; differ: {bad}" if bad else ""),
    )

    data = ROOT / "data" / "total_dataset.csv"
    if data.is_file():
        actual = sha256(data)
        report(
            actual in expected_datasets,
            f"dataset {data.relative_to(ROOT).as_posix()}: SHA-256 {actual[:12]}",
        )
    else:
        print(f"[skip] dataset not present ({data.relative_to(ROOT).as_posix()}); expected SHA-256 "
              f"{', '.join(sorted(expected_datasets))}")  # fmt: skip
    print("verify_provenance: " + ("all checks passed" if not failures else f"{failures} failed"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
