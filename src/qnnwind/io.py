"""Configuration loading, LF-only file writing, run directories, and run metadata.

Every file the project writes goes through the helpers in this module, so that line endings
are LF on every platform.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Packages whose versions are recorded in every result.json.
RECORDED_PACKAGES = (
    "qiskit",
    "qiskit-machine-learning",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "torch",
    "optuna",
    "matplotlib",
    "PyYAML",
    "psutil",
)

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated recursively with ``override`` (neither is modified)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ValueError(f"Config {path} must contain a mapping")
    base = content.pop("base", None)
    if base is not None:
        content = _deep_merge(_read_yaml(path.parent / base), content)
    return content


def _set_dotted(tree: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = tree
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            raise KeyError(f"Override {dotted_key!r}: {key!r} is not a config section")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"Override {dotted_key!r} does not match an existing config key")
    node[keys[-1]] = value


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(text: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` (default used when VAR is unset or empty)."""

    def replace(match: re.Match[str]) -> str:
        value = os.environ.get(match.group(1), "")
        if value:
            return value
        return match.group(2) if match.group(2) is not None else match.group(0)

    return _ENV_PATTERN.sub(replace, text)


@dataclass(frozen=True)
class Config:
    """A resolved experiment configuration.

    Attributes:
        raw: The merged configuration tree (after ``base`` inheritance and overrides).
        source: Path of the YAML file that was loaded.
        root: Project root; relative paths in ``raw["paths"]`` are resolved against it.
    """

    raw: dict[str, Any]
    source: Path
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, name: str) -> Path:
        """Return the absolute path ``paths.<name>`` with environment variables expanded.

        ``${VAR}`` and ``${VAR:-default}`` are supported, so one config serves both the local
        machine and a cluster (``QNNWIND_SCRATCH`` set by the scripts or the job script).
        """
        expanded = expand_env(str(self.raw["paths"][name]))
        if "$" in expanded:
            raise ValueError(f"Unresolved environment variable in paths.{name}: {expanded}")
        path = Path(expanded)
        return path if path.is_absolute() else (self.root / path).resolve()

    @property
    def protocol(self) -> str:
        """Evaluation protocol: ``blocked`` (the default; experiment.yaml does not set the
        key) or ``random``."""
        return str(self.raw.get("protocol", "blocked"))

    def folds(self, n_rows: int) -> list:
        """The folds of this config's protocol."""
        from qnnwind.folds import folds_for

        return folds_for(self.raw["folds"], n_rows, self.protocol, self.raw.get("random_split"))

    @property
    def hash(self) -> str:
        """SHA-256 of the canonical JSON form of the merged configuration."""
        canonical = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config, apply ``base`` inheritance and dotted-key overrides.

    Args:
        path: Config file, e.g. ``configs/experiment.yaml``.
        overrides: Optional mapping such as ``{"qnn.optimizer.maxiter": 5}``. Keys must exist.

    Returns:
        The resolved :class:`Config`. The project root is the parent of the config directory.
    """
    source = Path(path).resolve()
    raw = _read_yaml(source)
    for key, value in (overrides or {}).items():
        _set_dotted(raw, key, value)
    return Config(raw=raw, source=source, root=source.parent.parent)


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``key=value`` where ``value`` is YAML (``qnn.optimizer.maxiter=5``)."""
    key, sep, value = text.partition("=")
    if not sep:
        raise ValueError(f"Override must look like key=value, got {text!r}")
    return key.strip(), yaml.safe_load(value)


# --------------------------------------------------------------------------------------
# LF-only writers
# --------------------------------------------------------------------------------------


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text with LF line endings, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _json_default(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as indented JSON with LF line endings."""
    write_text(path, json.dumps(obj, indent=2, default=_json_default, allow_nan=True) + "\n")


def add_run_metadata(result_file: Path, key: str, value: Any) -> None:
    """Add ``metadata[key]`` to a run's result.json, atomically (temporary file + replace).

    Used by the launcher to complete the metadata of the run it has just executed (the node
    load); existing keys are never replaced.
    """
    result = read_json(result_file)
    if key in result["metadata"]:
        raise KeyError(f"{result_file}: metadata.{key} already present")
    result["metadata"][key] = value
    tmp = result_file.with_name(f"{result_file.name}.tmp-{os.getpid()}")
    write_json(tmp, result)
    os.replace(tmp, result_file)


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a DataFrame as CSV with LF line endings and no index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# Run directories
# --------------------------------------------------------------------------------------


def run_dir(results: Path, model: str, n_train: int, fold: int, seed: int) -> Path:
    """``results/raw/<model>/N<N>/fold<k>/seed<s>``."""
    return results / "raw" / model / f"N{n_train}" / f"fold{fold}" / f"seed{seed}"


def tuning_dir(results: Path, model: str, n_train: int, fold: int) -> Path:
    """``results/tuning/<model>/N<N>/fold<k>``."""
    return results / "tuning" / model / f"N{n_train}" / f"fold{fold}"


def sqlite_url(path: Path) -> str:
    """Optuna storage URL for a SQLite file, built with ``Path.as_posix()``."""
    return "sqlite:///" + path.resolve().as_posix()


def validate_run_dir(directory: Path, required_files: tuple[str, ...]) -> bool:
    """Return True if ``directory`` holds a completed run with all ``required_files``."""
    result_file = directory / "result.json"
    if not result_file.is_file():
        return False
    try:
        result = read_json(result_file)
    except (OSError, json.JSONDecodeError):
        return False
    if result.get("status") != "complete":
        return False
    return all((directory / name).is_file() for name in required_files)


def staging_dir(final: Path) -> Path:
    """A fresh sibling directory in which a run writes its files before finalizing."""
    staging = final.with_name(f"{final.name}.tmp-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def finalize_dir(staging: Path, final: Path) -> None:
    """Move a staging directory to its final name, keeping any invalid previous output.

    Raw results are never overwritten: an existing final directory is renamed to
    ``<name>.invalid-<timestamp>`` rather than deleted. Callers only finalize after checking
    that the existing directory does not validate.
    """
    if final.exists():
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
        final.rename(final.with_name(f"{final.name}.invalid-{stamp}"))
    staging.rename(final)


# --------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------


def utc_now() -> str:
    """Current UTC time in ISO 8601 format."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def git_state(root: Path) -> dict[str, Any]:
    """Commit hash and working-tree state of the repository at ``root``.

    Inside the cluster container git is not available, so every job script reads the commit
    and ``git status --porcelain`` on the host and passes them in as QNNWIND_GIT_COMMIT,
    QNNWIND_GIT_DIRTY (0/1), and QNNWIND_GIT_STATUS (porcelain lines joined by "; "); these
    take precedence. Otherwise the ``git`` executable is used, and as a last resort
    ``.git`` is read directly (``dirty`` is then ``None``).
    """
    if os.environ.get("QNNWIND_GIT_COMMIT"):
        return {
            "commit": os.environ["QNNWIND_GIT_COMMIT"],
            "dirty": os.environ.get("QNNWIND_GIT_DIRTY") == "1",
            "status": os.environ.get("QNNWIND_GIT_STATUS", ""),
            "source": "host",
        }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        # Untracked files count as dirty (results/ is gitignored and does not).
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return {
            "commit": commit,
            "dirty": bool(status.strip()),
            "status": "; ".join(status.splitlines()),
            "source": "git",
        }
    except (OSError, subprocess.CalledProcessError):
        pass
    git_dir = root / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head[5:]
            ref_file = git_dir / ref
            if ref_file.is_file():
                return {
                    "commit": ref_file.read_text(encoding="utf-8").strip(),
                    "dirty": None,
                    "source": ".git",
                }
            for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + ref):
                    return {"commit": line.split(" ")[0], "dirty": None, "source": ".git"}
        return {"commit": head, "dirty": None, "source": ".git"}
    except OSError:
        return {"commit": None, "dirty": None, "source": ".git"}


def cpu_model() -> str:
    """Human-readable CPU model name on Linux and Windows."""
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def package_versions() -> dict[str, str | None]:
    """Installed versions of the packages in :data:`RECORDED_PACKAGES`."""
    versions: dict[str, str | None] = {}
    for name in RECORDED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def thread_settings() -> dict[str, Any]:
    """Thread-limit environment variables and torch's intra-op thread count."""
    settings: dict[str, Any] = {name: os.environ.get(name) for name in THREAD_ENV_VARS}
    if "torch" in sys.modules:
        settings["torch_num_threads"] = sys.modules["torch"].get_num_threads()
    return settings


def run_metadata(config: Config, dataset_sha256: str, workers: int) -> dict[str, Any]:
    """Metadata recorded in every result.json, except the end timestamp."""
    return {
        "git": git_state(config.root),
        "config_file": config.source.name,
        "config_hash": config.hash,
        "dataset_sha256": dataset_sha256,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model(),
        "workers_per_node": workers,
        "threads": thread_settings(),
        "affinity": _affinity(),
        "start_time": utc_now(),
    }


def peak_rss_mb() -> float | None:
    """Peak resident memory of this process in MiB (Linux: ru_maxrss; Windows: peak working set)."""
    if sys.platform == "win32":
        try:
            import psutil

            return psutil.Process().memory_info().peak_wset / 2**20
        except (ImportError, AttributeError):
            return None
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux
    except ImportError:
        return None


def _affinity() -> list[int] | None:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return None
