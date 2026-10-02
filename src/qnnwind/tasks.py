"""Task definitions for the task farm.

A task is one unit of work with a deterministic ID: one tuning study (``tune``), one final
run (``run``), or one copy of the timing-calibration probe (``calibrate``). Tasks are grouped
into task files ``<paths.tasks>/<group>.json``:

* ``tuning``: every (tuned model, N, fold);
* ``qnn``: every (QNN configuration, N, fold, seed);
* ``classical``: every other final run (LR, tuned classical and deep models, MLP-PM); the
  tuned ones need the ``tuning`` group to be complete;
* ``calibration_W<W>``: W concurrent copies of the calibration probe.

A task is *done* when its outputs validate (the same check the runner uses to skip it), so
tasks are idempotent and resumable.

Claiming (across nodes): before running a task, a worker creates the lock directory
``<paths.tasks>/locks/<id>.lock`` with ``mkdir``, which is atomic on a shared file system, so
only one worker on any node can hold a task. The holder touches ``owner.json`` inside the lock
every ``runtime.lock_heartbeat_seconds``. A lock whose heartbeat is older than
``runtime.lock_stale_seconds`` belongs to a killed job (e.g. a walltime cut): the next worker
breaks it with an atomic rename and reruns the task from the start. Failed tasks get a
``<paths.tasks>/state/<id>.failed`` marker, removed when the task later succeeds.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qnnwind.io import (
    Config,
    load_config,
    read_json,
    run_dir,
    tuning_dir,
    validate_run_dir,
    write_text,
)

RUN_FILES = ("result.json", "preds_val.csv", "preds_test.csv")
CURVE_FILES = ("curve_iter.csv", "curve_eval.csv")
FINAL_GROUPS = ("tuning", "qnn", "classical")

# Relative cost weights, used only to order tasks longest-first for better packing.
_TUNE_WEIGHT = {"LSTM": 60.0, "GRU": 60.0, "Transformer": 80.0, "MLP": 40.0, "SVR": 2.0}
_RUN_WEIGHT = {"LSTM": 2.0, "GRU": 2.0, "Transformer": 3.0, "MLP": 1.5, "MLP-PM": 0.5}


@dataclass(frozen=True)
class Task:
    """One unit of work.

    Attributes:
        id: Deterministic identifier, e.g. ``run__QNN-1__N750__fold0__seed0``.
        kind: ``tune``, ``run``, or ``calibrate``.
        group: Task-file group.
        model, n, fold, seed: What to run (``seed`` is None for tuning).
        cost: Relative cost estimate (ordering only).
        overrides: Config overrides applied for this task (calibration only).
        workers: Workers per node recorded in the metadata (calibration: the group's W).
    """

    id: str
    kind: str
    group: str
    model: str
    n: int
    fold: int
    seed: int | None = None
    cost: float = 1.0
    overrides: dict[str, Any] = field(default_factory=dict)
    workers: int | None = None


def model_kind(config: Config, model: str) -> str:
    for kind, names in config["models"].items():
        if model in names:
            return kind
    raise ValueError(f"Model {model!r} is not listed in the config")


def _qnn_gates(config: Config) -> dict[str, int]:
    from qnnwind.circuits import total_gates

    return total_gates(config["qnn"])


def build_tasks(config: Config) -> dict[str, list[Task]]:
    """The ``tuning``, ``qnn``, and ``classical`` task groups, longest tasks first."""
    folds, sizes, seeds = config["folds"]["run"], config["sizes"], config["seeds"]
    spaces = config["tuning"]["search_spaces"]
    models = [m for names in config["models"].values() for m in names]
    gates = _qnn_gates(config) if config["models"]["qnn"] else {}
    groups: dict[str, list[Task]] = {g: [] for g in FINAL_GROUPS}
    for n in sizes:
        for fold in folds:
            for model in models:
                if model in spaces:
                    groups["tuning"].append(
                        Task(
                            id=f"tune__{model}__N{n}__fold{fold}",
                            kind="tune",
                            group="tuning",
                            model=model,
                            n=n,
                            fold=fold,
                            cost=_TUNE_WEIGHT.get(model, 1.0) * n,
                        )
                    )
                is_qnn = model_kind(config, model) == "qnn"
                for seed in seeds:
                    group = "qnn" if is_qnn else "classical"
                    cost = gates[model] * n if is_qnn else _RUN_WEIGHT.get(model, 0.1) * n
                    groups[group].append(
                        Task(
                            id=f"run__{model}__N{n}__fold{fold}__seed{seed}",
                            kind="run",
                            group=group,
                            model=model,
                            n=n,
                            fold=fold,
                            seed=seed,
                            cost=float(cost),
                        )
                    )
    for tasks in groups.values():
        tasks.sort(key=lambda t: (-t.cost, t.id))
    return groups


def calibration_tasks(config: Config) -> dict[str, list[Task]]:
    """One group per W: W identical copies of the calibration probe, each with its own
    results directory (``<results>/calibration/W<W>/copy<i>``)."""
    cal = config["calibration"]
    results_raw = str(config.raw["paths"]["results"])
    groups = {}
    for w in cal["workers"]:
        group = f"calibration_W{w:03d}"
        groups[group] = [
            Task(
                id=f"calibrate__W{w:03d}__copy{i:03d}",
                kind="calibrate",
                group=group,
                model=cal["model"],
                n=int(cal["n_train"]),
                fold=int(cal["fold"]),
                seed=int(cal["seed"]),
                overrides={
                    "qnn.optimizer.maxiter": int(cal["maxiter"]),
                    "paths.results": f"{results_raw}/calibration/W{w:03d}/copy{i:03d}",
                },
                workers=int(w),
            )
            for i in range(w)
        ]
    return groups


# --------------------------------------------------------------------------------------
# Task files
# --------------------------------------------------------------------------------------


def task_file(config: Config, group: str) -> Path:
    return config.path("tasks") / f"{group}.json"


def write_task_file(config: Config, group: str, tasks: list[Task]) -> Path:
    """Write a group's task file (deterministic content: same config, same bytes)."""
    path = task_file(config, group)
    content = {
        "group": group,
        "config_file": config.source.name,
        "config_hash": config.hash,
        "n_tasks": len(tasks),
        "tasks": [asdict(t) for t in tasks],
    }
    write_text(path, json.dumps(content, indent=1) + "\n")
    return path


def read_task_file(config: Config, group: str) -> list[Task]:
    """Read a group's tasks; refuse a file written for a different configuration."""
    path = task_file(config, group)
    content = read_json(path)
    if content["config_hash"] != config.hash:
        raise RuntimeError(
            f"{path} was written for config hash {content['config_hash']}, but the current "
            f"config hash is {config.hash}. Re-run scripts/make_tasks.py."
        )
    return [Task(**t) for t in content["tasks"]]


# --------------------------------------------------------------------------------------
# Execution helpers
# --------------------------------------------------------------------------------------


def task_config(config: Config, task: Task) -> Config:
    """The configuration a task runs with (the base config plus the task's overrides)."""
    return load_config(config.source, task.overrides) if task.overrides else config


def task_argv(config: Config, task: Task, workers: int) -> list[str]:
    """Arguments for ``python -m qnnwind.runner`` that execute the task."""
    base = ["--config", str(config.source), "--model", task.model]
    base += ["--n", str(task.n), "--fold", str(task.fold)]
    for key, value in task.overrides.items():
        base += ["--set", f"{key}={json.dumps(value)}"]
    if task.kind == "tune":
        return ["tune", *base]
    return ["run", *base, "--seed", str(task.seed), "--workers", str(task.workers or workers)]


def task_output(config: Config, task: Task) -> Path:
    """The file or directory whose validity marks the task as done."""
    cfg = task_config(config, task)
    if task.kind == "tune":
        return tuning_dir(cfg.path("results"), task.model, task.n, task.fold) / "best_params.json"
    assert task.seed is not None
    return run_dir(cfg.path("results"), task.model, task.n, task.fold, task.seed)


def task_is_done(config: Config, task: Task) -> bool:
    """True if the task's outputs exist and validate (the runner would skip it)."""
    output = task_output(config, task)
    if task.kind == "tune":
        return output.is_file()
    files = RUN_FILES + (CURVE_FILES if model_kind(config, task.model) in ("qnn", "mlp_pm") else ())
    return validate_run_dir(output, files)


def state_dir(config: Config) -> Path:
    return config.path("tasks") / "state"


def lock_path(config: Config, task: Task) -> Path:
    return config.path("tasks") / "locks" / f"{task.id}.lock"


def lock_age(lock: Path) -> float | None:
    """Seconds since the lock's last heartbeat, or None if the lock does not exist."""
    owner = lock / "owner.json"
    for target in (owner, lock):
        try:
            return time.time() - target.stat().st_mtime
        except FileNotFoundError:
            continue
    return None


def lock_state(config: Config, task: Task) -> str:
    """``none``, ``live`` (heartbeat within lock_stale_seconds), or ``stale``."""
    age = lock_age(lock_path(config, task))
    if age is None:
        return "none"
    return "live" if age <= float(config["runtime"]["lock_stale_seconds"]) else "stale"


def try_claim(config: Config, task: Task, owner: dict[str, Any]) -> Path | None:
    """Atomically claim a task; break a stale lock first. Returns the lock, or None if the
    task is held by a live worker (on this or another node)."""
    lock = lock_path(config, task)
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            lock.mkdir()
        except FileExistsError:
            state = lock_state(config, task)
            if state == "live":
                return None
            if state == "stale":
                # Only one breaker wins the rename; everyone then retries the mkdir.
                broken = lock.with_name(f"{lock.name}.stale-{uuid.uuid4().hex[:12]}")
                try:
                    lock.rename(broken)
                    shutil.rmtree(broken, ignore_errors=True)
                except OSError:
                    pass
            continue
        write_text(lock / "owner.json", json.dumps(owner) + "\n")
        return lock
    return None


def heartbeat(lock: Path) -> None:
    """Refresh the lock's heartbeat (its owner.json modification time)."""
    try:
        os.utime(lock / "owner.json")
    except FileNotFoundError:
        pass


def release(lock: Path) -> None:
    """Remove a lock held by this worker."""
    shutil.rmtree(lock, ignore_errors=True)


def task_status(config: Config, task: Task) -> str:
    """``done``, ``running`` (live lock), ``failed``, or ``pending``.

    A stale lock (a task interrupted by a killed job) counts as pending: it is rerun.
    """
    if task_is_done(config, task):
        return "done"
    if lock_state(config, task) == "live":
        return "running"
    if (state_dir(config) / f"{task.id}.failed").exists():
        return "failed"
    return "pending"


def log_path(config: Config, task: Task) -> Path:
    """One log per task (appended to on every attempt)."""
    return config.path("logs") / task.group / f"{task.id}.log"
