"""Dynamic task claiming across nodes, stale locks, and git provenance."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from qnnwind.io import git_state, load_config
from qnnwind.tasks import (
    build_tasks,
    lock_path,
    lock_state,
    release,
    task_status,
    try_claim,
)

from conftest import ROOT


@pytest.fixture()
def farm_config(tmp_path):
    return load_config(
        ROOT / "configs" / "experiment.yaml",
        {
            "paths.tasks": str(tmp_path / "tasks"),
            "paths.results": str(tmp_path / "results"),
            "runtime.lock_stale_seconds": 60,
        },
    )


def _age(lock, seconds):
    old = time.time() - seconds
    os.utime(lock / "owner.json", (old, old))


def test_claim_is_exclusive_and_released(farm_config):
    task = build_tasks(farm_config)["qnn"][0]
    lock = try_claim(farm_config, task, {"who": "a"})
    assert lock is not None and lock_state(farm_config, task) == "live"
    assert try_claim(farm_config, task, {"who": "b"}) is None  # held by a live worker
    assert task_status(farm_config, task) == "running"
    release(lock)
    assert lock_state(farm_config, task) == "none"
    assert task_status(farm_config, task) == "pending"


def test_stale_lock_is_broken(farm_config):
    task = build_tasks(farm_config)["qnn"][1]
    lock = try_claim(farm_config, task, {"who": "killed job"})
    _age(lock, 120)  # heartbeat older than lock_stale_seconds (60)
    assert lock_state(farm_config, task) == "stale"
    assert task_status(farm_config, task) == "pending"  # an interrupted task is rerun
    new = try_claim(farm_config, task, {"who": "next job"})
    assert new == lock_path(farm_config, task)
    owner = json.loads((new / "owner.json").read_text(encoding="utf-8"))
    assert owner == {"who": "next job"}
    assert not [p for p in new.parent.iterdir() if ".stale-" in p.name]  # debris removed
    release(new)


def test_launcher_respects_live_locks_and_breaks_stale_ones(tmp_path, dataset):
    cfg = tmp_path / "lr.yaml"
    cfg.write_text(
        "\n".join(
            [
                f"base: {(ROOT / 'configs' / 'smoke.yaml').as_posix()}",
                "paths:",
                f"  data: {(ROOT / 'data' / 'total_dataset.csv').as_posix()}",
                f"  results: {(tmp_path / 'results').as_posix()}",
                f"  logs: {(tmp_path / 'logs').as_posix()}",
                f"  tasks: {(tmp_path / 'tasks').as_posix()}",
                "models: {classical: [LR], deep: [], mlp_pm: [], qnn: []}",
                "seeds: [0, 1]",
                "runtime: {lock_stale_seconds: 60}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    def script(name, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / name), "--config", str(cfg), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    assert script("make_tasks.py").returncode == 0
    config = load_config(cfg)
    held = next(t for t in build_tasks(config)["classical"] if t.seed == 1)
    lock = try_claim(config, held, {"host": "another-node"})  # a live worker elsewhere

    first = script("launcher.py", "--group", "classical", "--workers", "2")
    assert first.returncode == 0, first.stdout + first.stderr
    assert "second pass over 1 tasks held elsewhere" in first.stdout
    assert "'held_elsewhere': 1" in first.stdout and "'done': 1" in first.stdout
    assert "running" in script("status.py").stdout

    _age(lock, 600)  # the other job was killed: its heartbeat stopped
    second = script("launcher.py", "--group", "classical", "--workers", "2")
    assert second.returncode == 0 and "'done': 1" in second.stdout
    final = script("status.py")
    assert final.returncode == 0 and f"{'classical':<18}{2:>9}{0:>9}{0:>9}{0:>9}{2:>9}" in (
        final.stdout
    )


def test_git_state_from_host_environment(monkeypatch):
    monkeypatch.setenv("QNNWIND_GIT_COMMIT", "abc123")
    monkeypatch.setenv("QNNWIND_GIT_DIRTY", "1")
    monkeypatch.setenv("QNNWIND_GIT_STATUS", " M configs/experiment.yaml")
    assert git_state(ROOT) == {
        "commit": "abc123",
        "dirty": True,
        "status": " M configs/experiment.yaml",
        "source": "host",
    }
    monkeypatch.setenv("QNNWIND_GIT_DIRTY", "0")
    assert git_state(ROOT)["dirty"] is False
    monkeypatch.delenv("QNNWIND_GIT_COMMIT")
    local = git_state(ROOT)
    assert local["source"] in ("git", ".git") and len(local["commit"]) == 40
