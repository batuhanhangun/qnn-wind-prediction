"""Task generation, the task-farm launcher, and the status script."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from launcher import local_workers, physical_cores, spread
from qnnwind.io import load_config
from qnnwind.tasks import (
    build_tasks,
    calibration_tasks,
    read_task_file,
    task_argv,
    task_file,
    write_task_file,
)

from conftest import ROOT


def test_full_grid_task_counts_and_ids(config):
    groups = build_tasks(config)
    assert len(groups["tuning"]) == 9 * 4 * 6  # tuned models x N x folds
    assert len(groups["qnn"]) == 6 * 4 * 6 * 5  # 720 runs
    assert len(groups["classical"]) == 11 * 4 * 6 * 5  # LR, 9 tuned, MLP-PM
    ids = [t.id for tasks in groups.values() for t in tasks]
    assert len(ids) == len(set(ids))
    assert "run__QNN-1__N3000__fold5__seed4" in ids and "tune__SVR__N750__fold0" in ids
    # Deterministic, and the most expensive QNN tasks come first.
    assert [t.id for t in build_tasks(config)["qnn"]] == [t.id for t in groups["qnn"]]
    assert groups["qnn"][0].id.startswith("run__QNN-1__N3000")


def test_task_files_are_deterministic_and_config_bound(tmp_path):
    cfg_path = ROOT / "configs" / "experiment.yaml"
    config = load_config(cfg_path, {"paths.tasks": str(tmp_path)})
    tasks = build_tasks(config)["tuning"]
    first = write_task_file(config, "tuning", tasks).read_bytes()
    second = write_task_file(config, "tuning", tasks).read_bytes()
    assert first == second and b"\r\n" not in first
    assert read_task_file(config, "tuning") == tasks
    other = load_config(cfg_path, {"paths.tasks": str(tmp_path), "tuning.n_trials": 31})
    with pytest.raises(RuntimeError, match="config hash"):
        read_task_file(other, "tuning")


def test_calibration_tasks(config):
    groups = calibration_tasks(config)
    assert {g: len(t) for g, t in groups.items()} == {
        "calibration_W001": 1,
        "calibration_W032": 32,
        "calibration_W064": 64,
        "calibration_W128": 128,
    }
    task = groups["calibration_W064"][7]
    assert (task.model, task.n, task.fold, task.workers) == ("QNN-1", 750, 5, 64)
    assert task.overrides["qnn.optimizer.maxiter"] == 5
    assert task.overrides["paths.results"].endswith("/calibration/W064/copy007")
    argv = task_argv(config, task, workers=999)
    assert argv[:1] == ["run"] and "--workers" in argv and argv[argv.index("--workers") + 1] == "64"
    results = {t.overrides["paths.results"] for tasks in groups.values() for t in tasks}
    assert len(results) == 225


def test_spread_over_cores():
    cores = list(range(128))
    assert spread(cores, 64) == list(range(0, 128, 2))
    assert spread(cores, 128) == cores and spread(cores, 1) == [0]
    assert spread(cores, 32) == list(range(0, 128, 4))
    with pytest.raises(ValueError):
        spread(cores, 129)


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux only")
def test_physical_cores_linux():
    cores = physical_cores()
    assert cores and set(cores) <= os.sched_getaffinity(0)


def _script(name, *args, env=None):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_launcher_end_to_end_spawn_w2(tmp_path, dataset):
    """make_tasks -> launcher (W = 2, spawn) -> status, with a dependency failure and resume."""
    cfg = tmp_path / "farm.yaml"
    cfg.write_text(
        "\n".join(
            [
                f"base: {(ROOT / 'configs' / 'smoke.yaml').as_posix()}",
                "paths:",
                f"  data: {(ROOT / 'data' / 'total_dataset.csv').as_posix()}",
                f"  results: {(tmp_path / 'results').as_posix()}",
                f"  logs: {(tmp_path / 'logs').as_posix()}",
                f"  tasks: {(tmp_path / 'tasks').as_posix()}",
                f"  outputs: {(tmp_path / 'outputs').as_posix()}",
                "models: {classical: [LR, kNN], deep: [], mlp_pm: [MLP-PM], qnn: [QNN-6]}",
                "sizes: [60]",  # > max n_neighbors (50) of the kNN search space
                "seeds: [0, 1]",
                "qnn: {optimizer: {maxiter: 1}}",
                "mlp_pm: {optimizer: {maxiter: 2}}",
                "runtime: {lock_heartbeat_seconds: 1}",  # many node-load samples per task
                "",
            ]
        ),
        encoding="utf-8",
    )
    made = _script("make_tasks.py", "--config", str(cfg))
    assert made.returncode == 0, made.stderr
    config = load_config(cfg)
    counts = {g: len(read_task_file(config, g)) for g in ("tuning", "qnn", "classical")}
    assert counts == {"tuning": 1, "qnn": 2, "classical": 6}

    # Classical before tuning: the two kNN runs fail (no tuning result), the rest succeed.
    first = _script("launcher.py", "--config", str(cfg), "--group", "classical", "--workers", "2")
    assert first.returncode == 1, first.stdout + first.stderr
    assert first.stdout.count("failed  ") == 2 and first.stdout.count("done  ") == 4
    status = _script("status.py", "--config", str(cfg), "--group", "classical", "--failed")
    assert status.returncode == 1 and "FAILED run__kNN__N60" in status.stdout
    knn_log = tmp_path / "logs" / "classical" / "run__kNN__N60__fold5__seed0.log"
    assert "tuning result missing" in knn_log.read_text(encoding="utf-8")

    # Tuning, then everything: only the missing tasks run; the failure markers are cleared.
    second = _script(
        "launcher.py",
        "--config",
        str(cfg),
        "--workers",
        "2",
        "--group",
        "tuning",
        "--group",
        "qnn",
        "--group",
        "classical",
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert "'failed': 0" in second.stdout and "'done': 5" in second.stdout  # 1 + 2 + 2
    assert not list((tmp_path / "tasks" / "state").glob("*"))
    assert not list((tmp_path / "tasks" / "locks").glob("*"))  # every lock released

    # Idempotent: a third run skips everything without starting a pool.
    third = _script(
        "launcher.py",
        "--config",
        str(cfg),
        "--workers",
        "2",
        "--group",
        "tuning",
        "--group",
        "qnn",
        "--group",
        "classical",
    )
    assert third.returncode == 0 and "to run: 0" in third.stdout and "pool of" not in third.stdout
    final = _script("status.py", "--config", str(cfg))
    assert final.returncode == 0, final.stdout
    for group, n in counts.items():
        assert f"{group:<18}{n:>9}{0:>9}{0:>9}{0:>9}{n:>9}" in final.stdout

    # Workers per node recorded in the run metadata; one log per task.
    result = json.loads(
        (
            tmp_path / "results" / "raw" / "QNN-6" / "N60" / "fold5" / "seed1" / "result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["metadata"]["workers_per_node"] == 2
    assert result["metadata"]["threads"]["OMP_NUM_THREADS"] == "1"
    # Node load: busy workers sampled at the start and at every heartbeat.
    load = result["metadata"]["node_load"]
    assert load["workers"] == 2 and load["heartbeat_seconds"] == 1.0 and load["samples"] > 1
    assert 1 <= load["busy_min"] <= load["busy_mean"] <= load["busy_max"] <= 2
    tuned = tmp_path / "results" / "tuning" / "kNN" / "N60" / "fold5" / "best_params.json"
    assert "node_load" not in json.loads(tuned.read_text(encoding="utf-8"))
    assert len(list((tmp_path / "logs").rglob("*.log"))) == 9
    assert task_file(config, "qnn").is_file()


def test_calibration_round_trip(tmp_path, dataset):
    """make_tasks --calibration -> launcher per W -> status --calibration picks W."""
    cfg = tmp_path / "cal.yaml"
    cfg.write_text(
        "\n".join(
            [
                f"base: {(ROOT / 'configs' / 'experiment.yaml').as_posix()}",
                "paths:",
                f"  data: {(ROOT / 'data' / 'total_dataset.csv').as_posix()}",
                f"  results: {(tmp_path / 'results').as_posix()}",
                f"  logs: {(tmp_path / 'logs').as_posix()}",
                f"  tasks: {(tmp_path / 'tasks').as_posix()}",
                "calibration: {n_train: 30, maxiter: 1, workers: [1, 2], max_slowdown: 10.0}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert _script("make_tasks.py", "--config", str(cfg), "--calibration").returncode == 0
    for group in ("calibration_W001", "calibration_W002"):
        run = _script("launcher.py", "--config", str(cfg), "--group", group)
        assert run.returncode == 0, run.stdout + run.stderr
        assert f"W = {group[-1]}" in run.stdout and "load keeper off" in run.stdout
    summary = _script("status.py", "--config", str(cfg), "--calibration")
    assert summary.returncode == 0, summary.stdout
    content = json.loads(
        (tmp_path / "results" / "calibration" / "summary.json").read_text(encoding="utf-8")
    )
    assert content["complete"] and content["chosen_W"] == 2  # generous limit in this test
    assert [r["copies_completed"] for r in content["rows"]] == [1, 2]
    assert content["rows"][0]["slowdown"] == 0.0
    copy = tmp_path / "results" / "calibration" / "W002" / "copy001"
    result = json.loads(next(copy.rglob("result.json")).read_text(encoding="utf-8"))
    assert result["metadata"]["workers_per_node"] == 2
    assert result["model_summary"]["scipy"]["options"]["maxiter"] == 1
    assert result["metadata"]["node_load"]["workers"] == 2


def test_status_skips_stale_groups_except_final_groups(tmp_path):
    """All-groups status warns on a stale calibration file; final groups stay strict."""
    base = ROOT / "configs" / "experiment.yaml"
    cfg = tmp_path / "status.yaml"
    lines = [f"base: {base.as_posix()}", "paths:", f"  tasks: {tmp_path.as_posix()}", ""]
    cfg.write_text("\n".join(lines), encoding="utf-8")
    config = load_config(cfg)
    write_task_file(config, "tuning", build_tasks(config)["tuning"][:3])
    stale = load_config(cfg, {"calibration.maxiter": 7})
    write_task_file(stale, "calibration_W001", calibration_tasks(stale)["calibration_W001"])

    out = _script("status.py", "--config", str(cfg))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "WARNING: skipping calibration_W001" in out.stderr and "tuning" in out.stdout
    explicit = _script("status.py", "--config", str(cfg), "--group", "calibration_W001")
    assert explicit.returncode != 0 and "config hash" in explicit.stderr

    # A stale final-group task file is still an error.
    write_task_file(stale, "qnn", build_tasks(stale)["qnn"][:2])
    strict = _script("status.py", "--config", str(cfg))
    assert strict.returncode != 0 and "config hash" in strict.stderr


def test_local_worker_default_is_physical_cores_minus_one():
    import psutil

    cores = physical_cores()
    count = len(cores) if cores is not None else psutil.cpu_count(logical=False)
    assert local_workers() == max(1, count - 1)
