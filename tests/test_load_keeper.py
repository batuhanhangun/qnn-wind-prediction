"""The load keeper of QNN jobs (filler work while real tasks still run on a node)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from qnnwind.qnn import filler_simulation

from conftest import ROOT


def test_filler_simulation_stops_on_request_and_writes_nothing(config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def should_stop() -> bool:
        calls.append(1)
        return len(calls) > 3

    chunks = filler_simulation(config["qnn"], "QNN-1", should_stop, batch=2)
    assert chunks == 3 and len(calls) == 4
    assert not list(tmp_path.iterdir())  # filler writes no results
    assert filler_simulation(config["qnn"], "QNN-1", lambda: True, batch=2) == 0


def _farm(tmp_path, extra: list[str]):
    cfg = tmp_path / "keeper.yaml"
    cfg.write_text(
        "\n".join(
            [
                f"base: {(ROOT / 'configs' / 'smoke.yaml').as_posix()}",
                "paths:",
                f"  data: {(ROOT / 'data' / 'total_dataset.csv').as_posix()}",
                f"  results: {(tmp_path / 'results').as_posix()}",
                f"  logs: {(tmp_path / 'logs').as_posix()}",
                f"  tasks: {(tmp_path / 'tasks').as_posix()}",
                "sizes: [60]",
                "runtime: {lock_heartbeat_seconds: 1}",
                *extra,
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
    return script


def test_load_keeper_keeps_the_node_busy_until_the_last_real_task(tmp_path, dataset):
    """One QNN task on W = 3: two workers run filler until it ends, and count as busy."""
    script = _farm(
        tmp_path,
        [
            "models: {classical: [LR], deep: [], mlp_pm: [], qnn: [QNN-6]}",
            "qnn: {optimizer: {maxiter: 2}}",
        ],
    )
    run = script("launcher.py", "--group", "qnn", "--workers", "3", "--load-keeper")
    out = run.stdout
    assert run.returncode == 0, out + run.stderr
    assert "load keeper on" in out and "pool of 3 workers" in out
    assert out.count("load keeper: filler@") >= 2  # started (and stopped) lines
    assert "stopped after" in out and "filler_seconds" in out
    started = [line for line in out.splitlines() if "load keeper:" in line and "started" in line]
    assert 1 <= len(started) <= 2  # at most W - 1 fillers beside the one real task

    result_file = tmp_path / "results" / "raw" / "QNN-6" / "N60" / "fold5" / "seed0"
    load = json.loads((result_file / "result.json").read_text(encoding="utf-8"))["metadata"][
        "node_load"
    ]
    assert load["workers"] == 3 and load["busy_max"] == 3 and load["samples"] > 3
    assert load["busy_mean"] > 2.5  # filler counts as busy
    # Filler writes no results: the only run directory is the real task's.
    runs = [p for p in (tmp_path / "results" / "raw").rglob("result.json")]
    assert runs == [result_file / "result.json"]
    assert not list((tmp_path / "tasks" / "locks").glob("*"))


def test_no_load_keeper_for_classical_or_tuning(tmp_path, dataset):
    script = _farm(tmp_path, ["models: {classical: [LR, kNN], deep: [], mlp_pm: [], qnn: []}"])
    for group in ("tuning", "classical"):
        run = script("launcher.py", "--group", group, "--workers", "3", "--load-keeper")
        assert run.returncode == 0, run.stdout + run.stderr
        assert "load keeper off" in run.stdout and "load keeper:" not in run.stdout


def test_load_keeper_off_by_default(tmp_path, dataset):
    """On a PC the load keeper stays off unless --load-keeper or --cluster is given."""
    script = _farm(
        tmp_path,
        [
            "models: {classical: [], deep: [], mlp_pm: [], qnn: [QNN-6]}",
            "qnn: {optimizer: {maxiter: 2}}",
        ],
    )
    run = script("launcher.py", "--group", "qnn", "--workers", "2")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "load keeper off" in run.stdout and "load keeper:" not in run.stdout


def test_keeper_groups_config(config):
    assert config["runtime"]["load_keeper"]["groups"] == ["qnn"]
    assert os.environ.get("OMP_NUM_THREADS") == "1"  # filler inherits single-threading


# One-line check of the multinode job (every task log shows exactly one attempt).
ONCE_CHECK = (
    "grep -c 'attempt on' {logs}/*/*.log"
    ' | awk -F: \'$NF != 1 {{bad = 1; print "NOT ONCE: " $0}}'
    ' END {{if (!bad) print "OK: every task ran exactly once"; exit bad}}\''
)


def test_two_launchers_split_tasks_without_duplicates(tmp_path, dataset):
    """Two concurrent launchers (standing in for two nodes) share one task list."""
    _farm(
        tmp_path,
        [
            "models: {classical: [LR], deep: [], mlp_pm: [MLP-PM], qnn: [QNN-6]}",
            "seeds: [0, 1, 2]",
            "qnn: {optimizer: {maxiter: 1}}",
        ],
    )
    cfg = tmp_path / "keeper.yaml"
    cmd = [sys.executable, str(ROOT / "scripts" / "launcher.py"), "--config", str(cfg)]
    cmd += ["--group", "qnn", "--group", "classical", "--workers", "2", "--load-keeper"]
    procs = [
        subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(2)
    ]
    outputs = [p.communicate(timeout=900)[0] for p in procs]
    assert all(p.returncode == 0 for p in procs), outputs
    done = [{line.split()[2] for line in out.splitlines() if " done " in line} for out in outputs]
    assert done[0] and done[1], "both launchers must run tasks"
    assert not done[0] & done[1]  # no task run twice
    assert len(done[0] | done[1]) == 9  # 3 QNN + 3 LR + 3 MLP-PM
    assert any("load keeper:" in out for out in outputs)

    from test_slurm import BASH, _path

    if BASH is not None:
        logs = (tmp_path / "logs").as_posix()
        check = subprocess.run(
            [BASH, "-c", ONCE_CHECK.format(logs=logs)],
            env={"PATH": _path()},
            capture_output=True,
            text=True,
            check=False,
        )
        assert check.returncode == 0 and "OK: every task ran exactly once" in check.stdout
