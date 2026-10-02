"""Slurm templates, render.sh, git provenance, and in_container.sh (run with bash)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT

SLURM = ROOT / "slurm"
TEMPLATES = sorted(SLURM.glob("*.sbatch"))
STRICT = ("tune.sbatch", "qnn.sbatch", "classical.sbatch")  # refuse a dirty tree
PLACEHOLDERS = (
    "<ACCOUNT>",
    "<QOS>",
    "<WALLTIME>",
    "<NODES>",
    "<QNNWIND_REPO>",
    "<QNNWIND_SCRATCH>",
    "<QNNWIND_EXTRAS>",
)


def find_bash() -> str | None:
    """Git's bash on Windows (not WSL's), the system bash elsewhere."""
    if sys.platform == "win32":
        git = shutil.which("git")
        for parent in Path(git).resolve().parents if git else []:
            candidate = parent / "bin" / "bash.exe"  # e.g. D:\Git\bin\bash.exe
            if candidate.is_file():
                return str(candidate)
        return None
    return shutil.which("bash")


BASH = find_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="bash not available")


def bash(script: str, env: dict[str, str], cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [BASH, "-c", script], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )


def test_templates():
    names = {t.name for t in TEMPLATES}
    assert names == {
        "aggregate.sbatch", "calibrate.sbatch", "classical.sbatch", "multinode_check.sbatch",
        "qnn.sbatch", "smoke_debug.sbatch", "tune.sbatch",
    }  # fmt: skip
    for template in TEMPLATES:
        text = template.read_text(encoding="utf-8")
        assert "#SBATCH -C cpu\n" in text, template.name
        for placeholder in PLACEHOLDERS:
            assert placeholder in text, (template.name, placeholder)
        assert "#SBATCH -o <QNNWIND_SCRATCH>/logs/%x-%j.out" in text
        mode = "strict" if template.name in STRICT else "warn"
        assert f'slurm/git_provenance.sh" {mode}\n' in text, template.name
        assert "\r\n" not in text


def test_no_hardcoded_home_or_absolute_paths():
    files = [*SLURM.iterdir(), ROOT / "README.md", *ROOT.glob("configs/*.yaml")]
    files += [*ROOT.glob("scripts/*.py"), *ROOT.glob("src/qnnwind/*.py")]
    files += [*ROOT.glob("environment/*"), *ROOT.glob("tests/*.py")]
    for path in files:
        if path.name == "test_slurm.py":
            continue
        text = path.read_text(encoding="utf-8")
        for needle in ("$HOME", "${HOME}", "/global/homes", "/pscratch"):
            assert needle not in text, (path.name, needle)


@needs_bash
def test_render_fills_every_placeholder(tmp_path):
    scratch = tmp_path.as_posix()
    env = {"PATH": _path(), "SCRATCH": scratch}
    out = bash("slurm/render.sh slurm/qnn.sbatch m1234 regular 24:00:00 4", env)
    assert out.returncode == 0, out.stderr
    rendered = Path(out.stdout.strip())
    assert rendered == Path(scratch) / "qnn-wind-runs" / "jobs" / "qnn.sbatch"
    text = rendered.read_text(encoding="utf-8")
    assert "<" not in "".join(line for line in text.splitlines() if line.startswith("#SBATCH"))
    assert "#SBATCH -A m1234" in text and "#SBATCH -N 4" in text and "#SBATCH -t 24:00:00" in text
    assert f'export QNNWIND_REPO="{scratch}/qnn-wind-prediction"' in text
    assert f'export QNNWIND_SCRATCH="{scratch}/qnn-wind-runs"' in text
    assert f'export QNNWIND_EXTRAS="{scratch}/qnn-wind-pyextras"' in text
    assert f"#SBATCH -o {scratch}/qnn-wind-runs/logs/%x-%j.out" in text
    assert (tmp_path / "qnn-wind-runs" / "logs").is_dir()
    # Explicit directories override the defaults.
    env |= {"QNNWIND_REPO": f"{scratch}/code", "QNNWIND_SCRATCH": f"{scratch}/runs"}
    out = bash("slurm/render.sh slurm/smoke_debug.sbatch m1 debug 00:30:00 1", env)
    text = Path(out.stdout.strip()).read_text(encoding="utf-8")
    assert f'export QNNWIND_REPO="{scratch}/code"' in text and "#SBATCH -q debug" in text


@needs_bash
@pytest.mark.skipif(shutil.which("git") is None, reason="git not available (e.g. in the container)")
def test_git_provenance_strict_and_warn(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {"PATH": _path(), "QNNWIND_REPO": repo.as_posix(), "HOME": tmp_path.as_posix()}
    setup = "git init -q && git -c user.email=a@b -c user.name=t commit -q --allow-empty -m init"
    assert bash(setup, env, cwd=repo).returncode == 0
    src = (SLURM / "git_provenance.sh").as_posix()
    clean = bash(f'set -e; source "{src}" strict; echo "dirty=$QNNWIND_GIT_DIRTY"', env)
    assert clean.returncode == 0 and "git working tree: clean" in clean.stdout
    assert "dirty=0" in clean.stdout

    (repo / "notes.txt").write_text("x", encoding="utf-8")
    strict = bash(f'set -e; source "{src}" strict; echo REACHED', env)
    assert strict.returncode != 0 and "REACHED" not in strict.stdout
    assert "ERROR" in strict.stderr and "?? notes.txt" in strict.stdout
    warn = bash(
        f'set -e; source "{src}" warn; echo "dirty=$QNNWIND_GIT_DIRTY status=$QNNWIND_GIT_STATUS"',
        env,
    )
    assert warn.returncode == 0 and "WARNING" in warn.stderr
    assert "dirty=1 status=?? notes.txt" in warn.stdout


@needs_bash
def test_in_container_mounts_and_environment(tmp_path):
    fake = tmp_path / "fake-podman-hpc"
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n', encoding="utf-8", newline="\n")
    fake.chmod(0o755)
    scratch = tmp_path.as_posix()
    extras = tmp_path / "qnn-wind-pyextras"
    env = {
        "PATH": _path(),
        "SCRATCH": scratch,
        "QNNWIND_REPO": ROOT.as_posix(),
        "QNNWIND_PODMAN": fake.as_posix(),
        "QNNWIND_GIT_COMMIT": "abc123",
        "QNNWIND_GIT_DIRTY": "0",
        "SLURM_JOB_ID": "42",
    }
    command = "slurm/in_container.sh scripts/status.py --config configs/smoke.yaml"
    missing = bash(command, env)  # no extras directory yet
    assert missing.returncode != 0 and "extras directory" in missing.stderr
    (extras / "lightgbm").mkdir(parents=True)
    out = bash(command, env)
    assert out.returncode == 0, out.stderr
    args = out.stdout.splitlines()
    runs = f"{scratch}/qnn-wind-runs"
    assert args[:2] == ["run", "--rm"]
    for flag, value in [
        ("-v", f"{ROOT.as_posix()}:{ROOT.as_posix()}"),
        ("-v", f"{runs}:{runs}"),
        ("-v", f"{extras.as_posix()}:/opt/qnnwind-extras:ro"),
        ("-w", ROOT.as_posix()),
        ("-e", f"QNNWIND_SCRATCH={runs}"),
        ("-e", "QNNWIND_EXTRAS_MOUNT=/opt/qnnwind-extras"),
        ("-e", f"PYTHONPATH=/opt/qnnwind-extras:{ROOT.as_posix()}/src"),
        ("-e", "OMP_NUM_THREADS=1"),
        ("-e", "QNNWIND_GIT_COMMIT=abc123"),
        ("-e", "QNNWIND_GIT_DIRTY=0"),
        ("-e", "SLURM_JOB_ID=42"),
    ]:
        pairs = list(zip(args, args[1:], strict=False))
        assert (flag, value) in pairs, (flag, value)
    tail = args[args.index("quantum_ml:v4") :]  # the unchanged v4 image
    assert tail == ["quantum_ml:v4", "python", "scripts/status.py", "--config",
                    "configs/smoke.yaml"]  # fmt: skip
    assert (tmp_path / "qnn-wind-runs").is_dir()


@needs_bash
def test_shell_syntax():
    for script in [*SLURM.glob("*.sh"), *TEMPLATES]:
        out = bash(f'bash -n "{script.as_posix()}"', {"PATH": _path()})
        assert out.returncode == 0, (script.name, out.stderr)


def _path() -> str:
    """PATH for the bash subprocess: bash's own directory first (for sed, awk, git...)."""
    import os

    extra = [str(Path(BASH).parent)] if BASH else []
    if sys.platform == "win32" and BASH:
        extra.append(str(Path(BASH).parents[1] / "usr" / "bin"))
    return os.pathsep.join([*extra, os.environ.get("PATH", "")])


def test_multinode_check_uses_the_cluster_launch_path():
    text = (SLURM / "multinode_check.sbatch").read_text(encoding="utf-8")
    assert 'export QNNWIND_SCRATCH="<QNNWIND_SCRATCH>/multinode-check"' in text  # separate runs
    assert 'CONFIG="${QNNWIND_CONFIG:-configs/smoke.yaml}"' in text
    assert "hostname" in text
    launch = 'srun -N "$SLURM_NNODES" --ntasks-per-node=1 -c 256 --cpu-bind=none'
    assert launch in text and launch in (SLURM / "qnn.sbatch").read_text(encoding="utf-8")
    assert "for group in tuning qnn classical; do" in text and "--workers 64" in text
