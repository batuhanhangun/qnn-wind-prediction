"""Task-farm launcher: runs a group's tasks on W single-threaded, core-pinned workers.

One launcher runs per node (inside the container). It reads ``<paths.tasks>/<group>.json``
(tasks in decreasing order of estimated cost), skips tasks whose outputs already validate,
and runs the rest on a pool of W worker processes created with the spawn start method. Each
worker is pinned to its own physical core (Linux; skipped on Windows) and runs one task at a
time as a subprocess ``python -m qnnwind.runner ...`` that inherits the pinning and the
single-thread settings. Each task appends to its own log.

Across nodes, tasks are claimed dynamically (``qnnwind.tasks.try_claim``: an atomic ``mkdir``
of a per-task lock directory on the shared file system, with a heartbeat). Every node walks
the same longest-first list, so no node idles while long tasks are still unclaimed. Tasks held
by a live worker elsewhere are skipped and checked once more after the pass; locks left by a
killed job go stale and are broken, so interrupted tasks rerun from the start. Several groups
run one after another, with a barrier between them.

Load keeper (with ``--load-keeper`` or ``--cluster``, for the groups in
``runtime.load_keeper.groups``, i.e. QNN jobs): W filler payloads are
queued after the real tasks, so a worker reaches one only when no unclaimed task is left for
it. The filler runs QNN circuit simulation (``qnnwind.qnn.filler_simulation``, no results)
and counts as busy in the node load, until the node's last real task has finished. Then the
fillers stop and the node exits.

Defaults on a PC: W = physical cores minus one, load keeper off, run directory ``runs/``
(see scripts/runs_dir.py). On a cluster node, ``--cluster`` uses ``runtime.workers`` of the
config and turns the load keeper on (the Slurm templates pass it).

Usage::

    python scripts/launcher.py --config configs/quick.yaml --group tuning --group classical
    python scripts/launcher.py --config configs/experiment.yaml --group qnn --cluster
"""

from __future__ import annotations

import os

# Single-threaded workers: set before numpy or torch is imported, and inherited by
# the spawned workers and their task subprocesses.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import queue  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aggregate import require_dataset  # noqa: E402
from qnnwind.io import add_run_metadata, load_config, utc_now, write_text  # noqa: E402
from qnnwind.tasks import (  # noqa: E402
    Task,
    heartbeat,
    log_path,
    read_task_file,
    release,
    state_dir,
    task_argv,
    task_is_done,
    task_output,
    try_claim,
)
from runs_dir import add_runs_argument, use_runs_dir  # noqa: E402

_CORE: int | None = None  # the core this worker process is pinned to
_BUSY: Any = None  # shared: workers of this launcher running a real task or filler
_OUTSTANDING: Any = None  # shared: real payloads of the current pass not yet finished here


# --------------------------------------------------------------------------------------
# Core selection and pinning
# --------------------------------------------------------------------------------------


def physical_cores() -> list[int] | None:
    """One logical CPU per physical core among the CPUs this process may use (Linux).

    Hardware-thread siblings are identified by their ``thread_siblings_list`` in sysfs, and the
    lowest CPU number of each core is used. Returns None where affinity is unsupported.
    """
    if not hasattr(os, "sched_getaffinity"):
        return None
    cores: dict[str, int] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        siblings = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        try:
            key = siblings.read_text(encoding="utf-8").strip()
        except OSError:
            key = f"cpu{cpu}"  # topology unavailable: treat every CPU as its own core
        cores.setdefault(key, cpu)
    return sorted(cores.values())


def local_workers() -> int:
    """Default W on a PC: the physical cores available to this process, minus one."""
    cores = physical_cores()
    if cores is None:
        import psutil

        count = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    else:
        count = len(cores)
    return max(1, count - 1)


def spread(cores: list[int], workers: int) -> list[int]:
    """``workers`` cores spread evenly over ``cores`` (e.g. across both sockets)."""
    if workers > len(cores):
        raise ValueError(f"W = {workers} workers but only {len(cores)} physical cores available")
    return [cores[(i * len(cores)) // workers] for i in range(workers)]


def _init_worker(core_queue: Any, pin: bool, busy: Any, outstanding: Any) -> None:
    global _CORE, _BUSY, _OUTSTANDING
    _BUSY, _OUTSTANDING = busy, outstanding
    try:
        _CORE = core_queue.get(timeout=10)
    except queue.Empty:  # a replacement for a worker that died: run unpinned
        _CORE = None
    if pin and _CORE is not None:
        os.sched_setaffinity(0, {_CORE})


# --------------------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------------------


def _value(shared: Any) -> int:
    with shared.get_lock():
        return int(shared.value)


def _add(shared: Any, delta: int) -> None:
    with shared.get_lock():
        shared.value += delta


def _run_with_heartbeat(
    argv: list[str], log: Path, lock: Path, interval: float, samples: list[int]
) -> int:
    """Run the task subprocess; at every heartbeat refresh the lock and record how many
    workers on this node are busy (the node load)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT / "src"), env.get("PYTHONPATH", "")) if p
    )
    with log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"$ {' '.join(argv)}\n")
        handle.flush()
        proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        while True:
            try:
                code = proc.wait(timeout=interval)
                break
            except subprocess.TimeoutExpired:
                heartbeat(lock)
                samples.append(_value(_BUSY))
        handle.write(f"===== exit code {code}\n")
    return code


def execute_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one payload: a real task or load-keeper filler (top-level for pickling)."""
    if payload.get("filler"):
        return run_filler(payload)
    try:
        return _execute_real(payload)
    finally:
        _add(_OUTSTANDING, -1)  # whatever the outcome, this real payload is finished here


def _execute_real(payload: dict[str, Any]) -> dict[str, Any]:
    """Claim and run one real task."""
    config = load_config(payload["config"])
    task = Task(**payload["task"])
    if task_is_done(config, task):
        return {"id": task.id, "status": "skipped", "seconds": 0.0}
    if payload["deadline"] is not None and time.time() > payload["deadline"]:
        return {"id": task.id, "status": "deferred", "seconds": 0.0}
    owner = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "core": _CORE,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "start": utc_now(),
    }
    lock = try_claim(config, task, owner)
    if lock is None:
        return {"id": task.id, "status": "claimed", "seconds": 0.0}
    try:
        if task_is_done(config, task):  # finished elsewhere between the check and the claim
            return {"id": task.id, "status": "skipped", "seconds": 0.0}
        log = log_path(config, task)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"\n===== {utc_now()} attempt on {json.dumps(owner)}\n")
        argv = [sys.executable, "-m", "qnnwind.runner"]
        argv += task_argv(config, task, payload["workers"])
        interval = float(config["runtime"]["lock_heartbeat_seconds"])
        _add(_BUSY, 1)
        samples = [_value(_BUSY)]  # at the start, then at every heartbeat
        t0 = time.perf_counter()
        try:
            code = _run_with_heartbeat(argv, log, lock, interval, samples)
        finally:
            _add(_BUSY, -1)
        seconds = time.perf_counter() - t0

        failed = state_dir(config) / f"{task.id}.failed"
        if code == 0 and task_is_done(config, task):
            if task.kind != "tune":  # final and calibration runs: record the node load
                load = node_load(samples, payload["workers"], interval)
                add_run_metadata(task_output(config, task) / "result.json", "node_load", load)
            failed.unlink(missing_ok=True)
            return {"id": task.id, "status": "done", "seconds": seconds}
        reason = f"exit code {code}" if code else "exit code 0 but outputs do not validate"
        failed.parent.mkdir(parents=True, exist_ok=True)
        write_text(failed, json.dumps({**owner, "end": utc_now(), "reason": reason}) + "\n")
        return {"id": task.id, "status": "failed", "seconds": seconds, "reason": reason}
    finally:
        release(lock)


def run_filler(payload: dict[str, Any]) -> dict[str, Any]:
    """Load keeper: simulate QNN circuits while real tasks still run on this node."""
    name = f"filler@{socket.gethostname()}:core{_CORE}"
    if _value(_OUTSTANDING) == 0:
        return {"id": name, "status": "filler", "seconds": 0.0, "chunks": 0}
    from qnnwind.qnn import filler_simulation

    config = load_config(payload["config"])
    keeper = config["runtime"]["load_keeper"]
    _add(_BUSY, 1)
    t0 = time.perf_counter()
    print(
        f"[{utc_now()}] load keeper: {name} started, "
        f"{_value(_OUTSTANDING)} real task(s) still running on this node",
        flush=True,
    )
    try:
        chunks = filler_simulation(
            config["qnn"],
            config["calibration"]["model"],
            should_stop=lambda: _value(_OUTSTANDING) == 0,
            batch=int(keeper["batch"]),
            seed=_CORE or 0,
        )
    finally:
        _add(_BUSY, -1)
    seconds = time.perf_counter() - t0
    print(f"[{utc_now()}] load keeper: {name} stopped after {seconds:.1f} s", flush=True)
    return {"id": name, "status": "filler", "seconds": seconds, "chunks": chunks}


def node_load(samples: list[int], workers: int, interval: float) -> dict[str, Any]:
    """Busy workers on the node while the task ran (real tasks and load-keeper filler):
    sampled at its start and at every heartbeat. The timing analysis keeps QNN runs only when
    ``busy_mean >= 0.9 * workers``."""
    return {
        "workers": workers,
        "heartbeat_seconds": interval,
        "samples": len(samples),
        "busy_mean": sum(samples) / len(samples),
        "busy_min": min(samples),
        "busy_max": max(samples),
    }


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def _run_pass(
    pool: Any,
    payloads: list[dict],
    counts: dict[str, int],
    outstanding: Any,
    keeper: bool,
    workers: int,
) -> list[dict]:
    """Dispatch payloads longest-first (then fillers, if the load keeper is on); return the
    payloads held by a live worker elsewhere."""
    with outstanding.get_lock():
        outstanding.value = len(payloads)
    fillers = [{"filler": True, "config": payloads[0]["config"]}] * workers if keeper else []
    claimed = []
    results = pool.imap(execute_task, payloads + fillers, chunksize=1)  # ordered dispatch
    for payload, result in zip(payloads + fillers, results, strict=True):
        if result["status"] == "claimed":
            claimed.append(payload)
            continue
        if result["status"] == "filler":
            counts["filler_seconds"] = round(counts.get("filler_seconds", 0) + result["seconds"])
            continue
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        extra = f" ({result['reason']})" if "reason" in result else ""
        print(
            f"[{utc_now()}] {result['status']:<8} {result['id']} {result['seconds']:.1f} s{extra}",
            flush=True,
        )
    return claimed


def run_group(
    config_path: Path,
    group: str,
    workers: int,
    deadline: float | None,
    pin: bool,
    load_keeper: bool = False,
) -> dict[str, int]:
    """Run one group to completion on this node; return counts per outcome."""
    config = load_config(config_path)
    tasks = read_task_file(config, group)
    todo = [t for t in tasks if not task_is_done(config, t)]
    group_workers = {t.workers for t in tasks if t.workers is not None}
    if group_workers:  # calibration groups fix their own W
        (workers,) = group_workers
    keeper = load_keeper and group in config["runtime"]["load_keeper"]["groups"]
    print(
        f"[{utc_now()}] {socket.gethostname()}: group {group}: {len(tasks)} tasks, "
        f"to run: {len(todo)}, W = {workers}, load keeper {'on' if keeper else 'off'}",
        flush=True,
    )
    counts = {"done": 0, "skipped": len(tasks) - len(todo), "failed": 0, "deferred": 0}
    if not todo:
        return counts

    cores = physical_cores() if pin else None
    chosen = spread(cores, workers) if cores is not None else [None] * workers
    pool_size = workers if keeper else min(workers, len(todo))  # the keeper fills all W
    ctx = mp.get_context("spawn")
    core_queue = ctx.Queue()
    busy, outstanding = ctx.Value("i", 0), ctx.Value("i", 0)
    for core in chosen[:pool_size]:
        core_queue.put(core)
    payloads = [
        {"config": str(config.source), "task": t.__dict__, "workers": workers, "deadline": deadline}
        for t in todo
    ]
    print(f"  pool of {pool_size} workers, cores {chosen[:pool_size]}", flush=True)
    initargs = (core_queue, cores is not None, busy, outstanding)
    with ctx.Pool(pool_size, initializer=_init_worker, initargs=initargs) as pool:
        claimed = _run_pass(pool, payloads, counts, outstanding, keeper, pool_size)
        if claimed:
            # Tasks held elsewhere during the pass are, by now, done, still running on another
            # node (left alone), or orphaned by a killed job (stale lock: run here).
            print(f"  second pass over {len(claimed)} tasks held elsewhere", flush=True)
            still = _run_pass(pool, claimed, counts, outstanding, keeper, pool_size)
            counts["held_elsewhere"] = len(still)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--group", action="append", required=True, help="repeatable, in order")
    parser.add_argument(
        "--workers",
        type=int,
        help="W (default: physical cores minus one; with --cluster: runtime.workers)",
    )
    parser.add_argument(
        "--load-keeper",
        action="store_true",
        help="run filler work while the node's last tasks finish (runtime.load_keeper)",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="cluster node: W = runtime.workers of the config and --load-keeper",
    )
    add_runs_argument(parser)
    parser.add_argument(
        "--no-new-after-minutes",
        type=float,
        help="do not start new tasks after this many minutes (leave room before walltime)",
    )
    parser.add_argument("--no-pin", action="store_true", help="do not pin workers to cores")
    args = parser.parse_args(argv)

    use_runs_dir(args.runs)
    config = load_config(args.config)
    require_dataset(config)
    default = int(config["runtime"]["workers"]) if args.cluster else local_workers()
    workers = args.workers or default
    keeper = args.load_keeper or args.cluster
    deadline = time.time() + 60 * args.no_new_after_minutes if args.no_new_after_minutes else None
    pin = not args.no_pin and hasattr(os, "sched_setaffinity")
    total: dict[str, int] = {}
    for group in args.group:
        for key, value in run_group(args.config, group, workers, deadline, pin, keeper).items():
            total[key] = total.get(key, 0) + value
    print(f"[{utc_now()}] {socket.gethostname()}: finished: {total}", flush=True)
    return 1 if total.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
