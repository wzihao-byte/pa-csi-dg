"""Wait for the H-S unwrap queue, then run exclusive WiMANS GPU timing and the four-arm source queue.

Proceeds only if the unwrap queue reports all nine runs complete and its runner
has exited. A failed, stopped or vanished unwrap runner, or a failed timing,
ends the chain without starting WiMANS training. Every transition is recorded.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
UNWRAP = Path("D:/pa-csi-dg-revision-runtime/studies/h_phase_link_unwrap_20261023_v2")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def record(path, status, **fields):
    value = read(path) if path.exists() else {"events": []}
    value["events"].append({"at": now(), "status": status, **fields})
    value.update(status=status, updated_at=now(), pid=os.getpid())
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def alive(pid, create_time):
    try:
        return abs(psutil.Process(pid).create_time() - create_time) < 1.0
    except psutil.Error:
        return False


def unwrap_state():
    """'running', 'complete' (runner exited) or a terminal reason that blocks the chain."""
    status = read(UNWRAP / "queue/status.json")
    if status["status"] == "complete" and status["completed_jobs"] == status["total_jobs"] == 9:
        results = read(UNWRAP / "analysis/results.json")
        if results["status"] != "complete" or results["completed_new_runs"] != 9:
            return "blocked_unwrap_results_incomplete"
        lock = UNWRAP / ".factorial_runner.lock"
        try:
            holder = read(lock)["pid"]
        except FileNotFoundError:
            return "complete"
        return "running" if psutil.pid_exists(holder) else "complete"
    if status["status"] == "running":
        return "running" if alive(status["pid"], status["process_create_time"]) else "blocked_unwrap_runner_vanished"
    return "blocked_unwrap_" + status["status"]


def step(command, log):
    with log.open("ab") as handle:
        handle.write(f"\n=== {now()}: {' '.join(command)}\n".encode())
        handle.flush()
        return subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--check-only", action="store_true", help="Report the unwrap gate once and exit")
    args = parser.parse_args()
    study = args.study.resolve()
    if args.check_only:
        print(json.dumps({"unwrap_state": unwrap_state()}))
        return
    launcher = study / "launcher"
    launcher.mkdir(exist_ok=True)
    status_path, log = launcher / "chain_status.json", launcher / "chain.log"
    record(status_path, "waiting_for_unwrap_queue", unwrap_study=str(UNWRAP))
    while True:
        try:
            state = unwrap_state()
        except (OSError, ValueError, KeyError):
            state = "running"  # status files are replaced atomically; retry a transient read
        if state != "running":
            break
        time.sleep(args.poll_seconds)
    if state != "complete":
        record(status_path, state)
        sys.exit(1)
    record(status_path, "unwrap_queue_complete")
    module = [sys.executable, "-B", "-m", "experiments.wimans_20261014.execution"]
    timing = study / "gpu_timing.json"
    code = step(module + ["benchmark", "--study", str(study), "--output", str(timing)], log)
    if code or not timing.exists():
        record(status_path, "blocked_gpu_timing_failed", returncode=code)
        sys.exit(1)
    record(status_path, "gpu_timing_complete", receipt=str(timing))
    record(status_path, "four_arm_source_started")
    code = step(module + ["run", "--study", str(study), "--stage", "four_arm_source"], log)
    record(status_path, "four_arm_source_finished" if code == 0 else "four_arm_source_failed", returncode=code)
    sys.exit(code)


if __name__ == "__main__":
    main()
