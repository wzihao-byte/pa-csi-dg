"""Resume the frozen MultiEnv input study after a host interruption.

This coordinator preserves the previous owner records, verifies every completed
run and the last committed model/optimizer/RNG checkpoint, then re-enters the
immutable input-study engine. The engine itself performs a second exact-state
audit before taking another optimizer step.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.revision_validation.extended_engine import verify_extension
from experiments.revision_validation.study_io import bound_path, now, read, sha256, value_hash, verify_completed, write


def same_process(pid: int, birth: float) -> bool:
    try:
        return abs(psutil.Process(pid).create_time() - birth) < 0.01
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prior-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    # Retained in the command line for unchanged downstream identity gates.
    parser.add_argument("--compatibility-marker", required=True)
    args = parser.parse_args()
    bundle, prior, output = args.bundle.resolve(), args.prior_output.resolve(), args.output.resolve()
    require(args.compatibility_marker == "analysis/continue_inputs_after_controls.py",
            "Unexpected compatibility marker")
    require(not output.exists(), "Preserve an existing input recovery attempt")
    protocol, bundle_hash = verify_extension(bundle)
    require(protocol.get("study_kind") == "input_modality_controls" and len(protocol["jobs"]) == 63,
            "Wrong frozen input study")
    require(not (bundle / "queue_failure.json").exists(), "Input study has an unresolved recorded failure")
    require(not (bundle / "target_gate.json").exists(), "Target gate exists before source training completed")

    prior_status = read(prior / "status.json")
    require(prior_status.get("status") == "inputs_running", "Prior coordinator was not interrupted while inputs ran")
    require(not psutil.pid_exists(int(prior_status["pid"])) and not psutil.pid_exists(int(prior_status["child_pid"])),
            "Prior input owner or child is still live")
    for process in psutil.process_iter(["cmdline"]):
        if process.pid == os.getpid():
            continue
        command = process.info["cmdline"] or []
        if "--bundle" in command:
            try:
                if Path(command[command.index("--bundle") + 1]).resolve() == bundle:
                    raise RuntimeError("A runner already owns the frozen input bundle")
            except (ValueError, OSError):
                pass

    completed = []
    active = None
    for config in protocol["jobs"]:
        run = bundle / "runs" / config["job_id"]
        status = read(run / "status.json") if (run / "status.json").is_file() else None
        if status and status.get("status") == "completed":
            verify_completed(run, config)
            completed.append(config["job_id"])
        elif (run / "progress.json").is_file():
            require(active is None, "Multiple incomplete runs have committed checkpoints")
            progress = read(run / "progress.json")
            checkpoint = bound_path(run, progress["checkpoint"])
            require(sha256(checkpoint) == progress["sha256"], "Committed resume checkpoint hash mismatch")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            require(state["config_value_sha256"] == value_hash(config), "Checkpoint configuration binding differs")
            require(len(state["history"]) == progress["completed_epochs"], "Checkpoint history length differs")
            require(state["step"] == len(state["history"]) * config["updates_per_epoch"],
                    "Checkpoint optimizer-step count differs")
            active = {"job_id": config["job_id"], "completed_epochs": progress["completed_epochs"],
                      "checkpoint": progress["checkpoint"], "checkpoint_sha256": progress["sha256"],
                      "config_value_sha256": state["config_value_sha256"]}
    require(len(completed) == 61 and active is not None and active["job_id"] == "E3_I_TWIN_AP_602",
            "Interrupted input boundary differs from the observed 61-job/one-active state")

    stale_lock = bundle / "runner.lock.json"
    require(stale_lock.is_file(), "Interrupted runner lock is missing")
    lock = read(stale_lock)
    require(not same_process(int(lock["pid"]), float(lock["create_time"])), "Frozen input lock owner remains live")
    archived_lock = bundle / "runner.lock.interrupted_20260917.json"
    require(not archived_lock.exists(), "Input interruption lock was already archived")

    output.mkdir(parents=True)
    recovery = {"status": "validated", "validated_at": now(), "pid": os.getpid(),
                "process_create_time": psutil.Process().create_time(), "command": psutil.Process().cmdline(),
                "code_sha256": sha256(Path(__file__)), "bundle": str(bundle),
                "input_bundle_sha256": bundle_hash, "prior_output": str(prior),
                "prior_status_sha256": sha256(prior / "status.json"), "completed_jobs": completed,
                "completed_count": len(completed), "active_run": active,
                "stale_lock": lock, "target_data_loaded": False, "widar_dependency": False}
    write(output / "recovery_validation.json", recovery)
    shutil.move(stale_lock, archived_lock)
    recovery["archived_lock"] = str(archived_lock)
    recovery["archived_lock_sha256"] = sha256(archived_lock)
    write(output / "status.json", {**recovery, "status": "starting_exact_resume"})

    command = [sys.executable, "-X", "utf8", "-m", "experiments.revision_validation.input_study_engine",
               "--bundle", str(bundle)]
    try:
        with (bundle / "runner.attempt02.stdout.log").open("x", encoding="utf-8") as stdout, \
             (bundle / "runner.attempt02.stderr.log").open("x", encoding="utf-8") as stderr:
            child = subprocess.Popen(command, cwd=bundle / "code", stdout=stdout, stderr=stderr,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            write(output / "status.json", {**recovery, "status": "inputs_running", "child_pid": child.pid,
                  "resume_command": command, "started_at": now()})
            result = child.wait()
        require(result == 0, f"Frozen input engine exited with return code {result}")
        queue = read(bundle / "queue_status.json")
        require(queue.get("status") == "completed" and queue.get("jobs") == 63,
                "Frozen input study did not complete all 63 jobs")
        summary = bundle / "evaluation/summary.json"
        write(output / "status.json", {"status": "completed", "completed_at": now(),
              "input_bundle_sha256": bundle_hash, "input_evaluation_summary_sha256": sha256(summary),
              "recovery_validation_sha256": sha256(output / "recovery_validation.json"),
              "resumed_from_job": active["job_id"], "resumed_from_epoch": active["completed_epochs"],
              "target_data_loaded_during_source_resume": False})
    except BaseException:
        write(output / "status.json", {**recovery, "status": "failed", "failed_at": now(),
              "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
