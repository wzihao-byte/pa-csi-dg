"""Create an audited planned pause and stop the live residual-scale process tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tpp_residual_scale_ablation import run_scale


BUNDLE = HERE / "generated_h11_l1248_v1"
OUTPUT_ROOT = REPO_ROOT / "outputs_tpp_residual_scale_ablation" / "h11_l1248_v1"
PAUSE_DIR = OUTPUT_ROOT / "planned_pause_20260812"
INTENT_PATH = PAUSE_DIR / "planned_pause_intent.json"
PAUSE_PATH = PAUSE_DIR / "planned_pause.json"


class PauseError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PauseError(f"Expected JSON object: {path}")
    return value


def process_record(process: psutil.Process) -> dict[str, Any]:
    return {
        "pid": process.pid,
        "name": process.name(),
        "create_time_utc": datetime.fromtimestamp(process.create_time(), timezone.utc).isoformat(),
        "command_line": process.cmdline(),
    }


def require_command(process: psutil.Process, fragment: str, role: str) -> None:
    command = " ".join(process.cmdline()).lower()
    if fragment.lower() not in command:
        raise PauseError(f"{role} PID {process.pid} command does not contain {fragment!r}: {command}")


def epoch_progress(log_path: Path) -> dict[str, Any]:
    if not log_path.is_file():
        return {"completed_epochs": 0, "current_epoch": 1, "latest_progress": None}
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    completed = sum(bool(re.search(r"100%\|.*150/150", line)) for line in lines)
    latest = next((line for line in reversed(lines) if "/150" in line), None)
    return {
        "completed_epochs": completed,
        "current_epoch": min(200, completed + 1),
        "latest_progress": latest,
    }


def find_matrix_state(jobs: list[Any]) -> tuple[list[dict[str, Any]], list[tuple[Any, Path]]]:
    rows: list[dict[str, Any]] = []
    active: list[tuple[Any, Path]] = []
    for job in jobs:
        attempts = run_scale.base.attempt_dirs(job.root)
        status = "not_started"
        attempt_path: Path | None = None
        if attempts:
            _number, attempt_path = attempts[-1]
            if (attempt_path / "attempt_completed.json").is_file():
                status = "success"
            elif (attempt_path / "attempt_failed.json").is_file():
                status = "failed"
            elif (attempt_path / "attempt_started.json").is_file():
                status = "active_or_interrupted"
                active.append((job, attempt_path))
            else:
                raise PauseError(f"Malformed attempt lifecycle: {attempt_path}")
        rows.append(
            {
                "job_id": job.job_id,
                "target": job.target,
                "seed": job.seed,
                "scale": job.scale,
                "status": status,
                "attempt": str(attempt_path) if attempt_path else None,
            }
        )
    return rows, active


def stop_process(process: psutil.Process, role: str) -> dict[str, Any]:
    before = process_record(process)
    process.terminate()
    try:
        exit_code = process.wait(timeout=10)
        forced = False
    except psutil.TimeoutExpired:
        process.kill()
        exit_code = process.wait(timeout=10)
        forced = True
    return {"role": role, "before": before, "forced_kill": forced, "exit_code": exit_code}


def pause() -> None:
    if PAUSE_PATH.exists() or INTENT_PATH.exists():
        raise PauseError(f"A planned pause record already exists: {PAUSE_DIR}")
    bundle_root = BUNDLE.resolve()
    _manifest, protocol, _execution, jobs = run_scale.verify_bundle(bundle_root)
    rows, active = find_matrix_state(jobs)
    failed = [row for row in rows if row["status"] == "failed"]
    if failed:
        raise PauseError(f"Cannot convert failed jobs into a planned pause: {failed}")
    if len(active) != 1:
        raise PauseError(f"Expected exactly one active job, found {len(active)}")
    job, attempt = active[0]
    lock_path = OUTPUT_ROOT / ".factorial_runner.lock"
    if not lock_path.is_file():
        raise PauseError("Live runner lock is absent")
    lock = read_json(lock_path)
    runner = psutil.Process(int(lock["pid"]))
    require_command(runner, "run_scale.py", "runner")
    trainer_children = [child for child in runner.children(recursive=True) if "python" in child.name().lower()]
    if len(trainer_children) != 1:
        raise PauseError(f"Expected one trainer child, found {[child.pid for child in trainer_children]}")
    trainer = trainer_children[0]
    require_command(trainer, "train_factorial.py", "trainer")
    require_command(trainer, str(attempt / "job_config.json"), "trainer")
    pipeline = runner.parent()
    if pipeline is None:
        raise PauseError("Runner pipeline parent is absent")
    require_command(pipeline, "pipeline.py", "pipeline")

    completed = [row for row in rows if row["status"] == "success"]
    pending = [row for row in rows if row["status"] == "not_started"]
    progress = epoch_progress(attempt / "runner.stderr.log")
    PAUSE_DIR.mkdir(parents=True, exist_ok=False)
    intent = {
        "schema_version": "tpp-residual-scale-planned-pause/v1",
        "status": "pause_intent_recorded",
        "created_at_utc": utc_now(),
        "reason": "planned computer shutdown before 2026-08-14",
        "bundle": str(bundle_root),
        "bundle_manifest_sha256": sha256_file(bundle_root / "bundle_manifest.json"),
        "output_root": str(OUTPUT_ROOT.resolve()),
        "matrix": {
            "total": len(rows),
            "completed": len(completed),
            "active": 1,
            "failed": 0,
            "not_started": len(pending),
        },
        "active_job": {
            "job_id": job.job_id,
            "target": job.target,
            "seed": job.seed,
            "scale": job.scale,
            "attempt_number": 1,
            "attempt_dir": str(attempt.resolve()),
            "attempt_started_sha256": sha256_file(attempt / "attempt_started.json"),
            "job_config_sha256": sha256_file(attempt / "job_config.json"),
            "progress_before_stop": progress,
        },
        "processes": {
            "pipeline": process_record(pipeline),
            "runner": process_record(runner),
            "trainer": process_record(trainer),
        },
        "completed_job_ids": [row["job_id"] for row in completed],
    }
    run_scale.base.atomic_write_json(INTENT_PATH, intent)

    stopped = []
    stopped.append(stop_process(pipeline, "pipeline"))
    stopped.append(stop_process(runner, "runner"))
    stopped.append(stop_process(trainer, "trainer"))
    if any(psutil.pid_exists(record["before"]["pid"]) for record in stopped):
        raise PauseError("One or more experiment processes remain after pause")

    lock_snapshot = PAUSE_DIR / "runner_lock_snapshot.json"
    if not lock_path.is_file():
        raise PauseError("Runner lock disappeared unexpectedly during forced pause")
    lock_sha256 = sha256_file(lock_path)
    os.replace(lock_path, lock_snapshot)
    interruption_path = attempt / "planned_interruption.json"
    interruption = {
        "schema_version": "tpp-residual-scale-planned-interruption/v1",
        "status": "planned_shutdown_interruption",
        "created_at_utc": utc_now(),
        "job_id": job.job_id,
        "attempt": 1,
        "reason": "planned computer shutdown before 2026-08-14",
        "resume_policy": "do_not_resume_optimizer_state; rerun identical cell from epoch 1 as append-only attempt_002",
        "optimizer_state_available": False,
        "progress_at_pause": epoch_progress(attempt / "runner.stderr.log"),
        "preserved_files": {
            path.relative_to(attempt).as_posix(): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(attempt.rglob("*"))
            if path.is_file() and path.name != "planned_interruption.json"
        },
    }
    run_scale.base.atomic_write_json(interruption_path, interruption)
    pause_record = {
        "schema_version": "tpp-residual-scale-planned-pause/v1",
        "status": "paused_and_safe_for_shutdown",
        "created_at_utc": utc_now(),
        "reason": intent["reason"],
        "bundle": intent["bundle"],
        "bundle_manifest_sha256": intent["bundle_manifest_sha256"],
        "output_root": intent["output_root"],
        "matrix": intent["matrix"],
        "active_job": intent["active_job"],
        "progress_after_stop": interruption["progress_at_pause"],
        "completed_job_ids": intent["completed_job_ids"],
        "process_termination": stopped,
        "runner_lock_snapshot": str(lock_snapshot.resolve()),
        "runner_lock_snapshot_sha256": lock_sha256,
        "planned_interruption": str(interruption_path.resolve()),
        "planned_interruption_sha256": sha256_file(interruption_path),
        "resume": {
            "script": str((HERE / "resume_planned.py").resolve()),
            "command": "python experiments\\tpp_residual_scale_ablation\\resume_planned.py --execute",
            "policy": "validate all 10 sealed successes, preserve attempt_001, rerun only interrupted cell as attempt_002, then continue remaining jobs",
            "automatic_start_on_boot": False,
        },
        "safe_for_shutdown": True,
    }
    run_scale.base.atomic_write_json(PAUSE_PATH, pause_record)
    print(json.dumps(pause_record, ensure_ascii=False, indent=2, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and pause the live residual-scale experiment")
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    pause()


if __name__ == "__main__":
    main()
