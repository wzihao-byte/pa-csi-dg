"""Audit and recover an orphaned residual-scale runner without changing the protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import psutil

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tpp_residual_scale_ablation import analyze_scale, run_scale


BUNDLE = HERE / "generated_h11_l1248_v1"
OUTPUT_ROOT = REPO_ROOT / "outputs_tpp_residual_scale_ablation" / "h11_l1248_v1"
RUNNER_LOCK = OUTPUT_ROOT / ".factorial_runner.lock"
RESUME_STARTED = OUTPUT_ROOT / "planned_pause_20260812" / "resume_started.json"


class RecoveryError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryError(f"Expected a JSON object: {path}")
    return value


def active_experiment_processes() -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    markers = (
        "tpp_residual_scale_ablation\\run_scale.py",
        "tpp_residual_scale_ablation\\resume_planned.py",
        "tpp_residual_scale_ablation\\train_factorial.py",
        "tpp_residual_scale_ablation\\pipeline.py",
    )
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            if process.pid == os.getpid():
                continue
            command = " ".join(process.info.get("cmdline") or [])
            lowered = command.lower().replace("/", "\\")
            if any(marker in lowered for marker in markers):
                matches.append(
                    {
                        "pid": process.pid,
                        "name": process.info.get("name"),
                        "create_time": process.info.get("create_time"),
                        "command_line": command,
                    }
                )
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return matches


def audit_stale_lock() -> dict[str, Any]:
    if not RUNNER_LOCK.is_file():
        raise RecoveryError(f"Expected orphaned runner lock is missing: {RUNNER_LOCK}")
    lock = read_json(RUNNER_LOCK)
    pid = int(lock["pid"])
    if psutil.pid_exists(pid):
        try:
            process = psutil.Process(pid)
            command = " ".join(process.cmdline())
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            command = "<unavailable>"
        raise RecoveryError(f"Lock PID {pid} is still alive; refusing recovery: {command}")
    active = active_experiment_processes()
    if active:
        raise RecoveryError(f"Experiment processes are still alive; refusing recovery: {active}")
    return lock


def build_plan(jobs: list[Any], python: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    encountered_unfinished = False
    interrupted: list[dict[str, Any]] = []
    completed = 0
    untouched = 0
    for job in jobs:
        state = run_scale.base.scan(job, python)
        if state.successful_attempt is not None:
            if encountered_unfinished:
                raise RecoveryError(f"Successful job appears after an unfinished job: {job.job_id}")
            completed += 1
            plans.append(
                {
                    "job": job,
                    "action": "skip",
                    "next_attempt": state.next_attempt,
                    "successful_attempt": str(state.successful_attempt),
                }
            )
            continue

        encountered_unfinished = True
        statuses = [str(row["status"]) for row in state.prior]
        if statuses:
            if any(status != "interrupted" for status in statuses):
                raise RecoveryError(f"A failed or malformed lifecycle requires manual review: {job.job_id}: {state.prior}")
            interrupted.append(
                {
                    "job_id": job.job_id,
                    "target": job.target,
                    "seed": job.seed,
                    "scale": job.scale,
                    "next_attempt": state.next_attempt,
                    "prior": list(state.prior),
                }
            )
            action = "audited_rerun"
        else:
            untouched += 1
            action = "run"
        plans.append(
            {
                "job": job,
                "action": action,
                "next_attempt": state.next_attempt,
                "successful_attempt": None,
            }
        )

    if len(interrupted) != 1:
        raise RecoveryError(f"Expected exactly one interrupted job, found {len(interrupted)}: {interrupted}")
    summary = {
        "schema_version": "tpp-residual-scale-orphan-recovery/v1",
        "status": "ready",
        "created_at_utc": run_scale.base.utc_now(),
        "bundle_manifest_sha256": run_scale.base.sha256_file(BUNDLE / "bundle_manifest.json"),
        "prior_resume_started": str(RESUME_STARTED.resolve()),
        "prior_resume_started_sha256": run_scale.base.sha256_file(RESUME_STARTED),
        "completed_jobs_preserved": completed,
        "interrupted_job": interrupted[0],
        "untouched_jobs_remaining": untouched,
        "plan": [
            {
                "job_id": item["job"].job_id,
                "target": item["job"].target,
                "seed": item["job"].seed,
                "scale": item["job"].scale,
                "action": item["action"],
                "next_attempt": item["next_attempt"],
            }
            for item in plans
        ],
    }
    return plans, summary


def execute(plans: list[dict[str, Any]], summary: dict[str, Any], jobs: list[Any], python: Path) -> None:
    invocation_id = f"orphan-recovery-{uuid.uuid4().hex}"
    invocation = OUTPUT_ROOT / "runner_invocations" / invocation_id
    invocation.mkdir(parents=True, exist_ok=False)
    summary = {**summary, "status": "started", "invocation_id": invocation_id}
    run_scale.base.write_json_exclusive(invocation / "plan.json", summary)

    orphaned_lock = invocation / "orphaned_runner_lock.json"
    RUNNER_LOCK.replace(orphaned_lock)

    results: list[dict[str, Any]] = []
    with run_scale.base.RunnerLock(invocation_id):
        current_block: tuple[str, int] | None = None
        for item in plans:
            job = item["job"]
            block = (job.target, job.seed)
            if current_block is not None and block != current_block:
                barrier = run_scale.paired_barrier(jobs, python, current_block[0], current_block[1])
                run_scale.base.write_json_exclusive(
                    invocation / f"barrier_{current_block[0]}_seed_{current_block[1]}.json", barrier
                )
            current_block = block
            if item["action"] == "skip":
                results.append(
                    {
                        "job_id": job.job_id,
                        "status": "skipped_sealed_success",
                        "attempt": item["successful_attempt"],
                    }
                )
                continue
            attempt, health = run_scale.base.execute_job(job, int(item["next_attempt"]), python)
            results.append(
                {
                    "job_id": job.job_id,
                    "status": "success_after_audited_rerun" if item["action"] == "audited_rerun" else "success",
                    "attempt": str(attempt),
                    "health_sha256": run_scale.base.payload_sha256(health),
                }
            )
        if current_block is not None:
            barrier = run_scale.paired_barrier(jobs, python, current_block[0], current_block[1])
            run_scale.base.write_json_exclusive(
                invocation / f"barrier_{current_block[0]}_seed_{current_block[1]}.json", barrier
            )
        result_path = run_scale.finalize_results(jobs, python, BUNDLE.resolve())
        run_scale.base.write_json_exclusive(
            invocation / "matrix_completed.json",
            {
                "schema_version": "tpp-residual-scale-orphan-recovery/v1",
                "status": "matrix_complete",
                "results": results,
                "scale_results": str(result_path.resolve()),
                "scale_results_sha256": run_scale.base.sha256_file(result_path),
            },
        )

    analyze_scale.main()
    analysis_path = OUTPUT_ROOT / "analysis" / "residual_scale_analysis.json"
    run_scale.base.write_json_exclusive(
        invocation / "completed.json",
        {
            "schema_version": "tpp-residual-scale-orphan-recovery/v1",
            "status": "complete_with_analysis",
            "created_at_utc": run_scale.base.utc_now(),
            "scale_results": str((OUTPUT_ROOT / "scale_results.json").resolve()),
            "scale_results_sha256": run_scale.base.sha256_file(OUTPUT_ROOT / "scale_results.json"),
            "analysis": str(analysis_path.resolve()),
            "analysis_sha256": run_scale.base.sha256_file(analysis_path),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover an orphaned residual-scale runner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    stale_lock = audit_stale_lock()
    _manifest, _protocol, _execution, jobs = run_scale.verify_bundle(BUNDLE.resolve())
    python = Path(sys.executable).resolve()
    plans, summary = build_plan(jobs, python)
    summary["orphaned_lock"] = stale_lock
    if args.check_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return
    execute(plans, summary, jobs, python)


if __name__ == "__main__":
    main()
