"""Explicitly resume the single audited shutdown interruption, then continue the matrix."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tpp_residual_scale_ablation import analyze_scale, run_scale


BUNDLE = HERE / "generated_h11_l1248_v1"
OUTPUT_ROOT = REPO_ROOT / "outputs_tpp_residual_scale_ablation" / "h11_l1248_v1"
PAUSE_DIR = OUTPUT_ROOT / "planned_pause_20260812"
PAUSE_PATH = PAUSE_DIR / "planned_pause.json"
RESUME_STARTED = PAUSE_DIR / "resume_started.json"
RESUME_COMPLETED = PAUSE_DIR / "resume_completed.json"


class ResumeError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResumeError(f"Expected JSON object: {path}")
    return value


def validate_pause(jobs: list[Any]) -> tuple[dict[str, Any], Any, Any]:
    if not PAUSE_PATH.is_file():
        raise ResumeError(f"Planned pause record is missing: {PAUSE_PATH}")
    pause = read_json(PAUSE_PATH)
    if pause.get("status") != "paused_and_safe_for_shutdown" or pause.get("safe_for_shutdown") is not True:
        raise ResumeError("Pause record is not sealed as safe")
    if pause.get("bundle_manifest_sha256") != run_scale.base.sha256_file(BUNDLE / "bundle_manifest.json"):
        raise ResumeError("Pause/bundle binding mismatch")
    interruption_path = Path(pause["planned_interruption"])
    if run_scale.base.sha256_file(interruption_path) != pause["planned_interruption_sha256"]:
        raise ResumeError("Planned interruption record hash mismatch")
    if (OUTPUT_ROOT / ".factorial_runner.lock").exists():
        raise ResumeError("A live or stale runner lock is present")
    paused_id = pause["active_job"]["job_id"]
    matches = [job for job in jobs if job.job_id == paused_id]
    if len(matches) != 1:
        raise ResumeError(f"Paused job is not unique in the bundle: {paused_id}")
    paused_job = matches[0]
    state = run_scale.base.scan(paused_job, Path(sys.executable).resolve())
    if state.successful_attempt is not None or state.next_attempt != 2:
        raise ResumeError("Paused job lifecycle is not the expected interrupted attempt_001")
    if len(state.prior) != 1 or state.prior[0]["status"] != "interrupted":
        raise ResumeError(f"Unexpected paused lifecycle: {state.prior}")
    attempt = Path(state.prior[0]["path"])
    if str(attempt.resolve()) != str(Path(pause["active_job"]["attempt_dir"]).resolve()):
        raise ResumeError("Paused attempt path mismatch")
    return pause, paused_job, state


def plan_jobs(jobs: list[Any], paused_job: Any) -> list[dict[str, Any]]:
    plans = []
    passed_pause = False
    for job in jobs:
        if job.job_id == paused_job.job_id:
            state = run_scale.base.scan(job, Path(sys.executable).resolve())
            plans.append({"job": job, "action": "planned_rerun", "next_attempt": state.next_attempt})
            passed_pause = True
            continue
        state = run_scale.scan_no_restart(job, Path(sys.executable).resolve())
        if not passed_pause and state.successful_attempt is None:
            raise ResumeError(f"A pre-pause job is not sealed successful: {job.job_id}")
        if passed_pause and state.prior and state.successful_attempt is None:
            raise ResumeError(f"Unexpected post-pause lifecycle: {job.job_id}: {state.prior}")
        plans.append(
            {
                "job": job,
                "action": state.action,
                "next_attempt": state.next_attempt,
                "successful_attempt": str(state.successful_attempt) if state.successful_attempt else None,
            }
        )
    return plans


def compact_plan(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "job_id": item["job"].job_id,
            "target": item["job"].target,
            "seed": item["job"].seed,
            "scale": item["job"].scale,
            "action": item["action"],
            "next_attempt": item["next_attempt"],
        }
        for item in plans
    ]


def execute_resume(jobs: list[Any], plans: list[dict[str, Any]], pause: dict[str, Any]) -> None:
    if RESUME_STARTED.exists():
        raise ResumeError(f"Resume was already started; manual audit required: {RESUME_STARTED}")
    invocation_id = f"planned-resume-{uuid.uuid4().hex}"
    with run_scale.base.RunnerLock(invocation_id):
        invocation = OUTPUT_ROOT / "runner_invocations" / invocation_id
        invocation.mkdir(parents=True, exist_ok=False)
        start_record = {
            "schema_version": "tpp-residual-scale-planned-resume/v1",
            "status": "started",
            "created_at_utc": run_scale.base.utc_now(),
            "pause_record": str(PAUSE_PATH.resolve()),
            "pause_record_sha256": run_scale.base.sha256_file(PAUSE_PATH),
            "bundle_manifest_sha256": run_scale.base.sha256_file(BUNDLE / "bundle_manifest.json"),
            "invocation_id": invocation_id,
            "plan": compact_plan(plans),
        }
        run_scale.base.write_json_exclusive(RESUME_STARTED, start_record)
        run_scale.base.write_json_exclusive(invocation / "plan.json", start_record)
        results = []
        current_block: tuple[str, int] | None = None
        python = Path(sys.executable).resolve()
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
                    {"job_id": job.job_id, "status": "skipped_sealed_success", "attempt": item["successful_attempt"]}
                )
                continue
            attempt, health = run_scale.base.execute_job(job, int(item["next_attempt"]), python)
            results.append(
                {
                    "job_id": job.job_id,
                    "status": "success_after_planned_rerun" if item["action"] == "planned_rerun" else "success",
                    "attempt": str(attempt),
                    "health_sha256": run_scale.base.payload_sha256(health),
                }
            )
        if current_block is not None:
            barrier = run_scale.paired_barrier(jobs, python, current_block[0], current_block[1])
            run_scale.base.write_json_exclusive(
                invocation / f"barrier_{current_block[0]}_seed_{current_block[1]}.json", barrier
            )
        final = run_scale.finalize_results(jobs, python, BUNDLE.resolve())
        run_scale.base.write_json_exclusive(
            invocation / "completed.json",
            {
                "schema_version": "tpp-residual-scale-planned-resume/v1",
                "status": "matrix_complete",
                "results": results,
                "scale_results": str(final),
                "scale_results_sha256": run_scale.base.sha256_file(final),
            },
        )
    analyze_scale.main()
    completion = {
        "schema_version": "tpp-residual-scale-planned-resume/v1",
        "status": "complete_with_analysis",
        "created_at_utc": run_scale.base.utc_now(),
        "scale_results": str((OUTPUT_ROOT / "scale_results.json").resolve()),
        "scale_results_sha256": run_scale.base.sha256_file(OUTPUT_ROOT / "scale_results.json"),
        "analysis": str((OUTPUT_ROOT / "analysis" / "residual_scale_analysis.json").resolve()),
        "analysis_sha256": run_scale.base.sha256_file(OUTPUT_ROOT / "analysis" / "residual_scale_analysis.json"),
    }
    run_scale.base.write_json_exclusive(RESUME_COMPLETED, completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume the audited planned shutdown interruption")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    _manifest, _protocol, _execution, jobs = run_scale.verify_bundle(BUNDLE.resolve())
    pause, paused_job, _state = validate_pause(jobs)
    plans = plan_jobs(jobs, paused_job)
    summary = {
        "status": "ready_to_resume",
        "paused_job": paused_job.job_id,
        "completed_jobs_preserved": sum(item["action"] == "skip" for item in plans),
        "planned_reruns": sum(item["action"] == "planned_rerun" for item in plans),
        "new_jobs_remaining_after_rerun": sum(item["action"] == "run" for item in plans),
        "plan": compact_plan(plans),
    }
    if args.check_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return
    execute_resume(jobs, plans, pause)


if __name__ == "__main__":
    main()
