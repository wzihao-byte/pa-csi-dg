"""Run the frozen 18-job supplement through the original sealed-attempt runner."""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from common import ARMS, BUNDLE, HERE, OUTPUT, PARAMETERS, legacy, read, require, sha, verify_bundle, write_new


def backend():
    base = legacy("runner")
    base.OUTPUT_ROOT = OUTPUT
    base.EXPERIMENT_ROOT = HERE
    base.ARM_ORDER = ARMS
    base.SCHEMA_VERSION = "h-l1248-grouped-runner/v1"
    base.COMPLETION_VERSION = "h-l1248-grouped-completion/v1"
    base.SEAL_VERSION = "h-l1248-grouped-seal/v1"
    original = base.validate_run

    def validate(job, attempt, python):
        health = original(job, attempt, python)
        config = read(attempt / "job_config.json")
        run = attempt / "payload/run"
        initialization = read(run / "initialization_audit.json")
        require(initialization["parameter_count"] == health["parameter_count"] == PARAMETERS, "Actual parameter mismatch")
        require(initialization["initial_state_sha256"] == config["h_grouped"]["initial_state_sha256"], "Actual initialization mismatch")
        require(health["sampler_schedule_sha256"] == config["h_grouped"]["expected_sampler_sha256"], "Actual sampling differs from frozen 30,000-step replay")
        split = read(run / "split_manifest.json")
        require(split["grouped_source_manifest_sha256"] == config["h_grouped"]["source_manifest_sha256"], "Actual grouped split binding mismatch")
        for key in ("sample_id", "subject_id", "acquisition_id"):
            require(not set(split["group_identity"]["train"][key]) & set(split["group_identity"]["val"][key]), "Actual source identity overlap")
        metrics = read(run / "metrics.json")
        best = max(float(row["val"]["accuracy"]) for row in metrics["history"])
        require(metrics["best_val_accuracy"] == best == metrics["best_val_score"], "Checkpoint selection score drift")
        health["initial_state_sha256"] = initialization["initial_state_sha256"]
        health["last_tied_best_source_epoch"] = max(row["epoch"] for row in metrics["history"] if row["val"]["accuracy"] == best)
        health["group_identity_overlap"] = False
        return health

    base.validate_run = validate
    return base


def jobs_for(base, bundle, registry):
    return [base.Job(j["job_id"], j["target"], j["seed"], j["arm"], bundle / j["config"],
                     sha(bundle / j["config"]), base.payload_sha256(read(bundle / j["config"])))
            for j in registry["jobs"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    require(bundle == BUNDLE.resolve(), "Use the registered supplement bundle")
    _, registry = verify_bundle(bundle)
    base = backend()
    jobs = jobs_for(base, bundle, registry)
    python = Path(sys.executable)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    if args.dry_run:
        plans = [{"job_id": j.job_id, "action": base.scan(j, python).action} for j in jobs]
        print(json.dumps({"status": "dry_run_passed", "jobs": len(plans), "plan": plans}, indent=2))
        return
    # Recheck the upstream gate inside the actual runner, not only the coordinator.
    from queue_after_inputs import prerequisite_complete, gpu_idle
    evidence = prerequisite_complete(bundle)
    ready = read(OUTPUT / "queue/ready.json")
    require(ready["bundle_manifest_sha256"] == sha(bundle / "bundle_manifest.json"), "Readiness belongs to another bundle")
    require(ready["prerequisite"] == evidence, "Upstream completion changed after queue release")
    require(gpu_idle(), "GPU is occupied; do not overlap an existing experiment")
    invocation_id = uuid.uuid4().hex
    with base.RunnerLock(invocation_id):
        invocation = OUTPUT / "runner_invocations" / invocation_id
        invocation.mkdir(parents=True)
        write_new(invocation / "started.json", {"status": "started", "pid": os.getpid(), "bundle_manifest_sha256": sha(bundle / "bundle_manifest.json"), "jobs": 18})
        for job in jobs:
            verify_bundle(bundle)
            state = base.scan(job, python)
            if state.action == "run":
                base.execute_job(job, state.next_attempt, python)
            if job.arm == ARMS[-1]:
                barrier = base.quartet_barrier(jobs, python, job.target, job.seed)
                write_new(invocation / f"pair_{job.target}_{job.seed}.json", barrier)
        final = base.finalize_results(jobs, python, bundle)
        from analyze import analyze
        analysis = analyze(bundle, final)
        write_new(invocation / "completed.json", {"status": "complete", "jobs": 18,
                  "results_sha256": sha(final), "analysis": str(analysis)})
    print(json.dumps({"status": "complete", "jobs": 18, "results": str(final)}))


if __name__ == "__main__":
    main()
