"""Train frozen input controls only after their completed prerequisite studies."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import traceback

import psutil

from experiments.revision_validation.extended_engine import train_experiment, verify_extension
from experiments.revision_validation.extended_evaluate import evaluate
from experiments.revision_validation.study_engine import acquire_lock
from experiments.revision_validation.study_io import bound_path, now, read, sha256, verify_bundle, verify_completed, write


def verify_prerequisites(protocol):
    evidence = []
    for requirement in protocol["prerequisites"]:
        bundle = Path(requirement["bundle"]).resolve()
        # Do not inspect atomically replaced status JSON while a writer is live.
        for process in psutil.process_iter(["cmdline"]):
            command = process.info["cmdline"] or []
            if "--bundle" in command and Path(command[command.index("--bundle")+1]).resolve() == bundle:
                raise RuntimeError(f"Prerequisite runner remains active: {bundle}")
        verifier = verify_bundle if requirement["kind"] == "core" else verify_extension
        reference, digest = verifier(bundle)
        if digest != requirement["bundle_sha256"]:
            raise RuntimeError("A prerequisite bundle differs from the frozen declaration")
        if (bundle / "queue_failure.json").exists():
            raise RuntimeError("A prerequisite has an unresolved failure")
        for config in reference["jobs"]:
            verify_completed(bundle / "runs" / config["job_id"], config)
        summary_path = bundle / "evaluation/summary.json"
        summary = read(summary_path)
        if {r["job_id"] for r in summary["runs"]} != {c["job_id"] for c in reference["jobs"]}:
            raise RuntimeError("Prerequisite evaluation does not cover every frozen job")
        for relative, expected in summary["artifacts_sha256"].items():
            if sha256(bound_path(bundle, relative)) != expected:
                raise RuntimeError("Prerequisite evaluation artifact hash mismatch")
        evidence.append({**requirement, "evaluation_summary_sha256": sha256(summary_path), "verified_jobs": len(reference["jobs"])})
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    protocol, digest = verify_extension(bundle)
    if Path(__file__).resolve().parents[2] != bundle / "code":
        raise RuntimeError("Use the immutable input-study code snapshot")
    prerequisites = verify_prerequisites(protocol)
    lock, identity = acquire_lock(bundle)
    try:
        write(bundle / "prerequisite_gate.json", {"passed_at": now(), "prerequisites": prerequisites})
        for index, config in enumerate(protocol["jobs"]):
            verify_extension(bundle)
            write(bundle / "queue_status.json", {"status": "running", "job_index": index, "job_id": config["job_id"],
                  "total_jobs": len(protocol["jobs"]), "bundle_sha256": digest, "pid": os.getpid(), "updated_at": now()})
            train_experiment(config, bound_path(bundle, "runs/" + config["job_id"]))
        write(bundle / "queue_status.json", {"status": "source_training_completed", "jobs": len(protocol["jobs"]), "updated_at": now()})
        summary_hash = evaluate(bundle)
        write(bundle / "queue_status.json", {"status": "completed", "jobs": len(protocol["jobs"]), "updated_at": now(),
              "bundle_sha256": digest, "evaluation_summary_sha256": summary_hash})
    except BaseException:
        write(bundle / "queue_failure.json", {"status": "failed", "updated_at": now(), "error": traceback.format_exc()})
        raise
    finally:
        if lock.exists() and read(lock) == identity:
            lock.unlink()


if __name__ == "__main__":
    main()
