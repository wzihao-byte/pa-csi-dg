"""Run the frozen Widar source matrix after its prerequisite input study."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import traceback

from experiments.revision_validation.extended_engine import train_experiment, verify_extension
from experiments.revision_validation.extended_evaluate import evaluate
from experiments.revision_validation.input_study_engine import verify_prerequisites
from experiments.revision_validation.study_engine import acquire_lock
from experiments.revision_validation.study_io import bound_path, now, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    protocol, digest = verify_extension(bundle)
    if Path(__file__).resolve().parents[2] != bundle / "code":
        raise RuntimeError("Use the immutable Widar study code snapshot")
    prerequisites = verify_prerequisites(protocol)
    lock, identity = acquire_lock(bundle)
    try:
        write(bundle / "prerequisite_gate.json", {"passed_at": now(), "prerequisites": prerequisites})
        for index, config in enumerate(protocol["jobs"]):
            verify_extension(bundle)
            write(bundle / "queue_status.json", {"status": "running", "job_index": index,
                  "job_id": config["job_id"], "total_jobs": len(protocol["jobs"]),
                  "bundle_sha256": digest, "pid": os.getpid(), "updated_at": now()})
            train_experiment(config, bound_path(bundle, "runs/" + config["job_id"]))
        write(bundle / "queue_status.json", {"status": "source_training_completed", "jobs": len(protocol["jobs"]), "updated_at": now()})
        summary_hash = evaluate(bundle)
        write(bundle / "queue_status.json", {"status": "completed", "jobs": len(protocol["jobs"]),
              "updated_at": now(), "bundle_sha256": digest, "evaluation_summary_sha256": summary_hash})
    except BaseException:
        write(bundle / "queue_failure.json", {"status": "failed", "updated_at": now(), "error": traceback.format_exc()})
        raise
    finally:
        if lock.exists() and read(lock) == identity:
            lock.unlink()


if __name__ == "__main__":
    main()
