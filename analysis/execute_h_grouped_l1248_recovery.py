"""Execute the unchanged frozen H bundle through a recovered queue gate.

This wrapper replaces only the interrupted prerequisite-process binding. Model,
data, optimizer, sampler, checkpoint selection, sealed-attempt handling, result
finalization, and paired analysis are imported from the frozen v2 bundle.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h-root", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args()
    h_root, ready_path = args.h_root.resolve(), args.ready.resolve()
    bundle, code = h_root / "bundle", h_root / "bundle/code"
    if ready_path != h_root / "queue/ready.json":
        raise RuntimeError("Unexpected recovered readiness record")
    sys.path.insert(0, str(code))

    import common
    import run as frozen_run
    from analyze import analyze

    common.require(common.BUNDLE.resolve() == bundle, "Frozen H bundle path differs")
    _, registry = common.verify_bundle(bundle)
    ready = common.read(ready_path)
    owner_path = h_root / "queue/owner.json"
    owner = common.read(owner_path)
    common.require(ready.get("bundle_manifest_sha256") == common.sha(bundle / "bundle_manifest.json"),
                   "Recovered readiness belongs to another H bundle")
    common.require(ready.get("recovery_owner_sha256") == common.sha(owner_path),
                   "Recovered readiness and queue owner differ")
    common.require(owner.get("widar_dependency") is False, "Widar entered the recovered H queue")
    prerequisite = ready.get("prerequisite", {})
    common.require(prerequisite.get("input_bundle_sha256") == owner.get("input_bundle_sha256"),
                   "Recovered input bundle binding differs")
    common.require(prerequisite.get("input_evaluation_summary_sha256"),
                   "Recovered input evaluation binding is absent")

    gpu = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    common.require(not any(line.strip().isdigit() for line in gpu.stdout.splitlines()),
                   "GPU is occupied; recovered H execution must not overlap another experiment")

    base = frozen_run.backend()
    jobs = frozen_run.jobs_for(base, bundle, registry)
    python = Path(sys.executable)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    invocation_id = uuid.uuid4().hex
    with base.RunnerLock(invocation_id):
        invocation = h_root / "runner_invocations" / invocation_id
        invocation.mkdir(parents=True)
        common.write_new(invocation / "started.json", {
            "status": "started", "pid": os.getpid(), "jobs": 18,
            "bundle_manifest_sha256": common.sha(bundle / "bundle_manifest.json"),
            "queue_owner_sha256": common.sha(owner_path),
            "recovered_prerequisite_sha256": common.sha(ready_path),
            "execution_wrapper_sha256": common.sha(Path(__file__)),
        })
        for job in jobs:
            common.verify_bundle(bundle)
            state = base.scan(job, python)
            if state.action == "run":
                base.execute_job(job, state.next_attempt, python)
            if job.arm == common.ARMS[-1]:
                barrier = base.quartet_barrier(jobs, python, job.target, job.seed)
                common.write_new(invocation / f"pair_{job.target}_{job.seed}.json", barrier)
        final = base.finalize_results(jobs, python, bundle)
        analysis = analyze(bundle, final)
        common.write_new(invocation / "completed.json", {
            "status": "complete", "jobs": 18, "results_sha256": common.sha(final),
            "analysis": str(analysis), "queue_owner_sha256": common.sha(owner_path),
            "execution_wrapper_sha256": common.sha(Path(__file__)),
        })
    print(json.dumps({"status": "complete", "jobs": 18, "results": str(final),
                      "recovered_queue_gate": True}))


if __name__ == "__main__":
    main()
