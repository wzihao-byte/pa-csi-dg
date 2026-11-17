"""Wait for existing MultiEnv inputs, then run H; Widar is excluded by the user."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import psutil

from common import BUNDLE, HERE, OUTPUT, read, require, sha, verify_bundle, write_new


def same_process(pid, birth):
    try:
        return abs(psutil.Process(pid).create_time() - birth) < .01
    except psutil.NoSuchProcess:
        return False


def prerequisite_complete(bundle):
    prerequisite = read(bundle / "prerequisite.json")
    require(prerequisite["widar_cancelled_by_user"] is True, "Widar cancellation missing from current protocol")
    require(not same_process(prerequisite["pid"], prerequisite["process_create_time"]), "Existing MultiEnv input coordinator is still live")
    # Read mutable completion files only after the identified owner has exited.
    status_path = Path(prerequisite["coordinator_output"]) / "status.json"
    status = read(status_path)
    require(status["status"] == "completed", "Existing MultiEnv input chain did not complete successfully")
    inputs = Path(prerequisite["input_bundle"])
    manifest_hash = sha(inputs / "bundle_manifest.json")
    require(manifest_hash == prerequisite["input_bundle_sha256"], "Existing frozen input bundle changed")
    evaluation_hash = sha(inputs / "evaluation/summary.json")
    require(status["input_evaluation_summary_sha256"] == evaluation_hash, "Input evaluation completion binding mismatch")
    require(read(inputs / "queue_status.json")["status"] == "completed", "Input queue incomplete")
    require(status["input_bundle_sha256"] == manifest_hash, "Input completion belongs to another bundle")
    return {"coordinator_status_sha256": sha(status_path), "input_bundle_sha256": manifest_hash,
            "input_evaluation_summary_sha256": evaluation_hash}


def gpu_idle():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return not any(line.strip().isdigit() for line in result.stdout.splitlines())


def emit(event, **details):
    value = {"at": datetime.now(timezone.utc).isoformat(), "event": event, **details}
    with (OUTPUT / "queue/events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(value) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    require(bundle == BUNDLE.resolve(), "Unexpected supplement bundle")
    verify_bundle(bundle)
    prerequisite = read(bundle / "prerequisite.json")
    live = same_process(prerequisite["pid"], prerequisite["process_create_time"])
    if live:
        require(psutil.Process(prerequisite["pid"]).cmdline() == prerequisite["command"], "Upstream process command changed")
    if args.check_only:
        if not live:
            prerequisite_complete(bundle)
        print(json.dumps({"status": "prerequisite_wait_verified" if live else "prerequisite_complete", "formal_training_started": False}))
        return
    queue = OUTPUT / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    identity = {"pid": os.getpid(), "process_create_time": psutil.Process().create_time(),
                "bundle_manifest_sha256": sha(bundle / "bundle_manifest.json"),
                "command": psutil.Process().cmdline(), "waiting_for": prerequisite}
    write_new(queue / "owner.json", identity)
    try:
        emit("waiting_for_existing_multienv_inputs", upstream_pid=prerequisite["pid"], jobs=18, widar_excluded=True)
        while same_process(prerequisite["pid"], prerequisite["process_create_time"]):
            time.sleep(30)
        evidence = prerequisite_complete(bundle)
        verify_bundle(bundle)
        emit("upstream_completed", **evidence)
        while not gpu_idle():
            time.sleep(30)
        write_new(queue / "ready.json", {"bundle_manifest_sha256": sha(bundle / "bundle_manifest.json"), "prerequisite": evidence})
        command = [sys.executable, "-B", "-X", "utf8", str(HERE / "run.py"), "--execute", "--bundle", str(bundle)]
        emit("h_supplement_started", command=command)
        with (queue / "runner.log").open("xb") as log:
            result = subprocess.run(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        require(result.returncode == 0, f"H supplement runner failed: {result.returncode}")
        final = OUTPUT / "factorial_results.json"
        require(read(final)["job_count"] == 18 and read(final)["status"] == "complete", "H supplement result incomplete")
        write_new(queue / "completed.json", {"status": "completed", "jobs": 18, "results_sha256": sha(final)})
        emit("completed", jobs=18)
    except BaseException:
        error = traceback.format_exc()
        write_new(queue / "failed.json", {"status": "failed", "error": error})
        emit("failed", error=error)
        raise


if __name__ == "__main__":
    main()
