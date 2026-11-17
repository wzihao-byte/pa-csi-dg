"""One recovery pass after an in-flight acquisition ends; no GPU or training."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from experiments.wimans_20261014.acquisition import DEFAULT_ROOT, REPO, now, write_json


def main():
    path = DEFAULT_ROOT / "acquisition/status.json"
    record = {"started_at_utc": now(), "pid": os.getpid(), "status": "waiting_for_initial_acquisition",
              "recovery_kind": "one_sequential_pass_same_manifest_same_dataset_version_path_endpoint",
              "training_started": False, "gpu_used": False}
    write_json(DEFAULT_ROOT / "recovery_status.json", record)
    while True:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["status"] != "running":
            break
        # A crash must not be mistaken for an ongoing job indefinitely.
        import psutil
        launch = json.loads((DEFAULT_ROOT / "launch_record.json").read_text(encoding="utf-8-sig"))
        if not psutil.pid_exists(launch["pid"]):
            record.update(status="initial_job_crashed_requires_inspection", updated_at_utc=now())
            write_json(DEFAULT_ROOT / "recovery_status.json", record)
            raise SystemExit(1)
        time.sleep(2)
    if previous.get("complete"):
        record.update(status="not_needed_initial_acquisition_complete", updated_at_utc=now())
    else:
        # A failed initial pass exits before preprocessing. Preserve the old
        # attempt; verified existing bytes are reused, only missing files fetched.
        record.update(status="recovering", updated_at_utc=now())
        write_json(DEFAULT_ROOT / "recovery_status.json", record)
        result = subprocess.run([sys.executable, "-m", "experiments.wimans_20261014.prepare_workflow"], cwd=REPO)
        record.update(status="finished" if result.returncode == 0 else "failed",
                      exit_code=result.returncode, updated_at_utc=now())
    write_json(DEFAULT_ROOT / "recovery_status.json", record)


if __name__ == "__main__":
    main()
