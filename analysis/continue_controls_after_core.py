"""External serial coordinator; frozen control code starts after core exit.

This coordinator observes only process state while the core is live, and reads
JSON after core exit. It cannot clear failures or change models.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.runtime_observation import optional, read
from experiments.revision_validation.extended_engine import verify_extension
from experiments.revision_validation.study_io import now, sha256, verify_bundle, verify_completed, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-bundle", type=Path, required=True)
    parser.add_argument("--controls-bundle", type=Path, required=True)
    parser.add_argument("--core-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    core, controls, output = args.core_bundle.resolve(), args.controls_bundle.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new coordinator attempt; existing state is retained")
    core_protocol, core_hash = verify_bundle(core)
    control_protocol, control_hash = verify_extension(controls)
    if control_protocol["dependency_bundle_sha256"] != core_hash:
        raise RuntimeError("Frozen control dependency differs")
    process = psutil.Process(args.core_pid)
    birth = process.create_time()
    if "experiments.revision_validation.study_engine" not in process.cmdline() or str(core).lower() not in " ".join(process.cmdline()).lower():
        raise RuntimeError("Provided process is not the expected frozen core runner")
    identity = {"pid": os.getpid(), "started_at": now(), "core_pid": args.core_pid, "core_process_create_time": birth,
                "core_bundle_sha256": core_hash, "controls_bundle_sha256": control_hash,
                "code_sha256": {"coordinator": sha256(__file__), "observer": sha256(ROOT / "analysis/runtime_observation.py")}}
    output.mkdir(parents=True)
    write(output / "config.json", identity)
    write(output / "status.json", {"status": "waiting_for_core_exit", **identity})
    try:
        while True:
            try:
                live = psutil.Process(args.core_pid).create_time() == birth
            except psutil.NoSuchProcess:
                live = False
            if not live:
                if optional(core / "queue_failure.json"):
                    raise RuntimeError("Core recorded a failure; diagnosis is required")
                queue = optional(core / "queue_status.json")
                if not queue or queue["status"] != "source_training_completed":
                    raise RuntimeError("Core exited without completing its frozen source queue")
                break
            time.sleep(30)
        if verify_bundle(core)[1] != core_hash or verify_extension(controls)[1] != control_hash:
            raise RuntimeError("Frozen bundle changed while waiting")
        for config in core_protocol["jobs"]:
            verify_completed(core / "runs" / config["job_id"], config)
        if (controls / "queue_failure.json").exists() or (controls / "runner.lock.json").exists():
            raise RuntimeError("Control failure/lock remains; do not start a duplicate or hide a failure")
        command = [sys.executable, "-X", "utf8", "-m", "experiments.revision_validation.extended_engine", "--bundle", str(controls)]
        with (controls / "runner.attempt02.stdout.log").open("x", encoding="utf-8") as stdout, (controls / "runner.attempt02.stderr.log").open("x", encoding="utf-8") as stderr:
            child = subprocess.Popen(command, cwd=controls / "code", stdout=stdout, stderr=stderr,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            write(output / "status.json", {"status": "controls_running", "pid": os.getpid(), "child_pid": child.pid, "started_at": now(), "command": command})
            result = child.wait()
        if result != 0 or read(controls / "queue_status.json")["status"] != "completed":
            raise RuntimeError(f"Frozen controls exited with return code {result}; inspect its retained evidence")
        write(output / "status.json", {"status": "completed", "completed_at": now(), "core_bundle_sha256": core_hash,
              "controls_bundle_sha256": control_hash, "controls_evaluation_summary_sha256": sha256(controls / "evaluation/summary.json")})
    except BaseException:
        write(output / "status.json", {"status": "failed", "failed_at": now(), "error": traceback.format_exc(), **identity})
        raise


if __name__ == "__main__":
    main()
