"""Start the frozen input matrix only after the core-to-controls chain exits successfully."""
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
from experiments.revision_validation.extended_engine import verify_extension
from experiments.revision_validation.study_io import now, read, sha256, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--transition-pid", type=int)
    source.add_argument("--controls-pid", type=int)
    parser.add_argument("--transition-output", type=Path)
    parser.add_argument("--controls-bundle", type=Path)
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    transition = args.transition_output.resolve() if args.transition_output else None
    controls = args.controls_bundle.resolve() if args.controls_bundle else None
    bundle, output = args.input_bundle.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve an existing controls-to-inputs coordinator attempt")
    protocol, bundle_hash = verify_extension(bundle)
    if protocol.get("study_kind") != "input_modality_controls" or len(protocol["jobs"]) != 63:
        raise RuntimeError("Wrong frozen input study")
    upstream_pid = args.transition_pid if args.transition_pid is not None else args.controls_pid
    process = psutil.Process(upstream_pid)
    birth, command = process.create_time(), process.cmdline()
    joined = " ".join(command).replace("\\", "/").lower()
    if args.transition_pid is not None:
        if transition is None or "continue_controls_after_core.py" not in joined or str(transition).replace("\\", "/").lower() not in joined:
            raise RuntimeError("Provided process is not the expected core-to-controls coordinator")
        recovery_mode = False
    else:
        if controls is None:
            raise RuntimeError("--controls-bundle is required with --controls-pid")
        if "experiments.revision_validation.extended_engine" not in joined or str(controls).replace("\\", "/").lower() not in joined:
            raise RuntimeError("Provided process is not the expected frozen controls runner")
        recovery_mode = True
    identity = {"pid": os.getpid(), "started_at": now(), "upstream_pid": upstream_pid,
                "upstream_process_create_time": birth, "upstream_command": command,
                "transition_output": str(transition) if transition else None,
                "controls_bundle": str(controls) if controls else None,
                "recovery_after_process_interruption": recovery_mode,
                "input_bundle": str(bundle), "input_bundle_sha256": bundle_hash,
                "code_sha256": sha256(__file__), "mutable_training_json_observed_while_live": False}
    output.mkdir(parents=True)
    write(output / "config.json", identity)
    write(output / "status.json", {"status": "waiting_for_core_and_controls_chain", **identity})
    try:
        while True:
            try:
                live = psutil.Process(upstream_pid).create_time() == birth
            except psutil.NoSuchProcess:
                live = False
            if not live:
                break
            time.sleep(30)
        if recovery_mode:
            control_protocol, control_hash = verify_extension(controls)
            control_status = read(controls / "queue_status.json")
            if control_status.get("status") != "completed" or len(control_protocol["jobs"]) != 81:
                raise RuntimeError("Recovered frozen controls runner did not complete its 81-job study")
            for config in control_protocol["jobs"]:
                from experiments.revision_validation.study_io import verify_completed
                verify_completed(controls / "runs" / config["job_id"], config)
            if control_status.get("bundle_sha256") != control_hash:
                raise RuntimeError("Recovered controls completion is bound to a different bundle")
        else:
            transition_status = read(transition / "status.json")
            if transition_status.get("status") != "completed":
                raise RuntimeError("Core-to-controls chain did not complete; input study remains gated")
        if verify_extension(bundle)[1] != bundle_hash:
            raise RuntimeError("Frozen input bundle changed while waiting")
        if (bundle / "queue_failure.json").exists() or (bundle / "runner.lock.json").exists():
            raise RuntimeError("Input bundle has a failure/lock; do not start a duplicate")
        if (bundle / "target_gate.json").exists():
            raise RuntimeError("Input target gate exists before source training")
        command = [sys.executable, "-X", "utf8", "-m", "experiments.revision_validation.input_study_engine",
                   "--bundle", str(bundle)]
        with (bundle / "runner.attempt01.stdout.log").open("x", encoding="utf-8") as stdout, \
             (bundle / "runner.attempt01.stderr.log").open("x", encoding="utf-8") as stderr:
            child = subprocess.Popen(command, cwd=bundle / "code", stdout=stdout, stderr=stderr,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            write(output / "status.json", {"status": "inputs_running", "pid": os.getpid(),
                  "child_pid": child.pid, "started_at": now(), "command": command,
                  "input_bundle_sha256": bundle_hash})
            result = child.wait()
        if result != 0 or read(bundle / "queue_status.json").get("status") != "completed":
            raise RuntimeError(f"Frozen input study exited with return code {result}; inspect retained evidence")
        write(output / "status.json", {"status": "completed", "completed_at": now(),
              "input_bundle_sha256": bundle_hash,
              "input_evaluation_summary_sha256": sha256(bundle / "evaluation/summary.json")})
    except BaseException:
        write(output / "status.json", {"status": "failed", "failed_at": now(),
              "error": traceback.format_exc(), **identity})
        raise


if __name__ == "__main__":
    main()
