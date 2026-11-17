"""Recover the queued H supplement after an interrupted input coordinator.

The frozen H bundle is not modified. This coordinator binds a new live
MultiEnv-input prerequisite to the existing v2 queue, preserves the
interrupted owner record, and launches the frozen H runner only after the new
input chain completes and the GPU is idle.
"""
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


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def require(condition, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def same_process(pid: int, birth: float) -> bool:
    try:
        return abs(psutil.Process(pid).create_time() - birth) < 0.01
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def emit(events: Path, event: str, **details) -> None:
    value = {"at": datetime.now(timezone.utc).isoformat(), "event": event, **details}
    with events.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(value, sort_keys=True), flush=True)


def verify_bundle(bundle: Path) -> tuple[dict, str]:
    manifest_path = bundle / "bundle_manifest.json"
    manifest = read(manifest_path)
    for relative, expected in manifest["files_sha256"].items():
        require(sha256(bundle / relative) == expected, f"Frozen H bundle member changed: {relative}")
    for name, expected in manifest["external_code_sha256"].items():
        require(sha256(Path(name)) == expected, f"Historical H dependency changed: {name}")
    return manifest, sha256(manifest_path)


def process_identity(pid: int, required: tuple[str, ...]) -> tuple[float, list[str]]:
    process = psutil.Process(pid)
    birth, command = process.create_time(), process.cmdline()
    joined = " ".join(command).replace("\\", "/").lower()
    require(all(item.replace("\\", "/").lower() in joined for item in required),
            "Provided process is not the declared MultiEnv input coordinator")
    return birth, command


def verify_input_completion(output: Path, bundle: Path, expected_bundle_hash: str) -> dict:
    status_path = output / "status.json"
    status = read(status_path)
    require(status.get("status") == "completed", "Recovered MultiEnv input chain did not complete")
    require(sha256(bundle / "bundle_manifest.json") == expected_bundle_hash,
            "Frozen MultiEnv input bundle changed while waiting")
    queue_status = read(bundle / "queue_status.json")
    require(queue_status.get("status") == "completed", "MultiEnv input queue is incomplete")
    evaluation_hash = sha256(bundle / "evaluation/summary.json")
    require(status.get("input_bundle_sha256") == expected_bundle_hash,
            "Input completion belongs to another frozen bundle")
    require(status.get("input_evaluation_summary_sha256") == evaluation_hash,
            "Input coordinator/evaluation binding differs")
    return {"input_coordinator_status_sha256": sha256(status_path),
            "input_bundle_sha256": expected_bundle_hash,
            "input_evaluation_summary_sha256": evaluation_hash}


def gpu_idle() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return not any(line.strip().isdigit() for line in result.stdout.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-pid", type=int, default=0)
    parser.add_argument("--inputs-output", type=Path, required=True)
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--h-root", type=Path, required=True)
    parser.add_argument("--superseded-owner", type=Path, required=True)
    args = parser.parse_args()

    h_root, bundle = args.h_root.resolve(), args.h_root.resolve() / "bundle"
    queue, events = h_root / "queue", h_root / "queue/events.jsonl"
    inputs_output, input_bundle = args.inputs_output.resolve(), args.input_bundle.resolve()
    superseded_owner = args.superseded_owner.resolve()
    require(not (queue / "owner.json").exists(), "A current H queue owner record already exists")
    require(superseded_owner.is_file(), "Interrupted H queue owner record is missing")
    old_owner = read(superseded_owner)
    require(not same_process(old_owner["pid"], old_owner["process_create_time"]),
            "Superseded H queue owner is still live")
    for name in ("ready.json", "completed.json", "failed.json"):
        require(not (queue / name).exists(), f"H queue already contains {name}")
    require(not (h_root / "factorial_results.json").exists(), "H results already exist")

    _, bundle_hash = verify_bundle(bundle)
    approval = read(h_root / "qa/approval.json")
    require(approval.get("status") == "passed" and approval.get("bundle_manifest_sha256") == bundle_hash,
            "H QA approval is not bound to the frozen bundle")
    input_hash = sha256(input_bundle / "bundle_manifest.json")
    # Inputs are sealed and their owners have exited; no identity-holder process.
    require(args.inputs_pid == 0, "Use completed-input binding, not a live prerequisite")
    completed_inputs = verify_input_completion(inputs_output, input_bundle, input_hash)
    verified_path = Path(__file__).resolve().parents[1] / "docs/revision/evidence/multienv_inputs_completion_verified_20260917.json"
    verified = read(verified_path)
    require(verified["completed_source_runs"] == verified["completed_target_results"] == 63,
            "Input verification is incomplete")
    require(verified["coordinator_status_sha256"] == completed_inputs["input_coordinator_status_sha256"],
            "Completed input provenance differs")
    require(verified["evaluation_summary_sha256"] == completed_inputs["input_evaluation_summary_sha256"],
            "Completed input evaluation differs")
    birth, command = None, []

    dry = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", str(bundle / "code/run.py"),
         "--dry-run", "--bundle", str(bundle)],
        cwd=bundle / "code", capture_output=True, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    require(dry.returncode == 0, "Frozen H runner dry-run failed before recovery")
    plan = json.loads(dry.stdout)
    require(plan.get("status") == "dry_run_passed" and len(plan.get("plan", [])) == 18 and
            all(row.get("action") == "run" for row in plan["plan"]),
            "H recovery did not start from an untouched 18-job grid")

    identity = {"pid": os.getpid(), "process_create_time": psutil.Process().create_time(),
                "command": psutil.Process().cmdline(), "code_sha256": sha256(Path(__file__)),
                "bundle_manifest_sha256": bundle_hash,
                "qa_approval_sha256": sha256(h_root / "qa/approval.json"),
                "superseded_owner": str(superseded_owner),
                "superseded_owner_sha256": sha256(superseded_owner),
                "inputs_pid": args.inputs_pid, "inputs_process_create_time": birth,
                "inputs_command": command, "inputs_output": str(inputs_output),
                "input_bundle": str(input_bundle), "input_bundle_sha256": input_hash,
                "widar_dependency": False, "training_started_at_recovery": False}
    guard = Path(__file__).with_name("h_gpu_availability_v2.py")
    wrapper = Path(__file__).with_name("execute_h_grouped_l1248_recovery_v2.py")
    identity.update(completed_input_verification_sha256=sha256(verified_path),
                    gpu_guard_sha256=sha256(guard), execution_wrapper_sha256=sha256(wrapper),
                    recovery_reason="WDDM reports desktop clients as compute processes; known desktop identity and idle-resource gate")
    write_new(queue / "owner.json", identity)
    try:
        emit(events, "queue_recovered_after_wddm_gate_block", upstream_pid=None,
             superseded_owner_sha256=identity["superseded_owner_sha256"], jobs=18,
             widar_excluded=True, h_training_started=False)
        evidence = verify_input_completion(inputs_output, input_bundle, input_hash)
        verify_bundle(bundle)
        emit(events, "recovered_upstream_completed", **evidence)
        from h_gpu_availability_v2 import verify_available
        while True:
            gpu_report = verify_available()
            if gpu_report["available"]:
                write_new(queue / "gpu_readiness_v2.json", gpu_report)
                break
            emit(events, "waiting_for_verified_gpu_resources", blockers=[x["blockers"] for x in gpu_report["observations"]])
            time.sleep(30)
        write_new(queue / "ready.json", {"bundle_manifest_sha256": bundle_hash,
                  "recovery_owner_sha256": sha256(queue / "owner.json"), "prerequisite": evidence})
        execution_wrapper = Path(__file__).resolve().with_name("execute_h_grouped_l1248_recovery_v2.py")
        require(execution_wrapper.is_file(), "Recovered H execution wrapper is absent")
        runner = [sys.executable, "-B", "-X", "utf8", str(execution_wrapper),
                  "--h-root", str(h_root), "--ready", str(queue / "ready.json")]
        emit(events, "h_supplement_started", command=runner, recovery=True)
        with (queue / "runner.recovery20260917_v5.log").open("xb") as log:
            result = subprocess.run(runner, cwd=bundle / "code", stdout=log,
                                    stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        require(result.returncode == 0, f"Frozen H supplement runner failed: {result.returncode}")
        result_path = h_root / "factorial_results.json"
        final = read(result_path)
        require(final.get("status") == "complete" and final.get("job_count") == 18,
                "H supplement result is incomplete")
        write_new(queue / "completed.json", {"status": "completed", "jobs": 18,
                  "results_sha256": sha256(result_path), "recovery": True,
                  "owner_sha256": sha256(queue / "owner.json"),
                  "execution_wrapper_sha256": sha256(execution_wrapper)})
        emit(events, "completed", jobs=18, recovery=True)
    except BaseException:
        error = traceback.format_exc()
        write_new(queue / "failed.json", {"status": "failed", "error": error,
                  "owner_sha256": sha256(queue / "owner.json")})
        emit(events, "failed", error=error, recovery=True)
        raise


if __name__ == "__main__":
    main()
