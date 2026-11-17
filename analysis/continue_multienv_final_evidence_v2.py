"""Wait for the revised MultiEnv and H chains, then compile current-scope evidence."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.revision_validation.freeze_inputs import INPUT_CODE
from experiments.revision_validation.study_io import now, read, sha256, write


FINAL_CODE = list(dict.fromkeys(INPUT_CODE + [
    "analysis/compile_formal_revision_evidence.py",
    "analysis/compile_multienv_revision_evidence.py",
]))


def normalized(value):
    return str(value).replace("\\", "/").lower()


def process_identity(pid, required):
    process = psutil.Process(pid)
    command = process.cmdline()
    joined = normalized(" ".join(command))
    if not all(normalized(item) in joined for item in required):
        raise RuntimeError(f"PID {pid} is not the declared current-scope upstream process")
    return process.create_time(), command


def live(pid, birth):
    try:
        return abs(psutil.Process(pid).create_time() - birth) < 0.01
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def append_event(path, event):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"at": now(), **event}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-pid", type=int, default=0)
    parser.add_argument("--inputs-output", type=Path, required=True)
    parser.add_argument("--source-pid", type=int, default=0)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--h-pid", type=int, required=True)
    parser.add_argument("--h-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compatibility-marker", required=True)
    args = parser.parse_args()
    if args.compatibility_marker != "analysis/continue_multienv_final_evidence.py":
        raise RuntimeError("Wrong downstream routing marker")

    inputs_output, source_output = args.inputs_output.resolve(), args.source_output.resolve()
    h_root, runtime = args.h_root.resolve(), args.runtime.resolve()
    evidence_output, output = args.evidence_output.resolve(), args.output.resolve()
    if output.exists() or evidence_output.exists():
        raise FileExistsError("Preserve the existing current-scope evidence attempt/version")
    if args.inputs_pid or args.source_pid:
        raise RuntimeError("Inputs/source must be bound through completed records")
    inputs_status = read(inputs_output / "status.json")
    source_completion_path = source_output / "completion.json"
    source_completion = read(source_completion_path)
    verified_input = read(ROOT / "docs/revision/evidence/multienv_inputs_completion_verified_20260917.json")
    if inputs_status.get("status") != "completed" or verified_input["coordinator_status_sha256"] != sha256(inputs_output / "status.json"):
        raise RuntimeError("Completed input binding differs")
    if sha256(source_completion_path) != "481767f5c2b05137c83e1ee61ec0528af2b6ef81bbb8bbaa5b13aa5fc54a383b":
        raise RuntimeError("Sealed source completion differs")
    if source_completion.get("completed_count") != 117 or source_completion.get("target_data_loaded"):
        raise RuntimeError("Completed source scope differs")
    inputs_birth = source_birth = None
    inputs_command = source_command = []
    h_birth, h_command = process_identity(args.h_pid, ("recover_h_grouped_l1248_queue_v2.py", str(h_root)))
    h_queue_recovery = True
    output.mkdir(parents=True)
    code_root = output / "code"
    code_hashes = {}
    for relative in FINAL_CODE:
        source, destination = ROOT / relative, code_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        code_hashes[relative] = sha256(destination)
    identity = {"pid": os.getpid(), "started_at": now(),
        "inputs_pid": args.inputs_pid, "inputs_process_create_time": inputs_birth,
        "inputs_command": inputs_command, "inputs_output": str(inputs_output),
        "source_pid": args.source_pid, "source_process_create_time": source_birth,
        "source_command": source_command, "source_output": str(source_output),
        "h_pid": args.h_pid, "h_process_create_time": h_birth, "h_command": h_command,
        "h_queue_recovery_after_host_interruption": h_queue_recovery,
        "h_root": str(h_root), "runtime": str(runtime), "evidence_output": str(evidence_output),
        "widar_training_in_scope": False, "mutable_upstream_json_observed_while_live": False,
        "code_sha256": code_hashes,
        "completed_upstreams_bound_without_identity_holders": True,
        "input_completion_sha256": sha256(inputs_output / "status.json"),
        "source_completion_sha256": sha256(source_completion_path)}
    events = output / "events.jsonl"
    write(output / "config.json", identity)
    write(output / "status.json", {"status": "waiting_for_multienv_source_and_h", **identity})
    append_event(events, {"event": "waiting_for_multienv_source_and_h"})
    try:
        while live(args.h_pid, h_birth):
            time.sleep(30)
        inputs_status = read(inputs_output / "status.json")
        if inputs_status.get("status") != "completed":
            raise RuntimeError("MultiEnv input chain did not complete")
        source_completion = read(source_output / "completion.json")
        if source_completion.get("completed_count") != 117 or source_completion.get("target_data_loaded"):
            raise RuntimeError("Source-only analysis chain did not complete with 117 jobs")
        h_completion = read(h_root / "queue/completed.json")
        if h_completion.get("status") != "completed" or h_completion.get("jobs") != 18:
            raise RuntimeError("Original-H grouped supplement did not complete with 18 jobs")
        append_event(events, {"event": "current_scope_upstreams_verified",
            "inputs_status_sha256": sha256(inputs_output / "status.json"),
            "source_completion_sha256": sha256(source_output / "completion.json"),
            "h_completion_sha256": sha256(h_root / "queue/completed.json")})

        command = [sys.executable, "-B", "-X", "utf8",
            str(code_root / "analysis/compile_multienv_revision_evidence.py"),
            "--runtime", str(runtime), "--h-root", str(h_root),
            "--source-completion", str(source_output / "completion.json"),
            "--output", str(evidence_output)]
        write(output / "status.json", {"status": "compiling_current_scope_evidence",
              "command": command, **identity})
        append_event(events, {"event": "current_scope_evidence_started", "command": command})
        with (output / "compile.stdout.log").open("x", encoding="utf-8") as stdout, \
             (output / "compile.stderr.log").open("x", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=code_root, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode:
            raise RuntimeError(f"Frozen current-scope evidence compiler exited {result.returncode}")
        verified = read(evidence_output / "verified_multienv_results.json")
        if (not verified.get("all_current_scope_studies_completed")
                or verified.get("multienv_formal_run_count") != 180
                or verified.get("formal_source_analysis_count") != 117
                or verified.get("h_supplement_run_count") != 18
                or not verified.get("widar_training_cancelled_by_user")):
            raise RuntimeError("Compiled evidence has the wrong revised scope")
        completion = {"status": "completed", "completed_at": now(),
            "evidence_output": str(evidence_output),
            "verified_multienv_results_sha256": sha256(evidence_output / "verified_multienv_results.json"),
            "manifest_sha256": sha256(evidence_output / "manifest.json"),
            "multienv_formal_runs": 180, "formal_source_analyses": 117,
            "h_supplement_runs": 18, "widar_training_cancelled_by_user": True}
        write(output / "status.json", completion)
        append_event(events, {"event": "completed", **completion})
    except BaseException:
        failure = {"status": "failed", "failed_at": now(), "error": traceback.format_exc(), **identity}
        write(output / "status.json", failure)
        append_event(events, {"event": "failed", "error": failure["error"]})
        raise


if __name__ == "__main__":
    main()
