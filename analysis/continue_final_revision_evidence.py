"""Wait for both terminal formal chains and run a frozen evidence compiler."""
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
from experiments.revision_validation.freeze_widar import WIDAR_CODE
from experiments.revision_validation.study_io import now, read, sha256, write


FINAL_CODE = list(dict.fromkeys(WIDAR_CODE + ["analysis/compile_formal_revision_evidence.py"]))


def process_identity(pid: int, required_text: tuple[str, ...]):
    process = psutil.Process(pid)
    command = process.cmdline()
    joined = " ".join(command).replace("\\", "/").lower()
    if not all(text.replace("\\", "/").lower() in joined for text in required_text):
        raise RuntimeError(f"PID {pid} is not the expected formal coordinator")
    return process.create_time(), command


def live(pid: int, birth: float):
    try:
        return psutil.Process(pid).create_time() == birth
    except psutil.NoSuchProcess:
        return False


def append_event(path: Path, event: dict):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"at": now(), **event}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widar-coordinator-pid", type=int, required=True)
    parser.add_argument("--widar-coordinator-output", type=Path, required=True)
    parser.add_argument("--source-analysis-pid", type=int, required=True)
    parser.add_argument("--source-analysis-output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    widar_output = args.widar_coordinator_output.resolve()
    source_output = args.source_analysis_output.resolve()
    runtime = args.runtime.resolve()
    evidence_output = args.evidence_output.resolve()
    output = args.output.resolve()
    if output.exists() or evidence_output.exists():
        raise FileExistsError("Preserve the existing final-evidence attempt/version")
    widar_birth, widar_command = process_identity(
        args.widar_coordinator_pid,
        ("analysis/continue_widar_after_preparation.py", str(widar_output)),
    )
    source_birth, source_command = process_identity(
        args.source_analysis_pid,
        ("analysis/continue_source_analyses.py", str(source_output)),
    )
    output.mkdir(parents=True)
    code_root = output / "code"
    code_hashes = {}
    for relative in FINAL_CODE:
        source, destination = ROOT / relative, code_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        code_hashes[relative] = sha256(destination)
    identity = {
        "pid": os.getpid(),
        "started_at": now(),
        "widar_coordinator_pid": args.widar_coordinator_pid,
        "widar_process_create_time": widar_birth,
        "widar_command": widar_command,
        "widar_coordinator_output": str(widar_output),
        "source_analysis_pid": args.source_analysis_pid,
        "source_analysis_process_create_time": source_birth,
        "source_analysis_command": source_command,
        "source_analysis_output": str(source_output),
        "runtime": str(runtime),
        "evidence_output": str(evidence_output),
        "mutable_upstream_json_observed_while_live": False,
        "code_sha256": code_hashes,
    }
    events = output / "events.jsonl"
    write(output / "config.json", identity)
    write(output / "status.json", {"status": "waiting_for_formal_chains", **identity})
    append_event(events, {"event": "waiting_for_formal_chains"})
    try:
        while live(args.widar_coordinator_pid, widar_birth) or live(args.source_analysis_pid, source_birth):
            time.sleep(30)
        if read(widar_output / "status.json").get("status") != "completed":
            raise RuntimeError("Widar formal chain did not complete")
        analysis_completion = read(source_output / "completion.json")
        if analysis_completion.get("completed_count") != 117 or analysis_completion.get("target_data_loaded"):
            raise RuntimeError("Formal source-analysis chain did not complete with source-only scope")
        append_event(events, {"event": "formal_chains_verified",
                              "widar_status_sha256": sha256(widar_output / "status.json"),
                              "source_completion_sha256": sha256(source_output / "completion.json")})
        command = [sys.executable, "-X", "utf8", str(code_root / "analysis/compile_formal_revision_evidence.py"),
                   "--runtime", str(runtime), "--output", str(evidence_output)]
        write(output / "status.json", {"status": "compiling_formal_evidence", "command": command, **identity})
        with (output / "compile.stdout.log").open("x", encoding="utf-8") as stdout, \
             (output / "compile.stderr.log").open("x", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=code_root, stdout=stdout, stderr=stderr,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode:
            raise RuntimeError(f"Frozen formal evidence compiler exited {result.returncode}")
        verified = read(evidence_output / "verified_formal_results.json")
        if not verified.get("all_formal_studies_completed") or verified.get("formal_run_count") != 216 \
                or verified.get("formal_source_analysis_count") != 117:
            raise RuntimeError("Compiled formal evidence has the wrong completion scope")
        completion = {"status": "completed", "completed_at": now(),
                      "evidence_output": str(evidence_output),
                      "verified_formal_results_sha256": sha256(evidence_output / "verified_formal_results.json"),
                      "manifest_sha256": sha256(evidence_output / "manifest.json")}
        write(output / "status.json", completion)
        append_event(events, {"event": "completed", **completion})
    except BaseException:
        failure = {"status": "failed", "failed_at": now(), "error": traceback.format_exc(), **identity}
        write(output / "status.json", failure)
        append_event(events, {"event": "failed", "error": failure["error"]})
        raise


if __name__ == "__main__":
    main()
