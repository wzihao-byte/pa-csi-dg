"""Compile, QA, freeze, and run Widar after its declared prerequisites complete.

The coordinator observes only OS process identity while either upstream writer is
live.  Completed JSON is opened only after the corresponding writer exits.
Every failed attempt and negative gate log is retained in place.
"""
from __future__ import annotations

import argparse
import json
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


def process_identity(pid: int, required_text: tuple[str, ...]):
    process = psutil.Process(pid)
    command = process.cmdline()
    joined = " ".join(command).lower()
    if not all(text.lower() in joined for text in required_text):
        raise RuntimeError(f"PID {pid} is not the declared upstream process")
    return process.create_time(), command


def is_same_process(pid: int, birth: float) -> bool:
    try:
        return psutil.Process(pid).create_time() == birth
    except psutil.NoSuchProcess:
        return False


def append_event(path: Path, event: dict):
    event = {"at": now(), **event}
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_logged(command: list[str], cwd: Path, log: Path, *, require_success: bool = True):
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    if require_success and result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; inspect {log}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-pid", type=int)
    parser.add_argument("--completed-preparation", action="store_true")
    parser.add_argument("--resume-frozen-widar", action="store_true")
    parser.add_argument("--prior-coordinator-output", type=Path)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--qa-output", type=Path, required=True)
    parser.add_argument("--input-transition-pid", type=int, required=True)
    parser.add_argument("--input-transition-output", type=Path, required=True)
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--widar-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prepared = args.prepared.resolve()
    selection = args.selection.resolve()
    inventory = args.inventory.resolve()
    compiled = args.compiled.resolve()
    qa_output = args.qa_output.resolve()
    transition_output = args.input_transition_output.resolve()
    input_bundle = args.input_bundle.resolve()
    widar_bundle = args.widar_bundle.resolve()
    prior_output = args.prior_coordinator_output.resolve() if args.prior_coordinator_output else None
    output = args.output.resolve()
    guarded = (output,) if args.resume_frozen_widar else (output, compiled, qa_output, widar_bundle)
    for path in guarded:
        if path.exists():
            raise FileExistsError(f"Preserve existing attempt/output: {path}")
    if args.resume_frozen_widar:
        if prior_output is None:
            raise RuntimeError("--prior-coordinator-output is required when resuming a frozen Widar bundle")
        for path in (compiled, qa_output, widar_bundle, prior_output):
            if not path.exists():
                raise FileNotFoundError(path)
    for path in (selection, inventory):
        if not path.is_file():
            raise FileNotFoundError(path)
    _, input_bundle_hash = verify_extension(input_bundle)
    if args.completed_preparation:
        if args.preparation_pid is not None:
            raise RuntimeError("Use either --preparation-pid or --completed-preparation")
        preparation_birth, preparation_command = None, None
    else:
        if args.preparation_pid is None:
            raise RuntimeError("--preparation-pid is required unless preparation is already complete")
        preparation_birth, preparation_command = process_identity(
            args.preparation_pid,
            ("analysis/download_widar_cohort_v2.py", str(prepared)),
        )
    transition_birth, transition_command = process_identity(
        args.input_transition_pid,
        ("analysis/continue_inputs_after_controls.py", str(transition_output)),
    )
    identity = {
        "pid": os.getpid(),
        "started_at": now(),
        "preparation_pid": args.preparation_pid,
        "preparation_process_create_time": preparation_birth,
        "preparation_command": preparation_command,
        "preparation_already_completed": args.completed_preparation,
        "prepared": str(prepared),
        "selection": str(selection),
        "selection_sha256": sha256(selection),
        "inventory": str(inventory),
        "inventory_sha256": sha256(inventory),
        "input_transition_pid": args.input_transition_pid,
        "input_transition_process_create_time": transition_birth,
        "input_transition_command": transition_command,
        "input_transition_output": str(transition_output),
        "input_bundle": str(input_bundle),
        "input_bundle_sha256": input_bundle_hash,
        "compiled": str(compiled),
        "qa_output": str(qa_output),
        "widar_bundle": str(widar_bundle),
        "resumed_frozen_widar": args.resume_frozen_widar,
        "prior_coordinator_output": str(prior_output) if prior_output else None,
        "mutable_upstream_json_observed_while_live": False,
        "code_sha256": sha256(__file__),
    }
    output.mkdir(parents=True)
    events = output / "events.jsonl"
    write(output / "config.json", identity)
    write(output / "status.json", {"status": "waiting_for_widar_preparation", **identity})
    append_event(events, {"event": "waiting_for_widar_preparation", "pid": args.preparation_pid,
                          "already_completed": args.completed_preparation})

    try:
        if not args.completed_preparation:
            while is_same_process(args.preparation_pid, preparation_birth):
                time.sleep(30)
        preparation = read(prepared / "summary.json")
        if (
            not preparation.get("complete_inventory_accounting")
            or preparation.get("failures")
            or preparation.get("replacement_samples_added") != 0
            or preparation.get("inventory_sha256") != sha256(inventory)
        ):
            raise RuntimeError("Widar v2 preparation did not finish with complete frozen-inventory accounting")
        append_event(events, {
            "event": "widar_preparation_verified",
            "summary_sha256": sha256(prepared / "summary.json"),
            "valid_rows": preparation["valid_rows"],
            "signal_exclusion_count": len(preparation["signal_exclusions"]),
        })

        commands = {
            "compile": [sys.executable, "-X", "utf8", str(ROOT / "analysis/compile_widar_inputs_v2.py"),
                        "--prepared", str(prepared), "--output", str(compiled),
                        "--selection", str(selection), "--inventory", str(inventory)],
            "qa": [sys.executable, "-X", "utf8", "-m", "experiments.revision_validation.widar_study_qa",
                   "--data", str(compiled), "--output", str(qa_output)],
            "freeze": [sys.executable, "-X", "utf8", "-m", "experiments.revision_validation.freeze_widar",
                       "--bundle", str(widar_bundle), "--input-bundle", str(input_bundle),
                       "--data", str(compiled), "--selection", str(selection), "--inventory", str(inventory),
                       "--preparation-summary", str(prepared / "summary.json"),
                       "--qa-report", str(qa_output / "widar_study_qa.json")],
        }
        if not args.resume_frozen_widar:
            for phase in ("compile", "qa", "freeze"):
                write(output / "status.json", {"status": phase + "_running", **identity})
                append_event(events, {"event": phase + "_started", "command": commands[phase]})
                run_logged(commands[phase], ROOT, output / (phase + ".log"))
                append_event(events, {"event": phase + "_completed"})

        protocol, widar_bundle_hash = verify_extension(widar_bundle)
        if protocol.get("study_kind") != "external_widar_room_factorial" or len(protocol["jobs"]) != 36:
            raise RuntimeError("Frozen Widar study identity/job count differs")
        if protocol["prerequisites"] != [{"bundle": str(input_bundle), "bundle_sha256": input_bundle_hash, "kind": "extension"}]:
            raise RuntimeError("Frozen Widar prerequisite differs from the declared input study")

        frozen_cwd = widar_bundle / "code"
        negative_commands = {
            "target_negative_gate": [sys.executable, "-X", "utf8", "-m",
                                     "experiments.revision_validation.extended_evaluate", "--bundle", str(widar_bundle)],
            "prerequisite_negative_gate": [sys.executable, "-X", "utf8", "-m",
                                           "experiments.revision_validation.widar_study_engine", "--bundle", str(widar_bundle)],
        }
        if not args.resume_frozen_widar:
            for phase, command in negative_commands.items():
                code = run_logged(command, frozen_cwd, output / (phase + ".log"), require_success=False)
                if code == 0:
                    raise RuntimeError(f"{phase} unexpectedly passed")
                append_event(events, {"event": phase + "_passed", "nonzero_return_code": code})
        else:
            prior_config = read(prior_output / "config.json")
            prior_negative = read(prior_output / "negative_gate_qa.json")
            if (
                prior_config.get("compiled") != str(compiled)
                or prior_config.get("qa_output") != str(qa_output)
                or prior_config.get("widar_bundle") != str(widar_bundle)
                or prior_config.get("input_bundle_sha256") != input_bundle_hash
                or not prior_negative.get("passed")
                or prior_negative.get("widar_bundle_sha256") != widar_bundle_hash
            ):
                raise RuntimeError("Prior frozen-Widar attempt does not bind to the requested artifacts")
            qa_report = read(qa_output / "widar_study_qa.json")
            if not qa_report.get("passed") or qa_report.get("target_data_loaded"):
                raise RuntimeError("Prior Widar QA report does not satisfy its frozen source-only gate")
            append_event(events, {
                "event": "frozen_widar_artifacts_reused_after_process_interruption",
                "prior_config_sha256": sha256(prior_output / "config.json"),
                "prior_negative_gate_qa_sha256": sha256(prior_output / "negative_gate_qa.json"),
                "widar_study_qa_sha256": sha256(qa_output / "widar_study_qa.json"),
                "widar_bundle_sha256": widar_bundle_hash,
            })
        forbidden = ("target_gate.json", "prerequisite_gate.json", "queue_failure.json", "runner.lock.json")
        existing = [name for name in forbidden if (widar_bundle / name).exists()]
        if existing:
            raise RuntimeError(f"Negative gates left forbidden artifacts: {existing}")
        write(output / "negative_gate_qa.json", {
            "checked_at": now(),
            "passed": True,
            "target_evaluation_rejected_before_source_completion": True,
            "source_training_rejected_before_prerequisite_completion": True,
            "forbidden_artifacts_absent": list(forbidden),
            "widar_bundle_sha256": widar_bundle_hash,
            "reused_from": str(prior_output) if args.resume_frozen_widar else None,
        })

        write(output / "status.json", {"status": "waiting_for_input_study", **identity,
              "widar_bundle_sha256": widar_bundle_hash})
        append_event(events, {"event": "waiting_for_input_study", "pid": args.input_transition_pid})
        while is_same_process(args.input_transition_pid, transition_birth):
            time.sleep(30)
        transition_status = read(transition_output / "status.json")
        if transition_status.get("status") != "completed":
            raise RuntimeError("The core/control/input prerequisite chain did not complete")
        if verify_extension(input_bundle)[1] != input_bundle_hash:
            raise RuntimeError("Frozen input prerequisite changed while waiting")
        if verify_extension(widar_bundle)[1] != widar_bundle_hash:
            raise RuntimeError("Frozen Widar bundle changed while waiting")
        append_event(events, {"event": "input_study_verified",
                              "evaluation_summary_sha256": sha256(input_bundle / "evaluation/summary.json")})

        command = [sys.executable, "-X", "utf8", "-m", "experiments.revision_validation.widar_study_engine",
                   "--bundle", str(widar_bundle)]
        with (widar_bundle / "runner.attempt01.stdout.log").open("x", encoding="utf-8") as stdout, \
             (widar_bundle / "runner.attempt01.stderr.log").open("x", encoding="utf-8") as stderr:
            child = subprocess.Popen(
                command,
                cwd=frozen_cwd,
                stdout=stdout,
                stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            write(output / "status.json", {"status": "widar_running", "pid": os.getpid(),
                  "child_pid": child.pid, "started_at": now(), "command": command,
                  "widar_bundle_sha256": widar_bundle_hash})
            append_event(events, {"event": "widar_started", "child_pid": child.pid, "command": command})
            result = child.wait()
        if result or read(widar_bundle / "queue_status.json").get("status") != "completed":
            raise RuntimeError(f"Frozen Widar study exited with return code {result}; retain and inspect evidence")
        completion = {
            "status": "completed",
            "completed_at": now(),
            "widar_bundle_sha256": widar_bundle_hash,
            "widar_evaluation_summary_sha256": sha256(widar_bundle / "evaluation/summary.json"),
            "valid_rows": preparation["valid_rows"],
            "signal_exclusion_count": len(preparation["signal_exclusions"]),
        }
        write(output / "status.json", completion)
        append_event(events, {"event": "completed", **completion})
    except BaseException:
        failure = {"status": "failed", "failed_at": now(), "error": traceback.format_exc(), **identity}
        write(output / "status.json", failure)
        append_event(events, {"event": "failed", "error": failure["error"]})
        raise


if __name__ == "__main__":
    main()
