"""Continue verified dataset preparation after the existing downloaders exit.

Watches process identities, never mutable JSON files while their writers run.
Every compiler/preparer enforces its own full-data gate. No model is trained here.
"""
from __future__ import annotations

import argparse
import concurrent.futures
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
from experiments.revision_validation.study_io import now, read, sha256, write

CODE_FILES = (
    "analysis/continue_input_preparation.py", "analysis/compile_revision_inputs.py", "analysis/compile_modality_inputs.py",
    "analysis/native_phase_processing.py", "analysis/verify_native_phase.py", "analysis/prepare_native_phase.py",
    "analysis/prepare_native_multienv.py", "analysis/audit_multienv_timestamps.py", "analysis/verify_multienv_raw_mapping.py",
    "experiments/revision_validation/modality_models.py", "experiments/revision_validation/models.py",
    "experiments/revision_validation/prepare_verified_protocol.py", "experiments/revision_validation/study_io.py",
)


def process_identity(pid, script, directory):
    process = psutil.Process(pid)
    command = " ".join(process.cmdline()).replace("\\", "/").lower()
    if script.lower() not in command or str(directory).replace("\\", "/").lower() not in command:
        raise RuntimeError("Preparation process does not match its declared script/output")
    return {"pid": pid, "created_at_unix": process.create_time(), "command": process.cmdline()}


def wait_for_exit(identity):
    while True:
        try:
            live = psutil.Process(identity["pid"]).create_time() == identity["created_at_unix"]
        except psutil.NoSuchProcess:
            live = False
        if not live:
            return
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--native-pid", type=int, required=True)
    parser.add_argument("--widar-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime, output = args.runtime.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Keep previous pipeline attempt; use a new output")
    native = process_identity(args.native_pid, "prepare_native_multienv.py", runtime / "data/native_amp_v1")
    widar = process_identity(args.widar_pid, "download_widar_cohort.py", runtime / "data/widar_native_amp_v1")
    code_hashes = {name: sha256(ROOT / name) for name in CODE_FILES}
    output.mkdir(parents=True)
    for name in CODE_FILES:
        destination = output / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        if sha256(destination) != code_hashes[name]:
            raise RuntimeError("Pipeline code snapshot differs")
    config = {"pid": os.getpid(), "started_at": now(), "native_process": native, "widar_process": widar,
              "code_sha256": code_hashes, "classifier_training_or_selection": False,
              "plan_completed_by_this_pipeline": False}
    write(output / "config.json", config)
    write(output / "status.json", {"status": "waiting_for_preparation_processes", **config})

    def stage(name, script, arguments, result_path):
        for relative, expected in code_hashes.items():
            if sha256(ROOT / relative) != expected:
                raise RuntimeError(f"Preparation pipeline dependency changed: {relative}")
        command = [sys.executable, "-X", "utf8", str(ROOT / script), *[str(x) for x in arguments]]
        with (output / (name + ".stdout.log")).open("x", encoding="utf-8") as stdout, (output / (name + ".stderr.log")).open("x", encoding="utf-8") as stderr:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            write(output / (name + ".launch.json"), {"started_at": now(), "pid": child.pid, "command": command})
            result = child.wait()
        if result != 0:
            raise RuntimeError(f"Stage {name} failed with exit {result}; see retained logs")
        for relative, expected in code_hashes.items():
            if sha256(ROOT / relative) != expected:
                raise RuntimeError("Preparation dependency changed during a stage")
        write(output / (name + ".completed.json"), {"completed_at": now(), "result_path": str(result_path), "result_sha256": sha256(result_path)})

    def native_branch():
        wait_for_exit(native)
        summary = read(runtime / "data/native_amp_v1/summary.json")
        if not summary.get("complete_9000") or summary.get("failures"):
            raise RuntimeError("Native downloader exited without complete verified data")
        native_arrays = runtime / "data/native_amp_arrays_v1"
        stage("native_compile", "analysis/compile_revision_inputs.py", ["--kind", "native", "--prepared", runtime / "data/native_amp_v1", "--output", native_arrays], native_arrays / "dataset_manifest.json")
        phase = runtime / "data/native_phase_v2"
        stage("native_phase", "analysis/prepare_native_phase.py", ["--native-prepared", runtime / "data/native_amp_v1", "--qa-report", runtime / "qa/phase_v2/phase_qa.json", "--output", phase, "--workers", "4"], phase / "summary.json")
        modalities = runtime / "data/modality_arrays_v1"
        stage("modality_compile", "analysis/compile_modality_inputs.py", ["--phase-prepared", phase, "--native-compiled", native_arrays, "--output", modalities], modalities / "dataset_manifest.json")
        return {"native_input_manifest": str(native_arrays / "dataset_manifest.json"), "modality_input_manifest": str(modalities / "dataset_manifest.json")}

    def widar_branch():
        wait_for_exit(widar)
        summary = read(runtime / "data/widar_native_amp_v1/summary.json")
        if not summary.get("complete_selected_cohort") or summary.get("failures"):
            raise RuntimeError("Widar downloader exited without the complete verified cohort")
        widar_arrays = runtime / "data/widar_arrays_v1"
        stage("widar_compile", "analysis/compile_revision_inputs.py", ["--kind", "widar", "--prepared", runtime / "data/widar_native_amp_v1", "--output", widar_arrays, "--widar-selection", runtime / "data/widar_inventory_v1/selection_before_inventory.json"], widar_arrays / "dataset_manifest.json")
        return {"widar_input_manifest": str(widar_arrays / "dataset_manifest.json")}

    results, failures = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        tasks = {executor.submit(native_branch): "native", executor.submit(widar_branch): "widar"}
        for task in concurrent.futures.as_completed(tasks):
            name = tasks[task]
            try:
                results[name] = task.result()
                write(output / (name + ".branch.json"), {"status": "completed", "at": now(), **results[name]})
            except BaseException:
                failures[name] = traceback.format_exc()
                write(output / (name + ".branch.json"), {"status": "failed", "at": now(), "error": failures[name]})
    write(output / "status.json", {"status": "failed" if failures else "completed", "at": now(), "results": results,
          "failures": failures, "classifier_training_or_selection": False, "revision_plan_completed": False})
    if failures:
        raise RuntimeError("A dataset preparation branch failed; other completed evidence is preserved")


if __name__ == "__main__":
    main()
