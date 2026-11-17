"""Stage current-scope MultiEnv/H paper assets after its evidence gate."""
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
from experiments.revision_validation.study_io import now, read, sha256, write


FORMAL_TABLES = ("core_table.tex", "controls_table.tex", "inputs_table.tex", "h_grouped_table.tex")
FORMAL_FIGURES = (
    "formal_factorials.pdf",
    "formal_factorials.png",
    "formal_control_effects.pdf",
    "formal_control_effects.png",
)


def normalize(value: object) -> str:
    return str(value).replace("\\", "/").lower()


def process_identity(pid: int, required: tuple[str, ...]):
    process = psutil.Process(pid)
    command = process.cmdline()
    joined = normalize(" ".join(command))
    if not all(normalize(item) in joined for item in required):
        raise RuntimeError(f"PID {pid} is not the declared final-evidence coordinator")
    return process.create_time(), command


def live(pid: int, birth: float) -> bool:
    try:
        return psutil.Process(pid).create_time() == birth
    except psutil.NoSuchProcess:
        return False


def append_event(path: Path, event: dict):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"at": now(), **event}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_formal_evidence(evidence: Path):
    report = read(evidence / "verified_multienv_results.json")
    if (
        not report.get("all_current_scope_studies_completed")
        or report.get("multienv_formal_run_count") != 180
        or report.get("formal_source_analysis_count") != 117
        or report.get("h_supplement_run_count") != 18
        or not report.get("multienv_target_gates_verified")
        or report.get("population_inference_performed")
        or not report.get("widar_training_cancelled_by_user")
    ):
        raise RuntimeError("Formal evidence does not satisfy the complete revision gate")
    manifest = read(evidence / "manifest.json")
    for name, expected in manifest["files_sha256"].items():
        if sha256(evidence / name) != expected:
            raise RuntimeError(f"Formal evidence artifact differs: {name}")
    for name in FORMAL_TABLES:
        if name not in manifest["files_sha256"]:
            raise RuntimeError(f"Formal evidence omits paper table: {name}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-evidence-pid", type=int, required=True)
    parser.add_argument("--final-evidence-output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    final_output = args.final_evidence_output.resolve()
    evidence = args.evidence.resolve()
    staging = args.staging.resolve()
    assets = args.assets.resolve()
    output = args.output.resolve()
    for path in (output, staging, assets):
        if path.exists():
            raise FileExistsError(f"Preserve existing paper-assets attempt/output: {path}")
    birth, command = process_identity(
        args.final_evidence_pid,
        ("analysis/continue_multienv_final_evidence.py", str(final_output)),
    )

    output.mkdir(parents=True)
    frozen_script = output / "code/paper/sensors_manuscript/build_multienv_revision_figures.py"
    frozen_script.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "paper/sensors_manuscript/build_multienv_revision_figures.py", frozen_script)
    shutil.copy2(ROOT / "paper/sensors_manuscript/build_formal_revision_figures.py",
                 frozen_script.parent / "build_formal_revision_figures.py")
    identity = {
        "pid": os.getpid(),
        "started_at": now(),
        "final_evidence_pid": args.final_evidence_pid,
        "final_evidence_process_create_time": birth,
        "final_evidence_command": command,
        "final_evidence_output": str(final_output),
        "evidence": str(evidence),
        "staging": str(staging),
        "assets": str(assets),
        "frozen_figure_builder_sha256": sha256(frozen_script),
        "frozen_shared_figure_helpers_sha256": sha256(frozen_script.parent / "build_formal_revision_figures.py"),
        "mutable_formal_evidence_observed_while_writer_live": False,
    }
    events = output / "events.jsonl"
    write(output / "config.json", identity)
    write(output / "status.json", {"status": "waiting_for_formal_evidence", **identity})
    append_event(events, {"event": "waiting_for_formal_evidence"})

    try:
        while live(args.final_evidence_pid, birth):
            time.sleep(30)
        final_status = read(final_output / "status.json")
        if final_status.get("status") != "completed":
            raise RuntimeError("Final-evidence coordinator did not complete")
        report = verify_formal_evidence(evidence)
        if final_status.get("verified_multienv_results_sha256") != sha256(evidence / "verified_multienv_results.json"):
            raise RuntimeError("Final coordinator and formal evidence have different hashes")
        append_event(events, {
            "event": "formal_evidence_verified",
            "verified_multienv_results_sha256": sha256(evidence / "verified_multienv_results.json"),
            "multienv_runs": report["multienv_formal_run_count"],
            "formal_source_analyses": report["formal_source_analysis_count"],
            "h_supplement_runs": report["h_supplement_run_count"],
        })

        figure_command = [
            sys.executable,
            "-X",
            "utf8",
            str(frozen_script),
            "--evidence",
            str(evidence),
            "--output",
            str(staging),
        ]
        write(output / "status.json", {"status": "building_formal_figures", "command": figure_command, **identity})
        with (output / "figures.stdout.log").open("x", encoding="utf-8") as stdout, \
             (output / "figures.stderr.log").open("x", encoding="utf-8") as stderr:
            result = subprocess.run(
                figure_command,
                cwd=frozen_script.parents[2],
                stdout=stdout,
                stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        if result.returncode:
            raise RuntimeError(f"Frozen formal figure builder exited {result.returncode}")
        figure_manifest = read(staging / "manifest.json")
        for name in FORMAL_FIGURES:
            if sha256(staging / name) != figure_manifest["files_sha256"].get(name):
                raise RuntimeError(f"Generated figure differs from its manifest: {name}")

        shutil.copytree(staging, assets)
        (assets / "manifest.json").replace(assets / "figure_manifest.json")
        for name in FORMAL_TABLES:
            shutil.copy2(evidence / name, assets / name)
        files = [path for path in assets.iterdir() if path.is_file() and path.name != "manifest.json"]
        write(assets / "manifest.json", {
            "generated_at": now(),
            "formal_evidence": str(evidence),
            "verified_multienv_results_sha256": sha256(evidence / "verified_multienv_results.json"),
            "multienv_formal_runs": 180,
            "h_supplement_runs": 18,
            "widar_performance_included": False,
            "population_inference_performed": False,
            "files_sha256": {path.name: sha256(path) for path in sorted(files)},
        })
        completion = {
            "status": "completed",
            "completed_at": now(),
            "assets": str(assets),
            "asset_manifest_sha256": sha256(assets / "manifest.json"),
            "verified_multienv_results_sha256": sha256(evidence / "verified_multienv_results.json"),
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
