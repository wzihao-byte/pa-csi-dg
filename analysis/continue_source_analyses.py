"""Analyze each formal source run only after its append-only completion event."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.revision_validation.extended_engine import verify_extension
from experiments.revision_validation.freeze_inputs import INPUT_CODE
from experiments.revision_validation.study_io import now, read, sha256, write
from experiments.revision_validation.study_io import verify_bundle


ANALYSIS_CODE = INPUT_CODE + [
    "experiments/revision_validation/mechanism_metrics.py",
    "analysis/analyze_source_representation.py",
]


def has_completion_event(run):
    path = run / "events.jsonl"
    if not path.exists():
        return False
    # events.jsonl is append-only; status/progress/history are intentionally not
    # opened until the writer has emitted its terminal completion event.
    return any(json.loads(line).get("event") == "completed" for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def verify_analysis(output, job_id):
    summary = read(output / "summary.json")
    if summary["job_id"] != job_id or not summary["full_trained_model_analysis"] or summary["target_data_loaded"]:
        raise RuntimeError("Completed source analysis has the wrong identity or scope")
    for filename, expected in summary["artifacts_sha256"].items():
        if sha256(output / filename) != expected:
            raise RuntimeError("Completed source-analysis artifact hash mismatch")
    return sha256(output / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    core, controls = args.core.resolve(), args.controls.resolve()
    output, analysis_root = args.output.resolve(), args.analysis_root.resolve()
    if output.exists() or (analysis_root.exists() and not args.resume_existing):
        raise FileExistsError("Preserve the existing source-analysis coordinator/output version")
    output.mkdir(parents=True)
    analysis_root.mkdir(parents=True, exist_ok=args.resume_existing)
    core_protocol, core_hash = verify_bundle(core)
    control_protocol, control_hash = verify_extension(controls)
    code_root = output / "code"
    for relative in ANALYSIS_CODE:
        destination = code_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    write(output / "config.json", {"started_at": now(), "core": str(core), "core_bundle_sha256": core_hash,
          "controls": str(controls), "controls_bundle_sha256": control_hash,
          "analysis_root": str(analysis_root), "poll_seconds": 30,
          "resumed_existing_analyses": args.resume_existing,
          "target_data_loaded": False,
          "code_sha256": {relative: sha256(code_root / relative) for relative in ANALYSIS_CODE}})
    studies = (("r_core_v1", core, core_protocol), ("r_controls_v1", controls, control_protocol))
    completed = {}
    try:
        while True:
            progress = False
            for study_name, bundle, protocol in studies:
                for config in protocol["jobs"]:
                    job_id = config["job_id"]
                    run = bundle / "runs" / job_id
                    destination = analysis_root / study_name / job_id
                    key = study_name + "/" + job_id
                    if destination.exists():
                        completed[key] = verify_analysis(destination, job_id)
                        continue
                    if not has_completion_event(run):
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    log = output / "logs" / (key.replace("/", "__") + ".log")
                    log.parent.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, str(code_root / "analysis/analyze_source_representation.py"),
                               "--run", str(run), "--output", str(destination)]
                    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                    result = subprocess.run(command, cwd=code_root, text=True, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, creationflags=creationflags)
                    log.write_text(result.stdout, encoding="utf-8")
                    if result.returncode:
                        raise RuntimeError(f"Source analysis failed for {key}; see {log}")
                    completed[key] = verify_analysis(destination, job_id)
                    write(output / "progress.json", {"updated_at": now(), "completed": completed,
                          "completed_count": len(completed), "target_data_loaded": False})
                    progress = True
            expected = sum(len(protocol["jobs"]) for _, _, protocol in studies)
            if len(completed) == expected:
                write(output / "completion.json", {"completed_at": now(), "completed": completed,
                      "completed_count": len(completed), "target_data_loaded": False})
                return
            if not progress:
                time.sleep(30)
    except BaseException:
        write(output / "failure.json", {"failed_at": now(), "completed": completed,
              "target_data_loaded": False, "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
