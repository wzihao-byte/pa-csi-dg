"""Print a compact, read-only status for the active revision pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path("D:/pa-csi-dg-revision-runtime")
OWNERS = (
    "experiments.revision_validation.extended_engine --bundle D:\\pa-csi-dg-revision-runtime\\studies",
    "experiments.revision_validation.input_study_engine --bundle D:\\pa-csi-dg-revision-runtime\\studies",
    "analysis/continue_inputs_after_controls.py",
    "analysis/continue_source_analyses.py",
    "analysis/recover_h_grouped_l1248_queue",
    "analysis/execute_h_grouped_l1248_recovery",
    "outputs_h_grouped_l1248/v2/bundle/code/train_factorial.py",
    "analysis/continue_multienv_final_evidence.py",
    "analysis/continue_multienv_paper_assets_after_evidence.py",
    "analysis/continue_multienv_fragment_after_assets.py",
    "analysis/continue_multienv_manuscript_after_fragment.py",
    "analysis/continue_multienv_release_after_manuscript.py",
)


def last_json(path: Path):
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        stream.seek(0, 2)
        end = stream.tell()
        if not end:
            return None
        position = end - 1
        while position > 0:
            stream.seek(position)
            if stream.read(1) == b"\n" and position < end - 1:
                break
            position -= 1
        stream.seek(position + 1 if position else 0)
        value = stream.readline().decode("utf-8").strip()
    return json.loads(value) if value else None


def main() -> None:
    processes = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        command = " ".join(process.info["cmdline"] or [])
        processes.append((process.info, command.replace("/", "\\").lower()))
    owners = []
    for marker in OWNERS:
        normalized = marker.replace("/", "\\").lower()
        matches = [info for info, command in processes if normalized in command]
        owners.append({"marker": marker, "count": len(matches),
                       "pids": [item["pid"] for item in matches],
                       "creation_times": [item["create_time"] for item in matches]})
    widar = [info["pid"] for info, command in processes
             if (info["name"] or "").lower().startswith("python")
             and "widar_study_engine" in command]

    controls_root = RUNTIME / "studies/r_controls_v1/runs"
    completed, active = 0, []
    for run in sorted(controls_root.iterdir()):
        if not run.is_dir():
            continue
        event = last_json(run / "events.jsonl")
        if event and event.get("event") == "completed":
            completed += 1
        elif event:
            active.append({"job_id": run.name, "last_event": event})
    source_count = len(list((RUNTIME / "analyses/formal_source_v1").glob("*/*/summary.json")))
    inputs_bundle = RUNTIME / "studies/r_inputs_v1"
    input_completed, input_active = 0, []
    for run in sorted((inputs_bundle / "runs").glob("*")):
        if not run.is_dir():
            continue
        event = last_json(run / "events.jsonl")
        if event and event.get("event") == "completed":
            input_completed += 1
        elif event:
            input_active.append({"job_id": run.name, "last_event": event})
    h_root = ROOT / "outputs_h_grouped_l1248/v2"
    h_completed, h_failed, h_active = [], [], []
    for started in sorted((h_root / "runs").glob("*/*/*/attempt_*/attempt_started.json")):
        attempt = started.parent
        record = json.loads(started.read_text(encoding="utf-8-sig"))
        item = {"job_id": record["job_id"], "attempt": record["attempt"],
                "started_at": record["created_at_utc"], "directory": str(attempt)}
        if (attempt / "attempt_seal.json").is_file() and (attempt / "attempt_completed.json").is_file():
            h_completed.append(item)
        elif (attempt / "attempt_failed.json").is_file():
            h_failed.append(item)
        else:
            item["last_append_event"] = last_json(attempt / "payload/run/progress.jsonl")
            item["stderr_bytes"] = (attempt / "runner.stderr.log").stat().st_size
            item["stdout_bytes"] = (attempt / "runner.stdout.log").stat().st_size
            h_active.append(item)
    error_logs = [
        RUNTIME / "studies/r_controls_v1/runner.recovery20260911.stderr.log",
        inputs_bundle / "runner.attempt01.stderr.log",
        inputs_bundle / "runner.attempt02.stderr.log",
        RUNTIME / "logs/inputs_recovery_v4b.stderr.log",
        RUNTIME / "logs/h_recovery_v4.stderr.log",
        RUNTIME / "logs/final_evidence_v4b.stderr.log",
        RUNTIME / "logs/paper_assets_v4.stderr.log",
        RUNTIME / "logs/fragment_v4.stderr.log",
        RUNTIME / "logs/manuscript_v4.stderr.log",
        RUNTIME / "logs/release_v4.stderr.log",
        *[RUNTIME / f"logs/{name}_v5.stderr.log" for name in (
            "h_recovery", "final_evidence", "paper_assets", "fragment", "manuscript", "release")],
        RUNTIME / "launcher_logs/controls_to_inputs_v3.stderr.log",
        RUNTIME / "launcher_logs/formal_source_analysis_v3.stderr.log",
        h_root / "coordinator.recovery20260911.attempt02.stderr.log",
        *[RUNTIME / f"launcher_logs/{name}.stderr.log" for name in (
            "multienv_final_evidence_v3", "multienv_paper_assets_v3",
            "multienv_fragment_v3", "multienv_manuscript_v3", "multienv_release_v3")],
    ]
    nonempty = {str(path): path.stat().st_size for path in error_logs
                if path.is_file() and path.stat().st_size}
    print(json.dumps({
        "owners": owners,
        "all_nine_owners_singleton": all(item["count"] == 1 for item in owners
            if "input_study_engine" not in item["marker"]),
        "duplicate_owner_markers": [item["marker"] for item in owners if item["count"] > 1],
        "widar_training_pids": widar,
        "controls_completed": completed,
        "controls_current": active,
        "controls_target_results_committed": len(list((controls_root.parent / "evaluation").glob("*/result.json"))),
        "controls_evaluation_summary_exists": (controls_root.parent / "evaluation/summary.json").is_file(),
        "inputs_completed": input_completed,
        "inputs_current": input_active,
        "inputs_target_results_committed": len(list((inputs_bundle / "evaluation").glob("*/result.json"))),
        "inputs_evaluation_summary_exists": (inputs_bundle / "evaluation/summary.json").is_file(),
        "source_analyses_completed": source_count,
        "source_analysis_completion_exists": (RUNTIME / "coordinators/formal_source_analysis_v3/completion.json").is_file(),
        "h_last_queue_event": last_json(h_root / "queue/events.jsonl"),
        "h_results_exist": (h_root / "factorial_results.json").is_file(),
        "h_sealed_attempts": h_completed,
        "h_failed_attempts": h_failed,
        "h_unsealed_attempts": h_active,
        "nonempty_error_logs": nonempty,
    }, indent=2))


if __name__ == "__main__":
    main()
