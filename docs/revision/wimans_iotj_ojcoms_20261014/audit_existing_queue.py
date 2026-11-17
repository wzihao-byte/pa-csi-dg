"""Read-only audit of the existing H-S queue; write snapshots only beside this script.

No model import, inference, training, queue update, or checkpoint deserialization.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = Path(r"D:/pa-csi-dg-revision-runtime/studies/h_main_controls_20261008_v2")
JST = timezone(timedelta(hours=9))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path, digest=True):
    path = Path(path)
    stat = path.stat()
    result = dict(path=str(path), size_bytes=stat.st_size,
                  last_write_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat())
    if digest:
        result["sha256"] = sha(path)
    return result


def prediction_metrics(path):
    cm = [[0] * 6 for _ in range(6)]
    indices = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            a, b = row["label"], row["prediction"]
            assert 0 <= a < 6 and 0 <= b < 6
            cm[a][b] += 1
            indices.append(row["index"])
    assert indices == list(range(3000)), "Prediction row identity/order differs"
    support = [sum(row) for row in cm]
    assert support == [1200, 400, 400, 400, 400, 200]
    column = [sum(row[c] for row in cm) for c in range(6)]
    f1 = statistics.mean(200 * cm[c][c] / (support[c] + column[c]) for c in range(6))
    accuracy = 100 * sum(cm[c][c] for c in range(6)) / 3000
    return dict(prediction_count=len(indices), macro_f1_pct=f1, accuracy_pct=accuracy,
                support=support, confusion_matrix=cm)


def main():
    captured = datetime.now(timezone.utc)
    results_path = RUNTIME / "analysis/results.json"
    csv_path = RUNTIME / "analysis/validated_runs.csv"
    completed_path = RUNTIME / "queue/completed_jobs.json"
    status_path = RUNTIME / "queue/status.json"
    paths = [results_path, csv_path, completed_path, status_path]
    sources_before = [file_info(path) for path in paths]
    bundle_manifest_path = RUNTIME / "bundle/bundle_manifest.json"
    bundle_manifest = read(bundle_manifest_path)
    for rel, digest in bundle_manifest["files_sha256"].items():
        assert sha(RUNTIME / "bundle" / rel) == digest, "Frozen bundle member changed: " + rel
    results, completed, status = read(results_path), read(completed_path), read(status_path)
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    accepted = {entry["health"]["run_dir"]: entry for entry in completed["completed"]}
    assert len(rows) == len(accepted) == results["completed_new_runs"]
    assert len({(row["target"], row["seed"], row["arm"]) for row in rows}) == len(rows)
    verified = []
    max_error = 0.0
    for row in rows:
        run = Path(row["run"])
        entry = accepted[str(run)]
        attempt = Path(entry["attempt"])
        completion = read(attempt / "attempt_completed.json")
        seal = read(attempt / "attempt_seal.json")
        completion_sha = sha(attempt / "attempt_completed.json")
        assert completion["status"] == "success" and completion["returncode"] == 0
        assert completion_sha == entry["completion_sha256"] == seal["completion_manifest_sha256"]
        for rel, recorded in completion["artifacts"].items():
            path = attempt / rel
            assert path.stat().st_size == recorded["size_bytes"] and sha(path) == recorded["sha256"], rel
        metrics = read(run / "metrics.json")
        config = read(attempt / "job_config.json")
        alignment = read(run / "control_alignment_audit.json")
        initialization = read(run / "initialization_audit.json")
        split = read(run / "split_manifest.json")
        assert alignment["status"] == initialization["status"] == "passed"
        assert alignment["total_optimizer_steps"] == 30000
        assert initialization["parameter_count"] == entry["health"]["parameter_count"] == 4786294
        assert initialization["initial_state_sha256"] == config["h_controls"]["initial_state_sha256"]
        assert initialization["common_state_sha256"] == config["h_controls"]["common_state_sha256"]
        assert metrics["sampler_schedule_sha256"] == config["h_controls"]["expected_sampler_sha256"]
        assert split["grouped_source_manifest_sha256"] == config["h_controls"]["source_manifest_sha256"]
        for key in ("sample_id", "subject_id", "acquisition_id"):
            assert not set(split["group_identity"]["train"][key]) & set(split["group_identity"]["val"][key])
        assert sha(run / "metrics.json") == row["metrics_sha256"] == entry["health"]["metrics_sha256"]
        assert sha(run / "target_predictions.jsonl") == row["predictions_sha256"] == entry["health"]["target_predictions_sha256"]
        recomputed = prediction_metrics(run / "target_predictions.jsonl")
        for key, recorded in (("macro_f1_pct", float(row["macro_f1_pct"])),
                              ("accuracy_pct", float(row["accuracy_pct"]))):
            error = abs(recomputed[key] - recorded)
            assert error < 1e-9
            max_error = max(max_error, error)
        assert abs(recomputed["macro_f1_pct"] - 100 * metrics["test_metrics"]["f1_macro"]) < 1e-9
        verified.append(dict(job_id=entry["job_id"], attempt=str(attempt), arm=row["arm"],
                             completion_sha256=completion_sha, seal_sha256=sha(attempt / "attempt_seal.json"),
                             config_sha256=sha(attempt / "job_config.json"),
                             metrics_sha256=row["metrics_sha256"], predictions_sha256=row["predictions_sha256"],
                             recomputed=recomputed))
    sources_after = [file_info(path) for path in paths]
    assert sources_before == sources_after, "Queue snapshot changed during audit; rerun"
    active = next(job for job in read(RUNTIME / "bundle/registry.json")["jobs"]
                  if job["job_id"] == status["current_job"])
    attempt = RUNTIME / "runs" / active["target"] / f"seed_{active['seed']}" / f"HS_{active['control_arm']}" / f"attempt_{status['current_attempt']:03d}"
    current = dict(attempt=str(attempt), latest_files=sorted(
        [file_info(path, digest=False) for path in attempt.rglob("*") if path.is_file()],
        key=lambda info: info["last_write_utc"], reverse=True)[:8])
    for name in ("training_status.json", "source_progress.json"):
        path = attempt / "payload/run" / name
        if path.exists():
            current[name] = read(path)
    processes, process_error = [], None
    try:
        import psutil
        for process in psutil.process_iter(["pid", "ppid", "name", "cmdline", "create_time"]):
            info = process.info
            command = " ".join(info["cmdline"] or [])
            if "python" in (info["name"] or "").lower() and str(RUNTIME).lower() in command.lower():
                processes.append(dict(pid=info["pid"], parent_pid=info["ppid"], command=command,
                    created_utc=datetime.fromtimestamp(info["create_time"], timezone.utc).isoformat()))
    except Exception as error:
        process_error = repr(error)
    reference_path = RUNTIME / "bundle/historical_references.json"
    references = read(reference_path)["references"]
    for reference in references:
        run = Path(reference["attempt"]) / "payload/run"
        assert sha(reference["config"]) == reference["config_sha256"]
        assert sha(run / "metrics.json") == reference["metrics_sha256"]
        assert sha(run / "target_predictions.jsonl") == reference["predictions_sha256"]
    snapshot = dict(schema="existing-hs-queue-readonly-audit/v1", captured_at_utc=captured.isoformat(),
        captured_at_jst=captured.astimezone(JST).isoformat(), runtime=str(RUNTIME),
        audit_finished_at_utc=datetime.now(timezone.utc).isoformat(),
        status=results["status"], completed_new_runs=len(rows), total_new_runs=54, remaining_new_runs=54-len(rows),
        completed_counts=dict(Counter(row["arm"] for row in rows)), summaries=results["summaries"],
        complete_contrasts={key:value for key,value in results["contrasts"].items() if value["status"] == "complete"},
        source_files=sources_after, queue_status=status, current_job=current, processes=processes,
        frozen_bundle_manifest=file_info(bundle_manifest_path),
        frozen_bundle_members_verified=len(bundle_manifest["files_sha256"]),
        process_query_error=process_error, historical_reference_source=file_info(reference_path),
        historical_reference_count=len(references), verified_runs=verified,
        audit=dict(status="passed", artifact_manifest_and_seal_hashes_checked=True,
                   frozen_code_and_config_bundle_hashes_checked=True,
                   checkpoint_bytes_hashed_but_not_loaded=True, source_identity_disjointness_checked=True,
                   prediction_rows_recomputed=sum(run["recomputed"]["prediction_count"] for run in verified),
                   max_abs_metric_error_pp=max_error, checkpoint_selection_independently_replayed=False,
                   raw_arrays_read=False, new_training=False, new_inference=False, queue_mutated=False,
                   old_s1_run_count_not_expanded=True),
        interpretation_limits=["queue/status.at is a job-transition timestamp, not a training heartbeat",
            "partial arms have no equal-environment aggregate or final conclusion",
            "previously inspected MultiEnv targets remain development environments",
            "new accepted queue runs remain separate from the October 8 S1 package's 315 runs / 945000 rows"])
    (HERE / "existing_queue_snapshot.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(captured_at_jst=snapshot["captured_at_jst"], completed=len(rows), remaining=54-len(rows),
                         counts=snapshot["completed_counts"], current_job=status["current_job"],
                         audit=snapshot["audit"]), indent=2))


if __name__ == "__main__":
    main()
