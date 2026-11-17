"""Capture preparation progress without mutating training queues or protocols."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = Path("D:/pa-csi-dg-revision-runtime/data/wimans_20261014")
JST = timezone(timedelta(hours=9))


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    capture = datetime.now(timezone.utc)
    acquisition = read(RUNTIME / "acquisition/status.json")
    inputs = read(RUNTIME / "processed_storage_order_v2/input_audit.json")
    implementation = read(ROOT / "IMPLEMENTATION_QA.json")
    existing = read(ROOT / "existing_queue_snapshot.json")
    baseline = read(ROOT / "author_baseline_audit.json")
    candidates = read(ROOT / "candidate_configs_qa.json")
    artifact_names = ["PLAN.md", "protocol_plan.json", "protocol_plan_v1.json", "metadata_verification.json",
                      "subset_manifest.csv", "fold_manifest.json", "axis_storage_protocol.json", "input_axis_audit.json",
                      "preprocessing_qa.json", "IMPLEMENTATION_QA.json", "author_baseline_audit.json", "MANUSCRIPT_STRUCTURE.md",
                      "existing_queue_snapshot.json", "hs_four_arm_provenance.json", "peer_implementation_review.json",
                      "raw_acquisition_verification.json", "prepared_input_verification.json", "candidate_configs_qa.json",
                      "PREPARATION_EXECUTION.md"]
    snapshot = {"schema": "wimans-plan-progression/v1", "captured_at_utc": capture.isoformat(),
                "captured_at_jst": capture.astimezone(JST).isoformat(), "runtime": str(RUNTIME),
                "new_formal_training_started": False, "existing_training_queue_modified": False,
                "physical_MAT_export_status": "unresolved", "registered_input_protocol": "storage_order_v2",
                "raw_acquisition": {k: acquisition[k] for k in ("status", "expected_rows", "completed_rows", "failed_rows", "downloaded_bytes", "complete", "updated_at_utc")} if acquisition else None,
                "input_qa": {k: inputs.get(k) for k in ("status", "completed_rows", "candidate_rows", "full_input_qa_passed", "numerical_input_qa_passed", "updated_at_utc")} if inputs else None,
                "workflow_status": read(RUNTIME / "workflow_status.json"),
                "recovery_status": read(RUNTIME / "recovery_status.json"),
                "synthetic_implementation_qa": {k: implementation.get(k) for k in ("status", "tests_run", "failures", "errors", "real_source_qa_passed", "execution_frozen", "expected_nine_class_parameters")} if implementation else None,
                "existing_queue_last_independent_audit": {k: existing[k] for k in ("captured_at_jst", "completed_new_runs", "total_new_runs", "remaining_new_runs", "completed_counts")} if existing else None,
                "author_baseline_qualification": baseline["status"] if baseline else None,
                "candidate_configuration_qa": {k: candidates.get(k) for k in ("status", "config_count", "formal_four_arm_candidates",
                    "source_coral_search_candidates", "formal_executable_configs", "input_qa_passed", "real_source_training_qa_passed",
                    "source_shape_qa")} if candidates else None,
                "author_optimizer_prototype": {"path": str(ROOT.parents[2] / "experiments/arcfi_hs_20261014/PROTOTYPE_QA.json"),
                    "sha256": sha(ROOT.parents[2] / "experiments/arcfi_hs_20261014/PROTOTYPE_QA.json")},
                "remaining_acceptance": ["per-fold real source W11 full-batch short-training and strict reload QA",
                                         "exclusive GPU source timing after existing queue", "MultiEnv MC and ME adapter/source QA",
                                         "author baseline execution adaptation and source QA", "source CORAL gamma search before WC/MC formal jobs",
                                         "formal code/config/job bundle freeze", "72 formal training jobs and offline target evaluation",
                                         "system measurement, updated manuscripts/PDFs and rebuilt supplement"],
                "artifact_sha256": {name: sha(ROOT / name) for name in artifact_names if (ROOT / name).exists()}}
    (ROOT / "progress_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"raw": snapshot["raw_acquisition"], "input": snapshot["input_qa"],
                      "implementation": snapshot["synthetic_implementation_qa"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
