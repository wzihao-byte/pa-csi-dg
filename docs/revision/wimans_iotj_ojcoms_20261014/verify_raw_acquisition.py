"""Independently verify raw acquisition bytes and annotation identity without decoding CSI.

Single-threaded read-only access to receipts/raw; output only in this document folder.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

DOCS = Path(__file__).resolve().parent
REPO = DOCS.parents[2]
ROOT = Path(r"D:/pa-csi-dg-revision-runtime/data/wimans_20261014")
CLASSES = ("nothing", "walk", "rotation", "jump", "wave", "lie_down", "pick_up", "sit_down", "stand_up")
EXPECTED_GROUPS = dict(classroom=list(range(19, 25)), meeting_room=list(range(55, 61)), empty_room=list(range(91, 97)))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def info(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    started = datetime.now(timezone.utc)
    manifest_path, annotation_path = DOCS / "subset_manifest.csv", DOCS / "annotation_source.csv"
    meta = read(DOCS / "metadata_verification.json")
    assert sha(manifest_path) == meta["artifacts_sha256"]["subset_manifest.csv"]
    assert sha(annotation_path) == meta["annotation_sha256"]
    rows, annotations = csv_rows(manifest_path), csv_rows(annotation_path)
    selected = [row for row in annotations if float(row["wifi_band"]) == 5 and int(row["number_of_users"]) == 1]
    selected_by_id = {row["label"]: row for row in selected}
    manifest_by_id = {row["sample_id"]: row for row in rows}
    assert len(rows) == len(manifest_by_id) == len(selected) == len(selected_by_id) == 1782
    assert set(manifest_by_id) == set(selected_by_id)
    for sid, row in manifest_by_id.items():
        original = selected_by_id[sid]
        users = [i for i in range(1, 7) if original[f"user_{i}_activity"].strip()]
        assert len(users) == 1 and int(row["subject_id"]) == users[0]
        activity = original[f"user_{users[0]}_activity"]
        assert row["activity"] == activity and int(row["class_id"]) == CLASSES.index(activity)
        assert row["environment"] == original["environment"] and float(row["wifi_band"]) == 5
        assert row["position"] == original[f"user_{users[0]}_location"]
        group, instance = re.fullmatch(r"act_(\d+)_(\d+)", sid).groups()
        assert int(row["acquisition_group"]) == int(group) and int(row["instance_id"]) == int(instance)
        assert row["raw_relative_path"] == "wifi_csi/mat/" + sid + ".mat"
    room_counts = Counter(row["environment"] for row in rows)
    room_class, room_subject, groups = defaultdict(Counter), defaultdict(Counter), defaultdict(set)
    group_counts = Counter()
    for row in rows:
        room_class[row["environment"]][row["activity"]] += 1
        room_subject[row["environment"]][row["subject_id"]] += 1
        groups[row["environment"]].add(int(row["acquisition_group"]))
        group_counts[row["acquisition_group"]] += 1
    assert set(room_counts) == set(EXPECTED_GROUPS) and all(value == 594 for value in room_counts.values())
    for room in EXPECTED_GROUPS:
        assert sorted(groups[room]) == EXPECTED_GROUPS[room]
        assert dict(room_class[room]) == {name: 66 for name in CLASSES}
        assert dict(room_subject[room]) == {str(i): 99 for i in range(1, 7)}
    assert len(group_counts) == 18 and all(value == 99 for value in group_counts.values())
    status_path = ROOT / "acquisition/status.json"
    status_hash_before = sha(status_path)
    status = read(status_path)
    assert status["dataset_ref"] == "shuokanghuang/wimans" and status["dataset_version"] == 1
    assert status["manifest_sha256"] == sha(manifest_path)
    assert status["status"] == "complete" and status["complete"] is True and status["failed_rows"] == 0
    assert status["completed_rows"] == status["requested_rows"] == status["expected_rows"] == 1782
    assert status["training_started"] is False and not status["failures"]
    status_receipts = {row["sample_id"]: row for row in status["records"]}
    receipt_paths = list((ROOT / "acquisition/records").glob("*.json"))
    assert len(status["records"]) == len(status_receipts) == len(receipt_paths) == 1782
    assert set(status_receipts) == set(manifest_by_id) == {path.stem for path in receipt_paths}
    expected_paths = {str(Path(row["raw_relative_path"])) for row in rows}
    actual_paths = {str(path.relative_to(ROOT / "raw")) for path in (ROOT / "raw").rglob("*.mat")}
    assert actual_paths == expected_paths
    byte_sum, byte_duplicates, origins, verified = 0, defaultdict(list), Counter(), []
    for index, row in enumerate(rows, 1):
        sid = row["sample_id"]
        receipt_path = ROOT / "acquisition/records" / (sid + ".json")
        receipt = read(receipt_path)
        assert receipt == status_receipts[sid]
        assert receipt["sample_id"] == sid and receipt["relative_path"] == row["raw_relative_path"]
        assert receipt["dataset_version"] == 1 and receipt["model_scores_read"] is False
        assert receipt["target_access_kind"] == "raw_identity_and_numerical_audit_only"
        path = ROOT / "raw" / row["raw_relative_path"]
        stat_before = path.stat()
        digest = sha(path)
        with path.open("rb") as handle:
            header = handle.read(128)
        stat_after = path.stat()
        assert (stat_before.st_size, stat_before.st_mtime_ns) == (stat_after.st_size, stat_after.st_mtime_ns)
        assert stat_after.st_size == receipt["bytes"] and digest == receipt["sha256"]
        assert header.startswith(b"MATLAB 5.0 MAT-file") and len(header) == 128 and header[126:128] in (b"IM", b"MI")
        byte_sum += stat_after.st_size
        byte_duplicates[digest].append(sid)
        origins[receipt["origin"]] += 1
        verified.append(dict(sample_id=sid, relative_path=row["raw_relative_path"], dataset_version=1,
            bytes=stat_after.st_size, sha256=digest, receipt_sha256=sha(receipt_path), origin=receipt["origin"]))
        if index % 600 == 0:
            print(json.dumps(dict(verified_files=index, verified_bytes=byte_sum)), flush=True)
    assert byte_sum == status["downloaded_bytes"] == 4317606012
    assert sha(status_path) == status_hash_before, "Acquisition status changed during audit"
    attempts = []
    for path in sorted((ROOT / "acquisition/attempts").glob("*.json")):
        old = read(path)
        assert old["dataset_ref"] == status["dataset_ref"] and old["dataset_version"] == 1
        assert old["manifest_sha256"] == status["manifest_sha256"]
        old_successes = {row["sample_id"]: row for row in old["records"]}
        failures = old["failures"]
        assert len(old_successes) == old["completed_rows"] and len(failures) == old["failed_rows"]
        assert set(old_successes).isdisjoint({row["sample_id"] for row in failures})
        assert set(old_successes) | {row["sample_id"] for row in failures} == set(manifest_by_id)
        assert all(value == status_receipts[sid] for sid, value in old_successes.items())
        attempts.append(dict(source=info(path), status=old["status"], completed_rows=old["completed_rows"],
            failed_rows=old["failed_rows"], same_version_and_manifest=True,
            original_success_receipts_unchanged=True,
            all_failures_recovered_in_final_receipts=all(row["sample_id"] in status_receipts for row in failures),
            all_failures_query_endpoint_404=all("404 Client Error" in row["error"] and "?fileName=" in row["error"] for row in failures),
            failed_sample_ids=sorted(row["sample_id"] for row in failures)))
    assert len(attempts) == 1 and attempts[0]["completed_rows"] == 1584 and attempts[0]["failed_rows"] == 198
    code_path = REPO / "experiments/wimans_20261014/acquisition.py"
    saved = [info(path) for path in (REPO / "work/wimans_20261014/preprocessing_source_versions").glob("*_acquisition.py")]
    for item in saved:
        assert Path(item["path"]).name.startswith(item["sha256"] + "_")
    report = dict(schema="independent-wimans-raw-acquisition-verification/v1", status="passed",
        audit_started_at_utc=started.isoformat(), audit_finished_at_utc=datetime.now(timezone.utc).isoformat(),
        raw_root=str(ROOT / "raw"), data_version=1, author_repository_commit=meta["author_repository_commit"],
        subset_manifest=info(manifest_path), author_annotation=info(annotation_path),
        acquisition_status=info(status_path), raw_files_verified=1782, receipt_files_verified=1782,
        raw_total_bytes=byte_sum, selected_author_annotation_rows=1782, raw_path_set_exactly_matches_manifest=True,
        original_identity_fields_verified=["sample_id", "environment", "wifi_band", "single_active_user", "subject_id", "activity", "class_id", "position", "acquisition_group", "instance_id", "raw_relative_path"],
        room_counts=dict(room_counts), room_class_counts={room:dict(counts) for room,counts in room_class.items()},
        room_subject_counts={room:dict(counts) for room,counts in room_subject.items()},
        room_acquisition_groups={room:sorted(value) for room,value in groups.items()},
        acquisition_group_counts=dict(group_counts), unique_raw_sha256=len(byte_duplicates),
        exact_byte_duplicate_groups={digest:ids for digest,ids in byte_duplicates.items() if len(ids)>1},
        receipt_origins=dict(origins), preserved_acquisition_attempts=attempts,
        files=verified, code_provenance=dict(current_workspace_source=info(code_path), saved_sources=saved,
            final_status_reported_software_sha256=status.get("software_sha256"),
            executed_inflight_code_hash_independently_confirmed=False,
            limitation="The saved recovery version computes software_sha256 by reading its source path at status-writing time. A later on-disk edit can change that field without changing code already loaded in the running process. Current source and final reported hash must not be asserted as the inflight executed source version. Receipt/raw-byte verification is independent of this uncertainty."),
        audit_scope=dict(full_raw_byte_sha256_recomputed=True, receipt_lengths_and_hashes_checked=True,
            matlab_v5_header_and_endianness_checked=True, original_annotation_distribution_rederived=True,
            no_exclusions_or_sample_substitutions=True, no_data_version_replacement=True,
            scipy_or_mat_csi_decode_performed=False, signal_array_numerical_QA_complete=False,
            numerical_signal_duplicate_check_performed=False, physical_axis_assignment_confirmed=False,
            target_model_scores_read=False, target_inference_performed=False, training_started=False,
            GPU_used=False, acquisition_or_workflow_state_mutated=False),
        limitations=["Byte uniqueness does not establish uniqueness of decoded numerical CSI arrays.",
            "Full MAT decoding, timestamp/CSI numerical QA, storage-order adapter and model/config freeze are separate gates."])
    (DOCS / "raw_acquisition_verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], files=1782, bytes=byte_sum, unique_raw_sha256=len(byte_duplicates),
        duplicate_groups=report["exact_byte_duplicate_groups"], receipt_origins=dict(origins)), indent=2))


if __name__ == "__main__":
    main()
