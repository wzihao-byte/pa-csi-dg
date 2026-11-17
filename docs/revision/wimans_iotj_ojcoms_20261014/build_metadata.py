"""Build a metadata-only WiMANS experiment plan; no CSI or model is loaded."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROOMS = ("classroom", "meeting_room", "empty_room")
CLASSES = ("nothing", "walk", "rotation", "jump", "wave", "lie_down", "pick_up", "sit_down", "stand_up")
VALIDATION_USERS = {"classroom": [1, 2], "meeting_room": [3, 4], "empty_room": [5, 6]}


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "WiMANS-metadata-plan/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    api = "https://api.github.com/repos/huangshk/WiMANS/commits/main"
    existing_path = ROOT / "metadata_verification.json"
    existing = json.loads(existing_path.read_text(encoding="utf-8")) if existing_path.exists() else None
    if existing and existing.get("current_protocol"):
        # Once execution preparation has advanced, this metadata entrypoint must
        # never silently restore the original prospective v1 protocol.
        for name, expected in existing["artifacts_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Pinned metadata artifact changed: " + name)
        current = existing["current_protocol"]
        if hashlib.sha256((ROOT / current["file"]).read_bytes()).hexdigest() != current["sha256"]:
            raise ValueError("Current protocol receipt changed; update its audited revision record")
        protocol = json.loads((ROOT / current["file"]).read_text(encoding="utf-8"))
        registration = protocol["input"]["axis_storage_protocol"]
        if hashlib.sha256((ROOT / registration["file"]).read_bytes()).hexdigest() != registration["sha256"]:
            raise ValueError("Registered storage protocol changed")
        print(json.dumps({"mode": "verify_advanced_existing_protocol", "rows": existing["rows"],
                          "author_repository_commit": existing["author_repository_commit"],
                          "current_protocol_schema": protocol["schema"], "training_started_by_this_script": False},
                         ensure_ascii=False))
        return
    commit = existing["author_repository_commit"] if existing else json.loads(fetch(api))["sha"]
    url = f"https://raw.githubusercontent.com/huangshk/WiMANS/{commit}/dataset/annotation.csv"
    raw = fetch(url)
    if existing and hashlib.sha256(raw).hexdigest() != existing["annotation_sha256"]:
        raise ValueError("Pinned author annotation changed; preserve this plan and create a new version")
    (ROOT / "annotation_source.csv").write_bytes(raw)
    rows = []
    for item in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        if float(item["wifi_band"]) != 5.0 or int(item["number_of_users"]) != 1:
            continue
        active = [u for u in range(1, 7) if item[f"user_{u}_activity"].strip()]
        if len(active) != 1:
            raise ValueError("Single-user row does not have exactly one active user")
        user = active[0]
        label = item["label"]
        _, group, instance = label.split("_")
        activity = item[f"user_{user}_activity"].strip()
        rows.append({"sample_id": label, "environment": item["environment"], "wifi_band": "5",
                     "subject_id": user, "acquisition_group": int(group), "instance_id": int(instance),
                     "activity": activity, "class_id": CLASSES.index(activity),
                     "position": item[f"user_{user}_location"],
                     "raw_relative_path": f"wifi_csi/mat/{label}.mat"})
    rows.sort(key=lambda r: (ROOMS.index(r["environment"]), r["acquisition_group"], r["instance_id"]))
    assert len(rows) == 1782 and len({r["sample_id"] for r in rows}) == 1782
    counts = Counter((r["environment"], r["activity"]) for r in rows)
    assert all(counts[(room, label)] == 66 for room in ROOMS for label in CLASSES)
    groups = Counter((r["environment"], r["subject_id"], r["acquisition_group"]) for r in rows)
    assert len(groups) == 18 and set(groups.values()) == {99}
    with (ROOT / "subset_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    folds = []
    for target in ROOMS:
        val_users = VALIDATION_USERS[target]
        train_users = [u for u in range(1, 7) if u not in val_users]
        parts = {
            "train": [r for r in rows if r["environment"] != target and r["subject_id"] in train_users],
            "validation": [r for r in rows if r["environment"] != target and r["subject_id"] in val_users],
            "test": [r for r in rows if r["environment"] == target],
        }
        assert [len(parts[k]) for k in ("train", "validation", "test")] == [792, 396, 594]
        ids = {k: {r["sample_id"] for r in values} for k, values in parts.items()}
        assert not (ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"])
        train_groups = {r["acquisition_group"] for r in parts["train"]}
        val_groups = {r["acquisition_group"] for r in parts["validation"]}
        assert not train_groups & val_groups
        assert not set(train_users) & set(val_users)
        for part, expected in (("train", 88), ("validation", 44), ("test", 66)):
            assert set(Counter(r["activity"] for r in parts[part]).values()) == {expected}
        folds.append({"target_environment": target, "source_environments": [r for r in ROOMS if r != target],
                      "source_training_users": train_users, "source_validation_users": val_users,
                      "target_users": list(range(1, 7)), "seeds": [42, 52, 62],
                      "counts": {k: len(v) for k, v in parts.items()},
                      "sample_ids": {k: [r["sample_id"] for r in v] for k, v in parts.items()},
                      "target_subjects_overlap_sources": True})
    write_json("fold_manifest.json", {"schema": "wimans-metadata-fold-plan/v1", "folds": folds})
    jobs = []
    for arm in ("W00", "W01", "W10", "W11", "WC"):
        for room in ROOMS:
            for seed in (42, 52, 62):
                jobs.append({"dataset": "WiMANS", "arm": arm, "target": room, "seed": seed})
    for arm in ("MC", "MA", "ME"):
        for environment in ("E1", "E2", "E3"):
            for seed in (42, 52, 62):
                jobs.append({"dataset": "MultiEnv_HS", "arm": arm, "target": environment, "seed": seed})
    assert len(jobs) == 72 and len({(j["dataset"], j["arm"], j["target"], j["seed"]) for j in jobs}) == 72
    write_json("protocol_plan.json", {
        "schema": "wimans-iotj-ojcoms-prospective-plan/v1",
        "status": "metadata_pinned_input_and_implementation_qa_pending",
        "venues": {"primary": "IEEE Internet of Things Journal", "backup": "IEEE Open Journal of the Communications Society"},
        "dataset": {"wifi_band": "5", "number_of_users": 1, "classes": list(CLASSES),
                    "expected_rows": 1782, "input_tx_index": 0, "input_rx_count": 3,
                    "input_subcarrier_count": 30, "input_length": 850,
                    "raw_axes_and_export_permutation_audit": "required_before_input_freeze"},
        "scope": {"primary_claim": "cross_room_method_validation_after_retraining",
                  "unseen_subject_claim_supported_by_primary_folds": False,
                  "physical_phase_calibration_claimed": False},
        "input": {"amplitude": "absolute complex CSI, arithmetic duplicate-time mean, native-time linear interpolation",
                  "phase": "float64 wrapped angle, arithmetic duplicate-time mean, same-grid linear interpolation, feature-axis unwrap, float32",
                  "phase_unwrap_axis": -1, "loader_unwrap_phase": False,
                  "train_fitted_normalization": "none in primary inherited representation",
                  "grid": "850 endpoint-inclusive points over actual within-record relative timestamps"},
        "main_training": {"seeds": [42, 52, 62], "batch_size": 32,
                          "source_examples_per_environment_per_batch": 16,
                          "optimizer_steps": 30000, "virtual_epochs": 200, "updates_per_virtual_epoch": 150,
                          "optimizer": "Adam", "initial_lr": 0.001, "weight_decay": 0,
                          "lr_schedule": "0.001*0.9**(step/10000)", "gradient_clip": 0,
                          "temperature": 0.2, "supcon_lambda": 0.05, "warmup_updates": 0,
                          "positive_mode": "historical_hybrid_global_class_instance_views",
                          "source_selection_metric": "accuracy", "tie_policy": "last_tied_best",
                          "target_evaluation": "after_config_and_checkpoint_freeze"},
        "arm_definitions": {
            "W00": "AP capacity-matched global MLP + CE", "W01": "AP capacity-matched global MLP + hybrid SC",
            "W10": "AP TPP + CE", "W11": "AP TPP + hybrid SC", "WC": "AP global MLP + CE + source CORAL",
            "MC": "MultiEnv H-S AP global MLP + CE + source CORAL",
            "MA": "MultiEnv H-S author CSI-DG baseline, ARC-Fi fully labeled DG candidate, qualification pending",
            "ME": "MultiEnv H-S AP TPP + CE + identical masked-forward exposure, SC weight zero"},
        "core_jobs": jobs, "core_formal_training_count": 72, "existing_queue_remaining_at_plan": 29,
        "source_search_budget": {"seed": 20261014, "coral_gamma_candidates": [0.1, 1.0, 10.0],
                                 "coral_two_datasets_max_runs": 18, "author_baseline_max_runs": 9,
                                 "max_total_search_runs": 27, "target_evaluation_permitted": False},
        "optional_formal_counts": {"Wimans_AA_final_model": 9, "Wimans_author_CSI_DG": 9,
                                   "Wimans_differential_phase_final_model": 9,
                                   "joint_unseen_room_and_subject_final_plus_baseline": 18},
        "launch": {"this_artifact_is_an_executable_training_bundle": False,
                   "raw_signal_subset_downloaded_by_this_script": False, "training_started": False},
    })
    write_json("metadata_verification.json", {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "author_repository_commit": commit, "annotation_url": url,
        "annotation_sha256": hashlib.sha256(raw).hexdigest(), "rows": len(rows),
        "room_counts": dict(Counter(r["environment"] for r in rows)),
        "all_room_class_counts_are_66": True, "source_train_validation_subjects_disjoint": True,
        "source_train_validation_acquisition_groups_disjoint": True, "fold_sample_sets_disjoint": True,
        "target_subjects_overlap_sources": True,
        "raw_csi_loaded_by_this_script": False, "training_started": False,
        "input_quality_audit_complete": False,
        "artifacts_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                             for name in ("annotation_source.csv", "subset_manifest.csv", "fold_manifest.json", "protocol_plan.json")},
    })
    print(json.dumps({"rows": len(rows), "fold_counts": [f["counts"] for f in folds],
                      "author_repository_commit": commit, "training_started": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
