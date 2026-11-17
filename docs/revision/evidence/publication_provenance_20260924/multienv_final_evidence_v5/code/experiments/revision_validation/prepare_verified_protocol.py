"""Bind verified raw identities to legacy Y splits and new source-only manifests.

This is a post-hoc provenance audit. It never changes historical run artifacts
and does not turn the development environments into untouched test data.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from dg_dataset import (
    _expand_and_validate_subject_ids,
    _hash_indices,
    _read_subject_sidecar,
    _validate_sidecar_hash_binding,
)
from experiments.supcon_tpp_joint.protocol import VALIDATION_GROUP_IDS
from experiments.supcon_tpp_joint.runner_utils import canonical_json_hash

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/revision/evidence"
OUT = Path(__file__).resolve().parent / "verified_protocol"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def overlap_audit(parts):
    result = {}
    for field in ("sample_id", "subject_id", "acquisition_id", "raw_path"):
        sets = {name: {row[field] for row in rows} for name, rows in parts.items()}
        result[field] = {f"{a}_{b}": sorted(sets[a] & sets[b]) for a, b in combinations(parts, 2)}
    if any(values for checks in result.values() for values in checks.values()):
        raise ValueError(f"Split overlap: {result}")
    return result


def describe(rows):
    return {
        "count": len(rows),
        "subjects": sorted({r["subject_id"] for r in rows}),
        "acquisitions": len({r["acquisition_id"] for r in rows}),
        "label_counts": dict(sorted(Counter(str(r["label"]) for r in rows).items())),
        "environment_counts": dict(sorted(Counter(r["environment"] for r in rows).items())),
        "sample_keys_sha256": canonical_json_hash(sorted(r["sample_id"] for r in rows)),
    }


def main():
    summary_path = EVIDENCE / "raw_mapping/summary.json"
    summary = read(summary_path)
    if not (summary["all_9000_verified"] and summary["subject_sidecars_emitted"]
            and summary["matched_count"] == 9000 and not summary["mismatches"]):
        raise RuntimeError("All 9000 unique raw-to-processed matches are required before emitting a protocol.")
    array_audit = read(EVIDENCE / "array_audit.json")
    environments, verified_files, blocks, timestamps = {}, {}, [], []
    all_raw_paths, all_raw_hashes, all_subjects = set(), set(), set()
    for env, specification in sorted(array_audit["environments"].items()):
        sidecar_path = EVIDENCE / f"subject_sidecars/{env}.json"
        sidecar = read(sidecar_path)
        values, metadata = _read_subject_sidecar(sidecar_path, env)
        # Rehash actual files rather than trusting a copied historical hash field.
        actual = {kind: sha256(spec["path"]) for kind, spec in specification["files"].items()}
        assert _validate_sidecar_hash_binding(metadata, actual, env) == "bound"
        expanded, block_subjects = _expand_and_validate_subject_ids(values, env)
        labels = np.load(specification["files"]["label"]["path"], allow_pickle=False)
        assert len(sidecar["raw_paths"]) == len(sidecar["acquisition_ids"]) == len(labels) == 3000
        subjects = set(values)
        assert not (subjects & all_subjects), "Original subject IDs unexpectedly overlap environments"
        all_subjects.update(subjects)
        rows = []
        for index, raw_path in enumerate(sidecar["raw_paths"]):
            record_path = EVIDENCE / "raw_mapping" / (Path(raw_path).name + ".json")
            record = read(record_path)
            assert record["matched"] and record["processed_index"] == index
            assert record["environment"] == env and record["subject_id"] == values[index]
            assert record["raw_path"] == raw_path and record["commit"] == summary["commit"]
            assert record["verifier_version"] == summary["verifier_version"]
            assert record["processed_hashes"] == actual
            assert record["acquisition_id"] == sidecar["acquisition_ids"][index]
            assert record["label_matches"]
            assert record["amplitude_max_absolute_error"] < 1e-10
            assert record["phase_max_absolute_error"] < 1e-10
            assert raw_path not in all_raw_paths and record["raw_sha256"] not in all_raw_hashes
            all_raw_paths.add(raw_path)
            all_raw_hashes.add(record["raw_sha256"])
            rows.append({"sample_id": f"{env}:{index:04d}", "environment": env, "index": index,
                         "label": int(labels[index]), "subject_id": str(expanded[index]),
                         "acquisition_id": record["acquisition_id"], "raw_path": raw_path,
                         "raw_sha256": record["raw_sha256"]})
            timestamps.append({key: record[key] for key in (
                "raw_path", "mat_timestamp_exceeds_uint32", "timestamp_text15_changed_values",
                "raw_timestamp_monotonic", "replay_duration_seconds",
                "amplitude_max_absolute_error", "phase_max_absolute_error")})
        environments[env] = rows
        verified_files[env] = {"data_files": specification["files"], "actual_file_sha256": actual,
                               "sidecar_path": str(sidecar_path), "sidecar_sha256": sha256(sidecar_path)}
        for block, subject in enumerate(block_subjects):
            block_rows = rows[block * 300:(block + 1) * 300]
            assert describe(block_rows)["label_counts"] == {"0": 120, "1": 40, "2": 40, "3": 40, "4": 40, "5": 20}
            blocks.append({"environment": env, "block_id": f"B{block:02d}", "subject_id": subject,
                           "validation": f"B{block:02d}" in VALIDATION_GROUP_IDS[env], **describe(block_rows)})
        print(f"Verified {env}: 3000 identities and three current file hashes", flush=True)
    assert len(all_raw_paths) == len(all_raw_hashes) == 9000 and len(all_subjects) == 30

    folds = {}
    for target in environments:
        sources = [env for env in environments if env != target]
        parts = {"train": [], "val": [], "test": environments[target]}
        for env in sources:
            for row in environments[env]:
                key = "val" if f"B{row['index'] // 300:02d}" in VALIDATION_GROUP_IDS[env] else "train"
                parts[key].append(row)
        assert [len(parts[key]) for key in ("train", "val", "test")] == [4800, 1200, 3000]
        folds[target] = {"parts": parts, "overlap_audit": overlap_audit(parts),
                         "descriptions": {key: describe(rows) for key, rows in parts.items()}}

    # Audit every completed Y run using its recorded sample-set hashes, without
    # editing a historical manifest or inferring membership from label templates.
    registry_path = ROOT / "outputs_supcon_tpp_joint/formal_interim_v12/run_registry.csv"
    registry = list(csv.DictReader(registry_path.open(encoding="utf-8-sig", newline="")))
    completed = [run for run in registry if run["status"] == "completed"]
    assert len(completed) == 36
    legacy_checks = []
    for run in completed:
        manifest_path = Path(run["output_dir"]) / "split_manifest.json"
        manifest_hash = sha256(manifest_path)
        assert manifest_hash == run["split_manifest_file_sha256"]
        manifest = read(manifest_path)
        target = run["target_env"]
        assert manifest["inner_validation"] == "inferred_group_block_holdout"
        assert not manifest["subject_sidecars_complete"]
        for env, files in manifest["file_sha256"].items():
            assert files == verified_files[env]["actual_file_sha256"]
        for part, description in folds[target]["descriptions"].items():
            # Y's combined manifest uses _manifest_identifier_sets, which
            # prefixes the concatenated-loader sample ID, not the per-environment
            # local ID returned by _stable_sample_maps for other Y diagnostics.
            # Restore that recorded convention for hash verification; the new
            # source manifests deliberately use ENV:local_processed_index.
            offsets = {env: i * 3000 for i, env in enumerate(manifest["source_envs"])} if part != "test" else {target: 0}
            legacy_ids = [(r["environment"], r["index"] + offsets[r["environment"]]) for r in folds[target]["parts"][part]]
            legacy_keys = sorted(f"{env}:{index:04d}" for env, index in legacy_ids)
            assert canonical_json_hash(legacy_keys) == manifest[part]["sample_keys_sha256"]
            assert _hash_indices(sorted(index for _, index in legacy_ids)) == manifest[part]["indices_sha256"]
            assert description["count"] == manifest[part]["count"]
            assert description["label_counts"] == manifest[part]["label_counts"]
        legacy_checks.append({"job_id": run["job_id"], "target_environment": target,
                              "unchanged_manifest_sha256": manifest_hash,
                              "sample_sets_match_verified_rows": True,
                              "legacy_source_key_encoding": "environment:concatenated_source_loader_index; target loaded separately starts at zero",
                              "subject_and_acquisition_overlap": False})

    # Source manifests contain only source file paths and source rows. Target
    # metadata is kept separately and is unnecessary for smoke/training.
    for target, fold in folds.items():
        sources = [env for env in environments if env != target]
        payload = {
            "schema": "revision-verified-source-split/v1", "outer_target": target,
            "source_environments": sources, "development_data": True,
            "split_rule": "Reuse frozen Y sample membership, now verified against all original signals",
            "historical_Y_artifacts_modified": False,
            "inner_validation": "subject_grouped_holdout",
            "data_version": "legacy_multienv_850_hash_bound",
            "physical_time_or_phase_calibration_claimed": False,
            "provenance_summary_sha256": sha256(summary_path),
            "files": {env: verified_files[env] for env in sources},
            "train": fold["parts"]["train"], "val": fold["parts"]["val"],
            "descriptions": {key: fold["descriptions"][key] for key in ("train", "val")},
            "overlap_audit": overlap_audit({key: fold["parts"][key] for key in ("train", "val")}),
        }
        write(OUT / f"source_{target}.json", payload)
        write(OUT / f"deferred_target_{target}.json", {
            "outer_target": target, "development_data": True,
            "files": {target: verified_files[target]}, "test": fold["parts"]["test"],
            "description": fold["descriptions"]["test"]})
    audit = {
        "schema": "revision-verified-subject-audit/v1", "all_9000_verified": True,
        "source_commit": summary["commit"], "verifier_version": summary["verifier_version"],
        "current_processed_file_hashes_verified": True, "subject_count": len(all_subjects),
        "acquisition_count": len({r["acquisition_id"] for rows in environments.values() for r in rows}),
        "blocks": blocks, "Y_run_count": len(legacy_checks), "Y_runs": legacy_checks,
        "folds": {target: {key: value for key, value in fold.items() if key != "parts"}
                  for target, fold in folds.items()},
        "timestamp_precision_changed_files": sum(r["timestamp_text15_changed_values"] > 0 for r in timestamps),
        "nonmonotonic_replay_timestamp_files": sum(not r["raw_timestamp_monotonic"] for r in timestamps),
        "max_amplitude_error": max(r["amplitude_max_absolute_error"] for r in timestamps),
        "max_phase_error": max(r["phase_max_absolute_error"] for r in timestamps),
        "source_manifest_hashes": {env: sha256(OUT / f"source_{env}.json") for env in environments},
        "historical_Y_manifests_modified": False,
        "interpretation": "Subject/acquisition separation verified post hoc; development status and legacy time/phase limitations remain.",
    }
    write(EVIDENCE / "verified_subject_audit.json", audit)
    write(EVIDENCE / "replay_timestamp_records.json", timestamps)
    print(json.dumps({k: v for k, v in audit.items() if k not in ("blocks", "Y_runs", "folds")}, indent=2))


if __name__ == "__main__":
    main()
