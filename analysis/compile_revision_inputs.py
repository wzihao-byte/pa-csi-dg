"""Pack fully verified prepared rows into immutable arrays and grouped manifests.

Incomplete preparation is rejected before any dataset output is created. Native
MultiEnv controls preserve the exact historical row partitions. Widar uses its
metadata-declared subjects; it never inherits MultiEnv's 300-row block convention.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.revision_validation.prepare_verified_protocol import describe, overlap_audit
from experiments.revision_validation.study_io import bound_path, now, read, sha256, value_hash, write


def publish_array(temporary, destination):
    digest = sha256(temporary)
    if destination.exists():
        if sha256(destination) != digest:
            raise RuntimeError("Existing generated array differs; retain both files and investigate")
        bound_path(destination.parent, temporary.name).unlink()
    else:
        os.replace(temporary, destination)
    return digest


def pack_environment(output, environment, rows, variants, reader, labels):
    matrices, temporary_paths, statistics = {}, {}, {}
    for variant in variants:
        path = output / variant / f"{environment}_amp.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npy.partial")
        temporary_paths[variant] = temporary
        matrices[variant] = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(rows), 850, 90))
        statistics[variant] = {"min": float("inf"), "max": float("-inf")}
    try:
        for index, row in enumerate(rows):
            values = reader(row)
            for variant in variants:
                value = values[variant]
                if value.shape != (850, 90) or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise RuntimeError("Prepared sample shape/dtype/finite-value check failed")
                matrices[variant][index] = value
                statistics[variant]["min"] = min(statistics[variant]["min"], float(value.min()))
                statistics[variant]["max"] = max(statistics[variant]["max"], float(value.max()))
    finally:
        for matrix in matrices.values():
            matrix.flush()
            matrix._mmap.close()
    result = {}
    for variant in variants:
        path = output / variant / f"{environment}_amp.npy"
        digest = publish_array(temporary_paths[variant], path)
        result[variant] = {"path": str(path), "sha256": digest, "shape": [len(rows), 850, 90],
                           "dtype": "float32", "nonfinite_count": 0, **statistics[variant]}
    label_path = output / f"{environment}_label.npy"
    temporary = label_path.with_suffix(".npy.partial")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(labels, dtype=np.int32), allow_pickle=False)
    label_hash = publish_array(temporary, label_path)
    label_spec = {"path": str(label_path), "sha256": label_hash, "shape": [len(rows)], "dtype": "int32",
                  "nonfinite_count": 0, "min": int(min(labels)), "max": int(max(labels))}
    return result, label_spec


def load_preparation(prepared, kind):
    summary = read(prepared / "summary.json")
    config = read(prepared / "config.json")
    required = "complete_9000" if kind == "native" else "complete_selected_cohort"
    expected_count = 9000 if kind == "native" else 6749
    if not summary.get(required) or summary.get("failures") or summary["rows"] != expected_count:
        raise RuntimeError("Full preparation gate is not satisfied")
    if kind == "native" and not summary.get("all_packet_signal_matches"):
        raise RuntimeError("Native DAT/MAT signal equivalence has not passed")
    if kind == "native" and summary["config_value_sha256"] != value_hash(config):
        raise RuntimeError("Preparation configuration binding mismatch")
    if kind == "widar" and summary["config_sha256"] != sha256(prepared / "config.json"):
        raise RuntimeError("Preparation configuration file mismatch")
    records = []
    for identifier, expected in summary["record_sha256"].items():
        name = Path(identifier).stem if kind == "native" else hashlib.sha256(identifier.encode()).hexdigest()
        path = prepared / "records" / (name + ".json")
        if sha256(path) != expected:
            raise RuntimeError("Preparation record differs from completed summary")
        record = read(path)
        if record["config_value_sha256"] != value_hash(config):
            raise RuntimeError("Mixed preparation versions")
        if kind == "native" and record["mat_path"] != identifier:
            raise RuntimeError("Native record identity mismatch")
        if kind == "native" and not (record.get("verified") and record.get("all_complex_csi_and_packet_ids_equal")):
            raise RuntimeError("A native row lacks signal/packet verification")
        if kind == "widar" and record["sample_id"] != identifier:
            raise RuntimeError("Widar record identity mismatch")
        records.append(record)
    if len(records) != expected_count:
        raise RuntimeError("Wrong number of prepared records")
    raw_hashes = [r["dat_sha256"] if kind == "native" else r["raw_sha256"] for r in records]
    if len(set(raw_hashes)) != len(raw_hashes):
        raise RuntimeError("Duplicate raw byte streams in the selected cohort; investigate before splitting")
    return summary, config, records


def new_sidecar(path, environment, rows, amp_spec, label_spec, provenance):
    value = {"env_name": environment, "subject_ids": [r["subject_id"] for r in rows],
             "acquisition_ids": [r["acquisition_id"] for r in rows], "raw_paths": [r["raw_path"] for r in rows],
             "data_file_sha256": {"amp": amp_spec["sha256"], "label": label_spec["sha256"]},
             "provenance": provenance}
    write(path, value)
    return {"data_files": {"amp": amp_spec, "label": label_spec},
            "actual_file_sha256": {"amp": amp_spec["sha256"], "label": label_spec["sha256"]},
            "sidecar_path": str(path), "sidecar_sha256": sha256(path)}


def compile_native(prepared, output, config, records):
    variants = ("native_time_legacy_merge", "native_time_linear")
    original_protocol = ROOT / "experiments/revision_validation/verified_protocol"
    mapping = {}
    files = {variant: {} for variant in variants}
    for environment in ("E1", "E2", "E3"):
        rows = sorted([r for r in records if r["environment"] == environment], key=lambda r: r["index"])
        if [r["index"] for r in rows] != list(range(3000)):
            raise RuntimeError("Native array order does not cover each historical row exactly once")
        target = read(original_protocol / f"deferred_target_{environment}.json")
        original_rows = target["test"]
        for original, record in zip(original_rows, rows):
            if original["index"] != record["index"] or original["raw_path"] != record["mat_path"] or original["raw_sha256"] != record["mat_sha256"]:
                raise RuntimeError("Native row is not bound to its original identity")
            # Released native record subject strings omit the environment prefix.
            if original["subject_id"] != environment + ":" + record["subject_id"] or original["acquisition_id"] != record["acquisition_id"]:
                raise RuntimeError("Native subject/acquisition identity changed")
            mapping[original["sample_id"]] = {**original,
                "identity_source_mat_path": original["raw_path"], "identity_source_mat_sha256": original["raw_sha256"],
                "raw_path": record["dat_path"], "raw_sha256": record["dat_sha256"]}
        label_spec = target["files"][environment]["data_files"]["label"]
        if sha256(label_spec["path"]) != target["files"][environment]["actual_file_sha256"]["label"]:
            raise RuntimeError("Original label bytes changed")
        labels = np.load(label_spec["path"], allow_pickle=False).reshape(-1)
        if labels.tolist() != [r["label"] for r in original_rows]:
            raise RuntimeError("Original label file and verified rows differ")
        def reader(row):
            sample_path = Path(row["sample_path"])
            if sha256(sample_path) != row["sample_sha256"]:
                raise RuntimeError("Prepared native sample hash mismatch")
            with np.load(sample_path, allow_pickle=False) as values:
                return {variants[0]: values["historical_merge_native_time"], variants[1]: values["linear_native_time"]}
        packed, new_labels = pack_environment(output, environment, rows, variants, reader, labels)
        for variant in variants:
            transformed_rows = [mapping[r["sample_id"]] for r in original_rows]
            files[variant][environment] = new_sidecar(output / variant / "subject_sidecars" / f"{environment}.json",
                environment, transformed_rows, packed[variant], new_labels,
                {"method": "Native DAT reconstruction with all complex CSI and packet IDs matched to the original MAT identity",
                 "preparation_config_sha256": sha256(prepared / "config.json"), "variant": variant, "verified_rows": len(rows)})
        print(f"Packed native controls: {environment}", flush=True)
    for variant in variants:
        manifest_directory = output / variant / "manifests"
        for path in original_protocol.glob("*.json"):
            original = read(path)
            manifest = copy.deepcopy(original)
            manifest["data_version"] = variant
            manifest["original_partition_manifest_sha256"] = sha256(path)
            manifest["input_preparation_summary_sha256"] = sha256(prepared / "summary.json")
            manifest["physical_time_or_phase_calibration_claimed"] = False
            manifest["phase_available"] = False
            manifest["files"] = {env: files[variant][env] for env in original["files"]}
            for part in ("train", "val", "test"):
                if part in manifest:
                    manifest[part] = [mapping[row["sample_id"]] for row in original[part]]
                    if [r["sample_id"] for r in manifest[part]] != [r["sample_id"] for r in original[part]]:
                        raise RuntimeError("Historical partition changed")
            if "train" in manifest:
                all_parts = {part: manifest[part] for part in ("train", "val")}
                target_rows = read(original_protocol / f"deferred_target_{manifest['outer_target']}.json")["test"]
                all_parts["test"] = [mapping[r["sample_id"]] for r in target_rows]
                manifest["overlap_audit"] = overlap_audit(all_parts)
            write(manifest_directory / path.name, manifest)
    return {"variants": list(variants), "rows": 9000, "historical_partition_sample_order_preserved": True,
            "phase_included": False, "manifest_directories": [str(output / variant / "manifests") for variant in variants]}


def compile_widar(prepared, output, config, records, selection_path):
    selection = read(selection_path)
    inventory_path = selection_path.with_name("inventory.json")
    if sha256(inventory_path) != config["inventory_sha256"] or sha256(selection_path) != read(inventory_path)["selection_sha256"]:
        raise RuntimeError("Widar selection/inventory binding mismatch")
    files, environments = {}, {}
    for room in (1, 2, 3):
        environment = f"W{room}"
        records_here = sorted([r for r in records if r["room"] == room], key=lambda r: (int(r["subject"]), r["archive"], r["label"], r["position"], r["orientation"], r["repetition"]))
        rows = [{"sample_id": environment + ":" + r["sample_id"], "environment": environment, "index": index,
                 "label": r["label"], "subject_id": "Widar:user" + r["subject"], "acquisition_id": r["acquisition_id"],
                 "raw_path": r["raw_path"], "raw_sha256": r["raw_sha256"], "original_record_id": r["sample_id"]}
                for index, r in enumerate(records_here)]
        def reader(row):
            path = Path(row["amplitude_path"])
            if sha256(path) != row["amplitude_sha256"] or sha256(row["raw_path"]) != row["raw_sha256"]:
                raise RuntimeError("Widar prepared/raw sample hash mismatch")
            return {"native_amplitude": np.load(path, allow_pickle=False)}
        packed, labels = pack_environment(output, environment, records_here, ("native_amplitude",), reader, [r["label"] for r in rows])
        files[environment] = new_sidecar(output / "native_amplitude/subject_sidecars" / f"{environment}.json",
            environment, rows, packed["native_amplitude"], labels,
            {"method": "Official-release member CRC/raw SHA and predeclared cohort metadata", "verified_rows": len(rows),
             "selection_sha256": sha256(selection_path), "preparation_summary_sha256": sha256(prepared / "summary.json")})
        environments[environment] = rows
        print(f"Packed Widar candidate: {environment}, rows {len(rows)}", flush=True)
    manifest_directory = output / "native_amplitude/manifests"
    for target in environments:
        sources = sorted(env for env in environments if env != target)
        parts = {"train": [], "val": [], "test": environments[target]}
        for source in sources:
            validation_subject = "Widar:user" + selection["validation_subjects_if_room_is_source"][source[1:]]
            for row in environments[source]:
                parts["val" if row["subject_id"] == validation_subject else "train"].append(row)
        overlap = overlap_audit(parts)
        if any({r["label"] for r in rows} != set(range(6)) for rows in parts.values()):
            raise RuntimeError("Widar fold is missing a class")
        common = {"outer_target": target, "data_version": "widar_native_amplitude_v1",
                  "class_names": selection["class_names"], "external_candidate": True,
                  "independent_confirmation_completed": False, "physical_time_or_phase_calibration_claimed": False,
                  "selection_sha256": sha256(selection_path), "input_preparation_summary_sha256": sha256(prepared / "summary.json")}
        source_manifest = {**common, "schema": "revision-verified-source/v2", "source_environments": sources,
                           "inner_validation": "subject_grouped_holdout", "split_rule": "Predeclared original subject IDs, no 300-row block inference",
                           "files": {env: files[env] for env in sources}, "train": parts["train"], "val": parts["val"],
                           "descriptions": {part: describe(rows) for part, rows in parts.items()}, "overlap_audit": overlap}
        target_manifest = {**common, "files": {target: files[target]}, "test": parts["test"], "description": describe(parts["test"])}
        write(manifest_directory / f"source_{target}.json", source_manifest)
        write(manifest_directory / f"deferred_target_{target}.json", target_manifest)
    return {"variants": ["native_amplitude"], "rows": len(records), "room_counts": {env: len(rows) for env, rows in environments.items()},
            "subjects": sorted({r["subject_id"] for rows in environments.values() for r in rows}),
            "manifest_directories": [str(manifest_directory)], "independent_confirmation_completed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["native", "widar"], required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widar-selection", type=Path)
    args = parser.parse_args()
    prepared, output = args.prepared.resolve(), args.output.resolve()
    if args.kind == "widar" and args.widar_selection is None:
        raise ValueError("Widar requires its predeclared cohort selection")
    if (output / "dataset_manifest.json").exists():
        previous_config = read(output / "compile_config.json")
        if previous_config["kind"] != args.kind or Path(previous_config["prepared"]).resolve() != prepared:
            raise RuntimeError("Existing output belongs to different preparation inputs")
        if args.widar_selection and Path(previous_config["widar_selection"]).resolve() != args.widar_selection.resolve():
            raise RuntimeError("Existing output belongs to a different cohort selection")
        manifest = read(output / "dataset_manifest.json")
        for relative, expected in manifest["artifacts_sha256"].items():
            if sha256(bound_path(output, relative)) != expected:
                raise RuntimeError("Completed dataset artifact hash mismatch")
        print("Completed dataset already verifies; no files changed")
        return
    summary, config, records = load_preparation(prepared, args.kind)
    output.mkdir(parents=True, exist_ok=True)
    compile_config = {"kind": args.kind, "prepared": str(prepared), "preparation_summary_sha256": sha256(prepared / "summary.json"),
                      "compiler_sha256": sha256(__file__), "widar_selection": str(args.widar_selection) if args.widar_selection else None}
    if (output / "compile_config.json").exists() and read(output / "compile_config.json") != compile_config:
        raise RuntimeError("Changed compiler inputs require a new output version")
    write(output / "compile_config.json", compile_config)
    if args.kind == "native":
        result = compile_native(prepared, output, config, records)
    else:
        if args.widar_selection is None:
            raise ValueError("Widar requires its predeclared cohort selection")
        result = compile_widar(prepared, output, config, records, args.widar_selection.resolve())
    files = [p for p in output.rglob("*") if p.is_file() and not p.name.endswith(".partial")]
    write(output / "dataset_manifest.json", {"completed_at": now(), "complete": True, **result,
          "artifacts_sha256": {p.relative_to(output).as_posix(): sha256(p) for p in files}})
    print(result)


if __name__ == "__main__":
    main()
