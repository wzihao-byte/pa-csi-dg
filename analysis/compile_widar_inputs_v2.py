"""Compile the fully accounted Widar v2 cohort after signal-level exclusions."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.compile_revision_inputs import new_sidecar, pack_environment
from experiments.revision_validation.prepare_verified_protocol import describe, overlap_audit
from experiments.revision_validation.study_io import bound_path, now, read, sha256, value_hash, write


def load_accounted(prepared, inventory_path):
    summary, config, inventory = read(prepared / "summary.json"), read(prepared / "config.json"), read(inventory_path)
    if config["schema"] != "widar-native-amplitude/v2" or summary["schema"] != "widar-native-amplitude/v2-summary":
        raise RuntimeError("Wrong Widar preparation version")
    if not summary.get("complete_inventory_accounting") or summary.get("failures") or summary.get("replacement_samples_added") != 0:
        raise RuntimeError("Widar v2 inventory accounting is incomplete")
    if summary["inventory_sha256"] != sha256(inventory_path) or config["inventory_sha256"] != sha256(inventory_path):
        raise RuntimeError("Widar preparation/inventory binding mismatch")
    if summary["config_sha256"] != sha256(prepared / "config.json"):
        raise RuntimeError("Widar v2 configuration binding mismatch")
    excluded_ids = {row["sample_id"] for row in summary["signal_exclusions"]}
    inventory_ids = {row["sample_id"] for row in inventory["samples"]}
    if not excluded_ids <= inventory_ids or len(excluded_ids) != len(summary["exclusion_sha256"]):
        raise RuntimeError("Signal exclusion identities are invalid")
    for identifier, expected in summary["exclusion_sha256"].items():
        path = prepared / "signal_exclusions" / (hashlib.sha256(identifier.encode()).hexdigest() + ".json")
        if sha256(path) != expected or read(path)["sample_id"] != identifier:
            raise RuntimeError("Signal exclusion evidence differs")
    records = []
    for identifier, expected in summary["record_sha256"].items():
        path = prepared / "records" / (hashlib.sha256(identifier.encode()).hexdigest() + ".json")
        if sha256(path) != expected:
            raise RuntimeError("Prepared Widar record differs from its summary")
        record = read(path)
        if record["sample_id"] != identifier or record["config_value_sha256"] != value_hash(config):
            raise RuntimeError("Prepared Widar record identity/configuration differs")
        if sha256(record["raw_path"]) != record["raw_sha256"] or sha256(record["amplitude_path"]) != record["amplitude_sha256"]:
            raise RuntimeError("Prepared Widar signal artifact hash mismatch")
        records.append(record)
    if {row["sample_id"] for row in records} & excluded_ids:
        raise RuntimeError("A signal exclusion also appears as a valid record")
    if {row["sample_id"] for row in records} | excluded_ids != inventory_ids:
        raise RuntimeError("Valid and excluded records do not exhaust the frozen inventory")
    if summary["valid_rows"] != len(records) or summary["rows"] != len(records):
        raise RuntimeError("Widar v2 valid-row count differs")
    raw_hashes = [row["raw_sha256"] for row in records]
    if len(set(raw_hashes)) != len(raw_hashes):
        raise RuntimeError("Duplicate valid raw byte streams in Widar v2")
    return summary, config, inventory, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    prepared, output = args.prepared.resolve(), args.output.resolve()
    selection_path, inventory_path = args.selection.resolve(), args.inventory.resolve()
    if (output / "dataset_manifest.json").exists():
        manifest = read(output / "dataset_manifest.json")
        for relative, expected in manifest["artifacts_sha256"].items():
            if sha256(bound_path(output, relative)) != expected:
                raise RuntimeError("Completed Widar v2 array artifact differs")
        print("Completed Widar v2 arrays already verify; no files changed")
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve incomplete Widar v2 compiler output; use a new version")
    summary, config, inventory, records = load_accounted(prepared, inventory_path)
    selection = read(selection_path)
    if sha256(selection_path) != inventory["selection_sha256"]:
        raise RuntimeError("Widar selection/inventory binding mismatch")
    output.mkdir(parents=True, exist_ok=True)
    write(output / "compile_config.json", {"schema": "widar-compiled-input/v2",
          "prepared": str(prepared), "preparation_summary_sha256": sha256(prepared / "summary.json"),
          "selection_sha256": sha256(selection_path), "inventory_sha256": sha256(inventory_path),
          "compiler_sha256": sha256(__file__), "target_model_evaluation": False})
    files, environments = {}, {}
    for room in (1, 2, 3):
        environment = f"W{room}"
        records_here = sorted([row for row in records if row["room"] == room],
                              key=lambda row: (int(row["subject"]), row["archive"], row["label"],
                                               row["position"], row["orientation"], row["repetition"]))
        rows = [{"sample_id": environment + ":" + row["sample_id"], "environment": environment, "index": index,
                 "label": row["label"], "subject_id": "Widar:user" + row["subject"],
                 "acquisition_id": row["acquisition_id"], "raw_path": row["raw_path"],
                 "raw_sha256": row["raw_sha256"], "original_record_id": row["sample_id"]}
                for index, row in enumerate(records_here)]
        def reader(row):
            path = Path(row["amplitude_path"])
            if sha256(path) != row["amplitude_sha256"] or sha256(row["raw_path"]) != row["raw_sha256"]:
                raise RuntimeError("Widar v2 prepared/raw sample hash mismatch")
            return {"native_amplitude": np.load(path, allow_pickle=False)}
        packed, labels = pack_environment(output, environment, records_here, ("native_amplitude",), reader,
                                          [row["label"] for row in rows])
        files[environment] = new_sidecar(output / "native_amplitude/subject_sidecars" / f"{environment}.json",
            environment, rows, packed["native_amplitude"], labels,
            {"method": "Official inventory-bound raw SHA/CRC and v2 per-record signal validation",
             "verified_rows": len(rows), "selection_sha256": sha256(selection_path),
             "preparation_summary_sha256": sha256(prepared / "summary.json"),
             "signal_exclusion_count": len(summary["signal_exclusions"]), "replacement_samples_added": 0})
        environments[environment] = rows
        print(f"Packed Widar v2: {environment}, rows {len(rows)}", flush=True)
    if any(len({row["subject_id"] for row in environments[a]}) != 3 for a in environments):
        raise RuntimeError("Widar v2 room no longer contains all three declared subjects")
    subjects = {room: {row["subject_id"] for row in rows} for room, rows in environments.items()}
    if any(subjects[a] & subjects[b] for a, b in (("W1", "W2"), ("W1", "W3"), ("W2", "W3"))):
        raise RuntimeError("Widar subjects cross rooms")
    manifest_directory = output / "native_amplitude/manifests"
    for target in environments:
        sources = sorted(set(environments) - {target})
        parts = {"train": [], "val": [], "test": environments[target]}
        for source in sources:
            validation_subject = "Widar:user" + selection["validation_subjects_if_room_is_source"][source[1:]]
            for row in environments[source]:
                parts["val" if row["subject_id"] == validation_subject else "train"].append(row)
        overlap = overlap_audit(parts)
        if any({row["label"] for row in rows} != set(range(6)) for rows in parts.values()):
            raise RuntimeError("Widar v2 fold is missing a class")
        common = {"outer_target": target, "data_version": "widar_native_amplitude_v2",
                  "class_names": selection["class_names"], "external_candidate": True,
                  "independent_confirmation_completed": False, "physical_time_or_phase_calibration_claimed": False,
                  "selection_sha256": sha256(selection_path), "input_preparation_summary_sha256": sha256(prepared / "summary.json"),
                  "signal_exclusion_count": len(summary["signal_exclusions"]), "replacement_samples_added": 0}
        source_manifest = {**common, "schema": "revision-verified-source/v2", "source_environments": sources,
            "inner_validation": "subject_grouped_holdout", "split_rule": "Predeclared original subject IDs; signal-corrupt records omitted without replacement",
            "files": {environment: files[environment] for environment in sources},
            "train": parts["train"], "val": parts["val"],
            "descriptions": {part: describe(rows) for part, rows in parts.items()}, "overlap_audit": overlap}
        target_manifest = {**common, "files": {target: files[target]}, "test": parts["test"],
                           "description": describe(parts["test"])}
        write(manifest_directory / f"source_{target}.json", source_manifest)
        write(manifest_directory / f"deferred_target_{target}.json", target_manifest)
    files_on_disk = [path for path in output.rglob("*") if path.is_file() and not path.name.endswith(".partial")]
    dataset = {"completed_at": now(), "complete": True, "schema": "widar-compiled-input/v2",
          "variants": ["native_amplitude"], "rows": len(records),
          "room_counts": {environment: len(rows) for environment, rows in environments.items()},
          "subjects": sorted(set().union(*subjects.values())),
          "signal_exclusions": [{key: row[key] for key in ("sample_id", "archive", "room", "subject", "label", "reason")}
                                for row in summary["signal_exclusions"]],
          "replacement_samples_added": 0, "manifest_directories": [str(manifest_directory)],
          "independent_confirmation_completed": False, "target_model_evaluation": False,
          "artifacts_sha256": {path.relative_to(output).as_posix(): sha256(path) for path in files_on_disk}}
    write(output / "dataset_manifest.json", dataset)
    print({"rows": dataset["rows"], "room_counts": dataset["room_counts"],
           "signal_exclusions": len(dataset["signal_exclusions"]), "target_model_evaluation": False})


if __name__ == "__main__":
    main()
