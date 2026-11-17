"""Compile full, hash-bound phase variants while preserving original partitions."""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.compile_revision_inputs import publish_array
from analysis.native_phase_processing import VARIANTS
from experiments.revision_validation.modality_models import historical_feature_phase
from experiments.revision_validation.study_io import bound_path, now, read, sha256, value_hash, write

HISTORICAL = "historical_feature_axis_unwrap"


def verify_compiled(directory):
    manifest = read(directory / "dataset_manifest.json")
    if not manifest.get("complete") or manifest["rows"] != 9000:
        raise RuntimeError("Require the complete native amplitude compilation")
    for name, digest in manifest["artifacts_sha256"].items():
        if sha256(bound_path(directory, name)) != digest:
            raise RuntimeError("Compiled native amplitude artifact differs")
    return manifest


def compile_arrays(prepared, native, output, records):
    originals = ROOT / "experiments/revision_validation/verified_protocol"
    native_manifests = native / "native_time_linear/manifests"
    variants = (HISTORICAL, *VARIANTS)
    files = {variant: {} for variant in variants}
    for environment in ("E1", "E2", "E3"):
        old = read(originals / f"deferred_target_{environment}.json")
        modern = read(native_manifests / f"deferred_target_{environment}.json")
        rows = sorted([r for r in records if r["environment"] == environment], key=lambda r:r["index"])
        if [r["index"] for r in rows] != list(range(3000)):
            raise RuntimeError("Phase records do not cover the exact native array indices")
        old_files = old["files"][environment]
        for kind in ("amp", "phase", "label"):
            if sha256(old_files["data_files"][kind]["path"]) != old_files["actual_file_sha256"][kind]:
                raise RuntimeError("Historical input bytes changed")
        old_phase = np.load(old_files["data_files"]["phase"]["path"], mmap_mode="r", allow_pickle=False)
        matrices, temporaries = {}, {}
        for variant in variants:
            path = output / variant / f"{environment}_phase.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporaries[variant] = path.with_suffix(".npy.partial")
            matrices[variant] = np.lib.format.open_memmap(temporaries[variant], mode="w+", dtype=np.float32, shape=(3000,850,90))
        stats = {variant: {"min": float("inf"), "max": -float("inf")} for variant in variants}
        try:
            for index, row in enumerate(rows):
                identity = modern["test"][index]
                if row["index"] != identity["index"] or row["dat_path"] != identity["raw_path"] or row["raw_sha256"] != identity["raw_sha256"]:
                    raise RuntimeError("Phase raw identity and native amplitude identity differ")
                if environment + ":" + row["subject_id"] != identity["subject_id"] or row["acquisition_id"] != identity["acquisition_id"]:
                    raise RuntimeError("Phase subject/acquisition identity differs")
                path = Path(row["sample_path"])
                if sha256(path) != row["sample_sha256"]:
                    raise RuntimeError("Prepared phase sample changed")
                with np.load(path, allow_pickle=False) as packed:
                    values = {variant: packed[variant] for variant in VARIANTS}
                values[HISTORICAL] = historical_feature_phase(old_phase[index:index+1])[0]
                for variant, value in values.items():
                    if value.dtype != np.float32 or value.shape != (850,90) or not np.isfinite(value).all():
                        raise RuntimeError("Wrong compiled phase shape/dtype/nonfinite input")
                    matrices[variant][index] = value
                    stats[variant]["min"] = min(stats[variant]["min"], float(value.min()))
                    stats[variant]["max"] = max(stats[variant]["max"], float(value.max()))
        finally:
            for matrix in matrices.values():
                matrix.flush()
                matrix._mmap.close()
            old_phase._mmap.close()
        for variant in variants:
            path = output / variant / f"{environment}_phase.npy"
            digest = publish_array(temporaries[variant], path)
            phase_spec = {"path": str(path), "sha256": digest, "shape": [3000,850,90], "dtype": "float32", "nonfinite_count": 0, **stats[variant]}
            reference = old if variant == HISTORICAL else modern
            specification = copy.deepcopy(reference["files"][environment])
            specification["data_files"]["phase"] = phase_spec
            specification["actual_file_sha256"]["phase"] = digest
            sidecar = output / variant / "subject_sidecars" / f"{environment}.json"
            write(sidecar, {"env_name": environment, "subject_ids": [r["subject_id"] for r in reference["test"]],
                "acquisition_ids": [r["acquisition_id"] for r in reference["test"]], "raw_paths": [r["raw_path"] for r in reference["test"]],
                "data_file_sha256": specification["actual_file_sha256"],
                "provenance": {"variant": variant, "source_partition_manifest_sha256": sha256((originals if variant == HISTORICAL else native_manifests) / f"deferred_target_{environment}.json"),
                               "phase_preparation_summary_sha256": sha256(prepared / "summary.json"), "verified_rows": 3000}})
            specification.update(sidecar_path=str(sidecar), sidecar_sha256=sha256(sidecar))
            files[variant][environment] = specification
        print(f"Packed three phase variants: {environment}", flush=True)
    for variant in variants:
        reference_directory = originals if variant == HISTORICAL else native_manifests
        for reference_path in reference_directory.glob("*.json"):
            manifest = read(reference_path)
            manifest["files"] = {env: files[variant][env] for env in manifest["files"]}
            manifest.update(data_version=variant, phase_available=True, phase_preprocessing_applied_before_loading=True,
                complete_hardware_calibration=False, phase_input_compile_config_sha256=sha256(output / "compile_config.json"),
                original_partition_manifest_sha256=sha256(reference_path))
            write(output / variant / "manifests" / reference_path.name, manifest)
    return {"variants": list(variants), "rows": 9000, "original_partition_order_preserved": True,
            "phase_loaded_before_historical_float32_cast": True, "complete_hardware_calibration": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-prepared", type=Path, required=True)
    parser.add_argument("--native-compiled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepared, native, output = args.phase_prepared.resolve(), args.native_compiled.resolve(), args.output.resolve()
    if (output / "dataset_manifest.json").exists():
        previous_config = read(output / "compile_config.json")
        if previous_config["phase_prepared"] != str(prepared) or previous_config["native_compiled"] != str(native):
            raise RuntimeError("Existing modality output has different inputs")
        previous = read(output / "dataset_manifest.json")
        for relative, digest in previous["artifacts_sha256"].items():
            if sha256(bound_path(output, relative)) != digest:
                raise RuntimeError("Completed modality artifact changed")
        print("Completed modality input files verify; no changes")
        return
    summary, parent_config = read(prepared / "summary.json"), read(prepared / "config.json")
    if not summary.get("complete_9000") or summary["rows"] != 9000 or summary.get("failures") or summary.get("smoke"):
        raise RuntimeError("Full phase preparation is required; smoke is not a completed dataset")
    if summary["config_value_sha256"] != value_hash(parent_config):
        raise RuntimeError("Phase preparation configuration changed")
    verify_compiled(native)
    if read(native / "compile_config.json")["preparation_summary_sha256"] != parent_config["native_summary_sha256"]:
        raise RuntimeError("Amplitude and phase are not based on the same verified native preparation")
    records = []
    for name, digest in summary["record_sha256"].items():
        record_path = prepared / "records" / (Path(name).stem + ".json")
        if sha256(record_path) != digest:
            raise RuntimeError("Phase preparation record changed")
        record = read(record_path)
        if record["mat_path"] != name or record["config_value_sha256"] != summary["config_value_sha256"]:
            raise RuntimeError("Phase identity/configuration mismatch")
        records.append(record)
    if len(records) != 9000:
        raise RuntimeError("Phase preparation record count mismatch")
    config = {"phase_prepared": str(prepared), "native_compiled": str(native),
              "phase_summary_sha256": sha256(prepared / "summary.json"), "native_dataset_manifest_sha256": sha256(native / "dataset_manifest.json"),
              "code_sha256": {name: sha256(ROOT / name) for name in ("analysis/compile_modality_inputs.py",
                  "analysis/compile_revision_inputs.py", "analysis/native_phase_processing.py",
                  "experiments/revision_validation/modality_models.py", "experiments/revision_validation/study_io.py")}}
    if (output / "compile_config.json").exists() and read(output / "compile_config.json") != config:
        raise RuntimeError("Changed compiler configuration; use a new version")
    write(output / "compile_config.json", config)
    result = compile_arrays(prepared, native, output, records)
    artifacts = [p for p in output.rglob("*") if p.is_file() and not p.name.endswith(".partial")]
    write(output / "dataset_manifest.json", {"completed_at": now(), "complete": True, **result,
          "artifacts_sha256": {p.relative_to(output).as_posix(): sha256(p) for p in artifacts}})
    print(result)


if __name__ == "__main__":
    main()
