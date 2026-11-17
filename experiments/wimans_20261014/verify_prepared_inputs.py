"""Independently verify full stored-order inputs and raw reconstruction examples."""
from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import scipy

from .acquisition import DEFAULT_ROOT, DOCS, manifest, now, sha, write_json
from .raw_io import read_stored_mat
from analysis.prepare_native_histstyle_phase_20260929 import independent_reconstruct
from analysis.prepare_native_multienv import uniform_interpolation


def matrix_sha(value):
    import hashlib
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes(order="C")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    output = args.root / "processed_storage_order_v2"
    audit_path = output / "input_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("status") != "passed_storage_order_v2" or audit.get("full_input_qa_passed") is not True
        or audit.get("completed_rows") != 1782 or audit.get("failures")):
        raise ValueError("Complete 1782-row stored-order input audit required")
    rows = manifest()
    ids = json.loads((output / "sample_ids.json").read_text(encoding="utf-8"))
    if ids != [r["sample_id"] for r in rows]:
        raise ValueError("Processed order differs from pinned manifest")
    for name, receipt in audit["processed_files"].items():
        if sha(output / name) != receipt["sha256"] or (output / name).stat().st_size != receipt["bytes"]:
            raise ValueError("Processed file changed: " + name)
    amp, phase = [np.load(output / name, mmap_mode="r", allow_pickle=False) for name in ("amp.npy", "phase.npy")]
    labels = np.load(output / "label.npy", allow_pickle=False)
    if (amp.shape != (1782, 850, 90) or phase.shape != amp.shape or amp.dtype != np.float32
        or phase.dtype != np.float32 or labels.dtype != np.int64 or labels.shape != (1782,)
        or not np.array_equal(labels, [int(r["class_id"]) for r in rows])):
        raise ValueError("Processed shape/dtype/label identity mismatch")
    records = json.loads((output / "numerical_records.json").read_text(encoding="utf-8"))
    if len(records) != 1782 or [r["sample_id"] for r in records] != ids:
        raise ValueError("Numerical receipt membership/order mismatch")
    for index, record in enumerate(records):
        receipt = json.loads((args.root / "acquisition/records" / (ids[index] + ".json")).read_text(encoding="utf-8"))
        if record["index"] != index or record["raw_sha256"] != receipt["sha256"]:
            raise ValueError("Raw-to-processed receipt mismatch")
        for name, values in (("amp", amp[index]), ("phase", phase[index])):
            if not np.isfinite(values).all() or matrix_sha(values) != record[name + "_matrix_sha256"]:
                raise ValueError("Numerical values/row hash mismatch: " + ids[index])
        if np.min(amp[index]) < 0:
            raise ValueError("Negative amplitude")
    # Predeclared metadata choice: first record per room/class, all 27 cells.
    # This is signal QA only, no models or target scores.
    chosen, seen = [], set()
    for index, row in enumerate(rows):
        key = row["environment"], row["class_id"]
        if key not in seen:
            seen.add(key)
            chosen.append(index)
    assert len(chosen) == 27
    independent = []
    for index in chosen:
        row = rows[index]
        raw = args.root / "raw" / row["raw_relative_path"]
        csi, elapsed, _stats = read_stored_mat(raw)
        refamp = uniform_interpolation(np.abs(csi), elapsed).astype(np.float32)
        refphase = independent_reconstruct(csi, elapsed)
        error_a = float(np.max(np.abs(amp[index] - refamp)))
        error_p = float(np.max(np.abs(phase[index] - refphase)))
        if error_a > 2e-5 or error_p > 2e-5:
            raise ValueError("Independent interpolation/unwrap differs: " + row["sample_id"])
        independent.append({"sample_id": row["sample_id"], "room": row["environment"], "class_id": int(row["class_id"]),
                            "amplitude_max_abs_error": error_a, "phase_max_abs_error": error_p})
    perm_counts = Counter()
    for r in records:
        perm_counts.update(r["perm_counts"])
    summary = {"schema": "wimans-full-prepared-input-verification/v1", "checked_at_utc": now(), "status": "passed",
               "representation_version": "storage_order_v2", "physical_MAT_export_status": "unresolved",
               "verified_rows": 1782, "all_row_matrix_hashes_verified": True, "no_model_evaluation": True,
               "raw_target_access": "27 raw records inspected across room/class metadata cells for input QA only",
               "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
               "input_audit_sha256": sha(audit_path), "processed_files": audit["processed_files"],
               "numerical_records_sha256": sha(output / "numerical_records.json"),
               "software_sha256": {"verifier": sha(Path(__file__)), "raw_reader": sha(Path(__file__).with_name("raw_io.py")),
                                    "independent_phase_reference": sha(Path("analysis/prepare_native_histstyle_phase_20260929.py")),
                                    "scipy_grid_reference": sha(Path("analysis/prepare_native_multienv.py"))},
               "packets_min_max": [min(r["packets"] for r in records), max(r["packets"] for r in records)],
               "duration_seconds_min_max": [min(r["duration_microseconds"] for r in records) / 1e6,
                                             max(r["duration_microseconds"] for r in records) / 1e6],
               "max_packet_gap_microseconds": max(r["max_gap_microseconds"] for r in records),
               "total_duplicate_timestamps": sum(r["duplicate_timestamp_count"] for r in records),
               "total_wraps": sum(r["wrap_count"] for r in records),
               "total_selected_zero_amplitude_values": sum(r["selected_zero_amplitude_values"] for r in records),
               "amp_range": [min(r["amp_min"] for r in records), max(r["amp_max"] for r in records)],
               "phase_range": [min(r["phase_min"] for r in records), max(r["phase_max"] for r in records)],
               "all_packet_perm_counts": dict(perm_counts), "independent_reconstruction_examples": independent,
               "formal_training_freeze_authorized": False, "real_source_short_training_qa_passed": False}
    write_json(DOCS / "prepared_input_verification.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "verified_rows", "packets_min_max", "duration_seconds_min_max",
                                             "max_packet_gap_microseconds", "total_duplicate_timestamps")}), flush=True)


if __name__ == "__main__":
    main()
