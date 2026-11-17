"""Prepare the registered stored-order input; no physical export claim."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.wimans_20261014.acquisition import DEFAULT_ROOT, DOCS, manifest, now, sha, write_json
from experiments.wimans_20261014.raw_io import read_stored_mat, reconstruct


def matrix_sha(value):
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes(order="C")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--samples", nargs="*", help="Explicit preview identities; full preparation requires no filter")
    args = parser.parse_args()
    protocol_path = DOCS / "axis_storage_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (protocol.get("status") != "registered" or protocol.get("representation_version") != "storage_order_v2"
        or protocol.get("target_model_scores_accessed_before_registration") is not False
        or protocol.get("perm_reapplied") is not False or protocol.get("selected_first_stored_axis_index") != 0):
        raise ValueError("Stored-order protocol is not registered before target scoring")
    rows = manifest()
    if args.samples:
        selected = set(args.samples)
        rows = [r for r in rows if r["sample_id"] in selected]
        if len(rows) != len(selected):
            raise ValueError("Preview identity absent from manifest")
    preview = bool(args.samples)
    output = args.root / ("preview_storage_order_v2" if preview else "processed_storage_order_v2")
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / name for name in ("amp.npy", "phase.npy", "label.npy", "sample_ids.json")}
    shape = (len(rows), 850, 90)
    arrays = {}
    for name in ("amp.npy", "phase.npy"):
        if paths[name].exists():
            arrays[name] = np.load(paths[name], mmap_mode="r+")
            if arrays[name].shape != shape or arrays[name].dtype != np.float32:
                raise ValueError("Existing candidate layout changed")
        else:
            arrays[name] = np.lib.format.open_memmap(paths[name], mode="w+", dtype=np.float32, shape=shape)
    labels = np.array([int(r["class_id"]) for r in rows], dtype=np.int64)
    np.save(paths["label.npy"], labels)
    write_json(paths["sample_ids.json"], [r["sample_id"] for r in rows])
    completed, failures = [], []
    def audit(final=False):
        numerical_passed = final and len(completed) == 1782 and not failures and not preview
        value = {"schema": "wimans-candidate-input-audit/v1", "updated_at_utc": now(),
                 "status": "passed_storage_order_v2" if numerical_passed else ("failed" if failures else ("preview_only" if final else "preparing")),
                 "full_input_qa_passed": numerical_passed, "numerical_input_qa_passed": numerical_passed,
                 "representation_version": "storage_order_v2",
                 "axis_storage_protocol": {"path": str(protocol_path), "sha256": sha(protocol_path)},
                 "phase_representation": "native_historical_feature_unwrap_v1", "loader_unwrap_phase": False,
                 "expected_rows": 1782, "candidate_rows": len(rows), "completed_rows": len(completed),
                 "preview": preview, "failures": failures,
                 "axis_export_verification": {"status": "storage_order_protocol_registered", "physical_export_status": "unresolved", "perm_reapplied": False,
                      "candidate_selection": "stored csi first axis index 0, all second-axis groups, C-order 90 columns",
                      "physical_tx_index_confirmed": False,
                      "reason": "Registered stored-order protocol; author physical export/perm provenance remains unresolved"},
                 "manifest_sha256": sha(DOCS / "subset_manifest.csv"),
                 "fold_manifest_sha256": sha(DOCS / "fold_manifest.json"),
                 "raw_data_access": "All prepared identities may be target in one fold; raw numerical/identity audit only",
                 "model_scores_read": False, "training_started": False,
                 "software_sha256": {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name("raw_io.py")]},
                 "processed_files": {}}
        if final:
            value["processed_files"] = {name: {"sha256": sha(path), "bytes": path.stat().st_size} for name, path in paths.items()}
        write_json(output / "input_audit.json", value)
    audit()
    for index, row in enumerate(rows):
        sid = row["sample_id"]
        raw = args.root / "raw" / row["raw_relative_path"]
        receipt_path = args.root / "acquisition/records" / (sid + ".json")
        record_path = output / "records" / (sid + ".json")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["sample_id"] != sid or sha(raw) != receipt["sha256"]:
                raise ValueError("Raw receipt/hash mismatch")
            if record_path.exists():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                if (record["raw_sha256"] != receipt["sha256"] or record["index"] != index
                    or record["software_sha256"] != {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name("raw_io.py")]}):
                    raise ValueError("Existing preprocessing receipt changed; create a new candidate version")
                for name in ("amp", "phase"):
                    if matrix_sha(arrays[name + ".npy"][index]) != record[name + "_matrix_sha256"]:
                        raise ValueError("Previously prepared candidate row changed")
            else:
                csi, elapsed, stats = read_stored_mat(raw)
                amp, phase = reconstruct(csi, elapsed)
                arrays["amp.npy"][index] = amp
                arrays["phase.npy"][index] = phase
                # Flush before recording acceptance, making interruptions recoverable.
                for array in arrays.values():
                    array.flush()
                record = {**row, **stats, "index": index, "raw_sha256": receipt["sha256"],
                          "amp_matrix_sha256": matrix_sha(amp), "phase_matrix_sha256": matrix_sha(phase),
                          "amp_min": float(amp.min()), "amp_max": float(amp.max()),
                          "phase_min": float(phase.min()), "phase_max": float(phase.max()),
                          "prepared_at_utc": now(), "model_scores_read": False,
                          "software_sha256": {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name("raw_io.py")]}}
                write_json(record_path, record)
            completed.append(record)
        except Exception as exc:
            failures.append({"sample_id": sid, "error": type(exc).__name__ + ": " + str(exc),
                             "excluded_from_protocol": False})
        if (index + 1) % 25 == 0:
            audit()
            print(json.dumps({"prepared": len(completed), "failed": len(failures), "total": len(rows)}), flush=True)
    for array in arrays.values():
        array.flush()
    write_json(output / "numerical_records.json", completed)
    audit(True)
    if failures:
        raise RuntimeError("Numerical QA incomplete; no records silently excluded")
    print(json.dumps({"output": str(output), "prepared": len(completed), "formal_freeze_allowed": False}), flush=True)


if __name__ == "__main__":
    main()
