"""Prepare phase controls only from already verified native MultiEnv DAT bytes."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.audit_multienv_timestamps import decode_dat_csi
from analysis.native_phase_processing import VERSION, VARIANTS, RX_PAIRS, build_phase_variants
from analysis.prepare_native_multienv import native_elapsed
from analysis.verify_multienv_raw_mapping import read_dat_timestamps
from experiments.revision_validation.study_io import now, read, sha256, value_hash, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-prepared", type=Path, required=True)
    parser.add_argument("--qa-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Allows only the existing 12-row native smoke; never marks full preparation complete")
    args = parser.parse_args()
    parent, output = args.native_prepared.resolve(), args.output.resolve()
    summary, parent_config = read(parent / "summary.json"), read(parent / "config.json")
    expected_rows = 12 if args.smoke else 9000
    if summary["rows"] != expected_rows or summary.get("failures") or not summary.get("all_packet_signal_matches"):
        raise RuntimeError("Native parent has not passed the required record/CSI gate")
    if not args.smoke and not summary.get("complete_9000"):
        raise RuntimeError("Full native preparation is required before full phase preparation")
    if summary["config_value_sha256"] != value_hash(parent_config):
        raise RuntimeError("Native parent configuration changed")
    qa = read(args.qa_report)
    if not qa.get("passed") or "pi_boundary_shared_error_invariance" not in qa:
        raise RuntimeError("Require the phase QA including the explicit pi boundary")
    for relative, expected in qa["code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Phase QA code changed: {relative}")
    code_files = {**qa["code_sha256"], "analysis/prepare_native_phase.py": sha256(__file__)}
    config = {"version": VERSION, "smoke": args.smoke, "native_prepared": str(parent),
              "native_summary_sha256": sha256(parent / "summary.json"), "phase_qa_sha256": sha256(args.qa_report),
              "code_sha256": code_files, "variants": list(VARIANTS), "rx_pairs": [list(pair) for pair in RX_PAIRS],
              "complete_hardware_calibration": False, "classifier_training_or_selection": False}
    if (output / "config.json").exists() and read(output / "config.json") != config:
        raise RuntimeError("Phase preparation inputs changed; use a separately versioned output")
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("records", "samples", "failures"):
        (output / folder).mkdir(exist_ok=True)
    write(output / "config.json", config)
    config_hash = value_hash(config)
    records = []
    for identifier, expected_hash in sorted(summary["record_sha256"].items()):
        path = parent / "records" / (Path(identifier).stem + ".json")
        if sha256(path) != expected_hash:
            raise RuntimeError("Parent record changed")
        row = read(path)
        if row["mat_path"] != identifier or row["config_value_sha256"] != value_hash(parent_config):
            raise RuntimeError("Parent row identity/configuration mismatch")
        if not row.get("verified") or not row.get("all_complex_csi_and_packet_ids_equal"):
            raise RuntimeError("Parent lacks packet-level CSI verification")
        records.append(row)
    if len(records) != expected_rows:
        raise RuntimeError("Parent summary row count differs")

    def process(row):
        stem = Path(row["mat_path"]).stem
        record_path, sample_path = output / "records" / (stem + ".json"), output / "samples" / (stem + ".npz")
        raw_path = parent / "dat_cache" / (stem + ".dat")
        if sha256(raw_path) != row["dat_sha256"]:
            raise RuntimeError("Verified native raw bytes changed")
        if record_path.exists():
            record = read(record_path)
            if record["config_value_sha256"] != config_hash or record["raw_sha256"] != row["dat_sha256"]:
                raise RuntimeError("Phase record version/identity changed")
            if sha256(sample_path) != record["sample_sha256"]:
                raise RuntimeError("Phase sample bytes changed")
            return record
        data = raw_path.read_bytes()
        csi = decode_dat_csi(data)
        ticks, _ = read_dat_timestamps(data)
        elapsed, _ = native_elapsed(ticks)
        if len(csi) != row["packet_count"] or elapsed[-1] / 1e6 != row["duration_seconds"]:
            raise RuntimeError("Native packet count/duration does not match parent")
        values, diagnostics = build_phase_variants(csi, elapsed)
        temporary = sample_path.with_suffix(".npz.partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **values)
        os.replace(temporary, sample_path)
        record = {key: row[key] for key in ("environment", "index", "mat_path", "dat_path", "subject_id", "acquisition_id")}
        record.update(config_value_sha256=config_hash, parent_record_sha256=sha256(parent / "records" / (stem + ".json")),
                      raw_path=str(raw_path), raw_sha256=row["dat_sha256"], sample_path=str(sample_path),
                      sample_sha256=sha256(sample_path), diagnostics=diagnostics, packet_count=len(csi),
                      complete_hardware_calibration=False)
        write(record_path, record)
        return record

    completed, failures = [], []
    write(output / "status.json", {"status": "preparing", "updated_at": now(), "completed": 0, "total": expected_rows})
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        tasks = {executor.submit(process, row): row for row in records}
        for task in concurrent.futures.as_completed(tasks):
            row = tasks[task]
            try:
                completed.append(task.result())
            except Exception:
                failure = {"raw_identity": row["dat_path"], "at": now(), "error": traceback.format_exc(), "config_value_sha256": config_hash}
                failures.append(failure)
                stem = Path(row["mat_path"]).stem
                number = len(list((output / "failures").glob(stem + ".attempt*.json"))) + 1
                write(output / "failures" / f"{stem}.attempt{number:04d}.json", failure)
            done = len(completed) + len(failures)
            if done % 100 == 0 or done == expected_rows:
                write(output / "status.json", {"status": "preparing", "updated_at": now(), "completed": len(completed), "failed": len(failures), "total": expected_rows})
                print(f"Prepared phase {len(completed)}/{expected_rows}; failures {len(failures)}", flush=True)
    aggregate = {}
    for variant in VARIANTS:
        details = [row["diagnostics"][variant] for row in completed]
        aggregate[variant] = {"zero_magnitude_elements": sum(r["raw_channel_zero_magnitude_count"] for r in details),
            "channel_elements": sum(r["raw_channel_element_count"] for r in details),
            "circular_cancellation_bins": sum(r["circular_cancellation_bins"] for r in details),
            "near_pi_steps": sum(r["wrapped_adjacent_steps_within_1e_minus6_of_pi"] for r in details),
            "max_usable_time_gap_seconds": max((r["max_usable_time_gap_seconds"] for r in details), default=None),
            "max_endpoint_constant_extension_seconds": max((r["max_endpoint_constant_extension_seconds"] for r in details), default=None),
            "max_abs_centered_phase": max((r["max_absolute_centered_phase"] for r in details), default=None)}
    passed = len(completed) == expected_rows and not failures
    final = {"completed_at": now(), "complete_9000": passed and not args.smoke, "smoke": args.smoke,
             "rows": len(completed), "failures": failures, "config_value_sha256": config_hash,
             "raw_signal_identity_parent_verified": True, "classifier_training_or_selection": False,
             "record_sha256": {r["mat_path"]: sha256(output / "records" / (Path(r["mat_path"]).stem + ".json")) for r in sorted(completed, key=lambda r:r["mat_path"])},
             "aggregate_diagnostics": aggregate, "complete_hardware_calibration": False}
    write(output / "summary.json", final)
    write(output / "status.json", {"status": "completed" if passed else "failed", "updated_at": now(), "completed": len(completed), "failed": len(failures), "total": expected_rows})
    if not passed:
        raise RuntimeError("Phase preparation failed; failures preserved and no cohort exclusions applied")
    print({"rows": len(completed), "complete_9000": final["complete_9000"]}, flush=True)


if __name__ == "__main__":
    main()
