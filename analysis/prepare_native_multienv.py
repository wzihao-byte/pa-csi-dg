"""Build separately versioned amplitude controls from native DAT timestamps.

The frozen historical arrays are never modified. The first control changes only
the timestamps in the historical merge; the second also replaces that merge by
linear interpolation on 850 uniformly spaced within-recording times. Neither
implies a fixed sampling frequency or recovers the missing historical export log.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.audit_multienv_timestamps import decode_dat_csi
from analysis.verify_multienv_raw_mapping import legacy_merge, read_dat_timestamps
from experiments.revision_validation.study_io import now, read, sha256, value_hash, write

VERSION = "native-dat-amplitude-controls-v1"


def native_elapsed(timestamps):
    values = timestamps.astype(np.int64)
    differences = (np.diff(values) + 2 ** 32) % (2 ** 32)
    if not len(differences) or np.any(differences >= 2 ** 31):
        raise ValueError("Ambiguous backward clock movement/reset; do not silently repair")
    elapsed = np.r_[0, differences.cumsum()].astype(np.float64)
    if elapsed[-1] <= 0:
        raise ValueError("Recording has no positive duration")
    return elapsed, differences


def uniform_interpolation(values, elapsed):
    unique, inverse, counts = np.unique(elapsed, return_inverse=True, return_counts=True)
    if len(unique) < 2:
        raise ValueError("At least two distinct timestamps are required")
    averaged = np.zeros((len(unique), values.shape[1]), dtype=np.float64)
    np.add.at(averaged, inverse, values)
    averaged /= counts[:, None]
    grid = np.linspace(0.0, elapsed[-1], 850)
    return interp1d(unique, averaged, axis=0, kind="linear", bounds_error=True)(grid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 processes all rows; a positive value is a smoke only")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = ROOT / "docs/revision/sources"
    inventory = read(sources / "raw_dat_inventory.json")
    dat_files = {row["path"]: row for row in inventory["files"]}
    if len(dat_files) != 9000:
        raise RuntimeError("Need the complete pinned DAT inventory")
    config = {"version": VERSION, "source_commit": inventory["commit"], "limit": args.limit,
              "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
                  Path(__file__), ROOT / "analysis/audit_multienv_timestamps.py",
                  ROOT / "analysis/verify_multienv_raw_mapping.py")},
              "controls": ["historical_merge_native_uint32_time", "linear_amplitude_native_uint32_time"],
              "length": 850, "dtype": "float32", "timestamp_units": "microseconds",
              "clock_policy": "modulo-2^32 forward deltas; reject steps >=2^31; aggregate duplicate timestamps for interpolation",
              "no_phase_calibration_claim": True, "fixed_physical_sampling_rate": False}
    if (output / "config.json").exists() and read(output / "config.json") != config:
        raise RuntimeError("Preprocessing version changed; use a new output directory")
    write(output / "config.json", config)
    config_hash = value_hash(config)
    for name in ("dat_cache", "records", "samples"):
        (output / name).mkdir(exist_ok=True)
    records = []
    for env in ("E1", "E2", "E3"):
        sidecar = read(ROOT / f"docs/revision/evidence/subject_sidecars/{env}.json")
        for index, path in enumerate(sidecar["raw_paths"]):
            records.append({"environment": env, "index": index, "mat_path": path,
                            "subject_id": sidecar["subject_ids"][index],
                            "acquisition_id": sidecar["acquisition_ids"][index]})
    if args.limit:
        # Include both originally problematic records, as well as ordinary rows.
        examples = {"E1_S03_C04_A01_T09.mat", "E1_S05_C04_A01_T16.mat"}
        selected = [r for r in records if Path(r["mat_path"]).name in examples]
        selected += [records[int(i)] for i in np.linspace(0, len(records) - 1, max(1, args.limit - len(selected)))]
        records = list({r["mat_path"]: r for r in selected}.values())

    def process(row):
        stem = Path(row["mat_path"]).stem
        record_path = output / "records" / f"{stem}.json"
        sample_path = output / "samples" / f"{stem}.npz"
        if record_path.exists():
            previous = read(record_path)
            if previous.get("verified") and previous["config_value_sha256"] == config_hash:
                if sha256(sample_path) == previous["sample_sha256"]:
                    return previous
                raise RuntimeError("Prepared sample hash mismatch")
        dat_relative = "DAT/" + row["environment"] + "/" + stem + ".dat"
        expected = dat_files[dat_relative]
        local = output / "dat_cache" / (stem + ".dat")
        if not local.exists():
            url = "https://raw.githubusercontent.com/lcsig/Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments/" + inventory["commit"] + "/" + dat_relative
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=40) as response:
                        data = response.read()
                    temporary = local.with_suffix(".dat.partial")
                    temporary.write_bytes(data)
                    os.replace(temporary, local)
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code < 500 or attempt == 2:
                        raise
                    time.sleep(attempt + 1)
                except (TimeoutError, urllib.error.URLError):
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        data = local.read_bytes()
        blob_hash = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if len(data) != expected["size"] or blob_hash != expected["sha"]:
            raise RuntimeError("DAT bytes differ from the pinned Git blob")
        mapping = read(ROOT / "docs/revision/evidence/raw_mapping" / (stem + ".mat.json"))
        mat_path = sources / "raw_mat_cache" / (stem + ".mat")
        if sha256(mat_path) != mapping["raw_sha256"]:
            raise RuntimeError("Verified MAT cache hash mismatch")
        packets = loadmat(io.BytesIO(mat_path.read_bytes()), simplify_cells=True)["data"]
        original = np.stack([r["csi"].reshape(90) for r in packets])
        native = decode_dat_csi(data)
        ticks, counts = read_dat_timestamps(data)
        if not np.array_equal(native, original) or not np.array_equal(counts, np.asarray([r["bfee_count"] for r in packets])):
            raise RuntimeError("DAT and verified MAT packet identities/complex CSI differ")
        elapsed, differences = native_elapsed(ticks)
        amplitude = abs(native)
        merged = legacy_merge(amplitude, elapsed)
        interpolated = uniform_interpolation(amplitude, elapsed)
        if merged.shape != (850, 90) or interpolated.shape != (850, 90):
            raise RuntimeError("Wrong reconstructed shape")
        if not np.isfinite(merged).all() or not np.isfinite(interpolated).all():
            raise ValueError("Nonfinite prepared amplitude")
        temporary = sample_path.with_suffix(".npz.partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, historical_merge_native_time=merged.astype(np.float32),
                                linear_native_time=interpolated.astype(np.float32))
        os.replace(temporary, sample_path)
        result = {**row, "verified": True, "config_value_sha256": config_hash,
                  "dat_path": dat_relative, "dat_sha256": hashlib.sha256(data).hexdigest(), "dat_git_blob_sha1": blob_hash,
                  "mat_sha256": mapping["raw_sha256"], "sample_sha256": sha256(sample_path),
                  "all_complex_csi_and_packet_ids_equal": True, "packet_count": len(native),
                  "duration_seconds": float(elapsed[-1] / 1e6), "max_gap_seconds": float(differences.max() / 1e6),
                  "duplicate_timestamp_steps": int(np.count_nonzero(differences == 0)),
                  "uint32_wraps": int(np.count_nonzero(np.diff(ticks.astype(np.int64)) < 0)),
                  "sample_path": str(sample_path), "label": mapping["activity_id"]}
        # The original activity ID is metadata; training labels come from the
        # verified six-class sidecars/manifests, never from this 12-ID field.
        result["original_activity_id"] = result.pop("label")
        write(record_path, result)
        return result

    completed, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, row): row for row in records}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                failures.append({**row, "error": repr(exc)})
            count = len(completed) + len(failures)
            if count % 100 == 0 or count == len(records):
                write(output / "status.json", {"status": "preparing", "updated_at": now(),
                      "completed": len(completed), "failed": len(failures), "total": len(records)})
                print(f"Prepared {len(completed)}/{len(records)}; failures {len(failures)}", flush=True)
    write(output / "summary.json", {"version": VERSION, "completed_at": now(), "config_value_sha256": config_hash,
          "complete_9000": len(completed) == 9000 and not failures, "rows": len(completed), "failures": failures,
          "all_packet_signal_matches": all(r["all_complex_csi_and_packet_ids_equal"] for r in completed),
          "duration_seconds_min_max": [min(r["duration_seconds"] for r in completed), max(r["duration_seconds"] for r in completed)] if completed else None,
          "record_sha256": {r["mat_path"]: sha256(output / "records" / (Path(r["mat_path"]).stem + ".json")) for r in completed}})
    if failures:
        raise RuntimeError(f"{len(failures)} records failed; see retained summary")


if __name__ == "__main__":
    main()
