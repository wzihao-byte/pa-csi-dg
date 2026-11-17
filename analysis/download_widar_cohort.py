"""Fetch the predeclared public Widar cohort and prepare time-normalized amplitude.

No training, model selection, target prediction, learned normalization or phase
processing occurs here. Each ZIP member is checked against frozen size and CRC,
then bound to a raw SHA256 and a separately stored float32 amplitude array.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import sys
import threading
import time
import urllib.error
import zipfile
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.prepare_widar_inventory import PublicRangeReader, public_download_url
from analysis.audit_multienv_timestamps import decode_dat_csi
from analysis.verify_multienv_raw_mapping import read_dat_timestamps
from analysis.prepare_native_multienv import native_elapsed, uniform_interpolation
from experiments.revision_validation.study_io import bound_path, now, read, sha256, value_hash, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit-per-archive", type=int, default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory = read(args.inventory)
    config = {"schema": "widar-native-amplitude/v1", "inventory_sha256": sha256(args.inventory),
              "limit_per_archive": args.limit_per_archive, "shape": [850, 90], "dtype": "float32",
              "processing": "Intel5300 raw complex CSI in receive-chain/subcarrier order; abs; native uint32 time unroll; duplicate-time average; linear interpolation to 850 equally spaced recording-relative times",
              "rssi_agc_scaling": False, "phase_used": False, "learned_normalization": False,
              "fixed_physical_sampling_frequency": False, "target_model_evaluation": False,
              "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
                  Path(__file__), ROOT / "analysis/prepare_widar_inventory.py",
                  ROOT / "analysis/prepare_native_multienv.py", ROOT / "analysis/audit_multienv_timestamps.py",
                  ROOT / "analysis/verify_multienv_raw_mapping.py")}}
    if (output / "config.json").exists() and read(output / "config.json") != config:
        raise RuntimeError("Changed preprocessing or inventory requires a new output version")
    write(output / "config.json", config)
    for folder in ("raw", "amplitude", "records"):
        (output / folder).mkdir(exist_ok=True)
    config_hash = value_hash(config)
    lock = threading.Lock()
    counts = {row["archive"]: 0 for row in inventory["archives"]}

    def process_archive(archive):
        name = archive["archive"]
        rows = [r for r in inventory["samples"] if r["archive"] == name]
        if args.limit_per_archive:
            rows = rows[:args.limit_per_archive]
        reader = PublicRangeReader(public_download_url("/CSI/" + name), archive["archive_size"])
        results = []
        with zipfile.ZipFile(reader) as zipped:
            for row in rows:
                identifier = hashlib.sha256(row["sample_id"].encode()).hexdigest()
                record_path = output / "records" / (identifier + ".json")
                data_path = bound_path(output / "raw", name.removesuffix(".zip") + "/" + row["name"].replace("\\", "/"))
                amplitude_path = output / "amplitude" / (identifier + ".npy")
                if record_path.exists():
                    previous = read(record_path)
                    if previous["config_value_sha256"] != config_hash:
                        raise RuntimeError("Stored sample configuration differs")
                    if sha256(data_path) != previous["raw_sha256"] or sha256(amplitude_path) != previous["amplitude_sha256"]:
                        raise RuntimeError("Stored sample bytes differ from their hashes")
                    results.append(previous)
                else:
                    info = zipped.getinfo(row["name"])
                    if (info.CRC, info.file_size, info.compress_size, info.header_offset) != (row["crc32"], row["size"], row["compressed_size"], row["header_offset"]):
                        raise RuntimeError("Remote member metadata changed since cohort freeze")
                    if data_path.exists():
                        data = data_path.read_bytes()
                    else:
                        for attempt in range(3):
                            try:
                                data = zipped.read(info)
                                break
                            except urllib.error.HTTPError as exc:
                                if exc.code < 500 or attempt == 2:
                                    raise
                                time.sleep(attempt + 1)
                            except (TimeoutError, urllib.error.URLError):
                                if attempt == 2:
                                    raise
                                time.sleep(attempt + 1)
                        data_path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = data_path.with_suffix(".dat.partial")
                        temporary.write_bytes(data)
                        os.replace(temporary, data_path)
                    if len(data) != row["size"] or (zlib.crc32(data) & 0xffffffff) != row["crc32"]:
                        raise RuntimeError("Raw member length/CRC mismatch")
                    csi = decode_dat_csi(data)
                    timestamps, packet_ids = read_dat_timestamps(data)
                    if csi.shape != (len(timestamps), 90) or len(packet_ids) != len(timestamps):
                        raise RuntimeError("Native decoder packet/feature shape mismatch")
                    elapsed, differences = native_elapsed(timestamps)
                    amplitude = uniform_interpolation(abs(csi), elapsed).astype(np.float32)
                    if not np.isfinite(amplitude).all() or amplitude.shape != (850, 90):
                        raise RuntimeError("Invalid prepared amplitude")
                    temporary = amplitude_path.with_suffix(".npy.partial")
                    with temporary.open("wb") as stream:
                        np.save(stream, amplitude, allow_pickle=False)
                    os.replace(temporary, amplitude_path)
                    record = {**row, "config_value_sha256": config_hash, "raw_path": str(data_path),
                              "raw_sha256": hashlib.sha256(data).hexdigest(), "amplitude_path": str(amplitude_path),
                              "amplitude_sha256": sha256(amplitude_path), "packet_count": len(timestamps),
                              "duration_seconds": float(elapsed[-1] / 1e6),
                              "max_gap_seconds": float(differences.max() / 1e6),
                              "acquisition_id": f"{name}:user{row['subject']}:pos{row['position']}:ori{row['orientation']}:rep{row['repetition']}",
                              "acquisition_rule": "Conservatively group all gestures with the same release/subject/position/orientation/repetition",
                              "target_model_evaluation": False}
                    write(record_path, record)
                    results.append(record)
                counts[name] = len(results)
                if len(results) % 25 == 0 or len(results) == len(rows):
                    with lock:
                        write(output / "status.json", {"status": "preparing", "updated_at": now(),
                              "completed": sum(counts.values()), "by_archive": counts.copy(), "target_model_evaluation": False})
                    print(f"{name}: prepared {len(results)}/{len(rows)}", flush=True)
        return results

    results, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_archive, archive): archive for archive in inventory["archives"]}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as exc:
                failure = {"archive": futures[future]["archive"], "error": repr(exc)}
                failures.append(failure)
                print(f"Archive failed: {failure}", flush=True)
    write(output / "summary.json", {"finished_at": now(), "rows": len(results), "failures": failures,
          "complete_selected_cohort": not args.limit_per_archive and not failures and len(results) == len(inventory["samples"]),
          "target_model_evaluation": False, "config_sha256": sha256(output / "config.json"),
          "inventory_sha256": sha256(args.inventory), "record_sha256": {r["sample_id"]: sha256(output / "records" / (hashlib.sha256(r["sample_id"].encode()).hexdigest() + ".json")) for r in results}})
    if failures:
        raise RuntimeError("Cohort is incomplete; inspect failures without outcome-based exclusions")


if __name__ == "__main__":
    main()
