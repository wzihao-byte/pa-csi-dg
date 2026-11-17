"""Resume the predeclared Widar cohort with per-record corruption accounting.

The failed v1 output remains immutable evidence. Valid cached bytes/arrays are
rehash-verified and hard-linked into this new version. Expired public download
sessions are refreshed; a member whose inventory-bound bytes cannot be decoded
is recorded as a signal exclusion and is never replaced based on model outcome.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import traceback
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


DEPENDENCIES = (
    "analysis/download_widar_cohort_v2.py", "analysis/prepare_widar_inventory.py",
    "analysis/prepare_native_multienv.py", "analysis/audit_multienv_timestamps.py",
    "analysis/verify_multienv_raw_mapping.py", "experiments/revision_validation/study_io.py",
)


def verified_bytes(path, row):
    if not path.exists():
        return None
    data = path.read_bytes()
    if len(data) != row["size"] or (zlib.crc32(data) & 0xffffffff) != row["crc32"]:
        raise RuntimeError("Cached raw bytes differ from the frozen inventory")
    return data


def link_verified(source, destination, expected):
    if sha256(source) != expected:
        raise RuntimeError("Parent cache artifact hash mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != expected:
            raise RuntimeError("Existing v2 cache artifact differs")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    if sha256(destination) != expected:
        raise RuntimeError("Copied v2 cache artifact hash mismatch")


class RefreshingArchive:
    def __init__(self, archive, retries=6):
        self.archive = archive
        self.retries = retries
        self.zipped = None

    def close(self):
        if self.zipped is not None:
            self.zipped.close()
            self.zipped = None

    def refresh(self):
        self.close()
        reader = PublicRangeReader(public_download_url("/CSI/" + self.archive["archive"]), self.archive["archive_size"])
        self.zipped = zipfile.ZipFile(reader)

    def read(self, row):
        errors = []
        for attempt in range(self.retries):
            try:
                if self.zipped is None:
                    self.refresh()
                info = self.zipped.getinfo(row["name"])
                actual = (info.CRC, info.file_size, info.compress_size, info.header_offset)
                expected = (row["crc32"], row["size"], row["compressed_size"], row["header_offset"])
                if actual != expected:
                    raise RuntimeError("Remote member metadata changed since cohort inventory")
                return self.zipped.read(info), attempt, errors
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError,
                    zipfile.BadZipFile, OSError) as exc:
                errors.append(repr(exc))
                self.close()
                if attempt + 1 == self.retries:
                    raise RuntimeError(f"Public archive read failed after {self.retries} refreshed sessions: {errors}") from exc
                time.sleep(min(attempt + 1, 5))
        raise AssertionError("unreachable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--parent-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    inventory_path, parent, output = args.inventory.resolve(), args.parent_cache.resolve(), args.output.resolve()
    inventory = read(inventory_path)
    parent_summary = read(parent / "summary.json")
    config = {"schema": "widar-native-amplitude/v2", "inventory_sha256": sha256(inventory_path),
              "parent_failed_version": str(parent), "parent_config_sha256": sha256(parent / "config.json"),
              "parent_failed_summary_sha256": sha256(parent / "summary.json"),
              "shape": [850, 90], "dtype": "float32",
              "processing": "Intel5300 complex CSI in receive-chain/subcarrier order; abs; native uint32 time unroll; duplicate-time average; linear interpolation to 850 equally spaced recording-relative times",
              "corruption_rule": "Inventory-bound nonempty bytes that fail DAT decode/time/finite checks are individually recorded and not replaced",
              "network_rule": "HTTP/session failures refresh the official public URL and are not sample exclusions",
              "rssi_agc_scaling": False, "phase_used": False, "learned_normalization": False,
              "fixed_physical_sampling_frequency": False, "target_model_evaluation": False,
              "code_sha256": {name: sha256(ROOT / name) for name in DEPENDENCIES}}
    if not parent_summary.get("failures") or parent_summary.get("complete_selected_cohort"):
        raise RuntimeError("Parent must be the retained incomplete v1 attempt")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists() and read(output / "config.json") != config:
        raise RuntimeError("Changed recovery inputs require a new v2 directory")
    write(output / "config.json", config)
    for folder in ("raw", "amplitude", "records", "signal_exclusions", "attempts"):
        (output / folder).mkdir(exist_ok=True)
    if (output / "summary.json").exists():
        previous = output / "summary.json"
        retained = output / "attempts" / ("summary_" + hashlib.sha256(previous.read_bytes()).hexdigest()[:16] + ".json")
        if not retained.exists():
            shutil.copy2(previous, retained)
    config_hash = value_hash(config)
    lock = threading.Lock()
    counts = {row["archive"]: {"valid": 0, "excluded": 0} for row in inventory["archives"]}

    def process_archive(archive):
        name = archive["archive"]
        rows = [row for row in inventory["samples"] if row["archive"] == name]
        remote = RefreshingArchive(archive)
        valid, excluded, refreshed_sessions = [], [], 0
        try:
            for row in rows:
                identifier = hashlib.sha256(row["sample_id"].encode()).hexdigest()
                record_path = output / "records" / (identifier + ".json")
                exclusion_path = output / "signal_exclusions" / (identifier + ".json")
                raw_relative = name.removesuffix(".zip") + "/" + row["name"].replace("\\", "/")
                raw_path = bound_path(output / "raw", raw_relative)
                amplitude_path = output / "amplitude" / (identifier + ".npy")
                if record_path.exists():
                    record = read(record_path)
                    if record["config_value_sha256"] != config_hash or record["sample_id"] != row["sample_id"]:
                        raise RuntimeError("Stored v2 record binding differs")
                    if sha256(record["raw_path"]) != record["raw_sha256"] or sha256(record["amplitude_path"]) != record["amplitude_sha256"]:
                        raise RuntimeError("Stored v2 record artifact differs")
                    valid.append(record)
                elif exclusion_path.exists():
                    record = read(exclusion_path)
                    if record["config_value_sha256"] != config_hash or record["sample_id"] != row["sample_id"] or sha256(record["raw_path"]) != record["raw_sha256"]:
                        raise RuntimeError("Stored v2 signal exclusion differs")
                    excluded.append(record)
                else:
                    parent_record = parent / "records" / (identifier + ".json")
                    parent_raw = bound_path(parent / "raw", raw_relative)
                    parent_amplitude = parent / "amplitude" / (identifier + ".npy")
                    if parent_record.exists():
                        old = read(parent_record)
                        if old["sample_id"] != row["sample_id"]:
                            raise RuntimeError("Parent record identity differs")
                        link_verified(Path(old["raw_path"]), raw_path, old["raw_sha256"])
                        link_verified(Path(old["amplitude_path"]), amplitude_path, old["amplitude_sha256"])
                        amplitude = np.load(amplitude_path, mmap_mode="r", allow_pickle=False)
                        if amplitude.shape != (850, 90) or amplitude.dtype != np.float32 or not np.isfinite(amplitude).all():
                            raise RuntimeError("Parent amplitude fails v2 shape/dtype/finite validation")
                        record = {**old, "config_value_sha256": config_hash, "raw_path": str(raw_path),
                                  "amplitude_path": str(amplitude_path), "reused_parent_record_sha256": sha256(parent_record),
                                  "target_model_evaluation": False}
                        write(record_path, record)
                        valid.append(record)
                    else:
                        data = verified_bytes(parent_raw, row)
                        read_attempt = 0
                        read_errors = []
                        if data is None:
                            data, read_attempt, read_errors = remote.read(row)
                            refreshed_sessions += read_attempt
                        if len(data) != row["size"] or (zlib.crc32(data) & 0xffffffff) != row["crc32"]:
                            raise RuntimeError("Downloaded bytes differ from the frozen inventory")
                        raw_path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = raw_path.with_suffix(".dat.partial")
                        temporary.write_bytes(data)
                        os.replace(temporary, raw_path)
                        try:
                            csi = decode_dat_csi(data)
                            timestamps, packet_ids = read_dat_timestamps(data)
                            if csi.shape != (len(timestamps), 90) or len(packet_ids) != len(timestamps):
                                raise ValueError("Native decoder packet/feature shape mismatch")
                            elapsed, differences = native_elapsed(timestamps)
                            amplitude = uniform_interpolation(abs(csi), elapsed).astype(np.float32)
                            if amplitude.shape != (850, 90) or not np.isfinite(amplitude).all():
                                raise ValueError("Prepared amplitude is invalid")
                        except (AssertionError, ValueError) as exc:
                            exclusion = {**row, "config_value_sha256": config_hash, "raw_path": str(raw_path),
                                         "raw_sha256": hashlib.sha256(data).hexdigest(),
                                         "reason": "inventory-bound nonempty DAT fails signal/time decoding",
                                         "error_type": type(exc).__name__, "error": str(exc),
                                         "traceback": traceback.format_exc(), "replacement_selected": False,
                                         "target_model_evaluation": False}
                            write(exclusion_path, exclusion)
                            excluded.append(exclusion)
                        else:
                            temporary = amplitude_path.with_suffix(".npy.partial")
                            with temporary.open("wb") as stream:
                                np.save(stream, amplitude, allow_pickle=False)
                            os.replace(temporary, amplitude_path)
                            record = {**row, "config_value_sha256": config_hash, "raw_path": str(raw_path),
                                      "raw_sha256": hashlib.sha256(data).hexdigest(), "amplitude_path": str(amplitude_path),
                                      "amplitude_sha256": sha256(amplitude_path), "packet_count": len(timestamps),
                                      "duration_seconds": float(elapsed[-1] / 1e6),
                                      "max_gap_seconds": float(differences.max() / 1e6),
                                      "acquisition_id": f"{name}:user{row['subject']}:pos{row['position']}:ori{row['orientation']}:rep{row['repetition']}",
                                      "acquisition_rule": "Group gestures with the same release/subject/position/orientation/repetition",
                                      "public_session_refreshes_before_read": read_attempt,
                                      "public_read_prior_errors": read_errors, "target_model_evaluation": False}
                            write(record_path, record)
                            valid.append(record)
                counts[name] = {"valid": len(valid), "excluded": len(excluded)}
                accounted = len(valid) + len(excluded)
                if accounted % 25 == 0 or accounted == len(rows):
                    with lock:
                        write(output / "status.json", {"status": "preparing", "updated_at": now(),
                              "accounted": sum(item["valid"] + item["excluded"] for item in counts.values()),
                              "by_archive": counts.copy(), "target_model_evaluation": False})
                    print(f"{name}: accounted {accounted}/{len(rows)}; valid {len(valid)}; excluded {len(excluded)}", flush=True)
        finally:
            remote.close()
        return valid, excluded, refreshed_sessions

    valid, exclusions, failures, refreshes = [], [], [], {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_archive, archive): archive for archive in inventory["archives"]}
        for future in concurrent.futures.as_completed(futures):
            try:
                rows, omitted, refreshed = future.result()
                valid.extend(rows)
                exclusions.extend(omitted)
                refreshes[futures[future]["archive"]] = refreshed
            except BaseException:
                failure = {"archive": futures[future]["archive"], "traceback": traceback.format_exc()}
                failures.append(failure)
                print(f"Archive failed: {failure['archive']}", flush=True)
    accounted = len(valid) + len(exclusions)
    summary = {"finished_at": now(), "schema": "widar-native-amplitude/v2-summary",
          "inventory_nonempty_rows": len(inventory["samples"]), "valid_rows": len(valid), "rows": len(valid),
          "signal_exclusions": sorted(exclusions, key=lambda row: row["sample_id"]), "failures": failures,
          "complete_inventory_accounting": not failures and accounted == len(inventory["samples"]),
          "complete_selected_cohort": not failures and accounted == len(inventory["samples"]),
          "replacement_samples_added": 0, "network_sessions_refreshed": refreshes,
          "target_model_evaluation": False, "config_sha256": sha256(output / "config.json"),
          "inventory_sha256": sha256(inventory_path),
          "record_sha256": {row["sample_id"]: sha256(output / "records" / (hashlib.sha256(row["sample_id"].encode()).hexdigest() + ".json")) for row in valid},
          "exclusion_sha256": {row["sample_id"]: sha256(output / "signal_exclusions" / (hashlib.sha256(row["sample_id"].encode()).hexdigest() + ".json")) for row in exclusions}}
    write(output / "summary.json", summary)
    if failures or not summary["complete_inventory_accounting"]:
        raise RuntimeError("Widar v2 inventory accounting is incomplete; preserve and resume this version")
    print({"accounted": accounted, "valid": len(valid), "signal_exclusions": len(exclusions), "target_model_evaluation": False})


if __name__ == "__main__":
    main()
