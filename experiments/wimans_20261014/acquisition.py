"""Acquire only pinned manifest MAT files; no model evaluation or GPU work."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
import time
import zipfile
from urllib.parse import quote
from datetime import datetime, timezone
from pathlib import Path

import requests
import psutil

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs/revision/wimans_iotj_ojcoms_20261014"
DEFAULT_ROOT = Path("D:/pa-csi-dg-revision-runtime/data/wimans_20261014")
BASE = "https://www.kaggle.com/api/v1/datasets"
VERSION = 1
EXECUTED_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PROBE_HASHES = {
    "act_19_1": "6cb50b8bfee290083bcdaf4cb23db3c82f8972cd4d18a83a04f0b075ce49f6f3",
    "act_55_1": "1f965487d0a495f2f609966a8bfd7cda95e4637b722c616c4c25705c2baa77da",
    "act_91_1": "a3a465e6d2e139d13491713b6746e44cecd85367df226a8ea4c67371945cfc5d",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def manifest():
    with (DOCS / "subset_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1782 or len({r["sample_id"] for r in rows}) != 1782:
        raise ValueError("Manifest size or identity changed")
    return rows


def acquire_one(root, row):
    sid = row["sample_id"]
    path = root / "raw" / row["raw_relative_path"]
    receipt_path = root / "acquisition/records" / (sid + ".json")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["dataset_version"] != VERSION or receipt["relative_path"] != row["raw_relative_path"]:
            raise ValueError("Existing receipt version/identity mismatch")
        if not path.exists() or path.stat().st_size != receipt["bytes"] or sha(path) != receipt["sha256"]:
            raise ValueError("Previously accepted raw bytes changed; preserve and investigate")
        return receipt
    # Initial three saved probes are accepted only against their predeclared hashes.
    existing_digest = sha(path) if path.exists() else None
    if existing_digest is not None and PROBE_HASHES.get(sid) == existing_digest:
        payload = path.read_bytes()
        origin = "saved_probe_reverified"
    else:
        last_error = None
        for attempt in range(1, 5):
            try:
                # Escape the entire nested filename as one path parameter.
                # The legacy query form intermittently returns a 134-byte 404
                # even for files already successfully downloaded.
                endpoint = BASE + "/download/shuokanghuang/wimans/" + quote(row["raw_relative_path"], safe="")
                response = requests.get(endpoint, params={"datasetVersionNumber": VERSION}, timeout=(15, 90))
                response.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    names = archive.namelist()
                    if names != [sid + ".mat"]:
                        raise ValueError("Unexpected download ZIP members")
                    payload = archive.read(names[0])  # Includes the archive CRC check.
                if not payload.startswith(b"MATLAB 5.0 MAT-file"):
                    raise ValueError("Downloaded bytes are not a MATLAB v5 file")
                last_error = None
                break
            except (requests.RequestException, zipfile.BadZipFile, EOFError) as exc:
                last_error = exc
                time.sleep(attempt)
        if last_error is not None:
            raise last_error
        if existing_digest is not None:
            # Recover an interruption between the raw rename and receipt write.
            # Keep the saved bytes; same-version redownload must agree exactly.
            if len(payload) != path.stat().st_size or hashlib.sha256(payload).hexdigest() != existing_digest:
                raise ValueError("Unreceipted saved bytes differ from pinned-version redownload; preserve and investigate")
            origin = "unreceipted_saved_bytes_verified_against_pinned_version_redownload"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".mat.part")
            temp.write_bytes(payload)
            os.replace(temp, path)
            origin = "kaggle_version_1_path_endpoint_individual_file_zip"
    digest = hashlib.sha256(payload).hexdigest()
    if sid in PROBE_HASHES and digest != PROBE_HASHES[sid]:
        raise ValueError("Pinned probe differs")
    receipt = {"sample_id": sid, "relative_path": row["raw_relative_path"], "dataset_version": VERSION,
               "bytes": len(payload), "sha256": digest, "acquired_at_utc": now(), "origin": origin,
               "target_access_kind": "raw_identity_and_numerical_audit_only", "model_scores_read": False}
    write_json(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Preview only; never marks full acquisition complete")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("Use 1..8 download workers")
    root = args.root
    previous = root / "acquisition/status.json"
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        if old.get("status") == "running":
            owner = old.get("owner")
            if owner is None:
                legacy_launch = root / "launch_record.json"
                if not legacy_launch.exists():
                    raise RuntimeError("Legacy running state has no owner record; inspect before recovery")
                owner = json.loads(legacy_launch.read_text(encoding="utf-8-sig"))
            active = psutil.pid_exists(owner["pid"])
            if active and "process_create_time" in owner:
                active = psutil.Process(owner["pid"]).create_time() == owner["process_create_time"]
            if active:
                raise RuntimeError("Another acquisition is running; do not overwrite its status")
        attempt_name = old.get("started_at_utc", now()).replace(":", "-")
        write_json(root / "acquisition/attempts" / (attempt_name + ".json"), old)
    rows = manifest()
    if args.limit:
        rows = rows[:args.limit]
    meta_path = root / "acquisition/kaggle_dataset_metadata.json"
    if not meta_path.exists():
        response = requests.get(BASE + "/view/shuokanghuang/wimans", timeout=30)
        response.raise_for_status()
        metadata = response.json()
        if not any(v["versionNumber"] == VERSION for v in metadata["versions"]):
            raise ValueError("Pinned dataset version unavailable")
        write_json(meta_path, metadata)
    receipts, failures = [], []
    started = now()
    def status(final=False):
        value = {"schema": "wimans-raw-acquisition/v1", "started_at_utc": started, "updated_at_utc": now(),
                 "dataset_ref": "shuokanghuang/wimans", "dataset_version": VERSION,
                 "manifest_sha256": sha(DOCS / "subset_manifest.csv"), "expected_rows": 1782,
                 "requested_rows": len(rows), "completed_rows": len(receipts), "failed_rows": len(failures),
                 "downloaded_bytes": sum(r["bytes"] for r in receipts), "training_started": False,
                 "software_sha256": EXECUTED_CODE_SHA256,
                 "owner": {"pid": os.getpid(), "process_create_time": psutil.Process().create_time()},
                 "complete": final and len(receipts) == 1782 and not failures,
                 "status": "complete" if final and len(receipts) == 1782 and not failures else ("partial" if final else "running"),
                 "failures": failures, "records": sorted(receipts, key=lambda r: r["sample_id"])}
        write_json(root / "acquisition/status.json", value)
    status()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(acquire_one, root, row): row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                receipts.append(future.result())
            except Exception as exc:
                failures.append({"sample_id": row["sample_id"], "error": type(exc).__name__ + ": " + str(exc)})
            if (len(receipts) + len(failures)) % 25 == 0:
                status()
                print(json.dumps({"acquired": len(receipts), "failed": len(failures), "expected": 1782}), flush=True)
    status(True)
    if failures:
        raise RuntimeError("Acquisition has failures; inspect status. No records silently excluded.")


if __name__ == "__main__":
    main()
