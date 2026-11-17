"""Verify processed MultiEnv rows against public, commit-pinned original MAT files.

The legacy timestamp averaging is reproduced for provenance only. It averages
wrapped angles arithmetically and is NOT a phase calibration recommendation.
Subject sidecars are emitted only after all 9,000 rows match original signals.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import re
import shutil
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/revision/evidence/raw_mapping"
SOURCE = ROOT / "docs/revision/sources"
LABELS = {1:0, 2:1, 3:0, 4:0, 5:1, 6:2, 7:4, 8:2, 9:4, 10:3, 11:3, 12:5}
VERIFIER_VERSION = "legacy-replay-v3-timestamp-text15"


def timestamp_text15(timestamps):
    """Reproduce the precision-equivalent timestamp text path, not calibration.

    Some released MAT timestamps are near 2**64 and have already lost precision.
    Formatting them to 15 significant decimal digits reproduces the two initial
    unmatched processed rows in every amplitude and phase element. This does not
    assert that an unavailable historical CSV/export log has been recovered.
    Ordinary 32-bit integer timestamps are unchanged by this operation.
    """
    return np.asarray([float(format(float(value), ".15g")) for value in timestamps])


def read_dat_timestamps(data):
    """Read native uint32 timestamps and packet IDs, following read_bf_file.m."""
    offset, timestamps, counts = 0, [], []
    while offset < len(data):
        if offset + 3 > len(data):
            raise ValueError("Truncated DAT record header")
        size = struct.unpack_from(">H", data, offset)[0]
        if size < 1 or offset + 2 + size > len(data):
            raise ValueError("Truncated or invalid DAT record")
        code = data[offset + 2]
        record = data[offset + 3:offset + 2 + size]
        offset += 2 + size
        if code == 187:
            if len(record) < 20:
                raise ValueError("Truncated beamforming record")
            timestamps.append(struct.unpack_from("<I", record)[0])
            counts.append(struct.unpack_from("<H", record, 4)[0])
    return np.asarray(timestamps, dtype=np.uint32), np.asarray(counts, dtype=np.uint16)


def legacy_merge(values, timestamps, length=850):
    """Exactly the loop boundaries in dataset.py:merge_timestamp, vectorized per bin."""
    edge = timestamps[0] + (timestamps[-1]-timestamps[0])/length
    interval = (timestamps[-1]-timestamps[0])/length
    start, result = 0, []
    for i, timestamp in enumerate(timestamps):
        if timestamp > edge:
            result.append(values[start:i].mean(axis=0) if start < i else values[i])
            start = i
            edge += interval
    if start < len(values):
        result.append(values[start:].mean(axis=0))
    if len(result) < length:
        result.extend(values[len(values)-i-1] for i in range(length-len(result)))
    return np.array(result[:length])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=90, help="Evenly spaced original files; 0 verifies all 9000.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    inventory = json.loads((SOURCE / "raw_file_inventory.json").read_text())
    audit = json.loads((OUT.parent / "array_audit.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    previous_summary = OUT / "summary.json"
    if previous_summary.exists():
        previous = json.loads(previous_summary.read_text(encoding="utf-8"))
        if previous.get("verifier_version") != VERIFIER_VERSION:
            saved = OUT.parent / "raw_mapping_v2_snapshot"
            saved.mkdir(exist_ok=True)
            for path in OUT.glob("*.json"):
                if not (saved / path.name).exists():
                    shutil.copy2(path, saved / path.name)
    cache = SOURCE / "raw_mat_cache"
    cache.mkdir(exist_ok=True)
    arrays = {env:{kind:np.load(spec["path"],mmap_mode="r",allow_pickle=False) for kind,spec in info["files"].items()}
              for env,info in audit["environments"].items()}
    first_amplitudes = {env:np.array(values["amp"][:,0,:]) for env,values in arrays.items()}
    jobs=[]
    for env in sorted(arrays):
        paths=sorted(p for p in inventory["paths"] if p.startswith(f"MAT/{env}/"))
        assert len(paths)==3000
        for index,path in enumerate(paths):
            jobs.append((env,index,path))
    if args.limit:
        jobs=[jobs[int(i)] for i in np.linspace(0,len(jobs)-1,min(args.limit,len(jobs)),dtype=int)]
    def verify(job):
        env,index,path=job
        filename=Path(path).name
        record_path=OUT/(filename+".json")
        if record_path.exists():
            record=json.loads(record_path.read_text())
            if record.get("verifier_version")==VERIFIER_VERSION and record.get("matched") and record.get("commit")==inventory["commit"] and record.get("processed_hashes")=={k:s["sha256"] for k,s in audit["environments"][env]["files"].items()}:
                return record
        base={"environment":env,"processed_index":None,"raw_inventory_index":index,"raw_path":path,"commit":inventory["commit"],
              "verifier_version":VERIFIER_VERSION,
              "processed_hashes":{k:s["sha256"] for k,s in audit["environments"][env]["files"].items()}}
        try:
            local=cache/filename
            if not local.exists():
                url=f"https://raw.githubusercontent.com/lcsig/Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments/{inventory['commit']}/{path}"
                for attempt in range(3):
                    try:
                        data=urllib.request.urlopen(url,timeout=30).read()
                        local.write_bytes(data)
                        break
                    except Exception:
                        if attempt==2:raise
                        time.sleep(1+attempt)
            data=local.read_bytes()
            packets=loadmat(io.BytesIO(data),simplify_cells=True)["data"]
            assert all(int(p["Nrx"])==3 and int(p["Ntx"])==1 for p in packets)
            csi=np.stack([p["csi"].reshape(90) for p in packets])
            original_timestamps=np.array([p["timestamp_low"] for p in packets],dtype=float)
            timestamps=timestamp_text15(original_timestamps)
            amp=legacy_merge(abs(csi),timestamps)
            phase=legacy_merge(np.angle(csi),timestamps)
            # The historical os.listdir order need not equal numeric subject order.
            # Locate a candidate by its first amplitude vector, then verify ALL
            # amplitude and phase elements below; a fingerprint alone is insufficient.
            candidate_errors=np.max(abs(first_amplitudes[env]-amp[0]),axis=1)
            candidates=np.flatnonzero(candidate_errors<1e-10)
            fully_matching=[int(i) for i in candidates
                            if np.max(abs(arrays[env]["amp"][i]-amp))<1e-10
                            and np.max(abs(arrays[env]["phase"][i]-phase))<1e-10]
            if len(fully_matching)==1:
                index=fully_matching[0]
            elif len(fully_matching)>1:
                raise ValueError(f"Ambiguous original-to-processed match: {fully_matching}")
            else:
                index=int(np.argmin(candidate_errors))
                base["closest_first_amplitude_index"]=index
            match=re.fullmatch(r"(E\d+)_(S\d+)_C(\d+)_A(\d+)_T(\d+)\.mat",filename)
            assert match
            _,subject,experiment,activity,trial=match.groups()
            amp_error=float(np.max(abs(arrays[env]["amp"][index]-amp)))
            phase_error=float(np.max(abs(arrays[env]["phase"][index]-phase)))
            label=int(arrays[env]["label"][index]); label_ok=label==LABELS[int(activity)]
            matched=bool(len(fully_matching)==1 and amp_error<1e-10 and phase_error<1e-10 and label_ok)
            if matched:
                base["processed_index"]=index
            base.update(subject_id=subject,experiment_id=int(experiment),activity_id=int(activity),trial_id=int(trial),
                        acquisition_id=f"{env}:{subject}:C{int(experiment):02d}:T{int(trial):02d}",
                        raw_sha256=hashlib.sha256(data).hexdigest(),packets=len(packets),
                        replay_timestamp_method="MAT values formatted to 15 significant decimal digits; precision-equivalent historical replay",
                        mat_timestamp_exceeds_uint32=bool((original_timestamps > 2**32-1).any()),
                        timestamp_text15_changed_values=int(np.count_nonzero(timestamps != original_timestamps)),
                        replay_duration_seconds=float((timestamps[-1]-timestamps[0])/1e6),
                        raw_timestamp_monotonic=bool((np.diff(timestamps)>=0).all()),
                        amplitude_max_absolute_error=amp_error,phase_max_absolute_error=phase_error,
                        label_matches=label_ok if matched else None,matched=matched)
        except Exception as error:
            base.update(matched=False,error=f"{type(error).__name__}: {error}")
        record_path.write_text(json.dumps(base,indent=2)+"\n",encoding="utf-8")
        return base
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(verify,jobs):
            results.append(record)
            if len(results)%args.progress_every==0 or not record["matched"]:
                print(f"Checked {len(results)}/{len(jobs)}; matched {sum(r['matched'] for r in results)}; last {record['raw_path']}",flush=True)
    complete=len(results)==9000 and all(r["matched"] for r in results)
    summary={"commit":inventory["commit"],"verifier_version":VERIFIER_VERSION,"requested_count":len(jobs),"matched_count":sum(r["matched"] for r in results),
             "all_9000_verified":complete,"tolerance_absolute":1e-10,"subject_sidecars_emitted":False,
             "mismatches":[r for r in results if not r["matched"]],
             "timestamp_text15_changed_files":sum(r.get("timestamp_text15_changed_values",0)>0 for r in results),
             "timestamp_exceeds_uint32_files":sum(r.get("mat_timestamp_exceeds_uint32",False) for r in results),
             "verifier_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "interpretation":"Precision-equivalent timestamp replay and arithmetic wrapped-phase averaging; not calibrated phase or recovered CSV export logs."}
    if complete:
        sidecars=OUT.parent/"subject_sidecars";sidecars.mkdir(exist_ok=True)
        for env in sorted(arrays):
            ordered=sorted((r for r in results if r["environment"]==env),key=lambda r:r["processed_index"])
            assert [r["processed_index"] for r in ordered]==list(range(3000))
            ids=[r["subject_id"] for r in ordered]
            assert len(set(ids))==10
            assert all(len(set(ids[start:start+300]))==1 for start in range(0,3000,300))
            payload={"env_name":env,"subject_ids":ids,"data_file_sha256":ordered[0]["processed_hashes"],
                     "acquisition_ids":[r["acquisition_id"] for r in ordered],
                     "raw_paths":[r["raw_path"] for r in ordered],
                     "provenance":{"method":"all-row original-signal replay","source_commit":inventory["commit"],
                                   "verifier_version":VERIFIER_VERSION,
                                   "matching_tolerance_absolute":1e-10,"verified_rows":3000}}
            (sidecars/f"{env}.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
        summary["subject_sidecars_emitted"]=True
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="mismatches"}),flush=True)


if __name__=="__main__":main()
