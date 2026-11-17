"""Explain the two initial replay mismatches using MAT, DAT and stored arrays.

The CSV-format equivalent is a verified numerical path, not a recovered export log.
"""
from pathlib import Path
import sys
import hashlib
import json
import struct
import urllib.request

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.verify_multienv_raw_mapping import legacy_merge, timestamp_text15, read_dat_timestamps


def decode_dat_csi(data):
    """CSI bit layout and Rx permutation in the pinned upstream C/MATLAB reader."""
    result, offset = [], 0
    while offset < len(data):
        size = struct.unpack_from(">H",data,offset)[0]
        code = data[offset+2]
        record = data[offset+3:offset+2+size]
        offset += size+2
        if code != 187:
            continue
        nrx,ntx = record[8],record[9]
        assert (nrx,ntx)==(3,1)
        payload_length = struct.unpack_from("<H",record,16)[0]
        assert payload_length==(30*(nrx*ntx*16+3)+7)//8
        assert len(record)==20+payload_length
        bits=np.unpackbits(np.frombuffer(record[20:],dtype=np.uint8),bitorder="little")
        starts=np.arange(30)[:,None]*(3+16*nrx*ntx)+3+16*np.arange(nrx*ntx)[None,:]
        weights=2**np.arange(8)
        real=bits[starts[:,:,None]+np.arange(8)]@weights
        imag=bits[starts[:,:,None]+8+np.arange(8)]@weights
        real=np.where(real>=128,real-256,real)
        imag=np.where(imag>=128,imag-256,imag)
        csi=(real+1j*imag).reshape(30,nrx,ntx).transpose(2,1,0)
        antenna=record[15]
        perm=np.array([antenna&3,(antenna>>2)&3,(antenna>>4)&3])
        assert sorted(perm.tolist())==[0,1,2]
        corrected=csi.copy()
        corrected[:,perm,:]=csi
        result.append(corrected.reshape(90))
    return np.asarray(result)


def main():
    source=ROOT/"docs/revision/sources"
    audit=json.loads((ROOT/"docs/revision/evidence/array_audit.json").read_text(encoding="utf-8"))
    files=audit["environments"]["E1"]["files"]
    commit=json.loads((source/"raw_file_inventory.json").read_text())["commit"]
    amp=np.load(files["amp"]["path"],mmap_mode="r")
    phase=np.load(files["phase"]["path"],mmap_mode="r")
    results=[]
    for stem,index in (("E1_S03_C04_A01_T09",1108),("E1_S05_C04_A01_T16",1715)):
        mat_path=source/"raw_mat_cache"/(stem+".mat")
        dat_path=source/"raw_dat_cache"/(stem+".dat")
        for path, folder in ((mat_path,"MAT"),(dat_path,"DAT")):
            if not path.exists():
                path.parent.mkdir(parents=True,exist_ok=True)
                url=f"https://raw.githubusercontent.com/lcsig/Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments/{commit}/{folder}/E1/{path.name}"
                with urllib.request.urlopen(url,timeout=60) as response:
                    path.write_bytes(response.read())
        raw=loadmat(mat_path,simplify_cells=True)["data"]
        mat_csi=np.stack([r["csi"].reshape(90) for r in raw])
        mat_times=np.asarray([r["timestamp_low"] for r in raw],dtype=float)
        dat_times,dat_counts=read_dat_timestamps(dat_path.read_bytes())
        dat_csi=decode_dat_csi(dat_path.read_bytes())
        assert np.array_equal(dat_csi,mat_csi)
        assert np.array_equal(dat_counts,np.asarray([r["bfee_count"] for r in raw]))
        converted=np.array([int(t) if t<2**31 else int(t)+2**64-2**32 for t in dat_times],dtype=float)
        assert np.array_equal(mat_times,converted)
        variants={"MAT_direct_float64":mat_times,"MAT_text15":timestamp_text15(mat_times),"DAT_uint32":dat_times.astype(float)}
        comparisons={}
        for name,times in variants.items():
            replay_amp=legacy_merge(abs(mat_csi),times)
            replay_phase=legacy_merge(np.angle(mat_csi),times)
            comparisons[name]={"amplitude_max_abs_error":float(np.max(abs(amp[index]-replay_amp))),
                               "phase_max_abs_error":float(np.max(abs(phase[index]-replay_phase))),
                               "duration_seconds":float((times[-1]-times[0])/1e6),
                               "first_timestamps":[float(t) for t in times[:4]]}
        assert comparisons["MAT_text15"]["amplitude_max_abs_error"]<1e-10
        assert comparisons["MAT_text15"]["phase_max_abs_error"]<1e-10
        results.append(dict(raw_stem=stem,processed_index=index,packet_count=len(raw),
                            mat_sha256=hashlib.sha256(mat_path.read_bytes()).hexdigest(),
                            dat_sha256=hashlib.sha256(dat_path.read_bytes()).hexdigest(),
                            all_packet_ids_and_complex_csi_equal=True,
                            mat_equals_signextended_uint32_cast_to_float64=True,
                            conversion_origin_verified=False,comparisons=comparisons))
    output=ROOT/"docs/revision/evidence/timestamp_precision_audit.json"
    output.write_text(json.dumps({"source_commit":commit,
                                 "historical_csv_or_export_log_recovered":False,
                                 "replay_does_not_correct_physical_time":True,"records":results},indent=2)+"\n",encoding="utf-8")
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
