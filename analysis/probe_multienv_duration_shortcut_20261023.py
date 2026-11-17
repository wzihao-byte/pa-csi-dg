"""Second probe: how much class information is carried by recording DURATION /
segment POSITION (protocol timing), and whether the static phase descriptor
encodes that timing.  Uses the feature cache written by probe_static_phase.py
and the native environment records (duration_seconds, packet_count).
"""
from __future__ import annotations
import json, re, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, r2_score, confusion_matrix

HERE = Path(__file__).resolve().parents[1] / "docs/revision/evidence/duration_shortcut_probe_20261023"; OUT = HERE  # needs feat_E*.npz caches from the static probe (not committed, ~13 MB each)
REC = Path(r"D:/pa-csi-dg-revision-runtime/data/native_histstyle_phase_v1/environment_records")
ROOT = Path(r"C:/Users/howie/Desktop/enhanced/PA-CSI-HAR/datasets/PA-CSI datasets/MultiEnv")
ENVS = ["E1", "E2", "E3"]
NAME_RE = re.compile(r"E(\d)_S(\d+)_C(\d+)_A(\d+)_T(\d+)")
# segment position within the continuous trial, from the dataset paper's timing diagrams
POS = {(1, 1): 0, (1, 2): 1, (1, 3): 2, (2, 4): 0, (2, 5): 1, (2, 3): 2,
       (3, 6): 0, (3, 7): 1, (3, 8): 2, (3, 9): 3, (4, 1): 0, (4, 10): 1, (4, 4): 2, (4, 11): 3, (5, 12): 0}


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(env):
    d = dict(np.load(OUT / f"feat_{env}.npz", allow_pickle=True))
    recs = json.load(open(REC / f"{env}.json", encoding="utf-8-sig"))["records"]
    d["dur"] = np.array([r["duration_seconds"] for r in recs]); d["pkt"] = np.array([r["packet_count"] for r in recs])
    d["pos"] = np.array([POS[(int(c), int(a))] for c, a in zip(d["C"], d["A"])])
    return d


def loeo(data, fn, target_key, model, metric="f1"):
    per = {}
    for tgt in ENVS:
        src = [e for e in ENVS if e != tgt]
        Xtr = np.vstack([fn(data[e]) for e in src]); ytr = np.concatenate([data[e][target_key] for e in src])
        Xte = fn(data[tgt]); yte = data[tgt][target_key]
        m = model().fit(Xtr, ytr); p = m.predict(Xte)
        per[tgt] = float(f1_score(yte, p, average="macro")) if metric == "f1" else float(r2_score(yte, p))
    return per, float(np.mean(list(per.values())))


def main():
    data = {e: load(e) for e in ENVS}
    res = {}
    LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    RF = lambda: RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4)
    # 1) class from duration alone / packet count alone (protocol-timing ceiling)
    for name, fn in {"duration_only": lambda d: d["dur"][:, None], "packet_count_only": lambda d: d["pkt"][:, None],
                     "duration+pkt": lambda d: np.c_[d["dur"], d["pkt"]]}.items():
        per, mean = loeo(data, fn, "label", RF); res[name] = {"per": per, "mean_macro_f1": mean}
        log(f"LOEO class <- {name:18s} macroF1 mean={mean*100:6.2f} " + " ".join(f"{t}={per[t]*100:5.1f}" for t in ENVS))
    # per-class recall from duration-only, E1 target
    src = ["E2", "E3"]; fn = lambda d: d["dur"][:, None]
    m = RF().fit(np.vstack([fn(data[e]) for e in src]), np.concatenate([data[e]["label"] for e in src]))
    cm = confusion_matrix(data["E1"]["label"], m.predict(fn(data["E1"])), normalize="true")
    res["duration_only_E1_rownorm_confusion"] = np.round(cm * 100, 1).tolist()
    log("duration-only E1 row-normalised confusion (%):\n" + str(np.round(cm * 100, 1)))

    # 2) does the STATIC phase descriptor encode duration / segment position / packet count?
    feats = {"phase_time_mean_90": lambda d: d["ph_mean"], "phase_link_slope_icpt_6": lambda d: np.hstack([d["slope"], d["icpt"]]),
             "phase_first_frame_90": lambda d: d["ph_t0"], "amp_time_mean_90": lambda d: d["amp_mean"], "amp_mean+std_180": lambda d: np.hstack([d["amp_mean"], d["amp_std"]])}
    RFR = lambda: RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=4)
    for fname, fn in feats.items():
        per, mean = loeo(data, fn, "dur", RFR, metric="r2"); res[f"R2_duration<-{fname}"] = {"per": per, "mean": mean}
        log(f"LOEO R2 duration <- {fname:26s} mean R2={mean:6.3f} " + " ".join(f"{t}={per[t]:6.3f}" for t in ENVS))
        per, mean = loeo(data, fn, "pos", RF); res[f"F1_position<-{fname}"] = {"per": per, "mean": mean}
        log(f"LOEO F1 segment-position <- {fname:18s} mean={mean*100:6.2f} " + " ".join(f"{t}={per[t]*100:5.1f}" for t in ENVS))
    # 3) class from static phase, restricted to the 4-second segments only (classes 1,2,4 share duration 4 s):
    #    if phase still separates falling/walking/turning here, it is not just duration.
    def sub4(d):
        keep = (d["label"] == 1) | (d["label"] == 2) | (d["label"] == 4)
        return keep
    for fname in ["phase_time_mean_90", "phase_link_slope_icpt_6", "amp_time_mean_90", "amp_mean+std_180"]:
        fn = feats[fname]; per = {}
        for tgt in ENVS:
            src = [e for e in ENVS if e != tgt]
            Xtr = np.vstack([fn(data[e])[sub4(data[e])] for e in src]); ytr = np.concatenate([data[e]["label"][sub4(data[e])] for e in src])
            k = sub4(data[tgt]); m = LR().fit(Xtr, ytr); p = m.predict(fn(data[tgt])[k])
            per[tgt] = float(f1_score(data[tgt]["label"][k], p, average="macro"))
        res[f"F1_fall/walk/turn_only<-{fname}"] = per
        log(f"LOEO F1 among falling/walking/turning (all 4 s) <- {fname:26s} " + " ".join(f"{t}={per[t]*100:5.1f}" for t in ENVS) + f"  mean={np.mean(list(per.values()))*100:5.1f}")
    # 4) class from static phase + duration jointly
    fn = lambda d: np.c_[d["ph_mean"], d["dur"]]
    per, mean = loeo(data, fn, "label", RF); res["class<-phase_time_mean_90+duration"] = {"per": per, "mean": mean}
    log(f"LOEO class <- phase_time_mean_90 + duration  macroF1 mean={mean*100:6.2f} " + " ".join(f"{t}={per[t]*100:5.1f}" for t in ENVS))
    json.dump(res, open(OUT / "duration_probe.json", "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
