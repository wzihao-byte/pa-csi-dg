"""Pre-check for the three-class (falling / walking / turning) study.

Question: once only the three 4 s activities remain, can segment duration or packet
count still identify the class?  If both score near the 33.3% uniform-random
reference, the three-class study is free of the protocol-timing shortcut.

Inputs (read only): the native environment records, which hold duration_seconds and
packet_count per segment, in the same order as the stored inputs.
Output: probe_three_class.json next to this file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

REC = Path(r"D:/pa-csi-dg-revision-runtime/data/native_histstyle_phase_v1/environment_records")
OUT = Path(__file__).resolve().parent / "probe_three_class.json"
ENVS = ["E1", "E2", "E3"]
KEEP = (1, 2, 4)  # falling, walking, turning
NAMES = {1: "falling", 2: "walking", 4: "turning"}


def load(env):
    recs = json.load(open(REC / f"{env}.json", encoding="utf-8-sig"))["records"]
    lab = np.array([r["label"] for r in recs])
    dur = np.array([r["duration_seconds"] for r in recs], dtype=float)
    pkt = np.array([r["packet_count"] for r in recs], dtype=float)
    k = np.isin(lab, KEEP)
    return {"label": lab[k], "dur": dur[k], "pkt": pkt[k]}


def main():
    data = {e: load(e) for e in ENVS}
    res = {"classes": NAMES, "per_class_stats": {}, "probes": {}}
    for e in ENVS:
        d = data[e]
        res["per_class_stats"][e] = {
            NAMES[c]: {
                "n": int((d["label"] == c).sum()),
                "duration_s": {q: float(f(d["dur"][d["label"] == c])) for q, f in
                               [("min", np.min), ("median", np.median), ("max", np.max), ("mean", np.mean)]},
                "packets": {q: float(f(d["pkt"][d["label"] == c])) for q, f in
                            [("min", np.min), ("median", np.median), ("max", np.max), ("mean", np.mean)]},
            } for c in KEEP}
    feats = {"duration_only": lambda d: d["dur"][:, None], "packet_count_only": lambda d: d["pkt"][:, None],
             "duration+packet_count": lambda d: np.c_[d["dur"], d["pkt"]]}
    for name, fn in feats.items():
        per = {}
        for tgt in ENVS:
            src = [e for e in ENVS if e != tgt]
            m = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4)
            m.fit(np.vstack([fn(data[e]) for e in src]), np.concatenate([data[e]["label"] for e in src]))
            per[tgt] = float(f1_score(data[tgt]["label"], m.predict(fn(data[tgt])), average="macro"))
        res["probes"][name] = {"per_target_macro_f1": per, "mean_macro_f1": float(np.mean(list(per.values())))}
        print(f"{name:24s} mean={np.mean(list(per.values())) * 100:5.1f}  "
              + " ".join(f"{t}={v * 100:5.1f}" for t, v in per.items()))
    for e in ENVS:
        print(e, {c: (s["n"], s["duration_s"]["min"], s["duration_s"]["max"], s["packets"]["median"])
                  for c, s in res["per_class_stats"][e].items()})
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
