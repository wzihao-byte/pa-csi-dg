"""Static-descriptor leakage probe for MultiEnv amplitude/phase inputs.

Question: how much of the cross-environment class information in the historical
phase representation is carried by a STATIC per-record descriptor (time-mean of
the unwrapped phase, or just a per-link slope/intercept), i.e. without any
temporal structure?  If a 6- or 90-number static descriptor already separates
the six activities across environments, the 'phase gain' is unlikely to be
motion information.

Read-only: historical processed arrays + record metadata.  No training data of
the project is modified.  CPU only.
"""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/Users/howie/Desktop/enhanced/PA-CSI-HAR/datasets/PA-CSI datasets/MultiEnv")
REC = Path(r"D:/pa-csi-dg-revision-runtime/data/native_histstyle_phase_v1/environment_records")
OUT = Path(__file__).resolve().parents[1] / "docs/revision/evidence/duration_shortcut_probe_20261023"
OUT.mkdir(parents=True, exist_ok=True)
ENVS = ["E1", "E2", "E3"]
CHUNK = 100
NAME_RE = re.compile(r"E(\d)_S(\d+)_C(\d+)_A(\d+)_T(\d+)")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def extract(env_idx: int, env: str) -> dict:
    cache = OUT / f"feat_{env}.npz"
    if cache.exists():
        log("cached", cache)
        return dict(np.load(cache, allow_pickle=True))
    lab = np.load(ROOT / f"label_6c_{env_idx}.npy")
    amp = np.load(ROOT / f"data_6c_{env_idx}.npy", mmap_mode="r")
    ph = np.load(ROOT / f"data_angle_6c_{env_idx}.npy", mmap_mode="r")
    recs = json.load(open(REC / f"{env}.json", encoding="utf-8-sig"))["records"]
    assert amp.shape == ph.shape == (3000, 850, 90), (amp.shape, ph.shape)
    assert [int(r["label"]) for r in recs] == [int(x) for x in lab], "label order mismatch"
    S, C, A, T = [], [], [], []
    for r in recs:
        m = NAME_RE.search(r["raw_path"])
        assert m, r["raw_path"]
        S.append(int(m.group(2))); C.append(int(m.group(3))); A.append(int(m.group(4))); T.append(int(m.group(5)))
    n = len(lab)
    ph_mean = np.zeros((n, 90), np.float64); ph_std = np.zeros((n, 90), np.float64)
    ph_t0 = np.zeros((n, 90), np.float64); ph_wrapped_mean = np.zeros((n, 90), np.float64)
    amp_mean = np.zeros((n, 90), np.float64); amp_std = np.zeros((n, 90), np.float64)
    ph_tstd_mean = np.zeros((n, 3), np.float64)  # mean over subcarriers of temporal std, per link
    for i in range(0, n, CHUNK):
        j = min(n, i + CHUNK)
        p = np.asarray(ph[i:j], dtype=np.float64)
        pu = np.unwrap(p, axis=-1)  # historical loader convention (crosses link boundaries)
        ph_mean[i:j] = pu.mean(1); ph_std[i:j] = pu.std(1); ph_t0[i:j] = pu[:, 0, :]
        ph_wrapped_mean[i:j] = np.angle(np.exp(1j * p).mean(1))  # circular mean of raw wrapped phase
        for l in range(3):
            ph_tstd_mean[i:j, l] = pu[:, :, l * 30:(l + 1) * 30].std(1).mean(1)
        a = np.asarray(amp[i:j], dtype=np.float64)
        amp_mean[i:j] = a.mean(1); amp_std[i:j] = a.std(1)
        if (i // CHUNK) % 5 == 0:
            log(env, i, "/", n)
    x = np.arange(30) - 14.5
    slope = np.zeros((n, 3)); icpt = np.zeros((n, 3))
    for l in range(3):
        seg = ph_mean[:, l * 30:(l + 1) * 30]
        slope[:, l] = (seg * x).sum(1) / (x ** 2).sum(); icpt[:, l] = seg.mean(1)
    out = dict(label=lab.astype(int), S=np.array(S), C=np.array(C), A=np.array(A), T=np.array(T),
               ph_mean=ph_mean, ph_std=ph_std, ph_t0=ph_t0, ph_wrapped_mean=ph_wrapped_mean,
               ph_tstd_mean=ph_tstd_mean, amp_mean=amp_mean, amp_std=amp_std, slope=slope, icpt=icpt)
    np.savez(cache, **out)
    log("saved", cache)
    return out


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import f1_score, accuracy_score

    data = {env: extract(k, env) for k, env in enumerate(ENVS, 1)}

    # ---- label structure: which (C, A) pairs exist and what class they carry
    rows = []
    for env in ENVS:
        d = data[env]
        for c, a, y in zip(d["C"], d["A"], d["label"]):
            rows.append((env, int(c), int(a), int(y)))
    rows = np.array(rows)
    struct = {}
    for c in np.unique(rows[:, 1]):
        sub = rows[rows[:, 1] == c]
        struct[int(c)] = {int(a): {"n": int((sub[:, 2] == a).sum()),
                                   "labels": sorted(set(int(v) for v in sub[sub[:, 2] == a, 3]))}
                          for a in np.unique(sub[:, 2])}
    json.dump(struct, open(OUT / "label_structure.json", "w"), indent=1)
    log("label structure by experiment C:", json.dumps(struct))

    feats = {
        "phase_time_mean_90": lambda d: d["ph_mean"],
        "phase_first_frame_90": lambda d: d["ph_t0"],
        "phase_link_slope_icpt_6": lambda d: np.hstack([d["slope"], d["icpt"]]),
        "phase_link_icpt_3": lambda d: d["icpt"],
        "phase_wrapped_circmean_90": lambda d: d["ph_wrapped_mean"],
        "phase_time_std_90": lambda d: d["ph_std"],
        "phase_mean+std_180": lambda d: np.hstack([d["ph_mean"], d["ph_std"]]),
        "amp_time_mean_90": lambda d: d["amp_mean"],
        "amp_time_std_90": lambda d: d["amp_std"],
        "amp_mean+std_180": lambda d: np.hstack([d["amp_mean"], d["amp_std"]]),
    }
    models = {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
        "rf300": lambda: RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4),
    }
    results = {}
    for fname, fn in feats.items():
        for mname, mk in models.items():
            per = {}
            for tgt in ENVS:
                src = [e for e in ENVS if e != tgt]
                Xtr = np.vstack([fn(data[e]) for e in src]); ytr = np.concatenate([data[e]["label"] for e in src])
                Xte = fn(data[tgt]); yte = data[tgt]["label"]
                clf = mk().fit(Xtr, ytr); pred = clf.predict(Xte)
                per[tgt] = {"macro_f1": float(f1_score(yte, pred, average="macro")),
                            "acc": float(accuracy_score(yte, pred))}
            mean_f1 = float(np.mean([v["macro_f1"] for v in per.values()]))
            results[f"{fname}|{mname}"] = {"per_target": per, "mean_macro_f1": mean_f1}
            log(f"LOEO {fname:28s} {mname:7s} macroF1 mean={mean_f1*100:6.2f}  " +
                " ".join(f"{t}={per[t]['macro_f1']*100:5.1f}" for t in ENVS))
    json.dump(results, open(OUT / "loeo_static_probe.json", "w"), indent=1)

    # ---- within-acquisition ordering: does the static phase offset track the segment index A?
    # For each (env,S,C,T) group with >=2 segments, Spearman-like rank agreement between A and link intercept.
    from scipy.stats import spearmanr
    order = {}
    for env in ENVS:
        d = data[env]
        keys = list(zip(d["S"], d["C"], d["T"]))
        groups = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(i)
        rhos = {l: [] for l in range(3)}
        for k, idx in groups.items():
            if len(idx) < 3:
                continue
            idx = np.array(idx)
            for l in range(3):
                r = spearmanr(d["A"][idx], d["icpt"][idx, l]).correlation
                if np.isfinite(r):
                    rhos[l].append(r)
        order[env] = {f"link{l}": {"mean_rho": float(np.mean(v)), "frac_pos": float(np.mean(np.array(v) > 0)), "groups": len(v)}
                      for l, v in rhos.items()}
    json.dump(order, open(OUT / "within_trial_order.json", "w"), indent=1)
    log("within-trial A-order vs phase intercept:", json.dumps(order))

    # ---- can we predict the EXPERIMENT index C (protocol block) from static phase, cross-env?
    cres = {}
    for fname in ["phase_time_mean_90", "phase_link_slope_icpt_6", "amp_time_mean_90"]:
        fn = feats[fname]; per = {}
        for tgt in ENVS:
            src = [e for e in ENVS if e != tgt]
            Xtr = np.vstack([fn(data[e]) for e in src]); ytr = np.concatenate([data[e]["C"] for e in src])
            clf = models["logreg"]().fit(Xtr, ytr); pred = clf.predict(fn(data[tgt]))
            per[tgt] = float(f1_score(data[tgt]["C"], pred, average="macro"))
        cres[fname] = per
        log("LOEO predict experiment-index C from", fname, per)
    json.dump(cres, open(OUT / "predict_C.json", "w"), indent=1)

    # ---- summary stats of the static descriptor by class (E1) for the report
    d = data["E1"]
    by_class = {int(c): {"icpt_mean": d["icpt"][d["label"] == c].mean(0).round(2).tolist(),
                         "slope_mean": d["slope"][d["label"] == c].mean(0).round(4).tolist(),
                         "n": int((d["label"] == c).sum())} for c in range(6)}
    json.dump(by_class, open(OUT / "e1_static_by_class.json", "w"), indent=1)
    log("E1 static phase by class:", json.dumps(by_class))
    log("DONE")


if __name__ == "__main__":
    main()
