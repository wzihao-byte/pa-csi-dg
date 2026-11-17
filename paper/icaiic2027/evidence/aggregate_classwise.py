"""Aggregate per-class target F1 for the H-S four-arm study and its controls.

Reads only stored metrics (no training, no target data). Sources:
  - docs/revision/evidence/multienv_revision_v2/h_grouped_runs.json  (H-S H10/H11, 18 runs)
  - D:/pa-csi-dg-revision-runtime/studies/h_main_controls_20261008_v2/analysis/validated_runs.csv (54 runs)
  - D:/pa-csi-dg-revision-runtime/studies/h_phase_link_unwrap_20261023_v2/analysis/validated_runs.csv (N_L11, 9 runs)
Writes classwise_runs.csv and classwise_summary.json next to this script.
"""
import csv
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
D = "D:/pa-csi-dg-revision-runtime/studies"
CLASSES = ["No movement", "Falling", "Walking", "Sitting/standing", "Turning", "Picking up a pen"]
FOUR_S = [1, 2, 4]  # classes whose segments are all 4.0 s long
DUR = [0, 3, 5]  # classes whose segment duration is class-specific
ENVS = ["E1", "E2", "E3"]
SEEDS = (42, 52, 62)
runs = []


def add(arm, env, seed, macro, per_class, src):
    f1 = [None] * 6
    for m in per_class:
        f1[int(m["class_id"])] = 100.0 * float(m["f1"])
    assert all(v is not None for v in f1), (arm, env, seed)
    runs.append(
        {
            "arm": arm,
            "env": env,
            "seed": int(seed),
            "macro_f1": float(macro),
            "f1": f1,
            "four_s_mean": sum(f1[i] for i in FOUR_S) / 3,
            "dur_mean": sum(f1[i] for i in DUR) / 3,
            "source": src,
        }
    )


# H-S historical TPP arms (HS10 / HS11)
grouped = json.load(
    open(os.path.join(ROOT, "docs/revision/evidence/multienv_revision_v2/h_grouped_runs.json"), encoding="utf-8")
)
for r in grouped["rows"]:
    add("HS" + r["arm"][1:], r["target_env"], r["seed"], 100.0 * r["test_metrics"]["f1_macro"],
        r["per_class_metrics"], "h_grouped_runs.json")

# control studies on the same H-S partition; the DG-baseline study is optional (may be absent or partial)
for study, keep, optional in [("h_main_controls_20261008_v2", None, False), ("h_phase_link_unwrap_20261023_v2", {"N_L11"}, False),
                              ("h_dg_baselines_20261105_v2", {"GC10", "GX10", "C10", "X10"}, True)]:
    path = os.path.join(D, study, "analysis", "validated_runs.csv")
    if optional and not os.path.exists(path):
        continue
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if keep and row["arm"] not in keep:
                continue
            m = json.load(open(os.path.join(row["run"], "metrics.json"), encoding="utf-8"))
            macro = 100.0 * m["test_metrics"]["f1_macro"]
            assert abs(macro - float(row["macro_f1_pct"])) < 1e-6, (study, row["arm"], row["target"], row["seed"])
            add(row["arm"], row["target"], row["seed"], macro, m["per_class_metrics"], study)

with open(os.path.join(HERE, "classwise_runs.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["arm", "env", "seed", "macro_f1"] + [f"f1_c{i}" for i in range(6)] + ["four_s_mean", "dur_mean", "source"])
    for r in sorted(runs, key=lambda x: (x["arm"], x["env"], x["seed"])):
        w.writerow([r["arm"], r["env"], r["seed"], f"{r['macro_f1']:.6f}"] + [f"{v:.6f}" for v in r["f1"]]
                   + [f"{r['four_s_mean']:.6f}", f"{r['dur_mean']:.6f}", r["source"]])


def env_mean(vals_by_env):
    return sum(statistics.mean(vals_by_env[e]) for e in ENVS) / 3


def by_env(rs, key):
    return {e: [key(r) for r in rs if r["env"] == e] for e in ENVS}


summary = {}
partial = {a: n for a, n in ((a, sum(r["arm"] == a for r in runs)) for a in set(r["arm"] for r in runs)) if n != 9}
arms = sorted(a for a in set(r["arm"] for r in runs) if a not in partial)
for a in arms:
    rs = [r for r in runs if r["arm"] == a]
    summary[a] = {
        "n_runs": len(rs),
        "macro_f1_equal_env_mean": env_mean(by_env(rs, lambda r: r["macro_f1"])),
        "macro_f1_by_env": {e: statistics.mean(v) for e, v in by_env(rs, lambda r: r["macro_f1"]).items()},
        "macro_f1_sd_by_env": {e: statistics.stdev(v) for e, v in by_env(rs, lambda r: r["macro_f1"]).items()},
        "class_f1_equal_env_mean": [env_mean(by_env(rs, lambda r, i=i: r["f1"][i])) for i in range(6)],
        "four_s_mean_equal_env": env_mean(by_env(rs, lambda r: r["four_s_mean"])),
        "four_s_by_env": {e: statistics.mean(v) for e, v in by_env(rs, lambda r: r["four_s_mean"]).items()},
        "dur_mean_equal_env": env_mean(by_env(rs, lambda r: r["dur_mean"])),
        "runs": [{"env": r["env"], "seed": r["seed"], "macro_f1": r["macro_f1"], "f1": r["f1"]} for r in rs],
    }


def paired(a, b):
    out = {"pairs": []}
    diffs_m = {e: [] for e in ENVS}
    diffs_4 = {e: [] for e in ENVS}
    cls = {e: [[] for _ in range(6)] for e in ENVS}
    for e in ENVS:
        for s in SEEDS:
            ra = next(r for r in runs if r["arm"] == a and r["env"] == e and r["seed"] == s)
            rb = next(r for r in runs if r["arm"] == b and r["env"] == e and r["seed"] == s)
            dm = ra["macro_f1"] - rb["macro_f1"]
            d4 = ra["four_s_mean"] - rb["four_s_mean"]
            diffs_m[e].append(dm)
            diffs_4[e].append(d4)
            for i in range(6):
                cls[e][i].append(ra["f1"][i] - rb["f1"][i])
            out["pairs"].append({"env": e, "seed": s, "macro_pp": dm, "four_s_pp": d4})
    out["macro_mean_pp"] = env_mean(diffs_m)
    out["macro_by_env_pp"] = {e: statistics.mean(diffs_m[e]) for e in ENVS}
    out["macro_positive"] = sum(p["macro_pp"] > 0 for p in out["pairs"])
    out["four_s_mean_pp"] = env_mean(diffs_4)
    out["four_s_positive"] = sum(p["four_s_pp"] > 0 for p in out["pairs"])
    out["class_mean_pp"] = [sum(statistics.mean(cls[e][i]) for e in ENVS) / 3 for i in range(6)]
    return out


contrasts = {}
for a, b in [("HS11", "HS10"), ("G11", "G10"), ("HS10", "G10"), ("HS11", "G11"), ("P_ALL", "HS11"),
             ("P_REMOVE", "HS11"), ("N_H11", "HS11"), ("N_D11", "N_H11"), ("N_L11", "N_H11"),
             ("GC10", "G10"), ("GX10", "G10"), ("HS11", "GC10"), ("HS11", "GX10"), ("HS10", "GC10"), ("HS10", "GX10"),
             ("C10", "HS10"), ("X10", "HS10"), ("HS11", "C10"), ("HS11", "X10")]:
    if a in arms and b in arms:
        contrasts[f"{a}-{b}"] = paired(a, b)
inter = contrasts["HS11-HS10"]["macro_mean_pp"] - contrasts["G11-G10"]["macro_mean_pp"]
json.dump(
    {"classes": CLASSES, "four_s_classes": FOUR_S, "summary": summary, "contrasts": contrasts,
     "four_arm_interaction_pp": inter, "partial_arms_excluded": partial},
    open(os.path.join(HERE, "classwise_summary.json"), "w", encoding="utf-8"), indent=1,
)
print("runs", len(runs), "partial arms excluded:", partial)
for a in arms:
    s = summary[a]
    print(f"{a:9s} macro {s['macro_f1_equal_env_mean']:6.2f}  4s {s['four_s_mean_equal_env']:6.2f}  "
          f"dur {s['dur_mean_equal_env']:6.2f}  cls " + " ".join(f"{v:5.1f}" for v in s["class_f1_equal_env_mean"]))
for k, c in contrasts.items():
    print(f"{k:14s} macro {c['macro_mean_pp']:+6.2f} ({c['macro_positive']}/9)  4s {c['four_s_mean_pp']:+6.2f} "
          f"({c['four_s_positive']}/9)  cls " + " ".join(f"{v:+5.1f}" for v in c["class_mean_pp"]))
print("interaction (HS11-HS10)-(G11-G10) =", f"{inter:+.3f}")
