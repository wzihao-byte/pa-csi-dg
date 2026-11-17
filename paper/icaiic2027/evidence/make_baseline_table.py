"""Build tables/baseline_table.tex from the frozen DG-baseline study (h_dg_baselines_20261105_v2).

Run aggregate_classwise.py first. The table is written only when the two
global-backbone baseline arms (GC10, GX10) have all nine runs; the TPP-backbone
rows (C10, X10) are appended when they are complete as well. Nothing is written
for a partial study, so the manuscript cannot pick up unfinished results.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tables", "baseline_table.tex")
STUDY = "D:/pa-csi-dg-revision-runtime/studies/h_dg_baselines_20261105_v2/analysis/results.json"
S = json.load(open(os.path.join(HERE, "classwise_summary.json"), encoding="utf-8"))
summ, con = S["summary"], S["contrasts"]
ENVS = ["E1", "E2", "E3"]


def f(v, nd=2):
    return f"{v:.{nd}f}"


def sgn(v, nd=2):
    return ("$-$" if v < 0 else "+") + f"{abs(v):.{nd}f}"


def row(arm, label, versus):
    s = summ[arm]
    cells = " & ".join(f(s["macro_f1_by_env"][e]) for e in ENVS)
    if versus is None:
        delta = r"\multicolumn{2}{c}{reference}"
    else:
        key = f"{arm}-{versus}" if f"{arm}-{versus}" in con else f"{versus}-{arm}"
        x = con[key]
        sign = 1 if key.startswith(arm + "-") else -1
        pos = x["macro_positive"] if sign == 1 else 9 - x["macro_positive"] - sum(p["macro_pp"] == 0 for p in x["pairs"])
        delta = f"{sgn(sign * x['macro_mean_pp'])} & {pos}/9"
    return f"{label} & {cells} & {f(s['macro_f1_equal_env_mean'])} & {f(s['four_s_mean_equal_env'])} & {delta}\\\\"


if not all(a in summ for a in ("GC10", "GX10")):
    print("global-backbone baseline arms incomplete; excluded:", S.get("partial_arms_excluded"), "-> no table written")
    raise SystemExit(0)
study = json.load(open(STUDY, encoding="utf-8"))
rows = [r"\multicolumn{8}{l}{\textit{Plain backbone: capacity-matched global MLP pooling}}\\",
        row("G10", "ERM (G-CE)", None),
        row("GC10", "ERM + CORAL", "G10"),
        row("GX10", "ERM + Mixup", "G10"),
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{TPP backbone}}\\",
        row("HS10", "TPP + CE", "G10")]
if all(a in summ for a in ("C10", "X10")):
    rows += [row("C10", "TPP + CORAL", "G10"), row("X10", "TPP + Mixup", "G10")]
rows += [row("HS11", "TPP + multiview SupCon", "G10")]
ci = {k: v.get("bootstrap_95_interval_pp") for k, v in study["contrasts"].items() if v.get("status") == "complete"}
tpp_rows = all(a in summ for a in ("C10", "X10"))
backbones = ("of the capacity-matched global backbone and of the TPP backbone" if tpp_rows
             else "of the capacity-matched global backbone (ERM)")
note = ""
if "tsc_minus_global_coral" in ci and "tsc_minus_global_mixup" in ci:
    note = (r" The paired difference of TPP + SupCon against ERM + CORAL is " + sgn(study["contrasts"]["tsc_minus_global_coral"]["paired_mean_pp"])
            + r"\,pp (95\% bootstrap interval [" + sgn(ci["tsc_minus_global_coral"][0]) + ", " + sgn(ci["tsc_minus_global_coral"][1])
            + r"]) and against ERM + Mixup " + sgn(study["contrasts"]["tsc_minus_global_mixup"]["paired_mean_pp"])
            + r"\,pp ([" + sgn(ci["tsc_minus_global_mixup"][0]) + ", " + sgn(ci["tsc_minus_global_mixup"][1]) + "]).")
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(r"""\begin{table}[t]
\caption{Generic domain-generalization baselines under the subject-disjoint split (target macro-F1, \%). CORAL (DomainBed penalty, weight 1.0, on the fused 256-d feature) and inter-domain Mixup (Beta(0.2, 0.2), amplitude and phase mixed with the same coefficients) are added to cross-entropy training """ + backbones + r"""; each run shares the split, sampling schedule and initial shared weights of its paired cell. ``4\,s'' is the mean class-F1 of falling, walking and turning. $\Delta$ is the paired difference against ERM with the number of positive pairs out of nine.""" + note + r"""}
\label{tab:baseline}
\centering\footnotesize
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{lrrrrrrr}
\toprule
Model & E1 & E2 & E3 & Mean & 4\,s & $\Delta$ & Pos.\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
print("wrote", OUT)
for k, v in study["contrasts"].items():
    if v.get("status") == "complete":
        print(f"{k:32s} {v['paired_mean_pp']:+6.2f} pp  pos {v['positive_pairs']}/9  CI [{v['bootstrap_95_interval_pp'][0]:+.2f}, {v['bootstrap_95_interval_pp'][1]:+.2f}]")
