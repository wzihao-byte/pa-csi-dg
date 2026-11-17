"""Generate the LaTeX table fragments in ../tables from stored evidence only.

Inputs
  evidence/classwise_summary.json                       (built by aggregate_classwise.py)
  paper/submission_versions_20261007/shared/h_core_evidence.json   (earlier sample-stratified four arms)
  docs/revision/evidence/duration_shortcut_probe_20261023/*.json   (protocol probes)

Names used in the paper (repository names in parentheses)
  G-CE / G-SC / T-CE / T-SC     main-study arms (G10 / G11 / HS10 / HS11): pooling (Global + MLP, TPP) x training
                                procedure (CE, CE + SupCon). "SupCon" means the multiview SupCon training
                                procedure: three extra single-link view passes plus the hybrid SupCon loss
                                (supcon_views 4, lambda 0.05); CE arms use supcon_views 1, lambda 0, and never
                                encode single-link views. The two arms therefore differ in loss AND views.
  G0-CE / G0-SC                 global arms of the earlier study (H00 / H01): plain global max pooling without the
                                matched MLP; its TPP arms (H10 / H11) are the same model as T-CE / T-SC.
  original inputs               the stored 850-frame arrays (historical inputs).
  uniform-grid inputs           amplitude and phase re-derived on a uniform time grid (native).
  unwrapped phase, across / within links; differential phase     phase representations (N_H11 / N_L11 / N_D11).
  "improved"                    a paired run whose difference is above zero; "positive pair" is reserved for the loss.
Captions are short; definitions and intervals are given in the manuscript text.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "..", "tables")
os.makedirs(OUT, exist_ok=True)
S = json.load(open(os.path.join(HERE, "classwise_summary.json"), encoding="utf-8"))
summ, con = S["summary"], S["contrasts"]
hcore = json.load(open(os.path.join(ROOT, "paper/submission_versions_20261007/shared/h_core_evidence.json"), encoding="utf-8"))
PROBE = os.path.join(ROOT, "docs/revision/evidence/duration_shortcut_probe_20261023")
dur = json.load(open(os.path.join(PROBE, "duration_probe.json"), encoding="utf-8"))
stat = json.load(open(os.path.join(PROBE, "loeo_static_probe.json"), encoding="utf-8"))
ENVS = ["E1", "E2", "E3"]
MAIN = ["G10", "G11", "HS10", "HS11"]
EARLIER = ["H00", "H01", "H10", "H11"]
MAIN_ARMS = [("G-CE", "Global + MLP", "CE"), ("G-SC", "Global + MLP", "CE + SupCon"),
             ("T-CE", "TPP", "CE"), ("T-SC", "TPP", "CE + SupCon")]
EARLIER_ARMS = [("G$_0$-CE", "Global (plain)", "CE"), ("G$_0$-SC", "Global (plain)", "CE + SupCon"),
                ("T-CE", "TPP", "CE"), ("T-SC", "TPP", "CE + SupCon")]


def f(v, nd=2):
    return f"{v:.{nd}f}"


def sgn(v, nd=2):
    return ("$-$" if v < 0 else "+") + f"{abs(v):.{nd}f}"


def write(name, text):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        fh.write(text)


def block(values, arms):
    """values: one dict per arm with E1/E2/E3 (mean, sd) and mean; the best mean of each column is set in bold."""
    best = {e: max(v[e][0] for v in values) for e in ENVS}
    best_mean = max(v["mean"] for v in values)
    rows = []
    for (label, pooling, objective), v in zip(arms, values):
        cells = []
        for e in ENVS:
            m, sd = v[e]
            cells.append((r"\textbf{%s}" % f(m) if m == best[e] else f(m)) + f" ({f(sd)})")
        mean = r"\textbf{%s}" % f(v["mean"]) if v["mean"] == best_mean else f(v["mean"])
        rows.append(f"{label} & {pooling} & {objective} & " + " & ".join(cells) + f" & {mean}\\\\")
    return "\n".join(rows)


def with_count(x, mean_key, count_key):
    return f"{sgn(x[mean_key])} ({x[count_key]}/9)"


# ---------------------------------------------------------------- Table: main four arms (double column)
main_values = [{**{e: (summ[a]["macro_f1_by_env"][e], summ[a]["macro_f1_sd_by_env"][e]) for e in ENVS},
                "mean": summ[a]["macro_f1_equal_env_mean"]} for a in MAIN]
hsum = {x["arm"]: x for x in hcore["summary"] if x["study"] == "H"}
earlier_values = [{**{e: (float(hsum[a][e]), float(hsum[a][e + "_seed_sd"])) for e in ENVS},
                   "mean": float(hsum[a]["equal_environment_mean"])} for a in EARLIER]
write("main_table.tex", r"""\begin{table*}[t]
\caption{Leave-one-environment-out target macro-F1 (\%) on MultiEnv: mean over three seeds (SD in parentheses).}
\label{tab:main}
\centering\footnotesize
\begin{tabular}{lllrrrr}
\toprule
Arm & Pooling & Training & E1 (LOS) & E2 (LOS) & E3 (NLOS) & Mean\\
\midrule
\multicolumn{7}{l}{\textit{Main study: participant-disjoint source split, capacity matched (4,786,294 parameters in every arm)}}\\
""" + block(main_values, MAIN_ARMS) + r"""
\midrule
\multicolumn{7}{l}{\textit{Earlier study: sample-stratified source split (plain global arms: 2,788,918 parameters)}}\\
""" + block(earlier_values, EARLIER_ARMS) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- Table: paired comparisons
inter4 = con["HS11-HS10"]["four_s_mean_pp"] - con["G11-G10"]["four_s_mean_pp"]
crow = []
for key, label, definition in [("HS11-HS10", "SupCon on TPP", "T-SC $-$ T-CE"), ("G11-G10", "SupCon on global", "G-SC $-$ G-CE"),
                               ("HS10-G10", "TPP under CE", "T-CE $-$ G-CE"), ("HS11-G11", "TPP under SupCon", "T-SC $-$ G-SC")]:
    x = con[key]
    crow.append(f"{label} & {definition} & {with_count(x, 'macro_mean_pp', 'macro_positive')} & "
                f"{with_count(x, 'four_s_mean_pp', 'four_s_positive')}\\\\")
crow.append(r"\midrule")
crow.append(f"Interaction & & {sgn(S['four_arm_interaction_pp'])}\\phantom{{ (0/9)}} & {sgn(inter4)}\\phantom{{ (0/9)}}\\\\")
write("contrast_table.tex", r"""\begin{table}[t]
\caption{Mean paired difference in the main study (pp; improved runs of nine in parentheses).}
\label{tab:contrast}
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrr}
\toprule
Comparison & Difference & Macro-F1 & 4\,s classes\\
\midrule
""" + "\n".join(crow) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Table: class F1
CLS = [("No movement", "3/4/5"), ("Falling", "4.0"), ("Walking", "4.0"), ("Sitting/standing", "3.2"), ("Turning", "4.0"), ("Pen pick-up", "4.5")]
rrows = []
for i, (name, d) in enumerate(CLS):
    rrows.append(f"{name} & {d} & " + " & ".join(f(summ[a]['class_f1_equal_env_mean'][i]) for a in MAIN) + "\\\\")
rrows.append(r"\midrule")
rrows.append("Mean, 4\\,s classes & 4.0 & " + " & ".join(f(summ[a]["four_s_mean_equal_env"]) for a in MAIN) + "\\\\")
rrows.append("Mean, other classes & -- & " + " & ".join(f(summ[a]["dur_mean_equal_env"]) for a in MAIN) + "\\\\")
write("class_table.tex", r"""\begin{table}[t]
\caption{Class F1 (\%) in the main study, averaged over seeds and targets. Dur.: segment length in the recording protocol (s). All class-wise F1 scores are computed from the six-class classifier. The 4\,s class mean averages the F1 scores for falling, walking, and turning.}
\label{tab:class}
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrr}
\toprule
Activity & Dur. & G-CE & G-SC & T-CE & T-SC\\
\midrule
""" + "\n".join(rrows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Table: controls
def ctrl_row(arm, label, key):
    s = summ[arm]
    cells = " & ".join(f(s["macro_f1_by_env"][e]) for e in ENVS)
    if key is None:
        return f"{label} & {cells} & {f(s['macro_f1_equal_env_mean'])} & reference\\\\"
    return f"{label} & {cells} & {f(s['macro_f1_equal_env_mean'])} & {with_count(con[key], 'macro_mean_pp', 'macro_positive')}\\\\"


ctrl = [r"\multicolumn{6}{l}{\textit{Positive-pair rule, original inputs}}\\",
        ctrl_row("HS11", "Hybrid rule (T-SC)", None),
        ctrl_row("P_ALL", "All same-class pairs", "P_ALL-HS11"),
        ctrl_row("P_REMOVE", "No same-class negatives", "P_REMOVE-HS11"),
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Phase representation, uniform-grid inputs}}\\",
        ctrl_row("N_H11", "Unwrapped across links", None),
        ctrl_row("N_L11", "Unwrapped within links", "N_L11-N_H11"),
        ctrl_row("N_D11", "Differential", "N_D11-N_H11")]
write("control_table.tex", r"""\begin{table}[t]
\caption{Controls on T-SC under the main-study split: target macro-F1 (\%) and mean paired difference $\Delta$ from the reference of each group (pp; improved runs of nine in parentheses).}
\label{tab:control}
\centering\footnotesize
\setlength{\tabcolsep}{3.4pt}
\begin{tabular}{lrrrrr}
\toprule
Variant & E1 & E2 & E3 & Mean & $\Delta$\\
\midrule
""" + "\n".join(ctrl) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Table: duration-shortcut probes
def probe_row(label, per, mean):
    return f"{label} & " + " & ".join(f(100 * per[e], 1) for e in ENVS) + f" & {f(100 * mean, 1)}\\\\"


def stat_row(label, key):
    x = stat[key]
    return probe_row(label, {e: x["per_target"][e]["macro_f1"] for e in ENVS}, x["mean_macro_f1"])


p = [probe_row("Segment duration only", dur["duration_only"]["per"], dur["duration_only"]["mean_macro_f1"]),
     probe_row("Packet count only", dur["packet_count_only"]["per"], dur["packet_count_only"]["mean_macro_f1"]),
     probe_row("Duration + packet count", dur["duration+pkt"]["per"], dur["duration+pkt"]["mean_macro_f1"]),
     stat_row("Phase: link slope + intercept (6-d)", "phase_link_slope_icpt_6|rf300"),
     stat_row("Phase: mean + SD over time (180-d)", "phase_mean+std_180|rf300"),
     stat_row("Amplitude: mean + SD over time (180-d)", "amp_mean+std_180|rf300")]
x = dur["class<-phase_time_mean_90+duration"]
p.append(probe_row("Phase: mean over time (90-d) + duration", x["per"], x["mean"]))
p.append(r"\midrule")
p.append(r"\multicolumn{5}{l}{\textit{Three 4\,s classes only (uniform-random reference 33.3)}}\\")
x = dur["F1_fall/walk/turn_only<-phase_link_slope_icpt_6"]
p.append(probe_row("Phase: link slope + intercept (6-d)", x, sum(x.values()) / 3))
x = dur["F1_fall/walk/turn_only<-amp_mean+std_180"]
p.append(probe_row("Amplitude: mean + SD over time (180-d)", x, sum(x.values()) / 3))
write("probe_table.tex", r"""\begin{table}[t]
\caption{Duration-shortcut probes: leave-one-environment-out macro-F1 (\%) of a random forest on scalar or time-aggregated per-segment features.}
\label{tab:probe}
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrr}
\toprule
Feature & E1 & E2 & E3 & Mean\\
\midrule
""" + "\n".join(p) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# Values quoted only in the manuscript text (kept here so that they are regenerated with the tables).
bi_path = "D:/pa-csi-dg-revision-runtime/studies/h_phase_link_unwrap_20261023_v2/analysis/results.json"
bi = json.load(open(bi_path, encoding="utf-8"))["contrasts"]["link_unwrap_minus_cross_link"]["bootstrap_95_interval_pp"]
text_values = {
    "uniform_grid_minus_original_pp": {"mean": con["N_H11-HS11"]["macro_mean_pp"], "by_env": con["N_H11-HS11"]["macro_by_env_pp"],
                                       "improved": con["N_H11-HS11"]["macro_positive"]},
    "within_minus_across_link_bootstrap_95_pp": bi,
    "r2_duration_from_phase_slope_intercept_6d": dur["R2_duration<-phase_link_slope_icpt_6"]["mean"],
    "r2_duration_from_amplitude_mean_sd_180d": dur["R2_duration<-amp_mean+std_180"]["mean"],
}
json.dump(text_values, open(os.path.join(HERE, "text_values.json"), "w", encoding="utf-8"), indent=1)
print("tables written:", sorted(os.listdir(OUT)))
print(json.dumps(text_values, indent=1))
