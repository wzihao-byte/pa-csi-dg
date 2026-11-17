"""Build figures/hs_four_arm.pdf from evidence/classwise_summary.json (no training, no target data).

The figure shows what the tables cannot: the individual target-seed pairs.
  (a) macro-F1 of the 36 main-study runs; a line joins the CE and SupCon runs of one target-seed pair.
  (b) paired class-F1 differences of the nine pairs for the two steps that lead to T-SC.
The figure is drawn at its printed size (text width of IEEEtran, 516 pt), so font sizes are the printed sizes.
Colours are slots 1-3 of a validated categorical palette, used in fixed order, and every series also has
its own marker shape so that identity never depends on colour alone (greyscale print).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "..", "evidence", "classwise_summary.json"), encoding="utf-8"))
SUMM, CON = S["summary"], S["contrasts"]

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
ENV = {"E1": ("#2a78d6", "o"), "E2": ("#eb6834", "s"), "E3": ("#1baf7a", "^")}
STEP = {"HS10-G10": ("#2a78d6", "o", "TPP gain: T-CE $-$ G-CE"),
        "HS11-HS10": ("#eb6834", "D", "SupCon gain: T-SC $-$ T-CE")}


def runs(arm):
    return {(r["env"], r["seed"]): r for r in SUMM[arm]["runs"]}


def signed(v, nd=2):
    return ("−" if v < 0 else "+") + f"{abs(v):.{nd}f}"


fig = plt.figure(figsize=(7.16, 2.0))
ax1 = fig.add_axes([0.062, 0.215, 0.375, 0.69])
ax2 = fig.add_axes([0.525, 0.215, 0.468, 0.69])

# ------------------------------------------------------------------ (a) the 36 runs, paired by target and seed
XPOS = {"G10": 0.0, "G11": 1.0, "HS10": 2.45, "HS11": 3.45}
for ce, sc, key in (("G10", "G11", "G11-G10"), ("HS10", "HS11", "HS11-HS10")):
    a, b = runs(ce), runs(sc)
    assert sorted(a) == sorted(b) and len(a) == 9
    for (env, seed) in sorted(a):
        colour, marker = ENV[env]
        ax1.plot([XPOS[ce], XPOS[sc]], [a[(env, seed)]["macro_f1"], b[(env, seed)]["macro_f1"]], color=colour, lw=0.8,
                 alpha=0.85, marker=marker, ms=4.2, mec="white", mew=0.5, zorder=3)
    diffs = [b[k]["macro_f1"] - a[k]["macro_f1"] for k in a]
    mean = sum(diffs) / 9
    improved = sum(d > 0 for d in diffs)
    assert abs(mean - CON[key]["macro_mean_pp"]) < 1e-6 and improved == CON[key]["macro_positive"]
    ax1.text((XPOS[ce] + XPOS[sc]) / 2, 94.45, f"SupCon gain: {signed(mean)} pp\n{improved} of 9 runs improved",
             ha="center", va="center", fontsize=8, color=INK, linespacing=1.15)
for arm, x in XPOS.items():
    ax1.hlines(SUMM[arm]["macro_f1_equal_env_mean"], x - 0.3, x + 0.3, color=INK, lw=2.0, zorder=4)
ax1.set_xlim(-0.6, 4.05)
ax1.set_ylim(75.5, 96.4)
ax1.set_yticks([76, 80, 84, 88, 92])
ax1.set_xticks(list(XPOS.values()))
ax1.set_xticklabels(["G-CE", "G-SC", "T-CE", "T-SC"])
ax1.set_ylabel("Target macro-F1 (%)")
ax1.set_title("(a) The 36 main-study runs, paired by target and seed", loc="left", pad=4)
ax1.grid(axis="y", color=GRID, lw=0.5)
ax1.set_axisbelow(True)
for x0, x1, label in ((0.0, 1.0, "Global pooling"), (2.45, 3.45, "TPP")):
    ax1.text((x0 + x1) / 2, -0.19, label, transform=ax1.get_xaxis_transform(), ha="center", va="top", fontsize=8, color=MUTED)
handles = [Line2D([0], [0], color=c, marker=m, ms=4.2, mec="white", mew=0.5, lw=0.8, label="Target " + e) for e, (c, m) in ENV.items()]
handles.append(Line2D([0], [0], color=INK, lw=2.0, label="Arm mean"))
ax1.legend(handles=handles, loc="lower right", frameon=False, handlelength=1.6, handletextpad=0.5, labelspacing=0.25,
           borderaxespad=0.2)

# ------------------------------------------------------------------ (b) paired class-F1 differences
ORDER = [(1, "Falling"), (2, "Walking"), (4, "Turning"), (0, "No\nmovement"), (3, "Sitting/\nstanding"), (5, "Pen\npick-up")]
JITTER = [-0.055, -0.04, -0.027, -0.013, 0.0, 0.013, 0.027, 0.04, 0.055]
ax2.axvspan(-0.5, 2.5, color="#f1f0ec", zorder=0, lw=0)
ax2.axhline(0, color=MUTED, lw=0.7, zorder=1)
for offset, (key, (colour, marker, label)) in zip((-0.17, 0.17), STEP.items()):
    upper, lower = key.split("-")
    a, b = runs(lower), runs(upper)
    for position, (index, _) in enumerate(ORDER):
        diffs = sorted(b[k]["f1"][index] - a[k]["f1"][index] for k in sorted(a))
        mean = sum(diffs) / 9
        assert abs(mean - CON[key]["class_mean_pp"][index]) < 1e-6
        ax2.plot([position + offset + j for j in JITTER], diffs, ls="none", marker=marker, ms=3.4 if marker == "D" else 3.8,
                 mfc=colour, mec="white", mew=0.4, alpha=0.75, zorder=3)
        ax2.hlines(mean, position + offset - 0.13, position + offset + 0.13, color=INK, lw=1.8, zorder=4)
ax2.set_xlim(-0.5, 5.5)
ax2.set_ylim(-13, 24)
ax2.set_yticks([-10, 0, 10, 20])
ax2.set_xticks(range(6))
ax2.set_xticklabels([name for _, name in ORDER])
ax2.set_ylabel("Paired class-F1 difference (pp)")
ax2.set_title("(b) Class-F1 gain of each component in the nine paired runs", loc="left", pad=4)
ax2.grid(axis="y", color=GRID, lw=0.5)
ax2.set_axisbelow(True)
ax2.text(1.0, 22.6, "4 s classes", ha="center", va="center", fontsize=8, color=MUTED)
ax2.text(4.0, 22.6, "other classes", ha="center", va="center", fontsize=8, color=MUTED)
handles = [Line2D([0], [0], ls="none", marker=m, ms=4, mfc=c, mec="white", mew=0.4, label=label) for c, m, label in STEP.values()]
handles.append(Line2D([0], [0], color=INK, lw=1.8, label="Mean of nine paired runs"))
ax2.legend(handles=handles, loc="upper right", frameon=False, handlelength=1.3, handletextpad=0.5, labelspacing=0.25,
           borderaxespad=0.2, bbox_to_anchor=(1.0, 0.89))
for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5)
fig.savefig(os.path.join(HERE, "hs_four_arm.pdf"))
fig.savefig(os.path.join(HERE, "hs_four_arm.png"), dpi=220)
print("wrote hs_four_arm.pdf")
