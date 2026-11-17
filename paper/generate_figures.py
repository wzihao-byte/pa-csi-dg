from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "amp": "#D97706",
    "phase": "#2563EB",
    "tpp": "#059669",
    "supcon": "#7C3AED",
    "neutral": "#334155",
    "light": "#E2E8F0",
    "muted": "#64748B",
    "red": "#DC2626",
    "class1": "#2563EB",
    "class2": "#D97706",
}


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "png", "pdf"):
        fig.savefig(
            FIG_DIR / f"{stem}.{suffix}",
            dpi=220,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    color: str,
    fontsize: float = 10,
    text_color: str = "white",
    linewidth: float = 1.2,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.035",
        linewidth=linewidth,
        edgecolor=color,
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        weight="medium",
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["neutral"],
    style: str = "-|>",
    linewidth: float = 1.5,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def figure_problem_setting() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    ax.text(5.6, 5.3, "Source-only cross-environment CSI recognition", ha="center", fontsize=16, weight="medium")
    ax.text(1.85, 4.75, "Labeled source environments", ha="center", fontsize=12, color=COLORS["neutral"])
    ax.text(9.65, 4.75, "Unseen target environment", ha="center", fontsize=12, color=COLORS["neutral"])

    environments = [
        (0.95, 3.05, "Source E1", "LOS room", COLORS["class1"]),
        (0.95, 1.45, "Source E2", "LOS corridor", COLORS["class2"]),
        (8.75, 2.25, "Target E3", "NLOS / new layout", COLORS["red"]),
    ]
    for x, y, title, subtitle, color in environments:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), 1.80, 1.35,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor="white", edgecolor=color, linewidth=2.0,
            )
        )
        ax.text(x + 0.90, y + 1.10, title, ha="center", fontsize=11, weight="medium", color=color)
        ax.text(x + 0.90, y + 0.87, subtitle, ha="center", fontsize=9.5, color=COLORS["muted"])
        t = np.linspace(0, 1, 80)
        amp = 0.16 * np.sin(2 * np.pi * (2.4 * t + 0.15 * x)) + 0.08 * np.sin(2 * np.pi * 7 * t)
        phase = 0.13 * np.sin(2 * np.pi * (3.2 * t + 0.2))
        ax.plot(x + 0.27 + 1.28 * t, y + 0.50 + amp, color=COLORS["amp"], linewidth=1.5)
        ax.plot(x + 0.27 + 1.28 * t, y + 0.20 + phase, color=COLORS["phase"], linewidth=1.5)
        ax.text(x + 0.17, y + 0.52, "A", ha="right", va="center", fontsize=9, color=COLORS["amp"])
        ax.text(x + 0.17, y + 0.22, "φ", ha="right", va="center", fontsize=9, color=COLORS["phase"])

    rounded_box(ax, (4.55, 2.35), 2.15, 1.22, "Learn class semantics\nwithout target access", color=COLORS["neutral"], fontsize=10)
    arrow(ax, (2.78, 3.72), (4.50, 3.23), connectionstyle="arc3,rad=-0.08")
    arrow(ax, (2.78, 2.12), (4.50, 2.70), connectionstyle="arc3,rad=0.08")
    arrow(ax, (6.73, 2.96), (8.70, 2.96), color=COLORS["red"])

    ax.text(6.0, 1.27, "Desired representation", ha="center", fontsize=11.5, weight="medium")
    ax.text(
        6.0, 0.87,
        "same activity across sources → nearby    |    different activities → separated",
        ha="center", fontsize=10.5, color=COLORS["neutral"],
    )
    ax.text(
        6.0, 0.30,
        "Target samples are excluded from training, validation, calibration, and model selection.",
        ha="center", fontsize=10, color=COLORS["muted"],
    )
    save(fig, "fig1_problem_setting")


def figure_architecture() -> None:
    fig, ax = plt.subplots(figsize=(14.2, 8.2))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    ax.text(7.1, 7.88, "Current PA-CSI-DG model pipeline", ha="center", fontsize=17, weight="medium")
    ax.text(
        7.1,
        7.50,
        "Amplitude-only residual TPP; one fused SupCon projection per sample",
        ha="center",
        fontsize=11,
        color=COLORS["muted"],
    )

    amp_y, phase_y = 5.35, 2.30
    rounded_box(ax, (0.18, amp_y), 1.18, 0.86, "Amplitude CSI\n[B x 850 x 90]", color=COLORS["amp"], fontsize=9.2)
    rounded_box(ax, (0.18, phase_y), 1.18, 0.86, "Raw phase CSI\n[B x 850 x 90]", color=COLORS["phase"], fontsize=9.2)

    for y in (amp_y, phase_y):
        rounded_box(ax, (1.72, y - 0.10), 1.45, 1.06, "Time reduction x2\nGaussian relative\nposition encoding", color=COLORS["neutral"], fontsize=8.8)
        rounded_box(
            ax,
            (3.53, y - 0.22),
            2.45,
            1.30,
            "Temporal path: 5x MCAT\n+ Conv1D kernels 20/40\nChannel path: 3x30 groups\n+ Transformer + kernels 2/4",
            color=COLORS["neutral"],
            fontsize=8.35,
        )

    rounded_box(
        ax,
        (6.38, amp_y - 0.22),
        2.38,
        1.30,
        "Both amplitude paths\nGlobal max g + 0.25 t\nTPP levels [1,2,4]\n7d -> d projection",
        color=COLORS["tpp"],
        fontsize=8.7,
    )
    rounded_box(
        ax,
        (6.38, phase_y - 0.22),
        2.38,
        1.30,
        "Phase paths unchanged\nGlobal pooling only\n(no residual TPP)",
        color=COLORS["phase"],
        fontsize=9.1,
    )
    rounded_box(ax, (9.12, amp_y - 0.02), 1.18, 0.90, "Temporal +\nchannel concat", color=COLORS["amp"], fontsize=8.1)
    rounded_box(ax, (9.12, phase_y - 0.02), 1.18, 0.90, "Temporal +\nchannel concat", color=COLORS["phase"], fontsize=8.1)

    rounded_box(ax, (10.76, 3.63), 1.42, 1.18, "GRN gated\namplitude-phase\nfusion (256-D)", color=COLORS["neutral"], fontsize=8.9)
    rounded_box(ax, (12.70, 5.16), 1.22, 0.86, "6-class\nclassifier", color=COLORS["neutral"], fontsize=9.3)
    rounded_box(ax, (12.46, 2.50), 1.70, 1.02, "Train-only projection\n256 -> 128 + L2\none vector/sample", color=COLORS["supcon"], fontsize=8.2)

    for start, end in [
        ((1.39, amp_y + 0.43), (1.68, amp_y + 0.43)),
        ((3.20, amp_y + 0.43), (3.49, amp_y + 0.43)),
        ((6.01, amp_y + 0.43), (6.34, amp_y + 0.43)),
        ((8.79, amp_y + 0.43), (9.08, amp_y + 0.43)),
        ((1.39, phase_y + 0.43), (1.68, phase_y + 0.43)),
        ((3.20, phase_y + 0.43), (3.49, phase_y + 0.43)),
        ((6.01, phase_y + 0.43), (6.34, phase_y + 0.43)),
        ((8.79, phase_y + 0.43), (9.08, phase_y + 0.43)),
    ]:
        arrow(ax, start, end)
    arrow(ax, (10.34, amp_y + 0.43), (10.72, 4.52), connectionstyle="arc3,rad=-0.10")
    arrow(ax, (10.34, phase_y + 0.43), (10.72, 3.92), connectionstyle="arc3,rad=0.10")
    arrow(ax, (12.22, 4.47), (12.66, 5.55), connectionstyle="arc3,rad=-0.12")
    arrow(ax, (12.22, 3.96), (12.42, 3.02), connectionstyle="arc3,rad=0.12")

    ax.text(13.30, 6.26, r"$\mathcal{L}_{CE}$", ha="center", fontsize=12, color=COLORS["neutral"])
    ax.text(13.30, 2.14, r"$\mathcal{L}_{SupCon}$", ha="center", fontsize=12, color=COLORS["supcon"])
    ax.text(11.46, 3.35, r"$h$", ha="center", fontsize=11)

    ax.add_patch(FancyBboxPatch((1.85, 0.88), 10.55, 0.48, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor="#F1F5F9", edgecolor=COLORS["light"]))
    ax.text(
        7.12,
        1.12,
        "Protocol correction from the reports: V=1 global fused view; antenna-masked views are disabled.",
        ha="center",
        va="center",
        fontsize=9.7,
        color=COLORS["neutral"],
    )
    ax.text(7.1, 0.42, r"Joint objective:  $\mathcal{L}=\mathcal{L}_{CE}+\lambda(t)\mathcal{L}_{SupCon}$", ha="center", fontsize=12.3, weight="medium")
    save(fig, "fig2_architecture")


def figure_residual_tpp() -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11.6, 7.8), gridspec_kw={"height_ratios": [2.2, 1, 1, 1.2]})
    t = np.linspace(0, 1, 320)
    response = (
        0.18
        + 0.18 * np.sin(2 * np.pi * 1.5 * t)
        + 0.95 * np.exp(-((t - 0.31) / 0.045) ** 2)
        + 0.62 * np.exp(-((t - 0.76) / 0.07) ** 2)
    )
    response = np.maximum(response, 0)

    ax = axes[0]
    ax.plot(t, response, color=COLORS["tpp"], linewidth=2.2)
    ax.fill_between(t, 0, response, color=COLORS["tpp"], alpha=0.10)
    ax.set_ylabel("Conv. response")
    ax.set_title("Temporal pyramid preserves where discriminative responses occur", fontsize=14, weight="medium")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, response.max() * 1.15)
    ax.grid(axis="y", color=COLORS["light"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([])

    levels = [1, 2, 4]
    for row_ax, level in zip(axes[1:4], levels):
        row_ax.set_xlim(0, 1)
        row_ax.set_ylim(0, 1)
        row_ax.axis("off")
        row_ax.text(-0.035, 0.50, f"Level {level}", ha="right", va="center", fontsize=10.5, weight="medium")
        for index in range(level):
            left, right = index / level, (index + 1) / level
            mask = (t >= left) & (t <= right)
            pooled = float(response[mask].max())
            normalized = pooled / response.max()
            row_ax.add_patch(Rectangle((left + 0.004, 0.12), 1 / level - 0.008, 0.72, facecolor="#ECFDF5", edgecolor=COLORS["tpp"], linewidth=1.3))
            row_ax.add_patch(Rectangle((left + 0.02 / level, 0.15), 0.11 / level, 0.64 * normalized, facecolor=COLORS["tpp"], alpha=0.75, edgecolor="none"))
            row_ax.text((left + right) / 2, 0.50, rf"$\max_{{t\in B_{index+1}}}r(t)$", ha="center", va="center", fontsize=9.5, color=COLORS["neutral"])
    axes[-1].text(0.5, -0.17, "Normalized time", ha="center", va="top", fontsize=10.5)
    fig.text(0.5, 0.015, r"Residual output:  $h_a=g_a+\alpha W[P_1,P_2,P_4]$,  with $\alpha=0.25$", ha="center", fontsize=12, weight="medium")
    fig.subplots_adjust(hspace=0.18, bottom=0.09)
    save(fig, "fig3_residual_tpp")


def figure_supcon_geometry() -> None:
    rng = np.random.default_rng(17)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), sharex=True, sharey=True)
    environments = {"E1": "o", "E2": "^"}
    class_colors = {"Falling": COLORS["class1"], "Turning": COLORS["class2"]}

    before_centers = {
        ("Falling", "E1"): (-1.9, 0.9), ("Falling", "E2"): (0.1, 1.9),
        ("Turning", "E1"): (-0.1, -1.5), ("Turning", "E2"): (1.9, -0.35),
    }
    after_centers = {
        ("Falling", "E1"): (-1.25, 1.05), ("Falling", "E2"): (-0.85, 1.25),
        ("Turning", "E1"): (0.90, -1.00), ("Turning", "E2"): (1.25, -0.80),
    }

    for ax, centers, title in zip(
        axes,
        (before_centers, after_centers),
        ("CE alone: class clusters may remain domain-split", "SupCon: cross-domain positives organize class geometry"),
    ):
        for (class_name, env), center in centers.items():
            points = rng.normal(loc=center, scale=(0.35, 0.30), size=(15, 2))
            ax.scatter(
                points[:, 0], points[:, 1], s=46, marker=environments[env],
                facecolor=class_colors[class_name], edgecolor="white", linewidth=0.7, alpha=0.85,
            )
        ax.axhline(0, color=COLORS["light"], linewidth=0.9)
        ax.axvline(0, color=COLORS["light"], linewidth=0.9)
        ax.set_title(title, fontsize=11.2, weight="medium")
        ax.set_xlabel("Embedding dimension 1")
        ax.grid(color=COLORS["light"], linewidth=0.7, alpha=0.55)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Embedding dimension 2")
    axes[0].set_xlim(-3.0, 3.0)
    axes[0].set_ylim(-2.7, 2.8)

    class_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=8, label=label)
        for label, color in class_colors.items()
    ]
    env_handles = [
        plt.Line2D([0], [0], marker=marker, color=COLORS["neutral"], linestyle="none", markersize=7, label=env)
        for env, marker in environments.items()
    ]
    fig.legend(handles=class_handles + env_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.text(
        0.5, 0.04,
        "Schematic only: color denotes activity; marker shape denotes source environment.",
        ha="center", fontsize=9.5, color=COLORS["muted"],
    )
    fig.subplots_adjust(bottom=0.19, wspace=0.16)
    save(fig, "fig4_supcon_geometry")


def figure_class_analysis() -> None:
    """Corrected class-wise signal diagnostic distilled from the reports."""
    labels = [
        "C0 No movement",
        "C1 Falling",
        "C2 Walking",
        "C3 Sit/stand",
        "C4 Turning",
        "C5 Pick up pen",
    ]
    # The original June plot swapped the display names for C2 and C3. These
    # values retain the verified numeric class IDs and use the corrected names.
    amp_ac_rms = np.array([1.486, 3.430, 6.512, 4.868, 5.729, 3.741])
    amp_lf_ratio = np.array([0.302, 0.826, 0.698, 0.810, 0.891, 0.863])
    phase_velocity_rms = np.array([1.305, 1.434, 1.451, 1.149, 1.442, 1.422])
    bar_colors = ["#94A3B8", COLORS["red"], "#60A5FA", "#10B981", COLORS["amp"], "#A78BFA"]

    fig = plt.figure(figsize=(13.6, 7.1))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.30)
    left = outer[0].subgridspec(3, 1, hspace=0.60)
    right = outer[1].subgridspec(3, 1, hspace=0.58)
    y = np.arange(len(labels))

    metrics = [
        (amp_ac_rms, "Amplitude AC RMS", "relative amplitude"),
        (amp_lf_ratio, "Amplitude low-frequency power ratio", "fraction"),
        (phase_velocity_rms, "Phase-velocity RMS", "radians / frame"),
    ]
    for index, (values, title, xlabel) in enumerate(metrics):
        ax = fig.add_subplot(left[index])
        ax.barh(y, values, color=bar_colors, alpha=0.90)
        ax.set_title(title, loc="left", fontsize=11.5, weight="medium")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_yticks(y, labels if index == 0 else [f"C{i}" for i in range(6)])
        ax.invert_yaxis()
        ax.grid(axis="x", color=COLORS["light"], linewidth=0.8, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8.7)
        ax.tick_params(axis="x", labelsize=8.5)

    t = np.linspace(0, 1, 320)
    fall = 0.08 + 1.10 * np.exp(-((t - 0.34) / 0.055) ** 2)
    turn = 0.10 + 0.32 * (1 + np.sin(2 * np.pi * 5.2 * t + 0.4)) * np.exp(-((t - 0.56) / 0.31) ** 4)
    ax = fig.add_subplot(right[0])
    ax.plot(t, fall, color=COLORS["red"], linewidth=2.0, label="C1 falling: localized transient")
    ax.plot(t, turn, color=COLORS["amp"], linewidth=2.0, label="C4 turning: sustained motion")
    ax.set_title("Hard pair: similar summaries, different locality", loc="left", fontsize=11.3, weight="medium")
    ax.set_ylabel("response")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    ax = fig.add_subplot(right[1])
    event_a = 0.08 + 0.92 / (1 + np.exp(-(t - 0.36) / 0.025))
    event_b = 0.08 + 0.92 / (1 + np.exp(-(t - 0.62) / 0.025))
    ax.plot(t, event_a, color=COLORS["tpp"], linewidth=2.0, label="source timing A")
    ax.plot(t, event_b, color=COLORS["tpp"], linewidth=2.0, linestyle="--", label="source timing B")
    ax.set_title("C3 sit/stand: transition timing can shift", loc="left", fontsize=11.3, weight="medium")
    ax.set_ylabel("response")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    ax = fig.add_subplot(right[2])
    rng = np.random.default_rng(23)
    static = 0.06 * rng.normal(size=t.size)
    static += 0.34 * np.exp(-((t - 0.78) / 0.012) ** 2)
    ax.plot(t, static, color=COLORS["muted"], linewidth=1.4)
    ax.axvspan(0.755, 0.805, color=COLORS["red"], alpha=0.08)
    ax.text(0.77, 0.38, "isolated noise spike", ha="center", fontsize=8.5, color=COLORS["red"])
    ax.set_title("C0 no movement: max pooling needs a global safety path", loc="left", fontsize=11.0, weight="medium")
    ax.set_ylabel("low-power response")
    ax.set_xlabel("normalized time")

    for trace_ax in fig.axes[-3:]:
        trace_ax.grid(axis="y", color=COLORS["light"], linewidth=0.8, alpha=0.75)
        trace_ax.spines[["top", "right"]].set_visible(False)
        trace_ax.set_xlim(0, 1)
        trace_ax.tick_params(labelsize=8.5)

    fig.suptitle("Class-informed motivation for residual temporal pooling", fontsize=16, weight="medium", y=0.995)
    fig.text(
        0.5,
        0.006,
        "Report-derived diagnostic: 240 examples per class (80 per environment); verified mapping C2=walking and C3=sit/stand.",
        ha="center",
        fontsize=9.4,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(top=0.92, bottom=0.09)
    save(fig, "fig5_class_analysis")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": COLORS["neutral"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["neutral"],
        }
    )
    figure_problem_setting()
    figure_architecture()
    figure_residual_tpp()
    figure_supcon_geometry()
    figure_class_analysis()


if __name__ == "__main__":
    main()
