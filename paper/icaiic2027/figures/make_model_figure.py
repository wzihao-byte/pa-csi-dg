"""Draw figures/model_overview.pdf: the network and its training views (schematic, no data).

Drawn at the printed size (IEEEtran text width, 516 pt; one drawing unit is one point), so the font sizes
below are the printed sizes (8 pt, 7.5 pt for secondary notes). Every label was checked against the
implementation used by the main-study runs:
  experiments/tpp_isolated/tpp_models.py  (stream encoder, residual TPP on both branches)
  dg_models.py                            (MCAT blocks, context encoder, gated fusion, projection head, view masks)
  dg_losses.py / train_dg.py              (CE on the complete view only, hybrid SupCon over all views)
  job_config.json of the T-SC runs        (5 MCAT blocks, kernels 20/40 and 2/4, 128/16 channels, levels 1-2-4-8)
Blue marks the pooling factor of the ablation (TPP), orange the objective factor (SupCon, training only),
matching the colours of the two steps in hs_four_arm.pdf.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8,
})
W, H = 516.0, 150.0
INK, MUTED = "#0b0b0b", "#52514e"
NEUTRAL, NEUTRAL_EDGE = "#f4f3ef", "#8d8c86"
BLUE, BLUE_BG = "#2a78d6", "#dfeaf9"
ORANGE, ORANGE_BG = "#eb6834", "#fdebe2"
FS, FS2 = 8.0, 7.5

fig = plt.figure(figsize=(W / 72, H / 72))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x0, y0, x1, y1, label="", fc=NEUTRAL, ec=NEUTRAL_EDGE, ls="-", lw=0.8, size=FS, weight="normal", z=2, colour=INK):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=3", fc=fc, ec=ec, ls=ls, lw=lw, zorder=z))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, ha="center", va="center", fontsize=size, weight=weight, color=colour,
                linespacing=1.12, zorder=z + 1)


def arrow(x0, y0, x1, y1, colour=INK, ls="-", lw=0.8, z=4, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=6.5, lw=lw, color=colour, ls=ls,
                                 shrinkA=0, shrinkB=0, zorder=z))


def line(xs, ys, colour=INK, ls="-", lw=0.8, z=3):
    ax.plot(xs, ys, color=colour, ls=ls, lw=lw, zorder=z, solid_capstyle="butt")


def note(x, y, label, size=FS2, colour=MUTED, ha="center", va="center", weight="normal", style="normal"):
    ax.text(x, y, label, ha=ha, va=va, fontsize=size, color=colour, weight=weight, style=style, linespacing=1.12, zorder=6)


def view(x, y, kept, colour, edge):
    """Three 30-feature receive-link groups; groups not in `kept` are set to zero (white)."""
    for g in range(3):
        ax.add_patch(Rectangle((x + 17 * g, y), 17, 10, fc=colour if g in kept else "white", ec=edge, lw=0.7, zorder=3))


# ------------------------------------------------------------------ input views
note(46, 143, "Input views", size=FS, colour=INK, weight="bold")
note(60.5, 130, "receive link", size=FS2)
for g in range(3):
    note(43.5 + 17 * g, 121.5, str(g + 1), size=FS2)
view(35, 105, (0, 1, 2), "#c9c8c2", NEUTRAL_EDGE)
note(31, 110, r"$v=0$", size=FS, colour=INK, ha="right")
note(46, 96.5, r"complete: $A$, $\Phi$ ($850\times90$)", size=FS2)
box(3, 22, 89, 86, fc="none", ec=ORANGE, ls=(0, (3, 2)), lw=0.8, z=1)
for k in range(3):
    view(35, 66 - 16 * k, (k,), ORANGE_BG, ORANGE)
    note(31, 71 - 16 * k, f"$v={k + 1}$", size=FS, colour=INK, ha="right")
note(46, 26.5, "other links set to zero", size=FS2)
note(46, 14, "single-link views:", size=FS2, colour=ORANGE)
note(46, 5.5, "training only", size=FS2, colour=ORANGE)
line([86, 98], [110, 110])
line([89, 98], [54, 54], colour=ORANGE)
line([98, 98], [54, 110])
arrow(98, 78, 108, 78)

# ------------------------------------------------------------------ stream encoder (drawn once, used twice)
box(113, 10, 377, 127, fc="white", ec=NEUTRAL_EDGE, lw=0.7, z=0)       # phase stream (behind)
box(108, 15, 372, 132, fc="white", ec=INK, lw=0.8, z=1)                # amplitude stream (front)
note(240, 141.5, r"Stream encoder (one for $A$, one for $\Phi$; separate weights)", size=FS, colour=INK, weight="bold")
box(114, 40, 170, 108, "Frame\naveraging\n850 → 425\n+ Gaussian\npositional\nencoding")
# temporal branch
note(196, 126, "Temporal branch", size=FS2, ha="left")
box(187, 84, 243, 120, "5 MCAT\nblocks")
box(252, 84, 304, 120, "2 convs,\nkernels 20, 40,\n128 ch each", size=FS2)
box(314, 84, 352, 120, "Residual\nTPP\n256-d", fc=BLUE_BG, ec=BLUE, lw=1.0)
# context branch
note(196, 21, "Context branch", size=FS2, ha="left")
box(187, 27, 243, 63, "Link mean\n+ Transformer\nlayer", size=FS2)
box(252, 27, 304, 63, "2 convs,\nkernels 2, 4,\n16 ch each", size=FS2)
box(314, 27, 352, 63, "Residual\nTPP\n32-d", fc=BLUE_BG, ec=BLUE, lw=1.0)
line([170, 178], [74, 74])
line([178, 178], [45, 102])
arrow(178, 102, 187, 102)
arrow(178, 45, 187, 45)
arrow(243, 102, 252, 102)
arrow(304, 102, 314, 102)
arrow(243, 45, 252, 45)
arrow(304, 45, 314, 45)
line([352, 366], [102, 102])
line([352, 366], [45, 45])
line([366, 366], [45, 102])

# ------------------------------------------------------------------ fusion and heads
arrow(366, 78, 404, 78)
arrow(377, 66, 404, 66, colour=NEUTRAL_EDGE)
note(390, 85, r"$u_A$", size=FS, colour=INK)
note(391, 58.5, r"$u_\Phi$", size=FS, colour=INK)
box(404, 54, 450, 92, "Gated\nfusion\n" + r"$f$, 256-d")
line([450, 457], [73, 73])
line([457, 457], [73, 107])
arrow(457, 107, 465, 107)
line([457, 457], [41, 73], colour=ORANGE)
arrow(457, 41, 465, 41, colour=ORANGE)
box(465, 92, 514, 122, "Linear\nclassifier")
note(489.5, 138, "CE loss on the", size=FS2, colour=INK)
note(489.5, 129.5, "complete view", size=FS2, colour=INK)
box(465, 26, 514, 56, "Projection\nhead, 128-d", fc=ORANGE_BG, ec=ORANGE, lw=1.0, size=FS2)
note(489.5, 17.5, "Hybrid SupCon", size=FS2, colour=ORANGE)
note(489.5, 9, "over all views", size=FS2, colour=ORANGE)

fig.savefig(os.path.join(HERE, "model_overview.pdf"))
fig.savefig(os.path.join(HERE, "model_overview.png"), dpi=220)
print("wrote model_overview.pdf")
