"""Plot the all-class confusion matrix for the completed H11 L1248 experiment.

The figure aggregates the nine target-by-seed confusion matrices from the
completed level-8 experiment, then normalizes each true-class row.  Keeping the
aggregation in count space gives the same weight to every evaluated sample.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "outputs_tpp_level8_ablation" / "l1248_v1"
DEFAULT_OUTPUT_DIR = HERE / "figures"

TARGETS = ("E1", "E2", "E3")
SEEDS = (42, 52, 62)
CLASS_LABELS = (
    "C0  No movement",
    "C1  Falling",
    "C2  Walking",
    "C3  Sitting / standing",
    "C4  Turning",
    "C5  Picking up a pen",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root containing level8_results.json and runs/ (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG, PDF, SVG, and CSV outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--stem",
        default="new_experiment_h11_confusion_matrix",
        help="Output filename stem",
    )
    return parser.parse_args()


def load_completed_confusions(results_root: Path) -> tuple[np.ndarray, list[Path]]:
    results_path = results_root / "level8_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("status") != "complete" or int(results.get("job_count", -1)) != 9:
        raise RuntimeError(f"Expected a complete nine-job experiment: {results_path}")

    matrices: list[np.ndarray] = []
    paths: list[Path] = []
    for target in TARGETS:
        for seed in SEEDS:
            path = (
                results_root
                / "runs"
                / target
                / f"seed_{seed}"
                / "H11"
                / "attempt_001"
                / "payload"
                / "run"
                / "confusion_matrix.csv"
            )
            matrix = np.loadtxt(path, delimiter=",", dtype=np.int64)
            if matrix.shape != (6, 6):
                raise RuntimeError(f"Expected a 6x6 confusion matrix: {path}")
            if np.any(matrix < 0) or np.any(matrix.sum(axis=1) == 0):
                raise RuntimeError(f"Invalid confusion counts: {path}")
            matrices.append(matrix)
            paths.append(path)

    total = np.sum(np.stack(matrices, axis=0), axis=0)
    return total, paths


def write_csv(path: Path, counts: np.ndarray, percentages: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["true_class", "support"]
            + [f"pred_{index}_count" for index in range(6)]
            + [f"pred_{index}_percent" for index in range(6)]
        )
        for index, label in enumerate(CLASS_LABELS):
            writer.writerow(
                [label, int(counts[index].sum())]
                + [int(value) for value in counts[index]]
                + [f"{value:.6f}" for value in percentages[index]]
            )


def plot_matrix(percentages: np.ndarray, output_dir: Path, stem: str) -> None:
    ink = "#263449"
    muted = "#5f6f85"

    fig, ax = plt.subplots(figsize=(9.6, 8.15), constrained_layout=True, facecolor="white")
    image = ax.imshow(percentages, cmap="Purples", vmin=0.0, vmax=100.0, aspect="equal")

    positions = np.arange(6)
    ax.set_xticks(positions, CLASS_LABELS, rotation=34, ha="right", rotation_mode="anchor")
    ax.set_yticks(positions, CLASS_LABELS)
    ax.set_xlabel("Predicted class", color=ink, labelpad=10)
    ax.set_ylabel("True class", color=ink, labelpad=10)
    ax.set_title(
        "H11 residual TPP + SupCon: all-class confusion matrix\n"
        "E1–E3 × seeds 42/52/62 · row-normalized",
        loc="left",
        color=ink,
        fontweight="bold",
        pad=16,
    )

    for row in range(6):
        for column in range(6):
            value = percentages[row, column]
            label = "<0.1" if 0.0 < value < 0.05 else f"{value:.1f}"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color="white" if value >= 55.0 else ink,
                fontsize=10.5,
                fontweight="bold" if row == column else "normal",
            )

    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", colors=ink, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(muted)
        spine.set_linewidth(0.8)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Samples within true class (%)", color=ink, labelpad=10)
    colorbar.ax.tick_params(colors=ink)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    counts, source_paths = load_completed_confusions(results_root)
    percentages = counts / counts.sum(axis=1, keepdims=True) * 100.0

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / f"{args.stem}.csv", counts, percentages)
    plot_matrix(percentages, output_dir, args.stem)

    print(
        json.dumps(
            {
                "status": "ok",
                "source_matrix_count": len(source_paths),
                "evaluated_samples": int(counts.sum()),
                "outputs": {
                    suffix: str((output_dir / f"{args.stem}.{suffix}").resolve())
                    for suffix in ("png", "pdf", "svg", "csv")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
