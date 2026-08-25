from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_tpp_isolated import hard_pair_report


REPORT_FIELDS = [
    "experiment_name",
    "target_env",
    "seed",
    "baseline_metrics_path",
    "tpp_metrics_path",
    "baseline_accuracy",
    "tpp_accuracy",
    "delta_accuracy",
    "baseline_f1_macro",
    "tpp_f1_macro",
    "delta_f1_macro",
    "baseline_class1_acc",
    "tpp_class1_acc",
    "delta_class1_acc",
    "baseline_class4_acc",
    "tpp_class4_acc",
    "delta_class4_acc",
    "baseline_true1_pred4_rate",
    "tpp_true1_pred4_rate",
    "delta_true1_pred4_rate",
    "baseline_true4_pred1_rate",
    "tpp_true4_pred1_rate",
    "delta_true4_pred1_rate",
    "baseline_pair_confusion_rate",
    "tpp_pair_confusion_rate",
    "delta_pair_confusion_rate",
    "relative_pair_confusion_reduction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare isolated TPP runs against baseline hard-pair confusion.")
    parser.add_argument(
        "--tpp-root",
        default="outputs_tpp_isolated/dg_loeo",
        help="Root containing isolated TPP outputs.",
    )
    parser.add_argument(
        "--baseline-root",
        action="append",
        default=None,
        help="Root containing baseline outputs with confusion matrices.",
    )
    parser.add_argument(
        "--analysis-root",
        default="analysis_tpp_isolated",
        help="Directory where comparison artifacts are written.",
    )
    parser.add_argument("--class-a", type=int, default=1, help="First hard-pair class index.")
    parser.add_argument("--class-b", type=int, default=4, help="Second hard-pair class index.")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_confusion(run_dir: Path) -> np.ndarray:
    npy_path = run_dir / "confusion_matrix.npy"
    if npy_path.exists():
        return np.load(npy_path)
    csv_path = run_dir / "confusion_matrix.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No confusion matrix found in {run_dir}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return np.array([[int(value) for value in row] for row in csv.reader(handle)], dtype=np.int64)


def infer_experiment_name(metrics_path: Path) -> str:
    parts = metrics_path.parts
    for index, part in enumerate(parts):
        if part.startswith("target_") and index > 0:
            return parts[index - 1]
    return ""


def row_key(metrics: Mapping[str, Any], metrics_path: Path) -> Tuple[str, int]:
    target_env = str(metrics.get("target_env") or "")
    seed = int(metrics.get("seed", 0))
    if not target_env:
        for part in metrics_path.parts:
            if part.startswith("target_"):
                target_env = part.removeprefix("target_")
    return target_env, seed


def metric_rows(root: Path, class_a: int, class_b: int) -> List[Dict[str, Any]]:
    rows = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        metrics = read_json(metrics_path)
        confusion = read_confusion(metrics_path.parent)
        hard_pair = hard_pair_report(confusion, class_a=class_a, class_b=class_b)
        test_metrics = metrics.get("test_metrics", {}) or {}
        target_env, seed = row_key(metrics, metrics_path)
        rows.append(
            {
                "metrics_path": metrics_path,
                "run_dir": metrics_path.parent,
                "experiment_name": metrics.get("experiment_name") or infer_experiment_name(metrics_path),
                "target_env": target_env,
                "seed": seed,
                "accuracy": float(test_metrics.get("accuracy", 0.0)),
                "f1_macro": float(test_metrics.get("f1_macro", 0.0)),
                **hard_pair,
                "confusion": confusion,
            }
        )
    return rows


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1, keepdims=True).astype(float)
    return np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0)


def safe_delta(left: float, right: float) -> float:
    return float(left) - float(right)


def comparison_rows(tpp_rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    baseline_by_key: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for row in baseline_rows:
        key = (str(row["target_env"]), int(row["seed"]))
        baseline_by_key.setdefault(key, row)

    rows = []
    for tpp in tpp_rows:
        key = (str(tpp["target_env"]), int(tpp["seed"]))
        baseline = baseline_by_key.get(key)
        if baseline is None:
            continue
        base_pair = float(baseline["pair_confusion_rate"])
        tpp_pair = float(tpp["pair_confusion_rate"])
        rows.append(
            {
                "experiment_name": tpp["experiment_name"],
                "target_env": key[0],
                "seed": key[1],
                "baseline_metrics_path": str(baseline["metrics_path"]),
                "tpp_metrics_path": str(tpp["metrics_path"]),
                "baseline_accuracy": baseline["accuracy"],
                "tpp_accuracy": tpp["accuracy"],
                "delta_accuracy": safe_delta(tpp["accuracy"], baseline["accuracy"]),
                "baseline_f1_macro": baseline["f1_macro"],
                "tpp_f1_macro": tpp["f1_macro"],
                "delta_f1_macro": safe_delta(tpp["f1_macro"], baseline["f1_macro"]),
                "baseline_class1_acc": baseline["class1_acc"],
                "tpp_class1_acc": tpp["class1_acc"],
                "delta_class1_acc": safe_delta(tpp["class1_acc"], baseline["class1_acc"]),
                "baseline_class4_acc": baseline["class4_acc"],
                "tpp_class4_acc": tpp["class4_acc"],
                "delta_class4_acc": safe_delta(tpp["class4_acc"], baseline["class4_acc"]),
                "baseline_true1_pred4_rate": baseline["true1_pred4_rate"],
                "tpp_true1_pred4_rate": tpp["true1_pred4_rate"],
                "delta_true1_pred4_rate": safe_delta(tpp["true1_pred4_rate"], baseline["true1_pred4_rate"]),
                "baseline_true4_pred1_rate": baseline["true4_pred1_rate"],
                "tpp_true4_pred1_rate": tpp["true4_pred1_rate"],
                "delta_true4_pred1_rate": safe_delta(tpp["true4_pred1_rate"], baseline["true4_pred1_rate"]),
                "baseline_pair_confusion_rate": base_pair,
                "tpp_pair_confusion_rate": tpp_pair,
                "delta_pair_confusion_rate": safe_delta(tpp_pair, base_pair),
                "relative_pair_confusion_reduction": ((base_pair - tpp_pair) / base_pair) if base_pair else 0.0,
                "_baseline_confusion": baseline["confusion"],
                "_tpp_confusion": tpp["confusion"],
            }
        )
    return rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_delta_heatmaps(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    for row in rows:
        experiment = str(row["experiment_name"])
        target_env = str(row["target_env"])
        seed = int(row["seed"])
        delta = normalize_rows(row["_tpp_confusion"]) - normalize_rows(row["_baseline_confusion"])
        stem = f"{experiment}_target_{target_env}_seed_{seed}_row_norm_delta"
        csv_path = output_dir / f"{stem}.csv"
        np.savetxt(csv_path, delta, delimiter=",", fmt="%.6f")
        written.append(csv_path)

        if plt is None:
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(delta, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_title(f"{experiment} target {target_env} seed {seed}")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_xticks(range(delta.shape[1]))
        ax.set_yticks(range(delta.shape[0]))
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        png_path = output_dir / f"{stem}.png"
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        written.append(png_path)
    return written


def format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_experiment: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_experiment.setdefault(str(row["experiment_name"]), []).append(row)

    lines = [
        "# TPP vs Baseline Hard-Pair Summary",
        "",
        "Baseline is matched by target environment and seed. Pair confusion is `(CM[1,4] + CM[4,1]) / (row1 + row4)`.",
        "",
        "| experiment | runs | pair confusion baseline | pair confusion TPP | relative reduction | delta macro F1 | delta class 1 recall | delta class 4 recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment, experiment_rows in sorted(by_experiment.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    experiment,
                    str(len(experiment_rows)),
                    format_float(mean(float(row["baseline_pair_confusion_rate"]) for row in experiment_rows)),
                    format_float(mean(float(row["tpp_pair_confusion_rate"]) for row in experiment_rows)),
                    format_float(mean(float(row["relative_pair_confusion_reduction"]) for row in experiment_rows)),
                    format_float(mean(float(row["delta_f1_macro"]) for row in experiment_rows)),
                    format_float(mean(float(row["delta_class1_acc"]) for row in experiment_rows)),
                    format_float(mean(float(row["delta_class4_acc"]) for row in experiment_rows)),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-Run Rows",
            "",
            "| experiment | target | seed | base pair | tpp pair | reduction | base F1 | tpp F1 | delta F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["experiment_name"]), str(item["target_env"]), int(item["seed"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["experiment_name"]),
                    str(row["target_env"]),
                    str(row["seed"]),
                    format_float(row["baseline_pair_confusion_rate"]),
                    format_float(row["tpp_pair_confusion_rate"]),
                    format_float(row["relative_pair_confusion_reduction"]),
                    format_float(row["baseline_f1_macro"]),
                    format_float(row["tpp_f1_macro"]),
                    format_float(row["delta_f1_macro"]),
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    tpp_root = (REPO_ROOT / args.tpp_root).resolve()
    baseline_roots = [
        (REPO_ROOT / baseline_root).resolve()
        for baseline_root in (args.baseline_root or ["outputs_anchor_ce_baseline"])
    ]
    analysis_root = (REPO_ROOT / args.analysis_root).resolve()

    tpp_rows = metric_rows(tpp_root, class_a=args.class_a, class_b=args.class_b)
    baseline_rows = []
    for baseline_root in baseline_roots:
        baseline_rows.extend(metric_rows(baseline_root, class_a=args.class_a, class_b=args.class_b))
    rows = comparison_rows(tpp_rows, baseline_rows)
    if not rows:
        raise SystemExit(
            f"No matched TPP/baseline rows found. TPP root={tpp_root}; baseline roots={baseline_roots}."
        )

    write_csv(analysis_root / "tpp_vs_baseline_hard_pair.csv", REPORT_FIELDS, rows)
    write_delta_heatmaps(rows, analysis_root / "confusion_delta_heatmaps")
    write_summary(analysis_root / "tpp_vs_baseline_summary.md", rows)
    print(f"Wrote {len(rows)} comparison rows to {analysis_root}")


if __name__ == "__main__":
    main()
