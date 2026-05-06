from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


CLASS_NAMES = ["bed", "fall", "run", "sitdown", "standup", "walk"]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_confusion(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if row:
                rows.append([int(value) for value in row])
    return np.asarray(rows, dtype=np.int64)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def class_rows(confusion: np.ndarray, class_names: Iterable[str]) -> List[Dict[str, Any]]:
    names = list(class_names)
    rows: List[Dict[str, Any]] = []
    for index in range(confusion.shape[0]):
        tp = float(confusion[index, index])
        support = float(confusion[index, :].sum())
        predicted = float(confusion[:, index].sum())
        precision = safe_div(tp, predicted)
        recall = safe_div(tp, support)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        rows.append(
            {
                "class": names[index] if index < len(names) else str(index),
                "support": int(support),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def top_confusions(confusion: np.ndarray, class_names: Iterable[str], limit: int) -> List[Dict[str, Any]]:
    names = list(class_names)
    items: List[Dict[str, Any]] = []
    for true_index in range(confusion.shape[0]):
        for pred_index in range(confusion.shape[1]):
            if true_index == pred_index:
                continue
            count = int(confusion[true_index, pred_index])
            if count <= 0:
                continue
            items.append(
                {
                    "true": names[true_index] if true_index < len(names) else str(true_index),
                    "pred": names[pred_index] if pred_index < len(names) else str(pred_index),
                    "count": count,
                }
            )
    return sorted(items, key=lambda item: item["count"], reverse=True)[:limit]


def metric_summary(metrics: Mapping[str, Any]) -> Dict[str, float]:
    test_metrics = metrics.get("test_metrics", {})
    return {
        "accuracy": float(test_metrics.get("accuracy", 0.0)),
        "precision_macro": float(test_metrics.get("precision_macro", 0.0)),
        "recall_macro": float(test_metrics.get("recall_macro", 0.0)),
        "f1_macro": float(test_metrics.get("f1_macro", 0.0)),
        "loss_total": float(test_metrics.get("loss_total", 0.0)),
        "loss_ce": float(test_metrics.get("loss_ce", 0.0)),
        "loss_supcon": float(test_metrics.get("loss_supcon", 0.0)),
        "loss_sep": float(test_metrics.get("loss_sep", 0.0)),
    }


def load_reference(name: str, path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    metrics = read_json(path)
    return {"name": name, "path": str(path), "summary": metric_summary(metrics)}


def format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def render_report(
    run_dir: Path,
    metrics: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    confusion: np.ndarray,
    references: List[Dict[str, Any]],
) -> str:
    summary = metric_summary(metrics)
    rows = class_rows(confusion, CLASS_NAMES)
    confusions = top_confusions(confusion, CLASS_NAMES, limit=8)

    lines = [
        "# SimMMDG Held-Out Report",
        "",
        "## Run",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Target environment: `{metrics.get('target_env')}`",
        f"- Source environments: `{', '.join(metrics.get('source_envs', []))}`",
        f"- Seed: `{metrics.get('seed')}`",
        f"- Selection metric: `{metrics.get('selection_metric')}`",
        f"- Best validation score: `{float(metrics.get('best_val_score', 0.0)):.6f}`",
        "",
        "## Overall Test Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {format_pct(summary['accuracy'])}% |",
        f"| Macro precision | {format_pct(summary['precision_macro'])}% |",
        f"| Macro recall | {format_pct(summary['recall_macro'])}% |",
        f"| Macro F1 | {format_pct(summary['f1_macro'])}% |",
        f"| Loss total | {summary['loss_total']:.6f} |",
        f"| Loss CE | {summary['loss_ce']:.6f} |",
        f"| Loss SupCon | {summary['loss_supcon']:.6f} |",
        f"| Loss separation | {summary['loss_sep']:.6f} |",
        "",
        "## Class Performance",
        "",
        "| Class | Support | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['class']} | {row['support']} | {format_pct(row['precision'])}% | "
            f"{format_pct(row['recall'])}% | {format_pct(row['f1'])}% |"
        )

    lines.extend(
        [
            "",
            "## Top Confusions",
            "",
            "| True | Predicted | Count |",
            "|---|---|---:|",
        ]
    )
    for item in confusions:
        lines.append(f"| {item['true']} | {item['pred']} | {item['count']} |")

    lines.extend(
        [
            "",
            "## Split Check",
            "",
            f"- Train env counts: `{split_manifest.get('train', {}).get('env_counts', {})}`",
            f"- Val env counts: `{split_manifest.get('val', {}).get('env_counts', {})}`",
            f"- Test env counts: `{split_manifest.get('test', {}).get('env_counts', {})}`",
            "",
            "## Reference Metrics",
            "",
        ]
    )
    if references:
        lines.extend(["| Reference | Accuracy | Macro F1 | Delta Acc. vs SimMMDG | Path |", "|---|---:|---:|---:|---|"])
        for ref in references:
            ref_summary = ref["summary"]
            delta = summary["accuracy"] - ref_summary["accuracy"]
            lines.append(
                f"| {ref['name']} | {format_pct(ref_summary['accuracy'])}% | "
                f"{format_pct(ref_summary['f1_macro'])}% | {100.0 * delta:+.2f} pp | `{ref['path']}` |"
            )
    else:
        lines.append("No reference metrics were provided or found.")

    losses_enabled = metrics.get("losses_enabled", {})
    shape_debug = metrics.get("shape_debug", {})
    uses_multiview_bridge = "multi_antenna_views" in shape_debug
    uses_translation = float(losses_enabled.get("lambda_translation", 0.0)) > 0.0

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- This run uses the same data, split, sampler, seed, optimizer schedule, batch size, and selection metric as the provided comparison configs.",
            "- The intended method difference is the isolated SimMMDG shared/private decomposition and `lambda_sep` separation loss.",
        ]
    )
    if uses_multiview_bridge:
        lines.append("- This run uses the retained multiview bridge path with antenna-view shared/private decomposition.")
    if uses_translation:
        lines.append("- Cross-view translation regularization is enabled in this run.")
    lines.extend(["", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SimMMDG PA-CSI held-out report.")
    parser.add_argument("--run-dir", required=True, help="Directory containing metrics.json and confusion_matrix.csv.")
    parser.add_argument("--output", help="Markdown report path. Defaults to run_dir/report.md.")
    parser.add_argument("--supcon-reference", help="Optional SupCon E1 metrics.json path.")
    parser.add_argument("--adain-reference", help="Optional SupCon+AdaIN E1 metrics.json path.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    metrics_path = run_dir / "metrics.json"
    split_path = run_dir / "split_manifest.json"
    confusion_path = run_dir / "confusion_matrix.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    if not confusion_path.exists():
        raise FileNotFoundError(confusion_path)

    metrics = read_json(metrics_path)
    split_manifest = read_json(split_path)
    confusion = read_confusion(confusion_path)

    references = [
        ref
        for ref in [
            load_reference("SupCon E1", Path(args.supcon_reference).resolve() if args.supcon_reference else None),
            load_reference("SupCon+AdaIN E1", Path(args.adain_reference).resolve() if args.adain_reference else None),
        ]
        if ref is not None
    ]

    report = render_report(run_dir, metrics, split_manifest, confusion, references)
    output_path = Path(args.output).resolve() if args.output else run_dir / "report.md"
    output_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
