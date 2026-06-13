from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


FIELDNAMES = [
    "metrics_path",
    "experiment_name",
    "seed",
    "target_env",
    "batch_size",
    "sampler_mode",
    "lambda_supcon",
    "temperature",
    "include_same_domain_same_class",
    "supcon_positive_mode",
    "best_val_score",
    "best_val_accuracy",
    "test_accuracy",
    "test_f1_macro",
    "test_precision_macro",
    "test_recall_macro",
    "loss_supcon_weighted",
    "loss_supcon_to_ce",
    "supcon_mean_positives_per_anchor",
    "supcon_frac_anchors_with_positive",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SupCon temperature sweep results.")
    parser.add_argument(
        "--output-root",
        default="outputs_supcon_temperature_sweep",
        help="Root directory containing sweep outputs.",
    )
    parser.add_argument(
        "--output-csv",
        default="supcon_temperature_sweep_summary.csv",
        help="CSV summary path to write.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return ""


def csv_cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def infer_experiment_name(metrics_path: Path) -> str:
    parts = metrics_path.parts
    for index, part in enumerate(parts):
        if part.startswith("target_") and index > 0:
            return parts[index - 1]
    return ""


def seed_from_config(config: Mapping[str, Any]) -> Any:
    seed_list = config.get("seed_list")
    if isinstance(seed_list, list) and seed_list:
        return seed_list[0]
    return None


def row_from_metrics(metrics_path: Path) -> Dict[str, Any]:
    metrics = read_json(metrics_path)
    resolved_config_path = metrics_path.parent / "resolved_config.json"
    resolved_config = read_json(resolved_config_path) if resolved_config_path.exists() else {}

    losses_enabled = metrics.get("losses_enabled", {}) or {}
    test_metrics = metrics.get("test_metrics", {}) or {}
    config_losses = resolved_config.get("losses", {}) or {}
    config_training = resolved_config.get("training", {}) or {}

    return {
        "metrics_path": str(metrics_path),
        "experiment_name": first_present(
            resolved_config.get("experiment_name"),
            metrics.get("experiment_name"),
            infer_experiment_name(metrics_path),
        ),
        "seed": first_present(metrics.get("seed"), seed_from_config(resolved_config)),
        "target_env": first_present(metrics.get("target_env"), resolved_config.get("target_env")),
        "batch_size": config_training.get("batch_size", ""),
        "sampler_mode": first_present(metrics.get("sampler_mode"), config_training.get("sampler")),
        "lambda_supcon": first_present(
            losses_enabled.get("lambda_supcon"),
            config_losses.get("lambda_supcon"),
        ),
        "temperature": first_present(
            losses_enabled.get("temperature"),
            config_losses.get("temperature"),
        ),
        "include_same_domain_same_class": first_present(
            losses_enabled.get("include_same_domain_same_class"),
            config_losses.get("include_same_domain_same_class"),
        ),
        "supcon_positive_mode": first_present(
            losses_enabled.get("supcon_positive_mode"),
            config_losses.get("supcon_positive_mode"),
        ),
        "best_val_score": metrics.get("best_val_score", ""),
        "best_val_accuracy": metrics.get("best_val_accuracy", ""),
        "test_accuracy": test_metrics.get("accuracy", ""),
        "test_f1_macro": test_metrics.get("f1_macro", ""),
        "test_precision_macro": test_metrics.get("precision_macro", ""),
        "test_recall_macro": test_metrics.get("recall_macro", ""),
        "loss_supcon_weighted": test_metrics.get("loss_supcon_weighted", ""),
        "loss_supcon_to_ce": test_metrics.get("loss_supcon_to_ce", ""),
        "supcon_mean_positives_per_anchor": test_metrics.get("supcon_mean_positives_per_anchor", ""),
        "supcon_frac_anchors_with_positive": test_metrics.get("supcon_frac_anchors_with_positive", ""),
    }


def sort_key(row: Mapping[str, Any]) -> tuple[float, float, int]:
    def as_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")

    def as_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    return (
        as_float(row.get("lambda_supcon")),
        as_float(row.get("temperature")),
        as_int(row.get("seed")),
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key, "")) for key in FIELDNAMES})


def format_value(value: Any, precision: int = 4) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def print_table(rows: List[Mapping[str, Any]]) -> None:
    headers = ["lambda", "temp", "seed", "batch", "sampler", "acc", "f1", "supcon/ce", "pos_frac"]
    table_rows = [
        [
            format_value(row.get("lambda_supcon"), precision=3),
            format_value(row.get("temperature"), precision=3),
            str(row.get("seed", "")),
            str(row.get("batch_size", "")),
            str(row.get("sampler_mode", "")),
            format_value(row.get("test_accuracy")),
            format_value(row.get("test_f1_macro")),
            format_value(row.get("loss_supcon_to_ce")),
            format_value(row.get("supcon_frac_anchors_with_positive")),
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table_rows))
        for index in range(len(headers))
    ]

    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in table_rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    metrics_paths = sorted(output_root.rglob("metrics.json"))
    if not metrics_paths:
        print(f"No metrics.json files found under {output_root}", file=sys.stderr)
        raise SystemExit(1)

    rows = sorted((row_from_metrics(path) for path in metrics_paths), key=sort_key)
    write_csv(Path(args.output_csv), rows)
    print_table(rows)
    print(f"\nWrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
