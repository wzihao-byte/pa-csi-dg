from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PER_RUN_FIELDNAMES = [
    "metrics_path",
    "experiment_name",
    "seed",
    "target_env",
    "source_envs",
    "lambda_supcon",
    "lambda_supcon_warmup_epochs",
    "temperature",
    "batch_size",
    "sampler_mode",
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

GROUP_KEYS = [
    "experiment_name",
    "lambda_supcon",
    "lambda_supcon_warmup_epochs",
    "temperature",
    "batch_size",
    "sampler_mode",
    "include_same_domain_same_class",
    "supcon_positive_mode",
]

AGGREGATE_FIELDNAMES = [
    "aggregate_scope",
    "target_env",
    *GROUP_KEYS,
    "num_runs",
    "num_target_envs",
    "num_seeds",
    "mean_test_accuracy",
    "std_test_accuracy",
    "mean_test_f1_macro",
    "std_test_f1_macro",
    "mean_test_precision_macro",
    "std_test_precision_macro",
    "mean_test_recall_macro",
    "std_test_recall_macro",
    "mean_best_val_accuracy",
    "std_best_val_accuracy",
    "mean_loss_supcon_to_ce",
    "mean_supcon_frac_anchors_with_positive",
]

COMPARISON_FIELDNAMES = [
    "target_env",
    "seed",
    "baseline_test_accuracy",
    "supcon_test_accuracy",
    "delta_accuracy",
    "baseline_test_f1_macro",
    "supcon_test_f1_macro",
    "delta_f1_macro",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate DG final evaluation results.")
    parser.add_argument(
        "--output-root",
        default="outputs_supcon_final_eval",
        help="Root directory containing final evaluation outputs.",
    )
    parser.add_argument(
        "--per-run-csv",
        default="supcon_final_eval_per_run.csv",
        help="Per-run CSV path to write.",
    )
    parser.add_argument(
        "--aggregate-csv",
        default="supcon_final_eval_aggregate.csv",
        help="Aggregate CSV path to write.",
    )
    parser.add_argument(
        "--baseline-root",
        help="Optional baseline output root for target_env/seed comparison.",
    )
    parser.add_argument(
        "--comparison-csv",
        default="supcon_final_eval_baseline_comparison.csv",
        help="CSV path for optional baseline comparison rows.",
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
        "source_envs": metrics.get("source_envs", ""),
        "lambda_supcon": first_present(
            losses_enabled.get("lambda_supcon"),
            config_losses.get("lambda_supcon"),
        ),
        "lambda_supcon_warmup_epochs": first_present(
            losses_enabled.get("lambda_supcon_warmup_epochs"),
            config_losses.get("lambda_supcon_warmup_epochs"),
        ),
        "temperature": first_present(
            losses_enabled.get("temperature"),
            config_losses.get("temperature"),
        ),
        "batch_size": config_training.get("batch_size", ""),
        "sampler_mode": first_present(metrics.get("sampler_mode"), config_training.get("sampler")),
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


def metric_rows(output_root: Path) -> List[Dict[str, Any]]:
    metrics_paths = sorted(output_root.rglob("metrics.json"))
    if not metrics_paths:
        return []
    return [row_from_metrics(path) for path in metrics_paths]


def to_float(value: Any) -> float:
    return float(value)


def maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_std(values: Sequence[Any]) -> Tuple[Any, Any]:
    numeric_values = [float(value) for value in values if maybe_float(value) is not None]
    if not numeric_values:
        return "", ""
    if len(numeric_values) == 1:
        return numeric_values[0], 0.0
    return mean(numeric_values), stdev(numeric_values)


def mean_only(values: Sequence[Any]) -> Any:
    numeric_values = [float(value) for value in values if maybe_float(value) is not None]
    if not numeric_values:
        return ""
    return mean(numeric_values)


def grouping_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return tuple(row.get(key, "") for key in GROUP_KEYS)


def aggregate_group(rows: Sequence[Mapping[str, Any]], scope: str, target_env: str) -> Dict[str, Any]:
    acc_mean, acc_std = mean_std([row.get("test_accuracy", "") for row in rows])
    f1_mean, f1_std = mean_std([row.get("test_f1_macro", "") for row in rows])
    precision_mean, precision_std = mean_std([row.get("test_precision_macro", "") for row in rows])
    recall_mean, recall_std = mean_std([row.get("test_recall_macro", "") for row in rows])
    val_acc_mean, val_acc_std = mean_std([row.get("best_val_accuracy", "") for row in rows])
    first_row = rows[0]
    aggregate = {
        "aggregate_scope": scope,
        "target_env": target_env,
        "num_runs": len(rows),
        "num_target_envs": len({str(row.get("target_env", "")) for row in rows}),
        "num_seeds": len({str(row.get("seed", "")) for row in rows}),
        "mean_test_accuracy": acc_mean,
        "std_test_accuracy": acc_std,
        "mean_test_f1_macro": f1_mean,
        "std_test_f1_macro": f1_std,
        "mean_test_precision_macro": precision_mean,
        "std_test_precision_macro": precision_std,
        "mean_test_recall_macro": recall_mean,
        "std_test_recall_macro": recall_std,
        "mean_best_val_accuracy": val_acc_mean,
        "std_best_val_accuracy": val_acc_std,
        "mean_loss_supcon_to_ce": mean_only([row.get("loss_supcon_to_ce", "") for row in rows]),
        "mean_supcon_frac_anchors_with_positive": mean_only(
            [row.get("supcon_frac_anchors_with_positive", "") for row in rows]
        ),
    }
    for key in GROUP_KEYS:
        aggregate[key] = first_row.get(key, "")
    return aggregate


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_config: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_config[grouping_key(row)].append(row)

    aggregates: List[Dict[str, Any]] = []
    for config_rows in by_config.values():
        aggregates.append(aggregate_group(config_rows, "overall", "all"))

        by_target: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in config_rows:
            by_target[str(row.get("target_env", ""))].append(row)
        for target_env, target_rows in sorted(by_target.items()):
            aggregates.append(aggregate_group(target_rows, "target_env", target_env))
    return aggregates


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key, "")) for key in fieldnames})


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def as_sort_float(value: Any, default: float = float("inf")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sort_per_run(row: Mapping[str, Any]) -> tuple[str, int]:
    return (str(row.get("target_env", "")), as_int(row.get("seed")))


def sort_aggregate(row: Mapping[str, Any]) -> tuple[float, str]:
    return (
        -as_sort_float(row.get("mean_test_accuracy"), default=float("-inf")),
        str(row.get("target_env", "")),
    )


def format_value(value: Any, precision: int = 4) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    if not rows:
        print("No rows to display.")
        return
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def print_per_run_table(rows: Sequence[Mapping[str, Any]]) -> None:
    headers = ["target_env", "seed", "acc", "f1", "val_acc", "supcon/ce"]
    table_rows = [
        [
            str(row.get("target_env", "")),
            str(row.get("seed", "")),
            format_value(row.get("test_accuracy")),
            format_value(row.get("test_f1_macro")),
            format_value(row.get("best_val_accuracy")),
            format_value(row.get("loss_supcon_to_ce")),
        ]
        for row in rows
    ]
    print("Per-run results:")
    print_table(headers, table_rows)


def print_aggregate_table(rows: Sequence[Mapping[str, Any]]) -> None:
    headers = ["scope", "target_env", "runs", "mean_acc", "std_acc", "mean_f1", "std_f1"]
    table_rows = [
        [
            str(row.get("aggregate_scope", "")),
            str(row.get("target_env", "")),
            str(row.get("num_runs", "")),
            format_value(row.get("mean_test_accuracy")),
            format_value(row.get("std_test_accuracy")),
            format_value(row.get("mean_test_f1_macro")),
            format_value(row.get("std_test_f1_macro")),
        ]
        for row in rows
    ]
    print("\nAggregate results:")
    print_table(headers, table_rows)


def comparison_rows(supcon_rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    baseline_by_key: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for row in sorted(baseline_rows, key=sort_per_run):
        baseline_by_key.setdefault((str(row.get("target_env", "")), str(row.get("seed", ""))), row)

    rows: List[Dict[str, Any]] = []
    for supcon_row in sorted(supcon_rows, key=sort_per_run):
        key = (str(supcon_row.get("target_env", "")), str(supcon_row.get("seed", "")))
        baseline_row = baseline_by_key.get(key)
        if baseline_row is None:
            continue

        baseline_acc = maybe_float(baseline_row.get("test_accuracy"))
        supcon_acc = maybe_float(supcon_row.get("test_accuracy"))
        baseline_f1 = maybe_float(baseline_row.get("test_f1_macro"))
        supcon_f1 = maybe_float(supcon_row.get("test_f1_macro"))
        rows.append(
            {
                "target_env": key[0],
                "seed": key[1],
                "baseline_test_accuracy": baseline_acc if baseline_acc is not None else "",
                "supcon_test_accuracy": supcon_acc if supcon_acc is not None else "",
                "delta_accuracy": (
                    supcon_acc - baseline_acc
                    if baseline_acc is not None and supcon_acc is not None
                    else ""
                ),
                "baseline_test_f1_macro": baseline_f1 if baseline_f1 is not None else "",
                "supcon_test_f1_macro": supcon_f1 if supcon_f1 is not None else "",
                "delta_f1_macro": (
                    supcon_f1 - baseline_f1
                    if baseline_f1 is not None and supcon_f1 is not None
                    else ""
                ),
            }
        )
    return rows


def print_comparison_table(rows: Sequence[Mapping[str, Any]]) -> None:
    headers = ["target_env", "seed", "base_acc", "supcon_acc", "delta_acc", "base_f1", "supcon_f1", "delta_f1"]
    table_rows = [
        [
            str(row.get("target_env", "")),
            str(row.get("seed", "")),
            format_value(row.get("baseline_test_accuracy")),
            format_value(row.get("supcon_test_accuracy")),
            format_value(row.get("delta_accuracy")),
            format_value(row.get("baseline_test_f1_macro")),
            format_value(row.get("supcon_test_f1_macro")),
            format_value(row.get("delta_f1_macro")),
        ]
        for row in rows
    ]
    print("\nBaseline comparison:")
    print_table(headers, table_rows)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    rows = sorted(metric_rows(output_root), key=sort_per_run)
    if not rows:
        print(f"No metrics.json files found under {output_root}", file=sys.stderr)
        raise SystemExit(1)

    aggregates = sorted(aggregate_rows(rows), key=sort_aggregate)
    write_csv(Path(args.per_run_csv), PER_RUN_FIELDNAMES, rows)
    write_csv(Path(args.aggregate_csv), AGGREGATE_FIELDNAMES, aggregates)
    print_per_run_table(rows)
    print_aggregate_table(aggregates)
    print(f"\nWrote {len(rows)} per-run rows to {args.per_run_csv}")
    print(f"Wrote {len(aggregates)} aggregate rows to {args.aggregate_csv}")

    if args.baseline_root:
        baseline_root = Path(args.baseline_root)
        baseline_rows = metric_rows(baseline_root)
        if not baseline_rows:
            print(f"No baseline metrics.json files found under {baseline_root}", file=sys.stderr)
            raise SystemExit(1)
        comparisons = comparison_rows(rows, baseline_rows)
        write_csv(Path(args.comparison_csv), COMPARISON_FIELDNAMES, comparisons)
        print_comparison_table(comparisons)
        print(f"\nWrote {len(comparisons)} comparison rows to {args.comparison_csv}")


if __name__ == "__main__":
    main()
