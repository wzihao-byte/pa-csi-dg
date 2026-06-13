from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_OUTPUT_ROOTS = [
    "outputs_supcon_lambda_sweep",
    "outputs_supcon_batch_sampler_sweep",
    "outputs_supcon_temperature_sweep",
    "outputs_supcon_positive_mask_sweep",
    "outputs_supcon_warmup_sweep",
    "outputs_supcon_final_eval",
    "outputs_subcenter_supcon_sweep",
]

PER_RUN_COLUMNS = [
    "sweep_root",
    "metrics_path",
    "experiment_name",
    "seed",
    "target_env",
    "source_envs",
    "contrastive_loss_type",
    "lambda_supcon",
    "lambda_supcon_warmup_epochs",
    "lambda_supcon_target",
    "lambda_supcon_effective",
    "temperature",
    "batch_size",
    "sampler_mode",
    "include_same_domain_same_class",
    "supcon_positive_mode",
    "lambda_pair_margin",
    "effective_pair_margin_weight",
    "prototype_num_subcenters",
    "selection_metric",
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

SUMMARY_METRIC_COLUMNS = [
    "num_runs",
    "mean_test_accuracy",
    "std_test_accuracy",
    "mean_test_f1_macro",
    "std_test_f1_macro",
    "mean_best_val_accuracy",
    "std_best_val_accuracy",
    "mean_loss_supcon_to_ce",
    "mean_supcon_frac_anchors_with_positive",
]

CONFIG_GROUP_KEYS = [
    "contrastive_loss_type",
    "lambda_supcon",
    "lambda_supcon_warmup_epochs",
    "temperature",
    "batch_size",
    "sampler_mode",
    "include_same_domain_same_class",
    "supcon_positive_mode",
    "lambda_pair_margin",
    "effective_pair_margin_weight",
    "prototype_num_subcenters",
]

GROUP_SPECS = {
    "by_lambda.csv": ["contrastive_loss_type", "lambda_supcon"],
    "by_temperature.csv": ["contrastive_loss_type", "lambda_supcon", "temperature"],
    "by_batch_sampler.csv": ["contrastive_loss_type", "lambda_supcon", "batch_size", "sampler_mode"],
    "by_positive_mask.csv": [
        "contrastive_loss_type",
        "lambda_supcon",
        "temperature",
        "include_same_domain_same_class",
        "supcon_positive_mode",
    ],
    "by_warmup.csv": ["contrastive_loss_type", "lambda_supcon", "lambda_supcon_warmup_epochs"],
    "by_subcenter.csv": [
        "contrastive_loss_type",
        "prototype_num_subcenters",
        "lambda_supcon",
        "lambda_pair_margin",
        "effective_pair_margin_weight",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SupCon sweep outputs across multiple rounds.")
    parser.add_argument(
        "--output-roots",
        nargs="*",
        help="One or more output roots to scan. Defaults to all known SupCon sweep roots.",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis_supcon_sweeps",
        help="Directory for CSV, Markdown, and plot outputs.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top configurations to show in the Markdown report.",
    )
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def nested_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


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


def to_float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def mean_std(values: Iterable[Any]) -> Tuple[Any, Any]:
    numbers = [number for value in values if (number := to_float_or_none(value)) is not None]
    if not numbers:
        return "", ""
    if len(numbers) == 1:
        return numbers[0], 0.0
    return mean(numbers), stdev(numbers)


def mean_only(values: Iterable[Any]) -> Any:
    numbers = [number for value in values if (number := to_float_or_none(value)) is not None]
    if not numbers:
        return ""
    return mean(numbers)


def flatten(prefix: str, value: Any, output: Dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten(next_prefix, nested_value, output)
    else:
        output[prefix] = value


def scan_metrics(output_roots: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    rows: List[Dict[str, Any]] = []
    scanned_roots: List[str] = []
    missing_roots: List[str] = []

    for root_name in output_roots:
        root = Path(root_name)
        if not root.exists():
            warn(f"output root not found, skipping: {root}")
            missing_roots.append(str(root))
            continue
        if not root.is_dir():
            warn(f"output root is not a directory, skipping: {root}")
            missing_roots.append(str(root))
            continue

        scanned_roots.append(str(root))
        metrics_paths = sorted(root.rglob("metrics.json"))
        if not metrics_paths:
            warn(f"no metrics.json files found under: {root}")
            continue
        for metrics_path in metrics_paths:
            try:
                rows.append(row_from_metrics(root, metrics_path))
            except Exception as exc:  # noqa: BLE001 - keep aggregation robust.
                warn(f"failed to read {metrics_path}: {exc}")

    return rows, scanned_roots, missing_roots


def row_from_metrics(root: Path, metrics_path: Path) -> Dict[str, Any]:
    metrics = read_json(metrics_path)
    resolved_config_path = metrics_path.parent / "resolved_config.json"
    resolved_config = read_json(resolved_config_path) if resolved_config_path.exists() else {}

    flattened: Dict[str, Any] = {}
    flatten("metrics", metrics, flattened)
    flatten("resolved_config", resolved_config, flattened)

    losses_enabled = metrics.get("losses_enabled", {}) or {}
    test_metrics = metrics.get("test_metrics", {}) or {}
    config_losses = resolved_config.get("losses", {}) or {}
    config_training = resolved_config.get("training", {}) or {}
    config_model = resolved_config.get("model", {}) or {}

    row = {
        "sweep_root": str(root),
        "metrics_path": str(metrics_path),
        "experiment_name": first_present(
            resolved_config.get("experiment_name"),
            metrics.get("experiment_name"),
            infer_experiment_name(metrics_path),
        ),
        "seed": first_present(metrics.get("seed"), seed_from_config(resolved_config)),
        "target_env": first_present(metrics.get("target_env"), resolved_config.get("target_env")),
        "source_envs": metrics.get("source_envs", ""),
        "contrastive_loss_type": first_present(
            losses_enabled.get("contrastive_loss_type"),
            config_losses.get("contrastive_loss_type"),
            "pairwise_supcon" if "supcon" in str(root).lower() else None,
        ),
        "lambda_supcon": first_present(
            losses_enabled.get("lambda_supcon"),
            config_losses.get("lambda_supcon"),
        ),
        "lambda_supcon_warmup_epochs": first_present(
            losses_enabled.get("lambda_supcon_warmup_epochs"),
            config_losses.get("lambda_supcon_warmup_epochs"),
        ),
        "lambda_supcon_target": test_metrics.get("lambda_supcon_target", ""),
        "lambda_supcon_effective": test_metrics.get("lambda_supcon_effective", ""),
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
        "lambda_pair_margin": first_present(
            losses_enabled.get("lambda_pair_margin"),
            config_losses.get("lambda_pair_margin"),
        ),
        "effective_pair_margin_weight": first_present(
            nested_get(metrics, "pair_margin_analysis", "effective_pair_margin_weight"),
            nested_get(metrics, "search_config", "effective_pair_margin_weight"),
            config_losses.get("effective_pair_margin_weight"),
            losses_enabled.get("lambda_pair_margin"),
            config_losses.get("lambda_pair_margin"),
        ),
        "prototype_num_subcenters": first_present(
            losses_enabled.get("prototype_num_subcenters"),
            config_model.get("prototype_num_subcenters"),
        ),
        "selection_metric": first_present(metrics.get("selection_metric"), config_training.get("selection_metric")),
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
    row["_flattened"] = flattened
    return row


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key, "")) for key in fieldnames})


def group_key(row: Mapping[str, Any], keys: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(row.get(key, "") for key in keys)


def applicable_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], filename: str) -> List[Mapping[str, Any]]:
    selected = [row for row in rows if all(row.get(key, "") not in ("", None) for key in keys)]
    if filename == "by_subcenter.csv":
        selected = [
            row for row in selected
            if row.get("contrastive_loss_type") == "subcenter_prototype"
            or row.get("prototype_num_subcenters", "") not in ("", None, 0, "0")
        ]
    return selected


def summarize_group(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[str, Any]:
    first_row = rows[0]
    accuracy_mean, accuracy_std = mean_std(row.get("test_accuracy", "") for row in rows)
    f1_mean, f1_std = mean_std(row.get("test_f1_macro", "") for row in rows)
    val_mean, val_std = mean_std(row.get("best_val_accuracy", "") for row in rows)
    summary = {key: first_row.get(key, "") for key in keys}
    summary.update(
        {
            "num_runs": len(rows),
            "mean_test_accuracy": accuracy_mean,
            "std_test_accuracy": accuracy_std,
            "mean_test_f1_macro": f1_mean,
            "std_test_f1_macro": f1_std,
            "mean_best_val_accuracy": val_mean,
            "std_best_val_accuracy": val_std,
            "mean_loss_supcon_to_ce": mean_only(row.get("loss_supcon_to_ce", "") for row in rows),
            "mean_supcon_frac_anchors_with_positive": mean_only(
                row.get("supcon_frac_anchors_with_positive", "") for row in rows
            ),
        }
    )
    return summary


def grouped_summary(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row, keys)].append(row)
    summaries = [summarize_group(group_rows, keys) for group_rows in grouped.values()]
    summaries.sort(key=lambda row: (-sort_float(row.get("mean_best_val_accuracy"), -math.inf), str(row)))
    return summaries


def write_group_summaries(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    outputs: Dict[str, List[Dict[str, Any]]] = {}
    for filename, keys in GROUP_SPECS.items():
        selected = applicable_rows(rows, keys, filename)
        if not selected:
            warn(f"no applicable rows for {filename}; skipping")
            continue
        summaries = grouped_summary(selected, keys)
        write_csv(out_dir / filename, [*keys, *SUMMARY_METRIC_COLUMNS], summaries)
        outputs[filename] = summaries
    return outputs


def sort_float(value: Any, default: float = math.inf) -> float:
    number = to_float_or_none(value)
    return number if number is not None else default


def config_summaries(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return grouped_summary(
        [row for row in rows if row.get("experiment_name", "") not in ("", None)],
        ["experiment_name", *CONFIG_GROUP_KEYS],
    )


def md_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], limit: int | None = None) -> str:
    selected = list(rows[:limit] if limit is not None else rows)
    if not selected:
        return "_No rows available._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        lines.append("| " + " | ".join(md_escape(format_value(row.get(column, ""))) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def format_value(value: Any, precision: int = 4) -> str:
    if value == "":
        return ""
    number = to_float_or_none(value)
    if number is None:
        return str(value)
    return f"{number:.{precision}f}"


def write_report(
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    scanned_roots: Sequence[str],
    missing_roots: Sequence[str],
    grouped_outputs: Mapping[str, Sequence[Mapping[str, Any]]],
    top_n: int,
) -> None:
    configs = config_summaries(rows)
    top_by_val = sorted(configs, key=lambda row: -sort_float(row.get("mean_best_val_accuracy"), -math.inf))
    top_by_test = sorted(configs, key=lambda row: -sort_float(row.get("mean_test_accuracy"), -math.inf))

    lines = [
        "# SupCon Sweep Report",
        "",
        "This report aggregates SupCon sweep outputs across available result roots.",
        "",
        "Source validation behavior should drive hyperparameter selection. Target test rankings below are diagnostic only and should not be used for repeated hand-tuning.",
        "",
        "## Scan Summary",
        "",
        f"- Total runs found: {len(rows)}",
        f"- Roots scanned: {', '.join(scanned_roots) if scanned_roots else 'None'}",
        f"- Missing roots: {', '.join(missing_roots) if missing_roots else 'None'}",
        "",
        "## Top Configurations By Best Validation Accuracy",
        "",
        markdown_table(
            top_by_val,
            [
                "experiment_name",
                "contrastive_loss_type",
                "lambda_supcon",
                "temperature",
                "batch_size",
                "sampler_mode",
                "mean_best_val_accuracy",
                "mean_test_accuracy",
                "num_runs",
            ],
            limit=top_n,
        ),
        "",
        "## Top Configurations By Test Accuracy (Diagnostic Only)",
        "",
        "These rows are for diagnostics only. Do not use target test rankings for repeated hyperparameter selection.",
        "",
        markdown_table(
            top_by_test,
            [
                "experiment_name",
                "contrastive_loss_type",
                "lambda_supcon",
                "temperature",
                "batch_size",
                "sampler_mode",
                "mean_test_accuracy",
                "mean_best_val_accuracy",
                "num_runs",
            ],
            limit=top_n,
        ),
        "",
    ]

    section_specs = [
        ("Lambda Sweep Summary", "by_lambda.csv", ["contrastive_loss_type", "lambda_supcon", "mean_best_val_accuracy", "mean_test_accuracy", "num_runs"]),
        ("Temperature Sweep Summary", "by_temperature.csv", ["contrastive_loss_type", "lambda_supcon", "temperature", "mean_best_val_accuracy", "mean_test_accuracy", "num_runs"]),
        ("Batch/Sampler Sweep Summary", "by_batch_sampler.csv", ["contrastive_loss_type", "lambda_supcon", "batch_size", "sampler_mode", "mean_best_val_accuracy", "mean_test_accuracy", "num_runs"]),
        ("Positive-Mask Sweep Summary", "by_positive_mask.csv", ["include_same_domain_same_class", "supcon_positive_mode", "lambda_supcon", "temperature", "mean_best_val_accuracy", "mean_test_accuracy", "mean_supcon_frac_anchors_with_positive", "num_runs"]),
        ("Warmup Sweep Summary", "by_warmup.csv", ["contrastive_loss_type", "lambda_supcon", "lambda_supcon_warmup_epochs", "mean_best_val_accuracy", "mean_test_accuracy", "num_runs"]),
        ("Subcenter Summary", "by_subcenter.csv", ["contrastive_loss_type", "prototype_num_subcenters", "lambda_supcon", "lambda_pair_margin", "mean_best_val_accuracy", "mean_test_accuracy", "num_runs"]),
    ]
    for title, filename, columns in section_specs:
        lines.extend([f"## {title}", ""])
        rows_for_section = grouped_outputs.get(filename, [])
        lines.append(markdown_table(rows_for_section, columns, limit=top_n))
        lines.append("")

    lines.extend(
        [
            "## Recommended Next Action",
            "",
            "Placeholder: choose candidate settings from source validation behavior, then run or inspect final multi-seed, multi-target evaluation once.",
            "",
        ]
    )
    (out_dir / "SUPCON_SWEEP_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def create_plots(out_dir: Path, grouped_outputs: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # noqa: BLE001
        warn(f"matplotlib unavailable; skipping plots ({exc})")
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_xy(
            plt,
            grouped_outputs.get("by_lambda.csv", []),
            x_key="lambda_supcon",
            y_key="mean_best_val_accuracy",
            title="Lambda vs Mean Best Validation Accuracy",
            output_path=plots_dir / "lambda_vs_mean_best_val_accuracy.png",
        )
        plot_xy(
            plt,
            grouped_outputs.get("by_lambda.csv", []),
            x_key="lambda_supcon",
            y_key="mean_test_accuracy",
            title="Lambda vs Mean Test Accuracy",
            output_path=plots_dir / "lambda_vs_mean_test_accuracy.png",
        )
        plot_xy(
            plt,
            grouped_outputs.get("by_temperature.csv", []),
            x_key="temperature",
            y_key="mean_best_val_accuracy",
            title="Temperature vs Mean Best Validation Accuracy",
            output_path=plots_dir / "temperature_vs_mean_best_val_accuracy.png",
        )
        plot_categorical(
            plt,
            grouped_outputs.get("by_batch_sampler.csv", []),
            label_keys=["batch_size", "sampler_mode"],
            y_key="mean_best_val_accuracy",
            title="Batch/Sampler Mean Best Validation Accuracy",
            output_path=plots_dir / "batch_sampler_mean_best_val_accuracy.png",
        )
        plot_xy(
            plt,
            grouped_outputs.get("by_warmup.csv", []),
            x_key="lambda_supcon_warmup_epochs",
            y_key="mean_best_val_accuracy",
            title="Warmup Epochs vs Mean Best Validation Accuracy",
            output_path=plots_dir / "warmup_epochs_vs_mean_best_val_accuracy.png",
        )
    except Exception as exc:  # noqa: BLE001
        warn(f"plotting failed; CSV/Markdown outputs are still valid ({exc})")


def plot_xy(plt: Any, rows: Sequence[Mapping[str, Any]], x_key: str, y_key: str, title: str, output_path: Path) -> None:
    points = [
        (to_float_or_none(row.get(x_key)), to_float_or_none(row.get(y_key)))
        for row in rows
    ]
    points = [(x, y) for x, y in points if x is not None and y is not None]
    if not points:
        warn(f"no numeric data for plot: {output_path.name}")
        return
    points.sort(key=lambda item: item[0])
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    plt.figure(figsize=(6, 4))
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_categorical(
    plt: Any,
    rows: Sequence[Mapping[str, Any]],
    label_keys: Sequence[str],
    y_key: str,
    title: str,
    output_path: Path,
) -> None:
    values = [
        (
            " / ".join(str(row.get(key, "")) for key in label_keys),
            to_float_or_none(row.get(y_key)),
        )
        for row in rows
    ]
    values = [(label, value) for label, value in values if value is not None]
    if not values:
        warn(f"no numeric data for plot: {output_path.name}")
        return
    values = values[:20]
    labels = [item[0] for item in values]
    y_values = [item[1] for item in values]
    plt.figure(figsize=(max(8, len(values) * 0.6), 4))
    plt.bar(range(len(values)), y_values)
    plt.xticks(range(len(values)), labels, rotation=45, ha="right")
    plt.ylabel(y_key)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    output_roots = args.output_roots if args.output_roots else DEFAULT_OUTPUT_ROOTS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, scanned_roots, missing_roots = scan_metrics(output_roots)
    if not rows:
        print(
            "No SupCon metrics.json files found in existing output roots. "
            f"Checked roots: {', '.join(output_roots)}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    rows.sort(key=lambda row: (str(row.get("sweep_root", "")), str(row.get("target_env", "")), str(row.get("seed", ""))))
    write_csv(out_dir / "all_supcon_runs.csv", PER_RUN_COLUMNS, rows)
    grouped_outputs = write_group_summaries(out_dir, rows)
    write_report(out_dir, rows, scanned_roots, missing_roots, grouped_outputs, max(1, int(args.top_n)))
    create_plots(out_dir, grouped_outputs)

    print(f"Found {len(rows)} runs across {len(scanned_roots)} existing roots.")
    print(f"Wrote unified analysis outputs to {out_dir}")


if __name__ == "__main__":
    main()
