from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
from scipy import stats


ARMS = ("Y00", "Y01", "Y10", "Y11")
ARM_LABELS = {
    "Y00": "global-basis / CE",
    "Y01": "global-basis / SupCon",
    "Y10": "residual-TPP / CE",
    "Y11": "residual-TPP / SupCon",
}
EFFECTS = {
    "TPP_without_SupCon": {"Y10": 1.0, "Y00": -1.0},
    "TPP_with_SupCon": {"Y11": 1.0, "Y01": -1.0},
    "SupCon_without_TPP": {"Y01": 1.0, "Y00": -1.0},
    "SupCon_with_TPP": {"Y11": 1.0, "Y10": -1.0},
    "interaction": {"Y11": 1.0, "Y10": -1.0, "Y01": -1.0, "Y00": 1.0},
    "joint_vs_control": {"Y11": 1.0, "Y00": -1.0},
}
CLASS_NAMES = {
    0: "no_movement",
    1: "falling",
    2: "walking",
    3: "sit_stand",
    4: "turning",
    5: "pick_up_pen",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze paired target margins in the formal SupCon x residual-TPP factorial."
    )
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path("outputs_supcon_tpp_joint/formal_interim_v12"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_supcon_tpp_joint_margin"),
    )
    parser.add_argument("--residual-scale", type=float, default=0.25)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def completed_attempt(formal_root: Path, target: str, seed: int, job_id: str) -> Path:
    job_dir = formal_root / "runs" / target / f"seed_{seed}" / job_id
    attempts = sorted(
        path
        for path in job_dir.glob("attempt_*")
        if (path / "target_predictions.jsonl").is_file()
    )
    if not attempts:
        raise FileNotFoundError(f"No completed attempt with target predictions under {job_dir}")
    return attempts[-1]


def read_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            probabilities = np.asarray(record["probability"], dtype=np.float64)
            true_class = int(record["true"])
            false_probabilities = probabilities.copy()
            false_probabilities[true_class] = -np.inf
            max_false = float(false_probabilities.max())
            true_probability = float(probabilities[true_class])
            rows[sample_id] = {
                "true": true_class,
                "pred": int(record["pred"]),
                "margin": true_probability - max_false,
                "true_probability": true_probability,
                "max_false_probability": max_false,
            }
    return rows


def population_variance(values: np.ndarray) -> float:
    return float(np.var(values, ddof=0))


def snr(mean: float, variance: float) -> float:
    return float(mean * mean / variance) if variance > 0 else math.inf


def linear_contrast(values: Mapping[str, float], coefficients: Mapping[str, float]) -> float:
    return float(sum(coefficients[arm] * values[arm] for arm in coefficients))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def target_fixed_effect_inference(
    rows: Iterable[Mapping[str, Any]], metric: str
) -> Dict[str, float]:
    """Balanced target-fixed-effect inference with three seeds per target.

    This reproduces the variance structure used by the locked formal analyzer:
    target is a fixed blocking factor, and the residual degrees of freedom are
    9 observations - 3 target means = 6.
    """

    rows = list(rows)
    grouped: Dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["target_env"])].append(float(row[metric]))
    if sorted(len(values) for values in grouped.values()) != [3, 3, 3]:
        raise ValueError(f"Expected three targets with three observations for {metric}: {grouped}")
    target_means = {target: float(np.mean(values)) for target, values in grouped.items()}
    marginal_mean = float(np.mean(list(target_means.values())))
    residuals = [
        value - target_means[target]
        for target, values in grouped.items()
        for value in values
    ]
    degrees_of_freedom = len(rows) - len(grouped)
    residual_mean_square = float(np.sum(np.square(residuals)) / degrees_of_freedom)
    standard_error = math.sqrt(residual_mean_square / len(rows))
    critical = float(stats.t.ppf(0.975, degrees_of_freedom))
    t_value = marginal_mean / standard_error if standard_error > 0 else math.inf
    p_value = float(2.0 * stats.t.sf(abs(t_value), degrees_of_freedom))
    return {
        "mean": marginal_mean,
        "standard_error": standard_error,
        "ci95_low": marginal_mean - critical * standard_error,
        "ci95_high": marginal_mean + critical * standard_error,
        "p_two_sided": p_value,
        "residual_df": float(degrees_of_freedom),
    }


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    formal_root = args.formal_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    formal = read_json(formal_root / "formal_results.json")
    result_rows = formal["results"]
    if len(result_rows) != 36:
        raise ValueError(f"Expected 36 formal jobs, found {len(result_rows)}")

    results: Dict[tuple[str, int, str], Mapping[str, Any]] = {}
    predictions: Dict[tuple[str, int, str], Dict[str, Dict[str, Any]]] = {}
    for result in result_rows:
        target = str(result["target_env"])
        seed = int(result["seed"])
        arm = str(result["arm"])
        key = (target, seed, arm)
        if key in results:
            raise ValueError(f"Duplicate formal result for {key}")
        attempt = completed_attempt(formal_root, target, seed, str(result["job_id"]))
        results[key] = result
        predictions[key] = read_predictions(attempt / "target_predictions.jsonl")

    blocks = sorted({(target, seed) for target, seed, _ in results})
    if len(blocks) != 9:
        raise ValueError(f"Expected 9 paired blocks, found {len(blocks)}")

    arm_rows = []
    effect_rows = []
    class_effect_rows = []
    theory_rows = []
    block_metrics: Dict[tuple[str, int], Dict[str, Dict[str, float]]] = {}

    for target, seed in blocks:
        sample_sets = [set(predictions[(target, seed, arm)]) for arm in ARMS]
        if not all(sample_set == sample_sets[0] for sample_set in sample_sets[1:]):
            raise ValueError(f"Sample identities differ across arms for {(target, seed)}")
        sample_ids = sorted(sample_sets[0])
        if len(sample_ids) != 3000:
            raise ValueError(f"Expected 3000 target samples for {(target, seed)}, found {len(sample_ids)}")

        true_labels = np.asarray(
            [predictions[(target, seed, "Y00")][sample_id]["true"] for sample_id in sample_ids],
            dtype=np.int64,
        )
        margins: Dict[str, np.ndarray] = {}
        block_metrics[(target, seed)] = {}
        for arm in ARMS:
            arm_predictions = predictions[(target, seed, arm)]
            arm_true = np.asarray([arm_predictions[sample_id]["true"] for sample_id in sample_ids])
            if not np.array_equal(arm_true, true_labels):
                raise ValueError(f"True labels differ across arms for {(target, seed, arm)}")
            arm_margin = np.asarray(
                [arm_predictions[sample_id]["margin"] for sample_id in sample_ids],
                dtype=np.float64,
            )
            margins[arm] = arm_margin
            per_class_means = [float(arm_margin[true_labels == label].mean()) for label in CLASS_NAMES]
            target_metrics = results[(target, seed, arm)]["target_metrics"]
            metrics = {
                "macro_f1": float(target_metrics["f1_macro"]),
                "accuracy": float(target_metrics["accuracy"]),
                "mean_margin": float(arm_margin.mean()),
                "class_balanced_mean_margin": float(np.mean(per_class_means)),
                "margin_variance": population_variance(arm_margin),
                "margin_snr": snr(float(arm_margin.mean()), population_variance(arm_margin)),
                "positive_margin_fraction": float(np.mean(arm_margin > 0.0)),
            }
            block_metrics[(target, seed)][arm] = metrics
            arm_rows.append(
                {
                    "target_env": target,
                    "seed": seed,
                    "arm": arm,
                    "arm_label": ARM_LABELS[arm],
                    **metrics,
                }
            )

        for effect_name, coefficients in EFFECTS.items():
            metrics_row: Dict[str, Any] = {
                "target_env": target,
                "seed": seed,
                "effect": effect_name,
            }
            for metric_name in (
                "macro_f1",
                "accuracy",
                "mean_margin",
                "class_balanced_mean_margin",
                "margin_snr",
            ):
                values = {arm: block_metrics[(target, seed)][arm][metric_name] for arm in ARMS}
                metrics_row[f"delta_{metric_name}"] = linear_contrast(values, coefficients)
            effect_rows.append(metrics_row)

            for label, class_name in CLASS_NAMES.items():
                mask = true_labels == label
                margin_values = {arm: float(margins[arm][mask].mean()) for arm in ARMS}
                f1_values = {
                    arm: float(results[(target, seed, arm)]["target_metrics"]["per_class"][str(label)]["f1"])
                    for arm in ARMS
                }
                class_effect_rows.append(
                    {
                        "target_env": target,
                        "seed": seed,
                        "class_id": label,
                        "class_name": class_name,
                        "effect": effect_name,
                        "delta_mean_margin": linear_contrast(margin_values, coefficients),
                        "delta_f1": linear_contrast(f1_values, coefficients),
                    }
                )

        # Empirical proxy for the sufficient condition in the margin theorem.
        # Y01 is the SupCon/global-basis arm and Y11 adds residual TPP.  Since
        # these are separately trained factorial arms, B/S/C are treatment-
        # effect proxies, not an exact within-checkpoint residual decomposition.
        alpha = float(args.residual_scale)
        for label, class_name in CLASS_NAMES.items():
            mask = true_labels == label
            global_margin = margins["Y01"][mask]
            scaled_tpp_effect = (margins["Y11"][mask] - margins["Y01"][mask]) / alpha
            a_value = float(global_margin.mean())
            b_value = float(scaled_tpp_effect.mean())
            v_value = population_variance(global_margin)
            s_value = population_variance(scaled_tpp_effect)
            c_value = float(np.cov(global_margin, scaled_tpp_effect, ddof=0)[0, 1])
            lhs = b_value / a_value if a_value != 0 else math.nan
            rhs = c_value / v_value if v_value > 0 else math.nan
            condition_gap = lhs - rhs
            base_snr = snr(a_value, v_value)
            joint_mean = a_value + alpha * b_value
            joint_variance = v_value + 2.0 * alpha * c_value + alpha * alpha * s_value
            theory_rows.append(
                {
                    "target_env": target,
                    "seed": seed,
                    "class_id": label,
                    "class_name": class_name,
                    "A_global_supcon_mean_margin": a_value,
                    "B_tpp_effect_per_unit_alpha": b_value,
                    "V_global_margin_variance": v_value,
                    "S_tpp_effect_variance": s_value,
                    "C_covariance": c_value,
                    "B_over_A": lhs,
                    "C_over_V": rhs,
                    "condition_gap": condition_gap,
                    "condition_pass": bool(condition_gap > 0.0),
                    "base_margin_snr": base_snr,
                    "joint_margin_snr": snr(joint_mean, joint_variance),
                    "delta_margin_snr": snr(joint_mean, joint_variance) - base_snr,
                }
            )

    write_csv(output_dir / "arm_block_metrics.csv", arm_rows)
    write_csv(output_dir / "paired_effects.csv", effect_rows)
    write_csv(output_dir / "paired_class_effects.csv", class_effect_rows)
    write_csv(output_dir / "margin_condition_proxy.csv", theory_rows)

    # Aggregate at the paired block level: each target-seed block receives the
    # same weight.  This is the inferential unit used by the formal experiment.
    arm_summary = []
    for arm in ARMS:
        selected = [row for row in arm_rows if row["arm"] == arm]
        summary: Dict[str, Any] = {"arm": arm, "arm_label": ARM_LABELS[arm]}
        for metric in (
            "macro_f1",
            "accuracy",
            "mean_margin",
            "class_balanced_mean_margin",
            "margin_snr",
        ):
            mean, sd = mean_sd(float(row[metric]) for row in selected)
            summary[f"mean_{metric}"] = mean
            summary[f"sd_{metric}"] = sd
        arm_summary.append(summary)

    effect_summary = []
    for effect_name in EFFECTS:
        selected = [row for row in effect_rows if row["effect"] == effect_name]
        summary = {"effect": effect_name}
        for metric in (
            "delta_macro_f1",
            "delta_accuracy",
            "delta_mean_margin",
            "delta_class_balanced_mean_margin",
            "delta_margin_snr",
        ):
            mean, sd = mean_sd(float(row[metric]) for row in selected)
            summary[f"mean_{metric}"] = mean
            summary[f"sd_{metric}"] = sd
        for metric in (
            "delta_macro_f1",
            "delta_class_balanced_mean_margin",
            "delta_margin_snr",
        ):
            inference = target_fixed_effect_inference(selected, metric)
            for key, value in inference.items():
                summary[f"{metric}_{key}"] = value
        effect_summary.append(summary)

    target_effect_summary = []
    for target in sorted({target for target, _ in blocks}):
        for effect_name in EFFECTS:
            selected = [
                row
                for row in effect_rows
                if row["target_env"] == target and row["effect"] == effect_name
            ]
            target_effect_summary.append(
                {
                    "target_env": target,
                    "effect": effect_name,
                    "mean_delta_macro_f1": mean_sd(row["delta_macro_f1"] for row in selected)[0],
                    "sd_delta_macro_f1": mean_sd(row["delta_macro_f1"] for row in selected)[1],
                    "mean_delta_class_balanced_margin": mean_sd(
                        row["delta_class_balanced_mean_margin"] for row in selected
                    )[0],
                }
            )

    class_summary = []
    for class_id, class_name in CLASS_NAMES.items():
        for effect_name in EFFECTS:
            selected = [
                row
                for row in class_effect_rows
                if row["class_id"] == class_id and row["effect"] == effect_name
            ]
            class_summary.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "effect": effect_name,
                    "mean_delta_f1": mean_sd(row["delta_f1"] for row in selected)[0],
                    "sd_delta_f1": mean_sd(row["delta_f1"] for row in selected)[1],
                    "mean_delta_margin": mean_sd(row["delta_mean_margin"] for row in selected)[0],
                }
            )

    write_csv(output_dir / "arm_summary.csv", arm_summary)
    write_csv(output_dir / "effect_summary.csv", effect_summary)
    write_csv(output_dir / "target_effect_summary.csv", target_effect_summary)
    write_csv(output_dir / "class_effect_summary.csv", class_summary)

    theory_passes = sum(bool(row["condition_pass"]) for row in theory_rows)
    theory_positive_snr = sum(float(row["delta_margin_snr"]) > 0 for row in theory_rows)
    report_lines = [
        "# SupCon × Residual-TPP Paired Margin Analysis",
        "",
        "This report uses the locked 36-run formal-interim factorial and all 108,000 target predictions. ",
        "The independent analysis unit is the paired `(target, seed)` block (9 blocks). The signed probability margin is ",
        "`p(true) - max p(false)`; it is positive exactly when the prediction is correct. All effects are paired within block.",
        "",
        "## Four-arm summary",
        "",
        "| arm | interpretation | macro-F1 | accuracy | balanced margin | margin SNR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in arm_summary:
        report_lines.append(
            f"| {row['arm']} | {row['arm_label']} | "
            f"{fmt(row['mean_macro_f1'])} ± {fmt(row['sd_macro_f1'])} | "
            f"{fmt(row['mean_accuracy'])} ± {fmt(row['sd_accuracy'])} | "
            f"{fmt(row['mean_class_balanced_mean_margin'])} | "
            f"{fmt(row['mean_margin_snr'])} |"
        )

    report_lines.extend(
        [
            "",
            "## Paired factorial effects",
            "",
            "| effect | Δ macro-F1 | 95% CI | p | Δ balanced margin | Δ margin SNR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in effect_summary:
        report_lines.append(
            f"| {row['effect']} | "
            f"{fmt(row['mean_delta_macro_f1'])} ± {fmt(row['sd_delta_macro_f1'])} | "
            f"[{fmt(row['delta_macro_f1_ci95_low'])}, {fmt(row['delta_macro_f1_ci95_high'])}] | "
            f"{row['delta_macro_f1_p_two_sided']:.4g} | "
            f"{fmt(row['mean_delta_class_balanced_mean_margin'])} | "
            f"{fmt(row['mean_delta_margin_snr'])} |"
        )

    report_lines.extend(
        [
            "",
            "## Target-specific macro-F1 effects",
            "",
            "| target | effect | mean Δ macro-F1 | SD |",
            "|---|---|---:|---:|",
        ]
    )
    for row in target_effect_summary:
        report_lines.append(
            f"| {row['target_env']} | {row['effect']} | "
            f"{fmt(row['mean_delta_macro_f1'])} | {fmt(row['sd_delta_macro_f1'])} |"
        )

    report_lines.extend(
        [
            "",
            "## Class-level joint-versus-control effect",
            "",
            "| class | mean Δ F1 | SD | mean Δ margin |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in class_summary:
        if row["effect"] != "joint_vs_control":
            continue
        report_lines.append(
            f"| {row['class_id']} {row['class_name']} | "
            f"{fmt(row['mean_delta_f1'])} | {fmt(row['sd_delta_f1'])} | "
            f"{fmt(row['mean_delta_margin'])} |"
        )

    report_lines.extend(
        [
            "",
            "## Empirical check of the sufficient-condition proxy",
            "",
            f"Using Y01 as the global-SupCon reference and `(Y11-Y01)/{args.residual_scale}` as the TPP treatment-effect proxy, ",
            f"`B/A > C/V` holds in **{theory_passes}/{len(theory_rows)}** target-seed-class cells. ",
            f"The empirical margin SNR increases in **{theory_positive_snr}/{len(theory_rows)}** cells.",
            "",
            "This is a factorial treatment-effect proxy because Y01 and Y11 are separately trained models. It is not an exact ",
            "within-checkpoint decomposition of the residual tensor. The raw cell statistics are in `margin_condition_proxy.csv`.",
            "",
            "## Interpretation guardrail",
            "",
            "A positive joint-versus-control contrast does not prove synergy. Synergy requires a positive interaction and the joint arm ",
            "to outperform both single-component arms. Target-specific effects must be reported because the formal study stopped for ",
            "harm on E1 in the joint-versus-TPP contrast.",
            "",
        ]
    )
    (output_dir / "MARGIN_ANALYSIS_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(json.dumps(
        {
            "formal_root": str(formal_root),
            "output_dir": str(output_dir),
            "blocks": len(blocks),
            "target_predictions": len(blocks) * len(ARMS) * 3000,
            "condition_passes": theory_passes,
            "condition_cells": len(theory_rows),
            "positive_snr_cells": theory_positive_snr,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
