from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = {
    0: "C0 no movement",
    1: "C1 falling",
    2: "C2 sit/stand",
    3: "C3 walking",
    4: "C4 turning",
    5: "C5 pick up pen",
}
TARGET_ENVS = ("E1", "E2", "E3")
FINAL_SEEDS = (42, 52, 62)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TPP C3-recovery candidates against seed-matched CE baselines."
    )
    parser.add_argument("--candidate-root", action="append", default=None)
    parser.add_argument("--baseline-root", action="append", default=None)
    parser.add_argument("--analysis-root", default="analysis_tpp_c3_recovery")
    parser.add_argument("--min-pair-reduction", type=float, default=0.10)
    parser.add_argument("--min-c1-f1-delta", type=float, default=0.05)
    parser.add_argument("--min-c3-f1-delta", type=float, default=-0.05)
    parser.add_argument("--min-macro-f1-delta", type=float, default=-0.01)
    parser.add_argument("--min-c0-precision-delta", type=float, default=-0.03)
    parser.add_argument("--min-e1-seed-wins", type=int, default=2)
    return parser.parse_args()


def resolve_roots(values: Sequence[str]) -> List[Path]:
    return [(REPO_ROOT / value).resolve() for value in values]


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
    return np.loadtxt(csv_path, delimiter=",", dtype=np.int64)


def infer_experiment_name(metrics_path: Path) -> str:
    for index, part in enumerate(metrics_path.parts):
        if part.startswith("target_") and index > 0:
            return metrics_path.parts[index - 1]
    return metrics_path.parent.parent.parent.name


def infer_target(metrics: Mapping[str, Any], metrics_path: Path) -> str:
    target = str(metrics.get("target_env") or "")
    if target:
        return target
    for part in metrics_path.parts:
        if part.startswith("target_"):
            return part.removeprefix("target_")
    return ""


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )


def class_metrics(confusion: np.ndarray) -> List[Dict[str, float]]:
    diagonal = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1).astype(float)
    predicted = confusion.sum(axis=0).astype(float)
    recall = safe_divide(diagonal, support)
    precision = safe_divide(diagonal, predicted)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    return [
        {
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "support": float(support[class_id]),
        }
        for class_id in range(confusion.shape[0])
    ]


def pair_confusion(confusion: np.ndarray, class_a: int = 1, class_b: int = 4) -> float:
    total = float(confusion[class_a].sum() + confusion[class_b].sum())
    if total == 0.0:
        return 0.0
    return float(confusion[class_a, class_b] + confusion[class_b, class_a]) / total


def row_rate(confusion: np.ndarray, true_class: int, predicted_class: int) -> float:
    total = float(confusion[true_class].sum())
    return float(confusion[true_class, predicted_class]) / total if total else 0.0


def load_runs(root: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not root.exists():
        return runs
    for metrics_path in sorted(root.rglob("metrics.json")):
        metrics = read_json(metrics_path)
        confusion = read_confusion(metrics_path.parent)
        per_class = class_metrics(confusion)
        runs.append(
            {
                "experiment_name": str(
                    metrics.get("experiment_name") or infer_experiment_name(metrics_path)
                ),
                "target_env": infer_target(metrics, metrics_path),
                "seed": int(metrics.get("seed", metrics_path.parent.name.removeprefix("seed_"))),
                "metrics_path": metrics_path,
                "confusion": confusion,
                "per_class": per_class,
                "accuracy": float(np.diag(confusion).sum() / confusion.sum()),
                "macro_f1": float(mean(item["f1"] for item in per_class)),
                "pair_confusion": pair_confusion(confusion),
                "c3_to_c0": row_rate(confusion, 3, 0),
                "c3_to_c4": row_rate(confusion, 3, 4),
            }
        )
    return runs


def baseline_index(roots: Sequence[Path]) -> Dict[Tuple[str, int], Mapping[str, Any]]:
    index: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for root in roots:
        for run in load_runs(root):
            key = (str(run["target_env"]), int(run["seed"]))
            index.setdefault(key, run)
    return index


def comparison_rows(
    candidates: Sequence[Mapping[str, Any]],
    baselines: Mapping[Tuple[str, int], Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, int]] = set()

    for candidate in candidates:
        key = (str(candidate["target_env"]), int(candidate["seed"]))
        baseline = baselines.get(key)
        if baseline is None:
            continue
        unique_key = (str(candidate["experiment_name"]), key[0], key[1])
        if unique_key in seen:
            continue
        seen.add(unique_key)

        run_rows.append(
            {
                "experiment_name": candidate["experiment_name"],
                "target_env": key[0],
                "seed": key[1],
                "baseline_macro_f1": baseline["macro_f1"],
                "candidate_macro_f1": candidate["macro_f1"],
                "delta_macro_f1": candidate["macro_f1"] - baseline["macro_f1"],
                "baseline_accuracy": baseline["accuracy"],
                "candidate_accuracy": candidate["accuracy"],
                "delta_accuracy": candidate["accuracy"] - baseline["accuracy"],
                "baseline_pair_confusion": baseline["pair_confusion"],
                "candidate_pair_confusion": candidate["pair_confusion"],
                "delta_pair_confusion": candidate["pair_confusion"] - baseline["pair_confusion"],
                "baseline_c3_to_c0": baseline["c3_to_c0"],
                "candidate_c3_to_c0": candidate["c3_to_c0"],
                "delta_c3_to_c0": candidate["c3_to_c0"] - baseline["c3_to_c0"],
                "baseline_c3_to_c4": baseline["c3_to_c4"],
                "candidate_c3_to_c4": candidate["c3_to_c4"],
                "delta_c3_to_c4": candidate["c3_to_c4"] - baseline["c3_to_c4"],
                "baseline_metrics_path": str(baseline["metrics_path"]),
                "candidate_metrics_path": str(candidate["metrics_path"]),
            }
        )

        for class_id, (base_class, candidate_class) in enumerate(
            zip(baseline["per_class"], candidate["per_class"])
        ):
            class_rows.append(
                {
                    "experiment_name": candidate["experiment_name"],
                    "target_env": key[0],
                    "seed": key[1],
                    "class_id": class_id,
                    "class_name": CLASS_NAMES.get(class_id, f"C{class_id}"),
                    "baseline_precision": base_class["precision"],
                    "candidate_precision": candidate_class["precision"],
                    "delta_precision": candidate_class["precision"] - base_class["precision"],
                    "baseline_recall": base_class["recall"],
                    "candidate_recall": candidate_class["recall"],
                    "delta_recall": candidate_class["recall"] - base_class["recall"],
                    "baseline_f1": base_class["f1"],
                    "candidate_f1": candidate_class["f1"],
                    "delta_f1": candidate_class["f1"] - base_class["f1"],
                    "support": base_class["support"],
                }
            )
    return run_rows, class_rows


def mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return mean(float(row[field]) for row in rows)


def aggregate_experiments(
    run_rows: Sequence[Mapping[str, Any]],
    class_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    grouped_runs: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    grouped_classes: Dict[Tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped_runs[str(row["experiment_name"])].append(row)
    for row in class_rows:
        grouped_classes[(str(row["experiment_name"]), int(row["class_id"]))].append(row)

    expected = {(env, seed) for env in TARGET_ENVS for seed in FINAL_SEEDS}
    aggregates: List[Dict[str, Any]] = []
    for experiment_name, rows in sorted(grouped_runs.items()):
        c0_rows = grouped_classes[(experiment_name, 0)]
        c1_rows = grouped_classes[(experiment_name, 1)]
        c3_rows = grouped_classes[(experiment_name, 3)]
        c4_rows = grouped_classes[(experiment_name, 4)]
        base_pair = mean_field(rows, "baseline_pair_confusion")
        candidate_pair = mean_field(rows, "candidate_pair_confusion")
        pair_reduction = (base_pair - candidate_pair) / base_pair if base_pair else 0.0
        e1_rows = [row for row in rows if row["target_env"] == "E1"]
        e1_seed_wins = sum(
            float(row["candidate_pair_confusion"]) < float(row["baseline_pair_confusion"])
            for row in e1_rows
        )
        available = {(str(row["target_env"]), int(row["seed"])) for row in rows}

        gate_macro = mean_field(rows, "delta_macro_f1") >= args.min_macro_f1_delta
        gate_c3 = mean_field(c3_rows, "delta_f1") >= args.min_c3_f1_delta
        gate_c1 = mean_field(c1_rows, "delta_f1") >= args.min_c1_f1_delta
        gate_pair = pair_reduction >= args.min_pair_reduction
        gate_c0_precision = (
            mean_field(c0_rows, "delta_precision") >= args.min_c0_precision_delta
        )
        gate_e1 = e1_seed_wins >= args.min_e1_seed_wins
        diagnostic_pass = all(
            [gate_macro, gate_c3, gate_c1, gate_pair, gate_c0_precision, gate_e1]
        )
        matrix_complete = expected.issubset(available)

        aggregates.append(
            {
                "experiment_name": experiment_name,
                "runs": len(rows),
                "matrix_complete": matrix_complete,
                "baseline_macro_f1": mean_field(rows, "baseline_macro_f1"),
                "candidate_macro_f1": mean_field(rows, "candidate_macro_f1"),
                "delta_macro_f1": mean_field(rows, "delta_macro_f1"),
                "delta_c0_precision": mean_field(c0_rows, "delta_precision"),
                "delta_c1_f1": mean_field(c1_rows, "delta_f1"),
                "delta_c3_f1": mean_field(c3_rows, "delta_f1"),
                "delta_c3_recall": mean_field(c3_rows, "delta_recall"),
                "delta_c4_f1": mean_field(c4_rows, "delta_f1"),
                "baseline_pair_confusion": base_pair,
                "candidate_pair_confusion": candidate_pair,
                "relative_pair_reduction": pair_reduction,
                "delta_c3_to_c0": mean_field(rows, "delta_c3_to_c0"),
                "delta_c3_to_c4": mean_field(rows, "delta_c3_to_c4"),
                "e1_seed_wins": e1_seed_wins,
                "e1_seed_runs": len(e1_rows),
                "gate_macro_f1": gate_macro,
                "gate_c3_f1": gate_c3,
                "gate_c1_f1": gate_c1,
                "gate_pair": gate_pair,
                "gate_c0_precision": gate_c0_precision,
                "gate_e1_seed_wins": gate_e1,
                "diagnostic_pass": diagnostic_pass,
                "promotion_ready": diagnostic_pass and matrix_complete,
            }
        )
    return aggregates


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    return f"{float(value):.4f}"


def write_report(
    path: Path,
    aggregates: Sequence[Mapping[str, Any]],
    class_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# TPP C3 Recovery Report",
        "",
        "Candidates are matched to the CE baseline by target environment and seed.",
        "A candidate is promotion-ready only after the full 3-target x 3-seed matrix is complete and every gate passes.",
        "",
        "## Acceptance Gates",
        "",
        f"- mean macro-F1 delta >= {args.min_macro_f1_delta:+.3f}",
        f"- mean C3 F1 delta >= {args.min_c3_f1_delta:+.3f}",
        f"- mean C1 F1 delta >= {args.min_c1_f1_delta:+.3f}",
        f"- relative C1/C4 pair-confusion reduction >= {args.min_pair_reduction:.1%}",
        f"- mean C0 precision delta >= {args.min_c0_precision_delta:+.3f}",
        f"- E1 pair confusion improves in at least {args.min_e1_seed_wins} seeds",
        "",
        "## Experiment Summary",
        "",
        "| experiment | runs | complete | delta macro F1 | delta C1 F1 | delta C3 F1 | delta C3 recall | pair reduction | E1 wins | diagnostic | promote |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in aggregates:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["experiment_name"]),
                    str(row["runs"]),
                    "yes" if row["matrix_complete"] else "no",
                    fmt(row["delta_macro_f1"]),
                    fmt(row["delta_c1_f1"]),
                    fmt(row["delta_c3_f1"]),
                    fmt(row["delta_c3_recall"]),
                    fmt(row["relative_pair_reduction"]),
                    f"{row['e1_seed_wins']}/{row['e1_seed_runs']}",
                    fmt(row["diagnostic_pass"]),
                    fmt(row["promotion_ready"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Mean Per-Class F1 Delta", ""])
    for aggregate in aggregates:
        experiment_name = str(aggregate["experiment_name"])
        lines.extend(
            [
                f"### {experiment_name}",
                "",
                "| class | baseline F1 | candidate F1 | delta F1 | delta recall | delta precision |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for class_id in sorted(CLASS_NAMES):
            rows = [
                row
                for row in class_rows
                if row["experiment_name"] == experiment_name and int(row["class_id"]) == class_id
            ]
            if not rows:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        CLASS_NAMES[class_id],
                        fmt(mean_field(rows, "baseline_f1")),
                        fmt(mean_field(rows, "candidate_f1")),
                        fmt(mean_field(rows, "delta_f1")),
                        fmt(mean_field(rows, "delta_recall")),
                        fmt(mean_field(rows, "delta_precision")),
                    ]
                )
                + " |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidate_roots = resolve_roots(
        args.candidate_root
        or [
            "outputs_tpp_isolated/dg_loeo/tpp_l124",
            "outputs_tpp_c3_recovery/dg_loeo",
        ]
    )
    baseline_roots = resolve_roots(
        args.baseline_root
        or [
            "outputs_anchor_ce_baseline/dg_loeo/anchor_ce_baseline",
            "outputs_tpp_isolated/baseline_ref/dg_loeo/anchor_ce_baseline_ref",
        ]
    )
    analysis_root = (REPO_ROOT / args.analysis_root).resolve()

    baselines = baseline_index(baseline_roots)
    candidates: List[Dict[str, Any]] = []
    for root in candidate_roots:
        candidates.extend(load_runs(root))
    run_rows, class_rows = comparison_rows(candidates, baselines)
    if not run_rows:
        raise SystemExit(
            f"No matched candidate/baseline runs found; candidates={candidate_roots}; "
            f"baselines={baseline_roots}."
        )

    aggregates = aggregate_experiments(run_rows, class_rows, args)
    write_csv(analysis_root / "c3_recovery_per_run.csv", run_rows)
    write_csv(analysis_root / "c3_recovery_per_class.csv", class_rows)
    write_csv(analysis_root / "c3_recovery_summary.csv", aggregates)
    write_report(analysis_root / "C3_RECOVERY_REPORT.md", aggregates, class_rows, args)
    print(f"Wrote C3 recovery analysis for {len(aggregates)} experiments to {analysis_root}")


if __name__ == "__main__":
    main()
