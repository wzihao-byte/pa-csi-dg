"""Prewritten paired analysis for the H11 L1248 residual-scale curve."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BUNDLE = HERE / "generated_h11_l1248_v1"
RESULTS = REPO_ROOT / "outputs_tpp_residual_scale_ablation" / "h11_l1248_v1" / "scale_results.json"
OUTPUT_DIR = REPO_ROOT / "outputs_tpp_residual_scale_ablation" / "h11_l1248_v1" / "analysis"
SCALES = (0.125, 0.5, 0.75, 1.0)
REFERENCE_SCALE = 0.25
BOOTSTRAP_SEED = 20260809
BOOTSTRAP_REPLICATES = 100_000
CLASS_NAMES = ("no_movement", "falling", "walking", "sit_stand", "turning", "pick_up_pen")


class AnalysisError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AnalysisError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def exact_sign_flip_p(differences: np.ndarray) -> float:
    observed = abs(float(differences.mean()))
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        value = abs(float(np.mean(differences * np.asarray(signs, dtype=np.float64))))
        count += int(value >= observed - 1e-15)
        total += 1
    return count / total


def paired_bootstrap_ci(differences: np.ndarray, seed_offset: int) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_REPLICATES, len(differences)))
    values = differences[indices].mean(axis=1)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def class_vector(row: Mapping[str, Any]) -> np.ndarray:
    records = sorted(row["per_class_metrics"], key=lambda item: int(item["class_id"]))
    if [int(item["class_id"]) for item in records] != list(range(6)):
        raise AnalysisError("Per-class metric IDs are incomplete")
    values = np.asarray([float(item["f1"]) for item in records], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise AnalysisError("Non-finite per-class metric")
    return values


def main() -> None:
    manifest = read_json(BUNDLE / "bundle_manifest.json")
    protocol = read_json(BUNDLE / "protocol_lock.json")
    references = read_json(BUNDLE / "reference_registry.json")
    results = read_json(RESULTS)
    if results.get("status") != "complete" or results.get("job_count") != 36:
        raise AnalysisError("The 36-job result matrix is not complete")
    if results.get("bundle_manifest_sha256") != sha256_file(BUNDLE / "bundle_manifest.json"):
        raise AnalysisError("Result/bundle binding mismatch")
    if references.get("row_count") != 9:
        raise AnalysisError("Reference matrix is incomplete")
    if manifest.get("job_count") != 36 or protocol.get("scales") != list(SCALES):
        raise AnalysisError("Protocol scale matrix drifted")
    if len(results.get("paired_barriers", [])) != 9 or any(
        barrier.get("status") != "passed" for barrier in results["paired_barriers"]
    ):
        raise AnalysisError("Paired barriers are incomplete")

    reference_lookup = {
        (row["target_env"], int(row["seed"])): row for row in references["rows"]
    }
    candidate_lookup = {
        (float(row["scale"]), row["target_env"], int(row["seed"])): row for row in results["rows"]
    }
    keys = [(target, seed) for target in protocol["targets"] for seed in protocol["seeds"]]
    if set(reference_lookup) != set(keys):
        raise AnalysisError("Reference target-by-seed cells drifted")
    expected_candidates = {(scale, target, seed) for scale in SCALES for target, seed in keys}
    if set(candidate_lookup) != expected_candidates:
        raise AnalysisError("Candidate target-by-seed-scale cells drifted")

    reference_macro = np.asarray(
        [float(reference_lookup[key]["test_metrics"]["f1_macro"]) for key in keys], dtype=np.float64
    )
    reference_classes = np.stack([class_vector(reference_lookup[key]) for key in keys])
    comparisons: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for scale_index, scale in enumerate(SCALES):
        candidate_rows = [candidate_lookup[(scale, target, seed)] for target, seed in keys]
        candidate_macro = np.asarray(
            [float(row["test_metrics"]["f1_macro"]) for row in candidate_rows], dtype=np.float64
        )
        candidate_classes = np.stack([class_vector(row) for row in candidate_rows])
        differences = candidate_macro - reference_macro
        ci_low, ci_high = paired_bootstrap_ci(differences, scale_index)
        p_value = exact_sign_flip_p(differences)
        raw_p_values.append(p_value)
        environment_rows = []
        for target in protocol["targets"]:
            indices = [index for index, key in enumerate(keys) if key[0] == target]
            environment_rows.append(
                {
                    "target_env": target,
                    "candidate_mean_macro_f1": float(candidate_macro[indices].mean()),
                    "reference_mean_macro_f1": float(reference_macro[indices].mean()),
                    "paired_delta_macro_f1": float(differences[indices].mean()),
                }
            )
        per_class = []
        for class_id, class_name in enumerate(CLASS_NAMES):
            per_class.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "candidate_mean_f1": float(candidate_classes[:, class_id].mean()),
                    "reference_mean_f1": float(reference_classes[:, class_id].mean()),
                    "paired_delta_f1": float(
                        (candidate_classes[:, class_id] - reference_classes[:, class_id]).mean()
                    ),
                }
            )
        comparisons.append(
            {
                "scale": scale,
                "reference_scale": REFERENCE_SCALE,
                "candidate_mean_macro_f1": float(candidate_macro.mean()),
                "reference_mean_macro_f1": float(reference_macro.mean()),
                "paired_delta_macro_f1": float(differences.mean()),
                "bootstrap_95_ci": [ci_low, ci_high],
                "exact_two_sided_sign_flip_p": p_value,
                "improved_cells": int(np.sum(differences > 0)),
                "tied_cells": int(np.sum(differences == 0)),
                "declined_cells": int(np.sum(differences < 0)),
                "environment_results": environment_rows,
                "per_class_results": per_class,
            }
        )
        for index, (target, seed) in enumerate(keys):
            paired_rows.append(
                {
                    "scale": scale,
                    "target_env": target,
                    "seed": seed,
                    "candidate_macro_f1": float(candidate_macro[index]),
                    "reference_scale": REFERENCE_SCALE,
                    "reference_macro_f1": float(reference_macro[index]),
                    "paired_delta_macro_f1": float(differences[index]),
                }
            )

    adjusted = holm_adjust(raw_p_values)
    for comparison, adjusted_p in zip(comparisons, adjusted):
        comparison["holm_adjusted_p"] = adjusted_p
        comparison["holm_significant_at_0_05"] = bool(adjusted_p < 0.05)
    scale_curve = [
        {"scale": REFERENCE_SCALE, "mean_macro_f1": float(reference_macro.mean()), "source": "sealed_reference"}
    ] + [
        {"scale": row["scale"], "mean_macro_f1": row["candidate_mean_macro_f1"], "source": "new_candidate"}
        for row in comparisons
    ]
    scale_curve.sort(key=lambda row: float(row["scale"]))
    descriptive_best = max(scale_curve, key=lambda row: float(row["mean_macro_f1"]))
    analysis = {
        "schema_version": "tpp-residual-scale-analysis/v1",
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "post_hoc_exploratory": True,
        "h11_only_scope": "combined-system sensitivity; no isolated TPP main effect or SupCon-by-TPP interaction",
        "bundle_manifest_sha256": sha256_file(BUNDLE / "bundle_manifest.json"),
        "results_sha256": sha256_file(RESULTS),
        "reference_scale": REFERENCE_SCALE,
        "candidate_scales": list(SCALES),
        "cell_count_per_scale": len(keys),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES},
        "multiple_comparison_adjustment": "Holm across four scale-vs-0.25 contrasts",
        "scale_curve": scale_curve,
        "descriptive_best_scale": descriptive_best["scale"],
        "descriptive_best_mean_macro_f1": descriptive_best["mean_macro_f1"],
        "best_scale_is_not_confirmatory_selection": True,
        "comparisons": comparisons,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_json = OUTPUT_DIR / "residual_scale_analysis.json"
    atomic_json(output_json, analysis)
    csv_path = OUTPUT_DIR / "paired_cells.csv"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{csv_path.name}.", suffix=".tmp", dir=csv_path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
            writer.writeheader()
            writer.writerows(paired_rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, csv_path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    if any(not math.isfinite(float(row["paired_delta_macro_f1"])) for row in comparisons):
        raise AnalysisError("Non-finite comparison escaped validation")
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(output_json.resolve()),
                "paired_cells": str(csv_path.resolve()),
                "descriptive_best_scale": analysis["descriptive_best_scale"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
