from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BUNDLE = HERE / "generated_l1248_v1"
OUTPUT = REPO_ROOT / "outputs_tpp_level8_ablation" / "l1248_v1"
RESULTS = OUTPUT / "level8_results.json"
ANALYSIS_DIR = OUTPUT / "analysis"
TARGETS = ("E1", "E2", "E3")
SEEDS = (42, 52, 62)
CLASS_NAMES = {
    0: "no_movement",
    1: "falling",
    2: "walking",
    3: "sit_stand",
    4: "turning",
    5: "pick_up_pen",
}


class AnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    require(array.size > 0 and bool(np.isfinite(array).all()), "Invalid summary values")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def paired_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    result = summary(array)
    rng = np.random.default_rng(20260804)
    indices = rng.integers(0, len(array), size=(50_000, len(array)))
    boot = array[indices].mean(axis=1)
    result["bootstrap_95_ci"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    observed = abs(float(array.mean()))
    sign_means = []
    for mask in range(1 << len(array)):
        signs = np.asarray([1.0 if mask & (1 << index) else -1.0 for index in range(len(array))])
        sign_means.append(abs(float((array * signs).mean())))
    result["exact_sign_flip_two_sided_p"] = float(
        sum(value >= observed - 1e-15 for value in sign_means) / len(sign_means)
    )
    return result


def pair_confusion(path: Path) -> float:
    confusion = np.load(path, allow_pickle=False)
    require(confusion.shape == (6, 6), f"Confusion shape: {path}")
    denominator = int(confusion[1, :].sum() + confusion[4, :].sum())
    return float((confusion[1, 4] + confusion[4, 1]) / max(1, denominator))


def main() -> None:
    manifest = read_json(BUNDLE / "bundle_manifest.json")
    references = read_json(BUNDLE / "reference_registry.json")
    results = read_json(RESULTS)
    require(results["status"] == "complete" and results["job_count"] == 9, "Incomplete result matrix")
    require(results["bundle_manifest_sha256"] == sha256_file(BUNDLE / "bundle_manifest.json"), "Bundle binding")
    require(len(results["paired_barriers"]) == 9, "Paired barrier count")
    require(all(item["status"] == "passed" for item in results["paired_barriers"]), "Paired barrier failure")
    reference_lookup = {(row["target_env"], int(row["seed"])): row for row in references["rows"]}
    candidate_lookup = {(row["target_env"], int(row["seed"])): row for row in results["rows"]}
    expected_keys = [(target, seed) for target in TARGETS for seed in SEEDS]
    require(list(reference_lookup) == expected_keys, "Reference order")
    require(list(candidate_lookup) == expected_keys, "Candidate order")

    paired_rows = []
    class_rows = []
    cell_values: dict[tuple[str, int, str], float] = {}
    class_values: dict[tuple[str, int, str, int], float] = {}
    pair_values: dict[tuple[str, int, str], float] = {}
    audit_rows = []
    for target, seed in expected_keys:
        reference = reference_lookup[(target, seed)]
        candidate = candidate_lookup[(target, seed)]
        candidate_attempt = Path(candidate["attempt_dir"])
        candidate_completion = candidate_attempt / "attempt_completed.json"
        candidate_metrics_path = candidate_attempt / "payload" / "run" / "metrics.json"
        candidate_metrics = read_json(candidate_metrics_path)
        require(candidate["completion_manifest_sha256"] == sha256_file(candidate_completion), "Completion hash")
        require(candidate["metrics_file_sha256"] == sha256_file(candidate_metrics_path), "Metrics hash")
        require(candidate_metrics["epochs"] == 200, "Epoch count")
        require(candidate_metrics["total_optimizer_steps"] == 30_000, "Step count")
        require(candidate_metrics["target_prediction_count"] == 3_000, "Prediction count")
        coverage = [float(item["train"]["supcon_frac_anchors_with_positive"]) for item in candidate_metrics["history"]]
        require(min(coverage) == 1.0, "Positive-anchor coverage")
        candidate_schedule = read_json(candidate_attempt / "payload" / "run" / "sampler_schedule.json")
        candidate_split = read_json(candidate_attempt / "payload" / "run" / "split_manifest.json")
        split_hashes = {
            "train": candidate_split["train"]["global_sample_id_sha256"],
            "val": candidate_split["val"]["global_sample_id_sha256"],
            "test": candidate_split["test"]["global_sample_id_sha256"],
        }
        require(candidate_schedule["global_sample_schedule_sha256"] == reference["sampler_schedule_sha256"], "Sampler pairing")
        require(split_hashes == reference["split_hashes"], "Split pairing")

        reference_attempt = Path(reference["attempt_dir"])
        ref_f1 = float(reference["test_metrics"]["f1_macro"])
        cand_f1 = float(candidate_metrics["test_metrics"]["f1_macro"])
        ref_acc = float(reference["test_metrics"]["accuracy"])
        cand_acc = float(candidate_metrics["test_metrics"]["accuracy"])
        cell_values[(target, seed, "L124")] = ref_f1
        cell_values[(target, seed, "L1248")] = cand_f1
        pair_values[(target, seed, "L124")] = pair_confusion(reference_attempt / "payload" / "run" / "confusion_matrix.npy")
        pair_values[(target, seed, "L1248")] = pair_confusion(candidate_attempt / "payload" / "run" / "confusion_matrix.npy")
        paired_rows.append({
            "target_env": target,
            "seed": seed,
            "reference_macro_f1": ref_f1,
            "candidate_macro_f1": cand_f1,
            "delta_macro_f1": cand_f1 - ref_f1,
            "reference_accuracy": ref_acc,
            "candidate_accuracy": cand_acc,
            "delta_accuracy": cand_acc - ref_acc,
            "reference_pair_confusion": pair_values[(target, seed, "L124")],
            "candidate_pair_confusion": pair_values[(target, seed, "L1248")],
            "delta_pair_confusion": pair_values[(target, seed, "L1248")] - pair_values[(target, seed, "L124")],
        })
        reference_classes = {int(item["class_id"]): item for item in reference["per_class_metrics"]}
        candidate_classes = {int(item["class_id"]): item for item in candidate_metrics["per_class_metrics"]}
        for class_id in range(6):
            ref_class = float(reference_classes[class_id]["f1"])
            cand_class = float(candidate_classes[class_id]["f1"])
            class_values[(target, seed, "L124", class_id)] = ref_class
            class_values[(target, seed, "L1248", class_id)] = cand_class
            class_rows.append({
                "target_env": target,
                "seed": seed,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "reference_f1": ref_class,
                "candidate_f1": cand_class,
                "delta_f1": cand_class - ref_class,
            })
        audit_rows.append({
            "target_env": target,
            "seed": seed,
            "candidate_attempt": str(candidate_attempt),
            "reference_attempt": str(reference_attempt),
            "split_hashes": split_hashes,
            "sampler_schedule_sha256": candidate_schedule["global_sample_schedule_sha256"],
            "epochs": candidate_metrics["epochs"],
            "total_optimizer_steps": candidate_metrics["total_optimizer_steps"],
            "minimum_positive_anchor_fraction": min(coverage),
            "target_prediction_count": candidate_metrics["target_prediction_count"],
        })

    arm_summary = []
    environment_summary = []
    class_summary = []
    for arm in ("L124", "L1248"):
        arm_values = [cell_values[(target, seed, arm)] for target, seed in expected_keys]
        arm_summary.append({
            "arm": arm,
            **summary(arm_values),
            "falling_turning_pair_confusion_mean": float(np.mean([pair_values[(target, seed, arm)] for target, seed in expected_keys])),
        })
        for target in TARGETS:
            environment_summary.append({
                "target_env": target,
                "arm": arm,
                **summary(cell_values[(target, seed, arm)] for seed in SEEDS),
            })
        for class_id in range(6):
            class_summary.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "arm": arm,
                **summary(class_values[(target, seed, arm, class_id)] for target, seed in expected_keys),
            })

    deltas = [float(row["delta_macro_f1"]) for row in paired_rows]
    env_deltas = {
        target: float(np.mean([row["delta_macro_f1"] for row in paired_rows if row["target_env"] == target]))
        for target in TARGETS
    }
    safety_deltas = {
        CLASS_NAMES[class_id]: float(np.mean([row["delta_f1"] for row in class_rows if row["class_id"] == class_id]))
        for class_id in (0, 3, 5)
    }
    decision_checks = {
        "mean_macro_f1_gain_at_least_0_5pp": float(np.mean(deltas)) >= 0.005,
        "no_environment_drop_over_1pp": min(env_deltas.values()) >= -0.01,
        "no_safety_class_drop_over_2pp": min(safety_deltas.values()) >= -0.02,
    }
    analysis = {
        "schema_version": "tpp-level8-analysis/v1",
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_manifest_sha256": sha256_file(BUNDLE / "bundle_manifest.json"),
        "results_sha256": sha256_file(RESULTS),
        "aggregation": "unweighted paired mean over target-by-seed cells",
        "parameters": {"L124": 3_721_334, "L1248": 4_786_294, "increment": 1_064_960},
        "overall_arm_summary": arm_summary,
        "environment_arm_summary": environment_summary,
        "class_arm_summary": class_summary,
        "paired_macro_f1_effect": paired_summary(deltas),
        "paired_accuracy_effect": paired_summary(row["delta_accuracy"] for row in paired_rows),
        "paired_pair_confusion_effect": paired_summary(row["delta_pair_confusion"] for row in paired_rows),
        "environment_macro_f1_deltas": env_deltas,
        "safety_class_f1_deltas": safety_deltas,
        "decision_checks": decision_checks,
        "promote_level8": all(decision_checks.values()),
        "audit_rows": audit_rows,
    }
    atomic_json(ANALYSIS_DIR / "level8_analysis.json", analysis)
    atomic_csv(
        ANALYSIS_DIR / "paired_cells.csv",
        list(paired_rows[0]),
        paired_rows,
    )
    atomic_csv(
        ANALYSIS_DIR / "paired_class_f1.csv",
        list(class_rows[0]),
        class_rows,
    )
    atomic_csv(
        ANALYSIS_DIR / "overall_arm_summary.csv",
        list(arm_summary[0]),
        arm_summary,
    )
    atomic_csv(
        ANALYSIS_DIR / "environment_arm_summary.csv",
        list(environment_summary[0]),
        environment_summary,
    )
    atomic_csv(
        ANALYSIS_DIR / "class_arm_summary.csv",
        list(class_summary[0]),
        class_summary,
    )
    manifest_out = {
        "schema_version": "tpp-level8-analysis-manifest/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_bundle_manifest_sha256": manifest["execution_lock_sha256"],
        "source_results_sha256": sha256_file(RESULTS),
        "artifacts": [
            {"path": path.name, "sha256": sha256_file(path), "size": path.stat().st_size}
            for path in sorted(ANALYSIS_DIR.iterdir())
            if path.is_file() and path.name != "analysis_manifest.json"
        ],
    }
    atomic_json(ANALYSIS_DIR / "analysis_manifest.json", manifest_out)
    print(json.dumps({
        "status": "passed",
        "mean_delta_macro_f1": analysis["paired_macro_f1_effect"]["mean"],
        "bootstrap_95_ci": analysis["paired_macro_f1_effect"]["bootstrap_95_ci"],
        "promote_level8": analysis["promote_level8"],
        "analysis": str((ANALYSIS_DIR / "level8_analysis.json").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
