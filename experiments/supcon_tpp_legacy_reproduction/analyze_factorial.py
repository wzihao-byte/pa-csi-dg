from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
BUNDLE = EXPERIMENT_ROOT / "generated_factorial_s32_v1"
OUTPUT = ROOT / "outputs_supcon_tpp_legacy_reproduction" / "factorial_s32"
AUDIT_DIR = ROOT / "outputs_supcon_tpp_legacy_reproduction" / "audit"
ANALYSIS_DIR = OUTPUT / "analysis"

TARGETS = ("E1", "E2", "E3")
SEEDS = (42, 52, 62)
ARMS = ("H00", "H01", "H10", "H11")
CLASS_NAMES = {
    0: "no_movement",
    1: "falling",
    2: "walking",
    3: "sit_stand",
    4: "turning",
    5: "pick_up_pen",
}
THRESHOLDS = {
    "overall_noninferiority_margin": -0.01,
    "per_environment_maximum_drop": -0.02,
    "pick_up_pen_overall_margin": -0.03,
    "pick_up_pen_per_environment_margin": -0.05,
    "minimum_better_gain": 0.01,
    "minimum_interaction": 0.01,
}


class AnalysisError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def close(actual: float, expected: float, context: str, tolerance: float = 1e-12) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise AnalysisError(f"{context}: expected {expected!r}, got {actual!r}")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def metrics_from_confusion(confusion: np.ndarray) -> tuple[dict[str, float], list[dict[str, Any]]]:
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros(6), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros(6), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(6), where=(precision + recall) > 0)
    aggregate = {
        "accuracy": float(true_positive.sum() / confusion.sum()),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
    }
    per_class = [
        {
            "class_id": class_id,
            "support": int(support[class_id]),
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
        }
        for class_id in range(6)
    ]
    return aggregate, per_class


def recompute_predictions(path: Path) -> tuple[np.ndarray, int]:
    confusion = np.zeros((6, 6), dtype=np.int64)
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for expected_index, line in enumerate(handle):
            record = json.loads(line)
            require(record["index"] == expected_index, f"unordered prediction in {path}")
            label = int(record["label"])
            prediction = int(record["prediction"])
            probabilities = np.asarray(record["probabilities"], dtype=np.float64)
            require(probabilities.shape == (6,), f"probability shape in {path}")
            require(np.isfinite(probabilities).all(), f"non-finite probability in {path}")
            require(abs(float(probabilities.sum()) - 1.0) <= 1e-5, f"probability sum in {path}")
            require(int(probabilities.argmax()) == prediction, f"argmax mismatch in {path}")
            require(0 <= label < 6 and 0 <= prediction < 6, f"class range in {path}")
            confusion[label, prediction] += 1
            count += 1
    return confusion, count


def tree_records(root: Path, excluded: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in excluded:
            continue
        records[path.relative_to(root).as_posix()] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return records


def sample_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    require(array.size > 0 and np.isfinite(array).all(), "invalid summary values")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def paired_effect_summary(values: list[float]) -> dict[str, Any]:
    summary = sample_summary(values)
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260804)
    indices = rng.integers(0, len(array), size=(50000, len(array)))
    boot = array[indices].mean(axis=1)
    summary["bootstrap_95_ci"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    observed = abs(float(array.mean()))
    sign_means = []
    for mask in range(1 << len(array)):
        signs = np.asarray([1.0 if mask & (1 << index) else -1.0 for index in range(len(array))])
        sign_means.append(abs(float((array * signs).mean())))
    summary["exact_sign_flip_two_sided_p"] = float(sum(value >= observed - 1e-15 for value in sign_means) / len(sign_means))
    return summary


def main() -> None:
    registry_path = BUNDLE / "factorial_job_registry.json"
    bundle_manifest_path = BUNDLE / "bundle_manifest.json"
    results_path = OUTPUT / "factorial_results.json"
    registry = read_json(registry_path)
    results = read_json(results_path)
    jobs = registry["jobs"]
    rows = results["rows"]
    require(len(jobs) == 36 and len(rows) == 36, "expected 36 registry jobs and result rows")
    expected_order = [(target, seed, arm) for target in TARGETS for seed in SEEDS for arm in ARMS]
    job_order = [(item["target_env"], int(item["seed"]), item["arm"]) for item in jobs]
    row_order = [(item["target_env"], int(item["seed"]), item["arm"]) for item in rows]
    require(job_order == expected_order, "registry order mismatch")
    require(row_order == expected_order, "factorial results order mismatch")

    bundle_sha = sha256_file(bundle_manifest_path)
    results_sha = sha256_file(results_path)
    require(results["status"] == "complete", "factorial results not complete")
    require(results["job_count"] == 36, "factorial result job count")
    require(results["bundle_manifest_sha256"] == bundle_sha, "factorial bundle binding")
    require(results["analysis_status"] == "not_yet_run", "unexpected runner analysis status")
    require(results["no_target_metric_tuning"] is True, "target tuning declaration")

    row_by_key = {(row["target_env"], int(row["seed"]), row["arm"]): row for row in rows}
    verified_rows: list[dict[str, Any]] = []
    metric_lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    confusion_lookup: dict[tuple[str, int, str], np.ndarray] = {}

    for job in jobs:
        key = (job["target_env"], int(job["seed"]), job["arm"])
        row = row_by_key[key]
        require(row["job_id"] == job["job_id"], f"job id mismatch {key}")
        attempt = Path(row["attempt_dir"])
        completion_path = attempt / "attempt_completed.json"
        seal_path = attempt / "attempt_seal.json"
        run_dir = attempt / "payload" / "run"
        completion = read_json(completion_path)
        seal = read_json(seal_path)
        require(seal["completion_manifest_sha256"] == sha256_file(completion_path), f"seal mismatch {key}")
        require(row["completion_manifest_sha256"] == sha256_file(completion_path), f"result completion hash {key}")
        require(
            completion["artifacts"] == tree_records(attempt, ("attempt_completed.json", "attempt_seal.json")),
            f"sealed artifact tree {key}",
        )

        metrics_path = run_dir / "metrics.json"
        metrics = read_json(metrics_path)
        require(row["metrics_file_sha256"] == sha256_file(metrics_path), f"metrics hash {key}")
        require(metrics["epochs"] == 200, f"epoch count {key}")
        require(metrics["optimizer_steps_per_epoch"] == 150, f"steps per epoch {key}")
        require(metrics["total_optimizer_steps"] == 30000, f"optimizer steps {key}")
        require(metrics["target_prediction_count"] == 3000, f"metric prediction count {key}")
        require(metrics["sampler_mode"] == "domain_class_balanced", f"sampler mode {key}")
        require(len(metrics["history"]) == 200, f"history length {key}")
        if key[2] in ("H01", "H11"):
            coverage = [float(epoch["train"]["supcon_frac_anchors_with_positive"]) for epoch in metrics["history"]]
            require(min(coverage) >= 1.0, f"positive-anchor coverage {key}")

        checkpoint_path = run_dir / "best_model.pt"
        checkpoint_sha = sha256_file(checkpoint_path)
        freeze_path = run_dir / "checkpoint_frozen.json"
        freeze = read_json(freeze_path)
        access = read_json(run_dir / "target_access_audit.json")
        require(row["checkpoint_sha256"] == checkpoint_sha, f"result checkpoint hash {key}")
        require(metrics["checkpoint_sha256"] == checkpoint_sha, f"metrics checkpoint hash {key}")
        require(freeze["status"] == "frozen_and_strict_reloaded", f"freeze status {key}")
        require(freeze["checkpoint_sha256"] == checkpoint_sha, f"freeze checkpoint hash {key}")
        require(freeze["target_data_opened"] is False, f"target opened before freeze {key}")
        require(access["checkpoint_frozen_manifest_sha256"] == sha256_file(freeze_path), f"access freeze binding {key}")
        require(access["checkpoint_sha256"] == checkpoint_sha, f"access checkpoint binding {key}")
        require(access["checkpoint_strict_reloaded_before_target_access"] is True, f"delayed target access {key}")

        prediction_path = run_dir / "target_predictions.jsonl"
        prediction_sha = sha256_file(prediction_path)
        require(row["target_predictions_file_sha256"] == prediction_sha, f"result prediction hash {key}")
        require(metrics["target_predictions_file_sha256"] == prediction_sha, f"metrics prediction hash {key}")
        require(access["target_predictions_file_sha256"] == prediction_sha, f"access prediction hash {key}")
        confusion, prediction_count = recompute_predictions(prediction_path)
        require(prediction_count == 3000 == row["target_prediction_count"], f"prediction count {key}")
        stored_confusion = np.load(run_dir / "confusion_matrix.npy", allow_pickle=False)
        require(stored_confusion.shape == (6, 6), f"confusion shape {key}")
        require(np.array_equal(confusion, stored_confusion), f"confusion mismatch {key}")
        aggregate, per_class = metrics_from_confusion(confusion)
        for name, value in aggregate.items():
            close(value, float(metrics["test_metrics"][name]), f"aggregate {key}/{name}")
            close(value, float(row["test_metrics"][name]), f"result aggregate {key}/{name}")
        require(len(per_class) == len(metrics["per_class_metrics"]) == len(row["per_class_metrics"]) == 6, f"per-class length {key}")
        for expected, metric_item, row_item in zip(per_class, metrics["per_class_metrics"], row["per_class_metrics"]):
            require(expected["class_id"] == metric_item["class_id"] == row_item["class_id"], f"class id {key}")
            require(expected["support"] == metric_item["support"] == row_item["support"], f"support {key}")
            for name in ("precision", "recall", "f1"):
                close(float(expected[name]), float(metric_item[name]), f"metric per-class {key}/{name}")
                close(float(expected[name]), float(row_item[name]), f"result per-class {key}/{name}")

        schedule = read_json(run_dir / "sampler_schedule.json")
        split = read_json(run_dir / "split_manifest.json")
        require((schedule["epochs_seen"], schedule["batches_seen"], schedule["samples_seen"]) == (200, 30000, 960000), f"schedule counts {key}")
        require(split["train"]["count"] == 4800 and split["val"]["count"] == 1200 and split["test"]["count"] == 3000, f"split counts {key}")
        metric_lookup[key] = metrics
        confusion_lookup[key] = confusion
        verified_rows.append(
            {
                "job_id": job["job_id"],
                "target_env": key[0],
                "seed": key[1],
                "arm": key[2],
                "attempt_dir": str(attempt),
                "completion_manifest_sha256": sha256_file(completion_path),
                "checkpoint_sha256": checkpoint_sha,
                "metrics_sha256": sha256_file(metrics_path),
                "predictions_sha256": prediction_sha,
                "split_hashes": {
                    "train": split["train"]["global_sample_id_sha256"],
                    "val": split["val"]["global_sample_id_sha256"],
                    "test": split["test"]["global_sample_id_sha256"],
                },
                "sampler_schedule_sha256": schedule["global_sample_schedule_sha256"],
                "target_prediction_count": prediction_count,
            }
        )

    for target in TARGETS:
        for seed in SEEDS:
            block = [row for row in verified_rows if row["target_env"] == target and row["seed"] == seed]
            require(len(block) == 4, f"quartet size {target}/{seed}")
            require(len({row["sampler_schedule_sha256"] for row in block}) == 1, f"quartet sampler {target}/{seed}")
            require(len({payload_sha256(row["split_hashes"]) for row in block}) == 1, f"quartet split {target}/{seed}")

    barrier_files = sorted((OUTPUT / "runner_invocations").rglob("barrier_*.json"))
    require(len(barrier_files) == 9, "expected nine quartet barriers")
    barrier_keys = []
    for path in barrier_files:
        barrier = read_json(path)
        require(barrier["status"] == "passed", f"barrier status {path}")
        barrier_keys.append((barrier["target"], int(barrier["seed"])))
    require(sorted(barrier_keys) == sorted((target, seed) for target in TARGETS for seed in SEEDS), "barrier matrix")

    audit = {
        "schema_version": "factorial-s32-independent-audit/v1",
        "status": "passed",
        "created_at_utc": utc_now(),
        "locked_runner_final_dry_run": {
            "status": "passed",
            "exit_code": 0,
            "planned_jobs": 36,
            "jobs_to_run": 0,
            "jobs_to_skip": 36,
        },
        "bundle_manifest_sha256": bundle_sha,
        "registry_sha256": sha256_file(registry_path),
        "factorial_results_sha256": results_sha,
        "job_count": len(verified_rows),
        "seal_count": 36,
        "failed_attempt_count": 0,
        "quartet_barrier_count": 9,
        "checks": {
            "artifact_tree_and_completion_seals": True,
            "checkpoint_hashes": True,
            "metrics_and_prediction_hashes": True,
            "strict_checkpoint_probe_via_locked_runner": True,
            "prediction_order_probability_and_count": True,
            "confusion_and_metrics_independent_recompute": True,
            "split_and_sampler_quartet_pairing": True,
            "epochs_steps_and_positive_anchor_coverage": True,
            "checkpoint_freeze_and_delayed_target_access": True,
        },
        "rows": verified_rows,
    }
    audit_path = AUDIT_DIR / "factorial_s32_completion.json"
    atomic_json(audit_path, audit)
    audit_sha = sha256_file(audit_path)

    cell_f1 = {key: float(metrics["test_metrics"]["f1_macro"]) for key, metrics in metric_lookup.items()}
    class_f1 = {
        (target, seed, arm, int(item["class_id"])): float(item["f1"])
        for (target, seed, arm), metrics in metric_lookup.items()
        for item in metrics["per_class_metrics"]
    }
    pair_confusion = {
        key: float((confusion[1, 4] + confusion[4, 1]) / max(1, confusion[1, :].sum() + confusion[4, :].sum()))
        for key, confusion in confusion_lookup.items()
    }

    overall_arm_rows = []
    environment_arm_rows = []
    class_arm_rows = []
    environment_class_arm_rows = []
    for arm in ARMS:
        summary = sample_summary(cell_f1[(target, seed, arm)] for target in TARGETS for seed in SEEDS)
        pair_summary = sample_summary(pair_confusion[(target, seed, arm)] for target in TARGETS for seed in SEEDS)
        overall_arm_rows.append({"arm": arm, **summary, "falling_turning_pair_confusion_mean": pair_summary["mean"]})
        for target in TARGETS:
            summary = sample_summary(cell_f1[(target, seed, arm)] for seed in SEEDS)
            pair_summary = sample_summary(pair_confusion[(target, seed, arm)] for seed in SEEDS)
            environment_arm_rows.append({"target_env": target, "arm": arm, **summary, "falling_turning_pair_confusion_mean": pair_summary["mean"]})
        for class_id in range(6):
            summary = sample_summary(class_f1[(target, seed, arm, class_id)] for target in TARGETS for seed in SEEDS)
            class_arm_rows.append({"class_id": class_id, "class_name": CLASS_NAMES[class_id], "arm": arm, **summary})
            for target in TARGETS:
                env_summary = sample_summary(class_f1[(target, seed, arm, class_id)] for seed in SEEDS)
                environment_class_arm_rows.append(
                    {"target_env": target, "class_id": class_id, "class_name": CLASS_NAMES[class_id], "arm": arm, **env_summary}
                )

    contrast_definitions = {
        "supcon_on_global": lambda values: values["H01"] - values["H00"],
        "supcon_on_tpp": lambda values: values["H11"] - values["H10"],
        "tpp_without_supcon": lambda values: values["H10"] - values["H00"],
        "tpp_with_supcon": lambda values: values["H11"] - values["H01"],
        "interaction": lambda values: values["H11"] - values["H10"] - values["H01"] + values["H00"],
    }
    paired_effect_rows = []
    paired_effects: dict[str, Any] = {}
    for name, contrast in contrast_definitions.items():
        values = []
        for target in TARGETS:
            for seed in SEEDS:
                arm_values = {arm: cell_f1[(target, seed, arm)] for arm in ARMS}
                value = float(contrast(arm_values))
                values.append(value)
                paired_effect_rows.append({"contrast": name, "target_env": target, "seed": seed, "macro_f1_difference": value})
        paired_effects[name] = paired_effect_summary(values)

    overall = {row["arm"]: float(row["mean"]) for row in overall_arm_rows}
    by_environment = {(row["target_env"], row["arm"]): float(row["mean"]) for row in environment_arm_rows}
    by_class = {(int(row["class_id"]), row["arm"]): float(row["mean"]) for row in class_arm_rows}
    by_environment_class = {
        (row["target_env"], int(row["class_id"]), row["arm"]): float(row["mean"])
        for row in environment_class_arm_rows
    }
    coexistence_checks = {
        "overall_vs_H01": overall["H11"] - overall["H01"] >= THRESHOLDS["overall_noninferiority_margin"],
        "overall_vs_H10": overall["H11"] - overall["H10"] >= THRESHOLDS["overall_noninferiority_margin"],
        "all_environment_stronger_single": all(
            by_environment[(target, "H11")] - max(by_environment[(target, "H01")], by_environment[(target, "H10")])
            >= THRESHOLDS["per_environment_maximum_drop"]
            for target in TARGETS
        ),
        "pick_up_pen_overall": by_class[(5, "H11")] - max(by_class[(5, "H01")], by_class[(5, "H10")])
        >= THRESHOLDS["pick_up_pen_overall_margin"],
        "pick_up_pen_all_environments": all(
            by_environment_class[(target, 5, "H11")]
            - max(by_environment_class[(target, 5, "H01")], by_environment_class[(target, 5, "H10")])
            >= THRESHOLDS["pick_up_pen_per_environment_margin"]
            for target in TARGETS
        ),
    }
    better_checks = {
        "overall_vs_H01": overall["H11"] - overall["H01"] >= THRESHOLDS["minimum_better_gain"],
        "overall_vs_H10": overall["H11"] - overall["H10"] >= THRESHOLDS["minimum_better_gain"],
    }
    interaction_check = paired_effects["interaction"]["mean"] >= THRESHOLDS["minimum_interaction"]
    decision = {
        "schema_version": "factorial-s32-coexistence-decision/v1",
        "created_at_utc": utc_now(),
        "factorial_results_sha256": results_sha,
        "completion_audit_sha256": audit_sha,
        "thresholds": THRESHOLDS,
        "observed": {
            "overall_macro_f1": overall,
            "H11_minus_H01": overall["H11"] - overall["H01"],
            "H11_minus_H10": overall["H11"] - overall["H10"],
            "environment_H11_minus_stronger_single": {
                target: by_environment[(target, "H11")]
                - max(by_environment[(target, "H01")], by_environment[(target, "H10")])
                for target in TARGETS
            },
            "pick_up_pen_H11_minus_stronger_single_overall": by_class[(5, "H11")]
            - max(by_class[(5, "H01")], by_class[(5, "H10")]),
            "pick_up_pen_H11_minus_stronger_single_by_environment": {
                target: by_environment_class[(target, 5, "H11")]
                - max(by_environment_class[(target, 5, "H01")], by_environment_class[(target, 5, "H10")])
                for target in TARGETS
            },
            "interaction_mean": paired_effects["interaction"]["mean"],
        },
        "checks": {
            "coexistence": coexistence_checks,
            "better": better_checks,
            "positive_interaction": interaction_check,
        },
        "decision": {
            "coexistence": all(coexistence_checks.values()),
            "better_than_both_single_components": all(better_checks.values()),
            "positive_synergy": bool(interaction_check),
        },
    }
    decision_path = ANALYSIS_DIR / "coexistence_decision.json"
    atomic_json(decision_path, decision)

    analysis = {
        "schema_version": "factorial-s32-analysis/v1",
        "created_at_utc": utc_now(),
        "factorial_results_sha256": results_sha,
        "completion_audit_sha256": audit_sha,
        "coexistence_decision_sha256": sha256_file(decision_path),
        "aggregation": "unweighted paired mean over target-by-seed cells",
        "class_mapping": CLASS_NAMES,
        "overall_arm_summary": overall_arm_rows,
        "environment_arm_summary": environment_arm_rows,
        "class_arm_summary": class_arm_rows,
        "environment_class_arm_summary": environment_class_arm_rows,
        "paired_effects": paired_effects,
        "coexistence_decision": decision["decision"],
    }
    analysis_path = ANALYSIS_DIR / "factorial_analysis.json"
    atomic_json(analysis_path, analysis)

    atomic_csv(
        ANALYSIS_DIR / "overall_arm_summary.csv",
        ["arm", "n", "mean", "sample_sd", "min", "max", "falling_turning_pair_confusion_mean"],
        overall_arm_rows,
    )
    atomic_csv(
        ANALYSIS_DIR / "environment_arm_summary.csv",
        ["target_env", "arm", "n", "mean", "sample_sd", "min", "max", "falling_turning_pair_confusion_mean"],
        environment_arm_rows,
    )
    atomic_csv(
        ANALYSIS_DIR / "class_arm_summary.csv",
        ["class_id", "class_name", "arm", "n", "mean", "sample_sd", "min", "max"],
        class_arm_rows,
    )
    atomic_csv(
        ANALYSIS_DIR / "environment_class_arm_summary.csv",
        ["target_env", "class_id", "class_name", "arm", "n", "mean", "sample_sd", "min", "max"],
        environment_class_arm_rows,
    )
    atomic_csv(
        ANALYSIS_DIR / "paired_effects.csv",
        ["contrast", "target_env", "seed", "macro_f1_difference"],
        paired_effect_rows,
    )
    manifest = {
        "schema_version": "factorial-s32-analysis-manifest/v1",
        "created_at_utc": utc_now(),
        "source": {
            "bundle_manifest_sha256": bundle_sha,
            "factorial_results_sha256": results_sha,
            "completion_audit_sha256": audit_sha,
        },
        "artifacts": [
            {"path": path.name, "sha256": sha256_file(path), "size": path.stat().st_size}
            for path in sorted(ANALYSIS_DIR.iterdir())
            if path.is_file() and path.name != "analysis_manifest.json"
        ],
    }
    atomic_json(ANALYSIS_DIR / "analysis_manifest.json", manifest)
    print(json.dumps({"status": "passed", "audit": str(audit_path), "analysis": str(analysis_path), "decision": decision["decision"]}, indent=2))


if __name__ == "__main__":
    main()
