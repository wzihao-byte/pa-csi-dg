from __future__ import annotations

import csv
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FACTORIAL_RESULTS = REPO_ROOT / "outputs_supcon_tpp_legacy_reproduction" / "factorial_s32" / "factorial_results.json"
H10_RESULTS = REPO_ROOT / "outputs_tpp_level8_ablation" / "l1248_h10_v1" / "level8_results.json"
H11_RESULTS = REPO_ROOT / "outputs_tpp_level8_ablation" / "l1248_v1" / "level8_results.json"
OUTPUT = REPO_ROOT / "outputs_tpp_level8_ablation" / "l1248_factorial_s32_v1" / "analysis"
TARGETS = ("E1", "E2", "E3")
SEEDS = (42, 52, 62)
ARMS = ("H00", "H01", "H10_L1248", "H11_L1248")
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
    result: dict[str, Any] = summary(array)
    rng = np.random.default_rng(20260807)
    indices = rng.integers(0, len(array), size=(50_000, len(array)))
    boot = array[indices].mean(axis=1)
    result["bootstrap_95_ci"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    observed = abs(float(array.mean()))
    sign_means = [
        abs(float(np.mean(array * np.asarray(signs, dtype=np.float64))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(array))
    ]
    result["exact_sign_flip_two_sided_p"] = float(
        np.mean(np.asarray(sign_means) >= observed - 1e-15)
    )
    return result


def main() -> None:
    historical = read_json(FACTORIAL_RESULTS)
    h10 = read_json(H10_RESULTS)
    h11 = read_json(H11_RESULTS)
    require(historical.get("status") == "complete" and historical.get("job_count") == 36, "Historical factorial incomplete")
    require(h10.get("status") == "complete" and h10.get("job_count") == 9, "H10 L1248 incomplete")
    require(h11.get("status") == "complete" and h11.get("job_count") == 9, "H11 L1248 incomplete")
    require(all(row.get("status") == "passed" for row in h10["paired_barriers"]), "H10 pairing failed")
    require(all(row.get("status") == "passed" for row in h11["paired_barriers"]), "H11 pairing failed")

    old = {(row["target_env"], int(row["seed"]), row["arm"]): row for row in historical["rows"]}
    new_h10 = {(row["target_env"], int(row["seed"])): row for row in h10["rows"]}
    new_h11 = {(row["target_env"], int(row["seed"])): row for row in h11["rows"]}
    old_barriers = {(row["target"], int(row["seed"])): row for row in historical["quartet_barriers"]}
    h10_barriers = {(row["target"], int(row["seed"])): row for row in h10["paired_barriers"]}
    h11_barriers = {(row["target"], int(row["seed"])): row for row in h11["paired_barriers"]}
    keys = [(target, seed) for target in TARGETS for seed in SEEDS]
    require(list(new_h10) == keys and list(new_h11) == keys, "Candidate row ordering")

    values: dict[tuple[str, int, str], float] = {}
    class_values: dict[tuple[str, int, str, int], float] = {}
    audit_rows = []
    for key in keys:
        quartet = old_barriers[key]
        b10 = h10_barriers[key]
        b11 = h11_barriers[key]
        require(quartet["sampler_schedule_sha256"] == b10["sampler_schedule_sha256"] == b11["sampler_schedule_sha256"], f"Sampler mismatch: {key}")
        require(quartet["split_hashes"] == b10["split_hashes"] == b11["split_hashes"], f"Split mismatch: {key}")
        rows = {
            "H00": old[key + ("H00",)],
            "H01": old[key + ("H01",)],
            "H10_L1248": new_h10[key],
            "H11_L1248": new_h11[key],
        }
        for arm, row in rows.items():
            values[key + (arm,)] = float(row["test_metrics"]["f1_macro"])
            per_class = {int(item["class_id"]): float(item["f1"]) for item in row["per_class_metrics"]}
            require(set(per_class) == set(CLASS_NAMES), f"Class metrics missing: {key}/{arm}")
            for class_id, value in per_class.items():
                class_values[key + (arm, class_id)] = value
        audit_rows.append({
            "target_env": key[0],
            "seed": key[1],
            "sampler_schedule_sha256": quartet["sampler_schedule_sha256"],
            "split_hashes": quartet["split_hashes"],
            "h10_l1248_attempt": new_h10[key]["attempt_dir"],
            "h11_l1248_attempt": new_h11[key]["attempt_dir"],
        })

    overall = [{"arm": arm, **summary(values[key + (arm,)] for key in keys)} for arm in ARMS]
    environment = [
        {
            "target_env": target,
            "arm": arm,
            **summary(values[(target, seed, arm)] for seed in SEEDS),
        }
        for target in TARGETS
        for arm in ARMS
    ]
    classes = [
        {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "arm": arm,
            **summary(class_values[key + (arm, class_id)] for key in keys),
        }
        for class_id in CLASS_NAMES
        for arm in ARMS
    ]
    effects = {
        "supcon_on_global": paired_summary(values[key + ("H01",)] - values[key + ("H00",)] for key in keys),
        "supcon_on_l1248_tpp": paired_summary(values[key + ("H11_L1248",)] - values[key + ("H10_L1248",)] for key in keys),
        "l1248_tpp_without_supcon": paired_summary(values[key + ("H10_L1248",)] - values[key + ("H00",)] for key in keys),
        "l1248_tpp_with_supcon": paired_summary(values[key + ("H11_L1248",)] - values[key + ("H01",)] for key in keys),
        "l1248_interaction": paired_summary(
            values[key + ("H11_L1248",)] - values[key + ("H10_L1248",)]
            - values[key + ("H01",)] + values[key + ("H00",)]
            for key in keys
        ),
        "level8_without_supcon": paired_summary(
            values[key + ("H10_L1248",)] - float(old[key + ("H10",)]["test_metrics"]["f1_macro"])
            for key in keys
        ),
        "level8_with_supcon": paired_summary(
            values[key + ("H11_L1248",)] - float(old[key + ("H11",)]["test_metrics"]["f1_macro"])
            for key in keys
        ),
    }
    overall_lookup = {row["arm"]: float(row["mean"]) for row in overall}
    decision = {
        "h11_l1248_vs_h01_at_least_minus_1pp": overall_lookup["H11_L1248"] - overall_lookup["H01"] >= -0.01,
        "h11_l1248_vs_h10_l1248_at_least_minus_1pp": overall_lookup["H11_L1248"] - overall_lookup["H10_L1248"] >= -0.01,
        "h11_l1248_better_than_h01_by_1pp": overall_lookup["H11_L1248"] - overall_lookup["H01"] >= 0.01,
        "h11_l1248_better_than_h10_l1248_by_1pp": overall_lookup["H11_L1248"] - overall_lookup["H10_L1248"] >= 0.01,
        "positive_interaction_at_least_1pp": float(effects["l1248_interaction"]["mean"]) >= 0.01,
    }
    analysis = {
        "schema_version": "factorial-s32-l1248-analysis/v1",
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "aggregation": "unweighted paired mean over target-by-seed cells",
        "arms": {
            "H00": "global pooling, SupCon off (sealed Factorial-S32 reference)",
            "H01": "global pooling, SupCon on (sealed Factorial-S32 reference)",
            "H10_L1248": "TPP [1,2,4,8], SupCon off",
            "H11_L1248": "TPP [1,2,4,8], SupCon on",
        },
        "parameters": {"H00": 2_788_918, "H01": 2_788_918, "H10_L1248": 4_786_294, "H11_L1248": 4_786_294},
        "overall_arm_summary": overall,
        "environment_arm_summary": environment,
        "class_arm_summary": classes,
        "paired_effects": effects,
        "decision_checks": decision,
        "audit_rows": audit_rows,
        "no_target_metric_tuning": True,
    }
    atomic_json(OUTPUT / "factorial_l1248_analysis.json", analysis)
    atomic_csv(OUTPUT / "overall_arm_summary.csv", ["arm", "n", "mean", "sample_sd", "min", "max"], overall)
    effect_rows = [{"effect": name, **value} for name, value in effects.items()]
    atomic_csv(
        OUTPUT / "paired_effects.csv",
        ["effect", "n", "mean", "sample_sd", "min", "max", "bootstrap_95_ci", "exact_sign_flip_two_sided_p"],
        effect_rows,
    )
    print(json.dumps({
        "status": "passed",
        "analysis": str((OUTPUT / "factorial_l1248_analysis.json").resolve()),
        "overall": overall_lookup,
        "level8_without_supcon": effects["level8_without_supcon"],
        "level8_with_supcon": effects["level8_with_supcon"],
        "supcon_on_l1248_tpp": effects["supcon_on_l1248_tpp"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

