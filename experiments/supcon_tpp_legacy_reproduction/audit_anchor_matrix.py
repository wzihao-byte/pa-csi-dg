from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

import protocol
import run_anchors


SCHEMA_VERSION = "legacy-anchor-matrix-audit/v1"
SEEDS = (42, 52, 62)
OUTPUT_PATH = protocol.OUTPUT_ROOT / "audit" / "anchor_matrix_completion.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise run_anchors.RunnerFailure(f"Expected JSON object: {path}")
    return payload


def unique_artifact(attempt: Path, name: str) -> Path:
    matches = sorted(attempt.rglob(name))
    if len(matches) != 1:
        raise run_anchors.RunnerFailure(
            f"Expected exactly one {name} below {attempt}, found {len(matches)}"
        )
    return matches[0]


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def supcon_advisory(cells: list[dict[str, Any]]) -> dict[str, Any]:
    tolerance = dict(protocol.REPRODUCTION_TOLERANCE)
    comparisons: list[dict[str, Any]] = []
    for cell in cells:
        if cell["arm"] != "anchor_supcon":
            continue
        key = (cell["target"], cell["seed"])
        reference = protocol.SUPCON_REFERENCE_CELLS[key]
        accuracy_difference = cell["accuracy"] - float(reference["accuracy"])
        f1_difference = cell["f1_macro"] - float(reference["f1_macro"])
        comparisons.append(
            {
                "target": cell["target"],
                "seed": cell["seed"],
                "observed_accuracy": cell["accuracy"],
                "reference_accuracy": float(reference["accuracy"]),
                "accuracy_difference": accuracy_difference,
                "accuracy_within_tolerance": abs(accuracy_difference)
                <= float(tolerance["per_run_accuracy_abs"]),
                "observed_f1_macro": cell["f1_macro"],
                "reference_f1_macro": float(reference["f1_macro"]),
                "f1_macro_difference": f1_difference,
                "f1_macro_within_tolerance": abs(f1_difference)
                <= float(tolerance["per_run_macro_f1_abs"]),
            }
        )

    target_means: list[dict[str, Any]] = []
    for target in protocol.TARGETS:
        observed = mean(
            item["observed_f1_macro"] for item in comparisons if item["target"] == target
        )
        reference = mean(
            item["reference_f1_macro"] for item in comparisons if item["target"] == target
        )
        difference = observed - reference
        target_means.append(
            {
                "target": target,
                "observed_f1_macro": observed,
                "reference_f1_macro": reference,
                "difference": difference,
                "within_tolerance": abs(difference)
                <= float(tolerance["per_target_mean_macro_f1_abs"]),
            }
        )
    observed_overall = mean(item["observed_f1_macro"] for item in comparisons)
    reference_overall = mean(item["reference_f1_macro"] for item in comparisons)
    overall_difference = observed_overall - reference_overall
    passed = (
        all(item["accuracy_within_tolerance"] for item in comparisons)
        and all(item["f1_macro_within_tolerance"] for item in comparisons)
        and all(item["within_tolerance"] for item in target_means)
        and abs(overall_difference) <= float(tolerance["overall_mean_macro_f1_abs"])
    )
    return {
        "gate_mode": "advisory_only_by_user_authorization_2026-07-23",
        "passed": passed,
        "blocks_factorial": False,
        "tolerance": tolerance,
        "cells": comparisons,
        "target_means": target_means,
        "overall": {
            "observed_f1_macro": observed_overall,
            "reference_f1_macro": reference_overall,
            "difference": overall_difference,
            "within_tolerance": abs(overall_difference)
            <= float(tolerance["overall_mean_macro_f1_abs"]),
        },
    }


def main() -> None:
    python_executable = Path(sys.executable).resolve()
    clean_worktree = protocol.DEFAULT_CLEAN_WORKTREE.resolve()
    tpp_snapshot_root = protocol.REPO_ROOT.resolve()
    generated_root = protocol.GENERATED_ROOT.resolve()
    preflight_audit, bundle = run_anchors.perform_preflight(
        generated_root, clean_worktree, tpp_snapshot_root, python_executable
    )
    arms = run_anchors.load_arm_specs(bundle, run_anchors.ARM_IDS)
    records: list[dict[str, Any]] = []
    metric_cells: list[dict[str, Any]] = []
    for arm in arms:
        for target in protocol.TARGETS:
            for seed in SEEDS:
                job = run_anchors.Job(arm=arm, target=target, seed=seed)
                scan = run_anchors.scan_attempts(
                    job,
                    clean_worktree,
                    tpp_snapshot_root,
                    python_executable,
                    bundle["lock"],
                )
                if scan.action != "skip" or scan.successful_attempt is None:
                    raise run_anchors.RunnerFailure(f"Missing healthy success: {job.job_id}")
                attempt = scan.successful_attempt
                completion = read_json(attempt / "attempt_completed.json")
                metrics = read_json(unique_artifact(attempt, "metrics.json"))
                test_metrics = metrics["test_metrics"]
                records.append(
                    {
                        "job_id": job.job_id,
                        "arm": arm.config_id,
                        "target": target,
                        "seed": seed,
                        "attempt_dir": str(attempt),
                        "completion_manifest_sha256": sha256_file(
                            attempt / "attempt_completed.json"
                        ),
                        "attempt_seal_sha256": sha256_file(attempt / "attempt_seal.json"),
                        "health_sha256": completion["health_sha256"],
                        "epochs_recorded": len(metrics["history"]),
                        "test_size": int(metrics["test_size"]),
                    }
                )
                metric_cells.append(
                    {
                        "arm": arm.config_id,
                        "target": target,
                        "seed": seed,
                        "accuracy": float(test_metrics["accuracy"]),
                        "f1_macro": float(test_metrics["f1_macro"]),
                    }
                )

    arm_aggregates: list[dict[str, Any]] = []
    for arm in run_anchors.ARM_IDS:
        arm_cells = [item for item in metric_cells if item["arm"] == arm]
        arm_aggregates.append(
            {
                "arm": arm,
                "runs": len(arm_cells),
                "accuracy_mean": mean(item["accuracy"] for item in arm_cells),
                "f1_macro_mean": mean(item["f1_macro"] for item in arm_cells),
                "per_target_f1_macro_mean": {
                    target: mean(
                        item["f1_macro"] for item in arm_cells if item["target"] == target
                    )
                    for target in protocol.TARGETS
                },
            }
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "created_at_utc": utc_now(),
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "audit_scope": "all four anchor arms x E1/E2/E3 x seeds 42/52/62",
        "integrity_gate": {
            "passed": True,
            "jobs_expected": 36,
            "jobs_validated": len(records),
            "failed_attempts_selected": 0,
            "checks": [
                "bundle and execution-lock hashes",
                "clean legacy worktree and TPP snapshot provenance",
                "attempt_completed/attempt_seal binding",
                "all artifact member hashes and sizes",
                "200-epoch finite metric history",
                "split counts and source/target environment order",
                "6x6 confusion matrix with support 3000",
                "derived target metrics",
                "checkpoint strict load and parameter count",
            ],
        },
        "preflight_sha256": run_anchors.payload_sha256(preflight_audit),
        "records": records,
        "metric_cells": metric_cells,
        "arm_aggregates": arm_aggregates,
        "supcon_historical_performance_gate": supcon_advisory(metric_cells),
        "authorization": {
            "performance_gate_is_report_only": True,
            "source": str(
                protocol.OUTPUT_ROOT
                / "audit"
                / "user_override_continue_after_performance_gate_20260723.md"
            ),
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        "execution_lock_sha256": sha256_file(generated_root / "execution_lock.json"),
    }
    if len(records) != 36:
        raise run_anchors.RunnerFailure(f"Expected 36 healthy records, got {len(records)}")
    atomic_write_json(OUTPUT_PATH, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(OUTPUT_PATH.resolve()),
                "jobs_validated": len(records),
                "supcon_performance_gate_passed": report["supcon_historical_performance_gate"]["passed"],
                "performance_gate_blocks_factorial": False,
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
