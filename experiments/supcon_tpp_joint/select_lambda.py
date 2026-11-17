"""Select one source-only SupCon lambda per outer fold from 18 screen results."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

try:  # Support module and direct-script execution.
    from . import protocol, screen_health
except ImportError:  # pragma: no cover
    import protocol  # type: ignore
    import screen_health  # type: ignore


class SelectionValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SelectionArtifact:
    target_env: str
    selected_lambda: float
    record: Mapping[str, object]
    path: Path
    file_sha256: str


def load_json_object(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SelectionValidationError(f"Expected a JSON object: {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _require(payload: Mapping[str, object], key: str, expected: object, context: str) -> None:
    actual = payload.get(key)
    if actual != expected:
        raise SelectionValidationError(f"{context}.{key} must be {expected!r}; received {actual!r}")


def validate_screen_registry_header(registry: Mapping[str, object]) -> List[Mapping[str, object]]:
    _require(registry, "schema_version", protocol.REGISTRY_SCHEMA, "registry")
    _require(registry, "protocol_name", protocol.PROTOCOL_NAME, "registry")
    _require(registry, "protocol_version", protocol.PROTOCOL_VERSION, "registry")
    _require(registry, "protocol_lock_sha256", protocol.protocol_lock_sha256(), "registry")
    _require(registry, "stage", "screen", "registry")
    _require(registry, "expected_job_count", protocol.BUDGETS["screen_runs"], "registry")
    jobs = registry.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != protocol.BUDGETS["screen_runs"]:
        raise SelectionValidationError("Screen registry must contain exactly 18 jobs.")
    if not all(isinstance(job, Mapping) for job in jobs):
        raise SelectionValidationError("Every screen registry job must be an object.")
    job_ids = [str(job.get("job_id")) for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise SelectionValidationError("Screen registry job IDs must be unique.")
    return jobs


def _validate_metric_row(result: Mapping[str, object], context: str) -> Mapping[str, object]:
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise SelectionValidationError(f"{context}.metrics must be an object.")
    for key in ("finite_training", "converged", "gradient_exploded"):
        if not isinstance(metrics.get(key), bool):
            raise SelectionValidationError(f"{context}.metrics.{key} must be boolean.")
    for key in (
        "best_source_val_macro_f1",
        "best_source_val_sit_stand_f1",
        "positive_anchor_fraction",
    ):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
            raise SelectionValidationError(f"{context}.metrics.{key} must be in [0,1].")
    try:
        screen_health.validate_screen_metrics(
            metrics,
            expected_epoch_count=protocol.EPOCHS,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SelectionValidationError(f"{context}.metrics health evidence is invalid: {exc}") from exc
    return metrics


def validate_screen_results(
    results_payload: Mapping[str, object],
    registry: Mapping[str, object],
    *,
    registry_file_sha256: str,
) -> Dict[str, Mapping[str, object]]:
    jobs = validate_screen_registry_header(registry)
    _require(results_payload, "schema_version", protocol.SCREEN_RESULTS_SCHEMA, "results")
    _require(results_payload, "protocol_name", protocol.PROTOCOL_NAME, "results")
    _require(results_payload, "protocol_version", protocol.PROTOCOL_VERSION, "results")
    _require(results_payload, "protocol_lock_sha256", protocol.protocol_lock_sha256(), "results")
    _require(results_payload, "screen_registry_file_sha256", registry_file_sha256, "results")
    raw_results = results_payload.get("results")
    if not isinstance(raw_results, list) or len(raw_results) != len(jobs):
        raise SelectionValidationError("Screen results must contain one row for every registry job.")
    if not all(isinstance(row, Mapping) for row in raw_results):
        raise SelectionValidationError("Every screen result row must be an object.")

    registry_by_id = {str(job["job_id"]): job for job in jobs}
    result_by_id: Dict[str, Mapping[str, object]] = {}
    for row in raw_results:
        job_id = str(row.get("job_id"))
        if job_id in result_by_id:
            raise SelectionValidationError(f"Duplicate screen result: {job_id}")
        if job_id not in registry_by_id:
            raise SelectionValidationError(f"Unknown screen result job: {job_id}")
        job = registry_by_id[job_id]
        for key in ("target_env", "arm", "screen_seed", "lambda_supcon", "config_content_sha256"):
            if row.get(key) != job.get(key):
                raise SelectionValidationError(
                    f"Result/registry mismatch for {job_id}.{key}: {row.get(key)!r} != {job.get(key)!r}"
                )
        _validate_metric_row(row, f"results[{job_id}]")
        result_by_id[job_id] = row
    if set(result_by_id) != set(registry_by_id):
        raise SelectionValidationError("Screen results do not cover the exact registry job set.")
    return result_by_id


def _healthy(metrics: Mapping[str, object]) -> bool:
    return (
        metrics["finite_training"] is True
        and metrics["converged"] is True
        and metrics["gradient_exploded"] is False
    )


def _row_for(
    jobs: Sequence[Mapping[str, object]],
    results: Mapping[str, Mapping[str, object]],
    *,
    target_env: str,
    arm: str,
    lambda_supcon: float,
) -> Mapping[str, object]:
    matches = [
        job
        for job in jobs
        if job.get("target_env") == target_env
        and job.get("arm") == arm
        and float(job.get("lambda_supcon")) == lambda_supcon
    ]
    if len(matches) != 1:
        raise SelectionValidationError(
            f"Expected one screen job for {target_env}/{arm}/lambda={lambda_supcon}; found {len(matches)}."
        )
    return results[str(matches[0]["job_id"])]


def select_target_lambda(
    target_env: str,
    jobs: Sequence[Mapping[str, object]],
    results: Mapping[str, Mapping[str, object]],
) -> dict:
    """Apply hard gates, S(lambda), and the 0.5 pp smallest-lambda rule."""

    y00 = _row_for(jobs, results, target_env=target_env, arm="Y00", lambda_supcon=0.0)
    y10 = _row_for(jobs, results, target_env=target_env, arm="Y10", lambda_supcon=0.0)
    y00_metrics = y00["metrics"]
    y10_metrics = y10["metrics"]
    assert isinstance(y00_metrics, Mapping) and isinstance(y10_metrics, Mapping)

    candidate_rows: List[dict] = []
    for value in protocol.LAMBDA_CANDIDATES:
        y01 = _row_for(jobs, results, target_env=target_env, arm="Y01", lambda_supcon=value)
        y11 = _row_for(jobs, results, target_env=target_env, arm="Y11", lambda_supcon=value)
        y01_metrics = y01["metrics"]
        y11_metrics = y11["metrics"]
        assert isinstance(y01_metrics, Mapping) and isinstance(y11_metrics, Mapping)

        failures: List[str] = []
        for label, metrics in (("Y00", y00_metrics), ("Y10", y10_metrics), ("Y01", y01_metrics), ("Y11", y11_metrics)):
            if not _healthy(metrics):
                failures.append(f"{label}_training_health")
        for label, metrics in (("Y01", y01_metrics), ("Y11", y11_metrics)):
            if float(metrics["positive_anchor_fraction"]) < protocol.SCREEN_THRESHOLDS["minimum_positive_anchor_fraction"]:
                failures.append(f"{label}_positive_anchor_fraction")

        combo_macro_delta = float(y11_metrics["best_source_val_macro_f1"]) - float(
            y10_metrics["best_source_val_macro_f1"]
        )
        combo_c3_delta = float(y11_metrics["best_source_val_sit_stand_f1"]) - float(
            y10_metrics["best_source_val_sit_stand_f1"]
        )
        if combo_macro_delta < -protocol.SCREEN_THRESHOLDS["maximum_combo_macro_f1_drop_vs_tpp"]:
            failures.append("Y11_macro_f1_drop_vs_Y10")
        if combo_c3_delta < -protocol.SCREEN_THRESHOLDS["maximum_combo_sit_stand_f1_drop_vs_tpp"]:
            failures.append("Y11_sit_stand_f1_drop_vs_Y10")

        score = 0.5 * (
            float(y01_metrics["best_source_val_macro_f1"])
            - float(y00_metrics["best_source_val_macro_f1"])
            + combo_macro_delta
        )
        candidate_rows.append(
            {
                "lambda_supcon": value,
                "passed_hard_gates": not failures,
                "hard_gate_failures": failures,
                "score": score,
                "components": {
                    "Y01_minus_Y00_macro_f1": float(y01_metrics["best_source_val_macro_f1"])
                    - float(y00_metrics["best_source_val_macro_f1"]),
                    "Y11_minus_Y10_macro_f1": combo_macro_delta,
                    "Y11_minus_Y10_sit_stand_f1": combo_c3_delta,
                },
                "input_job_ids": [str(y00["job_id"]), str(y10["job_id"]), str(y01["job_id"]), str(y11["job_id"])],
            }
        )

    passed = [row for row in candidate_rows if row["passed_hard_gates"]]
    selected_lambda: Optional[float] = None
    if passed:
        best_score = max(float(row["score"]) for row in passed)
        eligible = [
            row
            for row in passed
            if float(row["score"])
            >= best_score - protocol.SCREEN_THRESHOLDS["smallest_lambda_score_tolerance"] - 1e-12
        ]
        selected_lambda = min(float(row["lambda_supcon"]) for row in eligible)

    return {
        "status": "selected" if selected_lambda is not None else "failed",
        "target_env": target_env,
        "source_envs": list(protocol.source_environments(target_env)),
        "screen_seed": protocol.SCREEN_SEEDS[target_env],
        "selected_lambda": selected_lambda,
        "candidate_evaluations": candidate_rows,
    }


def build_selection_records(registry_path: Path, results_path: Path) -> Dict[str, dict]:
    registry = load_json_object(registry_path)
    results_payload = load_json_object(results_path)
    registry_sha = protocol.file_sha256(registry_path)
    results_sha = protocol.file_sha256(results_path)
    jobs = validate_screen_registry_header(registry)
    result_by_id = validate_screen_results(
        results_payload,
        registry,
        registry_file_sha256=registry_sha,
    )

    records: Dict[str, dict] = {}
    for target_env in protocol.ENVIRONMENTS:
        decision = select_target_lambda(target_env, jobs, result_by_id)
        records[target_env] = {
            "schema_version": protocol.SELECTION_RECORD_SCHEMA,
            "protocol_name": protocol.PROTOCOL_NAME,
            "protocol_version": protocol.PROTOCOL_VERSION,
            "protocol_lock_sha256": protocol.protocol_lock_sha256(),
            **decision,
            "lambda_candidates": list(protocol.LAMBDA_CANDIDATES),
            "screen_thresholds": dict(protocol.SCREEN_THRESHOLDS),
            "screen_registry": {
                "path": str(registry_path.resolve()),
                "file_sha256": registry_sha,
                "content_sha256": protocol.canonical_sha256(registry),
            },
            "screen_results": {
                "path": str(results_path.resolve()),
                "file_sha256": results_sha,
                "content_sha256": protocol.canonical_sha256(results_payload),
            },
        }
    return records


def validate_selection_record(record: Mapping[str, object], expected_target: Optional[str] = None) -> None:
    _require(record, "schema_version", protocol.SELECTION_RECORD_SCHEMA, "selection")
    _require(record, "protocol_name", protocol.PROTOCOL_NAME, "selection")
    _require(record, "protocol_version", protocol.PROTOCOL_VERSION, "selection")
    _require(record, "protocol_lock_sha256", protocol.protocol_lock_sha256(), "selection")
    _require(record, "status", "selected", "selection")
    target_env = record.get("target_env")
    if target_env not in protocol.ENVIRONMENTS or (expected_target is not None and target_env != expected_target):
        raise SelectionValidationError(f"Invalid selection target: {target_env}")
    _require(record, "source_envs", list(protocol.source_environments(str(target_env))), "selection")
    _require(record, "screen_seed", protocol.SCREEN_SEEDS[str(target_env)], "selection")
    _require(record, "lambda_candidates", list(protocol.LAMBDA_CANDIDATES), "selection")
    _require(record, "screen_thresholds", dict(protocol.SCREEN_THRESHOLDS), "selection")
    selected = record.get("selected_lambda")
    if selected not in protocol.LAMBDA_CANDIDATES:
        raise SelectionValidationError(f"Invalid selected lambda: {selected}")
    candidates = record.get("candidate_evaluations")
    if not isinstance(candidates, list) or len(candidates) != len(protocol.LAMBDA_CANDIDATES):
        raise SelectionValidationError("Selection record candidate_evaluations are incomplete.")
    candidate_lambdas: List[float] = []
    passed_rows: List[Mapping[str, object]] = []
    for index, row in enumerate(candidates):
        if not isinstance(row, Mapping):
            raise SelectionValidationError(f"candidate_evaluations[{index}] must be an object.")
        value = row.get("lambda_supcon")
        score = row.get("score")
        passed = row.get("passed_hard_gates")
        failures = row.get("hard_gate_failures")
        if value not in protocol.LAMBDA_CANDIDATES:
            raise SelectionValidationError(f"Invalid candidate lambda: {value!r}")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)):
            raise SelectionValidationError(f"Invalid candidate score for lambda={value!r}.")
        if not isinstance(passed, bool) or not isinstance(failures, list):
            raise SelectionValidationError(f"Invalid hard-gate fields for lambda={value!r}.")
        if passed == bool(failures):
            raise SelectionValidationError(f"Hard-gate status/failures disagree for lambda={value!r}.")
        candidate_lambdas.append(float(value))
        if passed:
            passed_rows.append(row)
    if sorted(candidate_lambdas) != sorted(protocol.LAMBDA_CANDIDATES):
        raise SelectionValidationError("Selection record must evaluate each lambda candidate exactly once.")
    if not passed_rows:
        raise SelectionValidationError("A selected record must contain at least one hard-gate passing candidate.")
    best_score = max(float(row["score"]) for row in passed_rows)
    tolerance = protocol.SCREEN_THRESHOLDS["smallest_lambda_score_tolerance"]
    eligible = [
        row for row in passed_rows if float(row["score"]) >= best_score - tolerance - 1e-12
    ]
    expected_selected = min(float(row["lambda_supcon"]) for row in eligible)
    if float(selected) != expected_selected:
        raise SelectionValidationError(
            f"Selected lambda violates the smallest-lambda-within-tolerance rule: "
            f"expected {expected_selected}, received {selected}."
        )
    selected_rows = [row for row in candidates if isinstance(row, Mapping) and row.get("lambda_supcon") == selected]
    if len(selected_rows) != 1 or selected_rows[0].get("passed_hard_gates") is not True:
        raise SelectionValidationError("Selected lambda did not pass the recorded hard gates.")
    for artifact_name in ("screen_registry", "screen_results"):
        artifact = record.get(artifact_name)
        if not isinstance(artifact, Mapping):
            raise SelectionValidationError(f"selection.{artifact_name} must be an object.")
        for hash_name in ("file_sha256", "content_sha256"):
            digest = artifact.get(hash_name)
            if not isinstance(digest, str) or len(digest) != 64:
                raise SelectionValidationError(f"selection.{artifact_name}.{hash_name} must be SHA256.")


def load_selection_record_files(paths: Sequence[Path]) -> Dict[str, SelectionArtifact]:
    if len(paths) != len(protocol.ENVIRONMENTS):
        raise SelectionValidationError("Exactly three selection record paths are required.")
    artifacts: Dict[str, SelectionArtifact] = {}
    for path in paths:
        if not path.is_file():
            raise SelectionValidationError(f"Selection record does not exist: {path}")
        record = load_json_object(path)
        validate_selection_record(record)
        target_env = str(record["target_env"])
        bound_paths: Dict[str, Path] = {}
        for artifact_name in ("screen_registry", "screen_results"):
            artifact = record[artifact_name]
            assert isinstance(artifact, Mapping)
            bound_path = Path(str(artifact["path"]))
            if not bound_path.is_file():
                raise SelectionValidationError(
                    f"Hash-bound {artifact_name} does not exist for {target_env}: {bound_path}"
                )
            payload = load_json_object(bound_path)
            if protocol.file_sha256(bound_path) != artifact["file_sha256"]:
                raise SelectionValidationError(f"Hash-bound {artifact_name} file changed for {target_env}.")
            if protocol.canonical_sha256(payload) != artifact["content_sha256"]:
                raise SelectionValidationError(f"Hash-bound {artifact_name} content changed for {target_env}.")
            bound_paths[artifact_name] = bound_path
        rebuilt = build_selection_records(
            bound_paths["screen_registry"],
            bound_paths["screen_results"],
        )[target_env]
        if record != rebuilt:
            raise SelectionValidationError(
                f"Selection record for {target_env} does not reproduce from its hash-bound 18 screen results."
            )
        if target_env in artifacts:
            raise SelectionValidationError(f"Duplicate selection record for {target_env}.")
        artifacts[target_env] = SelectionArtifact(
            target_env=target_env,
            selected_lambda=float(record["selected_lambda"]),
            record=record,
            path=path.resolve(),
            file_sha256=protocol.file_sha256(path),
        )
    if set(artifacts) != set(protocol.ENVIRONMENTS):
        raise SelectionValidationError("Selection records must cover exactly E1/E2/E3.")
    return artifacts


def write_selection_records(output_dir: Path, records: Mapping[str, Mapping[str, object]], *, clean: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {f"selection_{environment}.json" for environment in protocol.ENVIRONMENTS} | {
        "selection_summary.json"
    }
    existing = {path.name for path in output_dir.iterdir()}
    if existing and not clean:
        raise SelectionValidationError(
            f"Selection output is not empty: {output_dir}. Use --clean to replace known selection files."
        )
    unknown = existing - expected_names
    if clean and unknown:
        raise SelectionValidationError(f"Refusing to clean unknown selection artifacts: {sorted(unknown)}")
    if clean:
        for name in existing:
            (output_dir / name).unlink()
    for target_env, record in records.items():
        write_json(output_dir / f"selection_{target_env}.json", record)
    summary = {
        "schema_version": protocol.SELECTION_RECORD_SCHEMA,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "statuses": {environment: records[environment]["status"] for environment in protocol.ENVIRONMENTS},
        "selected_lambdas": {
            environment: records[environment]["selected_lambda"] for environment in protocol.ENVIRONMENTS
        },
    }
    write_json(output_dir / "selection_summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-registry", type=Path, required=True)
    parser.add_argument("--screen-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_selection_records(args.screen_registry, args.screen_results)
    write_selection_records(args.output_dir, records, clean=args.clean)
    failed = [environment for environment, record in records.items() if record["status"] != "selected"]
    print(
        json.dumps(
            {
                "records": {environment: records[environment]["selected_lambda"] for environment in records},
                "failed_targets": failed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if failed:
        raise SystemExit(f"Lambda selection failed closed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
