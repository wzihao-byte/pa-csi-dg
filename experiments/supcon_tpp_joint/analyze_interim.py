"""Validate and analyze the locked 36-run formal-interim matrix.

The analyzer is intentionally the first component allowed to read the formal
target predictions *after* all 36 jobs have passed structural validation.  It
recomputes target metrics from prediction rows, fits the preregistered balanced
target-effect model (residual df=6), evaluates only the three registered
futility conditions, and atomically emits a deterministic stop/continue
artifact.

Decision artifacts bind absolute formal-bundle and formal-output roots plus
root-relative paths and SHA256 digests.  Copying the decision file is safe, but
moving either evidence root invalidates it and requires regeneration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scipy.stats import t as student_t

try:  # Support module and direct-script execution.
    from . import check_protocol, protocol
    from .runner_utils import fixed_label_metrics
except ImportError:  # pragma: no cover
    import check_protocol  # type: ignore
    import protocol  # type: ignore
    from runner_utils import fixed_label_metrics  # type: ignore


FORMAL_RESULTS_SCHEMA = "supcon-tpp-formal-results/v1"
ATTEMPT_MANIFEST_SCHEMA = "supcon-tpp-formal-attempt-manifest/v1"
INTERIM_ANALYSIS_METHOD = "balanced_target_effect_coding/v1"

FORMAL_RESULTS_TOP_KEYS = {
    "schema_version",
    "protocol_name",
    "protocol_version",
    "protocol_lock_sha256",
    "stage",
    "formal_registry_file_sha256",
    "formal_registry_content_sha256",
    "execution_lock_file_sha256",
    "run_registry_file_sha256",
    "completed_job_count",
    "results",
}
FORMAL_RESULT_ROW_KEYS = {
    "job_id",
    "stage",
    "target_env",
    "seed",
    "arm",
    "selected_lambda",
    "lambda_supcon",
    "config_content_sha256",
    "selection_record_file_sha256",
    "interim_decision_file_sha256",
    "attempt_manifest_file_sha256",
    "metrics_file_sha256",
    "target_metrics_file_sha256",
    "target_predictions_file_sha256",
    "best_model_file_sha256",
    "target_metrics",
    "best_source_val_f1_macro",
    "best_epoch",
}
RUN_REGISTRY_FIELDS = [
    "job_id",
    "stage",
    "target_env",
    "source_envs",
    "seed",
    "arm",
    "lambda_supcon",
    "selected_lambda",
    "selection_record_file_sha256",
    "interim_decision_file_sha256",
    "config_path",
    "config_content_sha256",
    "config_file_sha256",
    "formal_registry_file_sha256",
    "formal_registry_content_sha256",
    "protocol_lock_file_sha256",
    "bundle_manifest_file_sha256",
    "execution_lock_file_sha256",
    "status",
    "attempt",
    "output_dir",
    "command",
    "stdout_log",
    "stderr_log",
    "started_at_utc",
    "completed_at_utc",
    "elapsed_seconds",
    "return_code",
    "metrics_path",
    "metrics_file_sha256",
    "target_metrics_file_sha256",
    "target_predictions_file_sha256",
    "best_model_file_sha256",
    "split_manifest_file_sha256",
    "attempt_manifest_file_sha256",
    "error",
]
ATTEMPT_MANIFEST_KEYS = {
    "schema_version",
    "protocol_name",
    "protocol_version",
    "protocol_lock_sha256",
    "stage",
    "job_id",
    "target_env",
    "source_envs",
    "seed",
    "arm",
    "selected_lambda",
    "lambda_supcon",
    "config_path",
    "config_content_sha256",
    "config_file_sha256",
    "selection_record_file_sha256",
    "interim_decision_file_sha256",
    "formal_registry_file_sha256",
    "formal_registry_content_sha256",
    "execution_lock_file_sha256",
    "artifact_file_sha256",
    "target_metrics",
}
REQUIRED_ATTEMPT_ARTIFACTS = {
    "metrics.json",
    "target_metrics.json",
    "target_predictions.jsonl",
    "best_model.pt",
    "source_split_manifest.json",
    "target_evaluation_manifest.json",
    "split_manifest.json",
    "phase_input_audit.json",
    "target_phase_input_audit.json",
    "history.json",
    "run_invocation.json",
}
DECISION_TOP_KEYS = {
    "schema_version",
    "protocol_name",
    "protocol_version",
    "protocol_lock_sha256",
    "decision",
    "completed_interim_job_count",
    "formal_interim_registry_content_sha256",
    "selection_record_file_sha256",
    "evidence",
    "analysis",
}


class InterimAnalysisError(RuntimeError):
    """Raised when formal evidence or a decision artifact fails closed."""


@dataclass(frozen=True)
class StructuredJob:
    registry: Mapping[str, Any]
    result: Mapping[str, Any]
    run_row: Mapping[str, str]
    manifest: Mapping[str, Any]
    output_dir: Path
    manifest_path: Path


def _load_json_object(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise InterimAnalysisError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InterimAnalysisError(f"Expected a JSON object: {path}")
    return value


def _sha(value: Any, *, context: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise InterimAnalysisError(f"{context} must be a lowercase SHA256 digest.")
    return text


def _finite_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InterimAnalysisError(f"{context} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise InterimAnalysisError(f"{context} must be finite.")
    return result


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_relative(path: Path, root: Path, *, context: str) -> str:
    resolved = path.resolve()
    if not _inside(resolved, root):
        raise InterimAnalysisError(f"{context} escapes evidence root {root}: {resolved}")
    return resolved.relative_to(root).as_posix()


def _resolve_relative(root: Path, raw: Any, *, context: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise InterimAnalysisError(f"{context} must be a non-empty relative path.")
    relative = Path(raw)
    if relative.is_absolute():
        raise InterimAnalysisError(f"{context} must be root-relative, not absolute.")
    path = (root / relative).resolve()
    if not _inside(path, root):
        raise InterimAnalysisError(f"{context} escapes its evidence root.")
    return path


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], *, context: str) -> None:
    expected_set = set(expected)
    if set(value) != expected_set:
        raise InterimAnalysisError(
            f"{context} schema drifted; extra={sorted(set(value)-expected_set)}, "
            f"missing={sorted(expected_set-set(value))}."
        )


def _expected_cells() -> set[Tuple[str, int, str]]:
    return {
        (target, seed, arm)
        for target in protocol.ENVIRONMENTS
        for seed in protocol.formal_seeds(target, "formal_interim")
        for arm in protocol.ARMS
    }


def _matrix_by_job(rows: Sequence[Mapping[str, Any]], *, context: str) -> Dict[str, Mapping[str, Any]]:
    expected_cells = _expected_cells()
    by_job: Dict[str, Mapping[str, Any]] = {}
    cells: set[Tuple[str, int, str]] = set()
    for row in rows:
        job_id = str(row.get("job_id", ""))
        cell = (str(row.get("target_env", "")), int(row.get("seed", -1)), str(row.get("arm", "")))
        if not job_id or job_id in by_job or cell in cells:
            raise InterimAnalysisError(f"{context} has a duplicate/empty job or cell: {job_id}/{cell}")
        by_job[job_id] = row
        cells.add(cell)
    if cells != expected_cells or len(by_job) != protocol.BUDGETS["formal_interim_jobs"]:
        raise InterimAnalysisError(
            f"{context} is not the exact paired 3-target x 3-seed x 4-arm matrix."
        )
    return by_job


def _read_run_registry(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != RUN_REGISTRY_FIELDS:
                raise InterimAnalysisError("run_registry.csv exact schema drifted.")
            rows = [{key: str(row.get(key, "")) for key in RUN_REGISTRY_FIELDS} for row in reader]
    except OSError as exc:
        raise InterimAnalysisError(f"Cannot read run registry {path}: {exc}") from exc
    return rows


def _validate_header(value: Mapping[str, Any], *, stage: str, context: str) -> None:
    expected = {
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "stage": stage,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise InterimAnalysisError(f"{context}.{key} is not locked to {wanted!r}.")


def _validate_structural_evidence(
    formal_registry_path: Path,
    formal_results_path: Path,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Path,
    List[StructuredJob],
    Dict[str, str],
]:
    """Validate the full 36-job structure before opening any prediction file."""

    formal_registry_path = formal_registry_path.resolve()
    formal_bundle_root = formal_registry_path.parent
    try:
        checked = check_protocol.check_bundle(formal_bundle_root)
    except Exception as exc:
        raise InterimAnalysisError(f"Formal interim bundle failed complete validation: {exc}") from exc
    if checked.get("manifest", {}).get("generation_mode") != "formal_interim":
        raise InterimAnalysisError("Interim analyzer requires a formal_interim generated bundle.")

    registry = _load_json_object(formal_registry_path)
    if registry.get("stage") != "formal_interim" or registry.get("expected_job_count") != 36:
        raise InterimAnalysisError("Formal registry is not the locked 36-job interim registry.")
    raw_registry_jobs = registry.get("jobs")
    if not isinstance(raw_registry_jobs, list) or not all(isinstance(row, Mapping) for row in raw_registry_jobs):
        raise InterimAnalysisError("Formal registry jobs must be a list of objects.")
    registry_by_job = _matrix_by_job(raw_registry_jobs, context="formal registry")
    registry_file_hash = protocol.file_sha256(formal_registry_path)
    registry_content_hash = protocol.canonical_sha256(registry)

    formal_results_path = formal_results_path.resolve()
    formal_output_root = formal_results_path.parent
    results = _load_json_object(formal_results_path)
    _exact_keys(results, FORMAL_RESULTS_TOP_KEYS, context="formal_results.json")
    _validate_header(results, stage="formal_interim", context="formal_results.json")
    if results.get("schema_version") != FORMAL_RESULTS_SCHEMA:
        raise InterimAnalysisError("formal_results.json schema version drifted.")
    if (
        results.get("formal_registry_file_sha256") != registry_file_hash
        or results.get("formal_registry_content_sha256") != registry_content_hash
        or results.get("completed_job_count") != 36
    ):
        raise InterimAnalysisError("formal_results.json is not bound to this complete interim registry.")
    execution_lock_hash = _sha(
        results.get("execution_lock_file_sha256"), context="formal_results execution lock"
    )
    result_rows = results.get("results")
    if not isinstance(result_rows, list) or not all(isinstance(row, Mapping) for row in result_rows):
        raise InterimAnalysisError("formal_results.json.results must be a list of objects.")
    for row in result_rows:
        _exact_keys(row, FORMAL_RESULT_ROW_KEYS, context=f"formal result {row.get('job_id')}")
    results_by_job = _matrix_by_job(result_rows, context="formal results")

    run_registry_path = formal_output_root / "run_registry.csv"
    if protocol.file_sha256(run_registry_path) != results.get("run_registry_file_sha256"):
        raise InterimAnalysisError("run_registry.csv changed after formal_results.json was finalized.")
    run_rows = _read_run_registry(run_registry_path)
    run_by_job = _matrix_by_job(run_rows, context="run registry")

    selection_hashes: Dict[str, str] = {}
    structured: List[StructuredJob] = []
    for job_id in sorted(registry_by_job):
        registry_job = registry_by_job[job_id]
        result = results_by_job.get(job_id)
        run_row = run_by_job.get(job_id)
        if result is None or run_row is None:
            raise InterimAnalysisError(f"Missing formal result/run row for {job_id}.")
        target = str(registry_job["target_env"])
        seed = int(registry_job["seed"])
        arm = str(registry_job["arm"])
        identity = {"job_id": job_id, "stage": "formal_interim", "target_env": target, "seed": seed, "arm": arm}
        if any(result.get(key) != value for key, value in identity.items()):
            raise InterimAnalysisError(f"Formal result identity mismatch for {job_id}.")
        if any(str(run_row.get(key)) != str(value) for key, value in identity.items()):
            raise InterimAnalysisError(f"Run registry identity mismatch for {job_id}.")
        selection_hash = _sha(
            registry_job.get("selection_record_file_sha256"), context=f"{job_id} selection hash"
        )
        previous = selection_hashes.setdefault(target, selection_hash)
        if previous != selection_hash:
            raise InterimAnalysisError(f"Selection record hash differs within target {target}.")
        expected_cross_fields = {
            "config_content_sha256": registry_job["config_content_sha256"],
            "selection_record_file_sha256": selection_hash,
            "interim_decision_file_sha256": None,
        }
        for key, wanted in expected_cross_fields.items():
            if result.get(key) != wanted:
                raise InterimAnalysisError(f"Formal result binding mismatch for {job_id}.{key}.")
        if (
            run_row["status"] != "completed"
            or run_row["return_code"] != "0"
            or run_row["error"] != ""
            or int(run_row["attempt"] or "0") < 1
            or run_row["formal_registry_file_sha256"] != registry_file_hash
            or run_row["formal_registry_content_sha256"] != registry_content_hash
            or run_row["execution_lock_file_sha256"] != execution_lock_hash
        ):
            raise InterimAnalysisError(f"Formal run is not a validated completion: {job_id}.")
        for field in (
            "config_content_sha256",
            "config_file_sha256",
            "selection_record_file_sha256",
            "metrics_file_sha256",
            "target_metrics_file_sha256",
            "target_predictions_file_sha256",
            "best_model_file_sha256",
            "attempt_manifest_file_sha256",
        ):
            expected_value = result.get(field) if field in result else registry_job.get(field)
            if run_row[field] != str(expected_value):
                raise InterimAnalysisError(f"Run/result hash binding mismatch for {job_id}.{field}.")

        output_dir = Path(run_row["output_dir"]).resolve()
        if not _inside(output_dir, formal_output_root):
            raise InterimAnalysisError(f"Formal output directory escapes its evidence root: {job_id}.")
        if Path(run_row["metrics_path"]).resolve() != output_dir / "metrics.json":
            raise InterimAnalysisError(f"Non-canonical metrics path for {job_id}.")
        manifest_path = output_dir / "attempt_manifest.json"
        if protocol.file_sha256(manifest_path) != run_row["attempt_manifest_file_sha256"]:
            raise InterimAnalysisError(f"Attempt manifest hash mismatch for {job_id}.")
        manifest = _load_json_object(manifest_path)
        _exact_keys(manifest, ATTEMPT_MANIFEST_KEYS, context=f"attempt manifest {job_id}")
        _validate_header(manifest, stage="formal_interim", context=f"attempt manifest {job_id}")
        if manifest.get("schema_version") != ATTEMPT_MANIFEST_SCHEMA:
            raise InterimAnalysisError(f"Attempt manifest schema drifted for {job_id}.")
        manifest_identity = {
            "job_id": job_id,
            "target_env": target,
            "source_envs": registry_job["source_envs"],
            "seed": seed,
            "arm": arm,
            "selected_lambda": registry_job["selected_lambda"],
            "lambda_supcon": registry_job["lambda_supcon"],
            "config_path": str((formal_bundle_root / str(registry_job["config_path"])).resolve()),
            "config_content_sha256": registry_job["config_content_sha256"],
            "config_file_sha256": registry_job["config_file_sha256"],
            "selection_record_file_sha256": selection_hash,
            "interim_decision_file_sha256": None,
            "formal_registry_file_sha256": registry_file_hash,
            "formal_registry_content_sha256": registry_content_hash,
            "execution_lock_file_sha256": execution_lock_hash,
        }
        if any(manifest.get(key) != wanted for key, wanted in manifest_identity.items()):
            raise InterimAnalysisError(f"Attempt manifest identity/binding mismatch for {job_id}.")
        artifact_hashes = manifest.get("artifact_file_sha256")
        if not isinstance(artifact_hashes, Mapping) or not REQUIRED_ATTEMPT_ARTIFACTS <= set(artifact_hashes):
            raise InterimAnalysisError(f"Attempt manifest lacks required artifacts for {job_id}.")
        for relative_name, raw_digest in artifact_hashes.items():
            artifact_path = _resolve_relative(output_dir, relative_name, context=f"{job_id} artifact path")
            digest = _sha(raw_digest, context=f"{job_id}/{relative_name} artifact hash")
            if not artifact_path.is_file() or protocol.file_sha256(artifact_path) != digest:
                raise InterimAnalysisError(f"Attempt artifact hash mismatch: {job_id}/{relative_name}.")
        linked = {
            "metrics.json": "metrics_file_sha256",
            "target_metrics.json": "target_metrics_file_sha256",
            "target_predictions.jsonl": "target_predictions_file_sha256",
            "best_model.pt": "best_model_file_sha256",
            "split_manifest.json": "split_manifest_file_sha256",
        }
        for artifact_name, row_field in linked.items():
            if artifact_hashes.get(artifact_name) != run_row[row_field]:
                raise InterimAnalysisError(f"Attempt/run hash mismatch: {job_id}/{artifact_name}.")
        if manifest.get("target_metrics") != result.get("target_metrics"):
            raise InterimAnalysisError(f"Attempt/result target metrics mismatch for {job_id}.")
        structured.append(
            StructuredJob(
                registry=registry_job,
                result=result,
                run_row=run_row,
                manifest=manifest,
                output_dir=output_dir,
                manifest_path=manifest_path,
            )
        )

    if set(selection_hashes) != set(protocol.ENVIRONMENTS) or len(structured) != 36:
        raise InterimAnalysisError("Formal evidence did not validate all three target selections and 36 jobs.")
    return registry, results, run_registry_path, structured, selection_hashes


def _compare_recomputed_metrics(
    stored: Mapping[str, Any], recomputed: Mapping[str, Any], *, job_id: str
) -> None:
    for key in (
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "per_class",
        "confusion_matrix",
        "c1_c4_pair_confusion",
    ):
        if stored.get(key) != recomputed.get(key):
            raise InterimAnalysisError(f"Prediction-recomputed target metric differs for {job_id}.{key}.")
    loss_ce = _finite_number(stored.get("loss_ce"), context=f"{job_id} target loss_ce")
    if loss_ce < 0.0:
        raise InterimAnalysisError(f"{job_id} target loss_ce must be nonnegative.")


def _recompute_prediction_metrics(job: StructuredJob) -> Tuple[Dict[str, Any], int, str]:
    target = str(job.registry["target_env"])
    labels: List[int] = []
    predictions: List[int] = []
    ordered_identity: List[Tuple[str, int, str]] = []
    seen: set[str] = set()
    path = job.output_dir / "target_predictions.jsonl"
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise ValueError("row is not an object")
                sample_id = str(record.get("sample_id", ""))
                group_id = str(record.get("group_id", ""))
                true = record.get("true")
                pred = record.get("pred")
                probabilities = record.get("probability")
                if (
                    not sample_id
                    or not sample_id.startswith(f"{target}:")
                    or not group_id
                    or not group_id.startswith(f"{target}:")
                    or sample_id in seen
                    or record.get("environment") != target
                    or isinstance(true, bool)
                    or not isinstance(true, int)
                    or true not in protocol.LABELS
                    or isinstance(pred, bool)
                    or not isinstance(pred, int)
                    or pred not in protocol.LABELS
                    or not isinstance(probabilities, list)
                    or len(probabilities) != len(protocol.LABELS)
                ):
                    raise ValueError("identity/label/probability schema validation failed")
                probability_values = [
                    _finite_number(value, context=f"{job.registry['job_id']} prediction probability")
                    for value in probabilities
                ]
                if (
                    any(value < 0.0 or value > 1.0 for value in probability_values)
                    or abs(sum(probability_values) - 1.0) > 1e-5
                    or max(range(len(probability_values)), key=probability_values.__getitem__) != pred
                ):
                    raise ValueError("invalid probability simplex or pred/argmax mismatch")
                seen.add(sample_id)
                labels.append(true)
                predictions.append(pred)
                ordered_identity.append((sample_id, true, group_id))
    except (OSError, json.JSONDecodeError, ValueError, InterimAnalysisError) as exc:
        raise InterimAnalysisError(f"Invalid target predictions {path}: {exc}") from exc

    expected_count = sum(item.support_per_environment for item in protocol.LABELS.values())
    expected_support = {
        label: item.support_per_environment for label, item in protocol.LABELS.items()
    }
    observed_support = {label: labels.count(label) for label in protocol.LABELS}
    if len(labels) != expected_count or observed_support != expected_support:
        raise InterimAnalysisError(
            f"Target predictions for {job.registry['job_id']} have wrong count/class support."
        )
    recomputed = fixed_label_metrics(labels, predictions, num_classes=len(protocol.LABELS))
    target_metrics_path = job.output_dir / "target_metrics.json"
    stored = _load_json_object(target_metrics_path)
    metrics = _load_json_object(job.output_dir / "metrics.json")
    if (
        metrics.get("target_metrics") != stored
        or job.result.get("target_metrics") != stored
        or job.manifest.get("target_metrics") != stored
    ):
        raise InterimAnalysisError(
            f"Target metrics copies disagree for {job.registry['job_id']}."
        )
    _compare_recomputed_metrics(stored, recomputed, job_id=str(job.registry["job_id"]))
    return recomputed, len(labels), protocol.canonical_sha256(ordered_identity)


def fit_target_effect_model(observations: Mapping[Tuple[str, int], float]) -> Dict[str, Any]:
    """Fit d[e,s] = mu + gamma[e] + error with balanced sum-to-zero coding."""

    expected = {
        (target, seed)
        for target in protocol.ENVIRONMENTS
        for seed in protocol.formal_seeds(target, "formal_interim")
    }
    if set(observations) != expected:
        raise InterimAnalysisError("Effect model requires exactly 3 targets x 3 paired seeds.")
    values = {key: _finite_number(value, context=f"effect observation {key}") for key, value in observations.items()}
    target_means = {
        target: sum(values[(target, seed)] for seed in protocol.formal_seeds(target, "formal_interim")) / 3.0
        for target in protocol.ENVIRONMENTS
    }
    marginal_mean = sum(target_means.values()) / len(protocol.ENVIRONMENTS)
    residuals = [
        values[(target, seed)] - target_means[target]
        for target in protocol.ENVIRONMENTS
        for seed in protocol.formal_seeds(target, "formal_interim")
    ]
    residual_df = int(protocol.BUDGETS["interim_residual_degrees_of_freedom"])
    if residual_df != 6:
        raise InterimAnalysisError("Interim residual degrees of freedom must remain locked at 6.")
    residual_ss = sum(value * value for value in residuals)
    residual_mse = residual_ss / residual_df
    marginal_se = math.sqrt(residual_mse / 9.0)
    target_se = math.sqrt(residual_mse / 3.0)
    critical_90 = float(student_t.ppf(float(protocol.FUTILITY_THRESHOLDS["futility_confidence_level"]), residual_df))
    critical_95 = float(student_t.ppf(float(protocol.FUTILITY_THRESHOLDS["harm_confidence_level"]), residual_df))
    return {
        "method": INTERIM_ANALYSIS_METHOD,
        "target_coding": "sum_to_zero",
        "residual_degrees_of_freedom": residual_df,
        "observations": [
            {"target_env": target, "seed": seed, "difference": values[(target, seed)]}
            for target in protocol.ENVIRONMENTS
            for seed in protocol.formal_seeds(target, "formal_interim")
        ],
        "marginal_mean": marginal_mean,
        "target_means": target_means,
        "target_effects": {target: target_means[target] - marginal_mean for target in protocol.ENVIRONMENTS},
        "residual_sum_squares": residual_ss,
        "residual_mean_square": residual_mse,
        "marginal_standard_error": marginal_se,
        "target_mean_standard_error": target_se,
        "critical_values": {"one_sided_90": critical_90, "one_sided_95": critical_95},
        "marginal_upper_bounds": {
            "one_sided_90": marginal_mean + critical_90 * marginal_se,
            "one_sided_95": marginal_mean + critical_95 * marginal_se,
        },
        "target_upper_bounds_one_sided_90": {
            target: target_means[target] + critical_90 * target_se for target in protocol.ENVIRONMENTS
        },
    }


def evaluate_futility(
    macro_f1: Mapping[Tuple[str, int, str], float],
    sit_stand_f1: Mapping[Tuple[str, int, str], float],
) -> Dict[str, Any]:
    contrasts = {
        "D_S": fit_target_effect_model(
            {
                (target, seed): macro_f1[(target, seed, "Y11")] - macro_f1[(target, seed, "Y01")]
                for target in protocol.ENVIRONMENTS
                for seed in protocol.formal_seeds(target, "formal_interim")
            }
        ),
        "D_T": fit_target_effect_model(
            {
                (target, seed): macro_f1[(target, seed, "Y11")] - macro_f1[(target, seed, "Y10")]
                for target in protocol.ENVIRONMENTS
                for seed in protocol.formal_seeds(target, "formal_interim")
            }
        ),
        "C3_Y11_minus_Y00": fit_target_effect_model(
            {
                (target, seed): sit_stand_f1[(target, seed, "Y11")]
                - sit_stand_f1[(target, seed, "Y00")]
                for target in protocol.ENVIRONMENTS
                for seed in protocol.formal_seeds(target, "formal_interim")
            }
        ),
    }

    c3_threshold = float(protocol.FUTILITY_THRESHOLDS["sit_stand_harm_margin_vs_control"])
    c3_upper = float(contrasts["C3_Y11_minus_Y00"]["marginal_upper_bounds"]["one_sided_95"])
    condition_c3 = {
        "condition_id": "sit_stand_harm_vs_control",
        "definition": "one-sided 95% UCB(C3 Y11-Y00) < -0.02",
        "threshold": c3_threshold,
        "upper_bound": c3_upper,
        "triggered": c3_upper < c3_threshold,
    }

    target_mean_threshold = float(protocol.FUTILITY_THRESHOLDS["target_specific_mean_harm"])
    target_upper_threshold = float(protocol.FUTILITY_THRESHOLDS["target_specific_upper_bound"])
    target_evaluations: List[Dict[str, Any]] = []
    for contrast_name in ("D_S", "D_T"):
        model = contrasts[contrast_name]
        for target in protocol.ENVIRONMENTS:
            mean = float(model["target_means"][target])
            upper = float(model["target_upper_bounds_one_sided_90"][target])
            target_evaluations.append(
                {
                    "contrast": contrast_name,
                    "target_env": target,
                    "mean": mean,
                    "mean_threshold": target_mean_threshold,
                    "one_sided_90_upper_bound": upper,
                    "upper_bound_threshold": target_upper_threshold,
                    "triggered": mean < target_mean_threshold and upper < target_upper_threshold,
                }
            )
    condition_target = {
        "condition_id": "target_specific_harm",
        "definition": "any target mean(D_S or D_T) < -0.05 and its one-sided 90% UCB < 0",
        "evaluations": target_evaluations,
        "triggered": any(item["triggered"] for item in target_evaluations),
    }

    practical_threshold = float(protocol.FUTILITY_THRESHOLDS["minimum_practical_effect"])
    practical_evaluations = [
        {
            "contrast": contrast_name,
            "one_sided_90_upper_bound": float(
                contrasts[contrast_name]["marginal_upper_bounds"]["one_sided_90"]
            ),
            "threshold": practical_threshold,
            "triggered": float(
                contrasts[contrast_name]["marginal_upper_bounds"]["one_sided_90"]
            )
            < practical_threshold,
        }
        for contrast_name in ("D_S", "D_T")
    ]
    condition_practical = {
        "condition_id": "minimum_practical_effect_futility",
        "definition": "one-sided 90% UCB(equal-target mean D_S or D_T) < +0.01",
        "evaluations": practical_evaluations,
        "triggered": any(item["triggered"] for item in practical_evaluations),
    }
    conditions = [condition_c3, condition_target, condition_practical]
    triggered_ids = [item["condition_id"] for item in conditions if item["triggered"]]
    core = {
        "method": INTERIM_ANALYSIS_METHOD,
        "metric_scope": ["target_macro_f1", "target_sit_stand_f1"],
        "residual_degrees_of_freedom": 6,
        "contrasts": contrasts,
        "futility_conditions": conditions,
        "triggered_condition_ids": triggered_ids,
    }
    return {**core, "analysis_core_sha256": protocol.canonical_sha256(core)}


def build_decision_payload(formal_registry_path: Path, formal_results_path: Path) -> Dict[str, Any]:
    registry, results, run_registry_path, jobs, selection_hashes = _validate_structural_evidence(
        formal_registry_path, formal_results_path
    )
    formal_registry_path = formal_registry_path.resolve()
    formal_results_path = formal_results_path.resolve()
    formal_bundle_root = formal_registry_path.parent
    formal_output_root = formal_results_path.parent

    # Prediction reads start only after every one of the 36 manifests, result
    # rows, and completed run-registry rows has passed the structural gate.
    macro_f1: Dict[Tuple[str, int, str], float] = {}
    sit_stand_f1: Dict[Tuple[str, int, str], float] = {}
    evidence_jobs: List[Dict[str, Any]] = []
    target_identity_hashes: Dict[str, str] = {}
    for job in sorted(
        jobs,
        key=lambda item: (
            protocol.ENVIRONMENTS.index(str(item.registry["target_env"])),
            int(item.registry["seed"]),
            list(protocol.ARMS).index(str(item.registry["arm"])),
        ),
    ):
        recomputed, prediction_count, prediction_identity_hash = _recompute_prediction_metrics(job)
        cell = (
            str(job.registry["target_env"]),
            int(job.registry["seed"]),
            str(job.registry["arm"]),
        )
        macro_f1[cell] = float(recomputed["f1_macro"])
        sit_stand_f1[cell] = float(recomputed["per_class"]["3"]["f1"])
        previous_identity = target_identity_hashes.setdefault(cell[0], prediction_identity_hash)
        if previous_identity != prediction_identity_hash:
            raise InterimAnalysisError(
                f"Ordered target sample/true/group identity differs across paired runs for {cell[0]}."
            )
        evidence_jobs.append(
            {
                "job_id": str(job.registry["job_id"]),
                "target_env": cell[0],
                "seed": cell[1],
                "arm": cell[2],
                "config_content_sha256": str(job.registry["config_content_sha256"]),
                "attempt_manifest_path": _safe_relative(
                    job.manifest_path, formal_output_root, context=f"{cell} attempt manifest"
                ),
                "attempt_manifest_file_sha256": protocol.file_sha256(job.manifest_path),
                "metrics_file_sha256": str(job.run_row["metrics_file_sha256"]),
                "target_metrics_file_sha256": str(job.run_row["target_metrics_file_sha256"]),
                "target_predictions_file_sha256": str(job.run_row["target_predictions_file_sha256"]),
                "target_prediction_count": prediction_count,
                "target_prediction_identity_sha256": prediction_identity_hash,
                "recomputed_target_macro_f1": macro_f1[cell],
                "recomputed_target_sit_stand_f1": sit_stand_f1[cell],
            }
        )

    evidence_core = {
        "portability": (
            "formal_bundle_root and formal_output_root are absolute; moving either root "
            "invalidates this artifact and requires regeneration"
        ),
        "formal_bundle_root": str(formal_bundle_root),
        "formal_registry": {
            "path": _safe_relative(formal_registry_path, formal_bundle_root, context="formal registry"),
            "file_sha256": protocol.file_sha256(formal_registry_path),
            "content_sha256": protocol.canonical_sha256(registry),
        },
        "formal_output_root": str(formal_output_root),
        "formal_results": {
            "path": _safe_relative(formal_results_path, formal_output_root, context="formal results"),
            "file_sha256": protocol.file_sha256(formal_results_path),
            "content_sha256": protocol.canonical_sha256(results),
        },
        "run_registry": {
            "path": _safe_relative(run_registry_path, formal_output_root, context="run registry"),
            "file_sha256": protocol.file_sha256(run_registry_path),
        },
        "execution_lock_file_sha256": str(results["execution_lock_file_sha256"]),
        "jobs": evidence_jobs,
    }
    evidence = {**evidence_core, "evidence_core_sha256": protocol.canonical_sha256(evidence_core)}
    analysis = evaluate_futility(macro_f1, sit_stand_f1)
    decision = "stop" if analysis["triggered_condition_ids"] else "continue"
    return {
        "schema_version": protocol.INTERIM_DECISION_SCHEMA,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "decision": decision,
        "completed_interim_job_count": protocol.BUDGETS["formal_interim_jobs"],
        "formal_interim_registry_content_sha256": protocol.canonical_sha256(registry),
        "selection_record_file_sha256": selection_hashes,
        "evidence": evidence,
        "analysis": analysis,
    }


def _paths_from_decision(record: Mapping[str, Any]) -> Tuple[Path, Path]:
    evidence = record.get("evidence")
    if not isinstance(evidence, Mapping):
        raise InterimAnalysisError("Decision evidence must be an object.")
    raw_bundle_root = evidence.get("formal_bundle_root")
    raw_output_root = evidence.get("formal_output_root")
    if (
        not isinstance(raw_bundle_root, str)
        or not raw_bundle_root
        or not Path(raw_bundle_root).is_absolute()
        or not isinstance(raw_output_root, str)
        or not raw_output_root
        or not Path(raw_output_root).is_absolute()
    ):
        raise InterimAnalysisError("Decision evidence roots must be absolute.")
    formal_bundle_root = Path(raw_bundle_root).resolve()
    formal_output_root = Path(raw_output_root).resolve()
    registry_ref = evidence.get("formal_registry")
    results_ref = evidence.get("formal_results")
    if not isinstance(registry_ref, Mapping) or not isinstance(results_ref, Mapping):
        raise InterimAnalysisError("Decision registry/results evidence references are missing.")
    registry_path = _resolve_relative(
        formal_bundle_root, registry_ref.get("path"), context="decision formal registry"
    )
    results_path = _resolve_relative(
        formal_output_root, results_ref.get("path"), context="decision formal results"
    )
    return registry_path, results_path


def validate_decision_artifact(
    decision_path: Path,
    *,
    expected_registry_content_sha256: Optional[str] = None,
    expected_selection_record_file_sha256: Optional[Mapping[str, str]] = None,
    require_continue: bool = False,
) -> Dict[str, Any]:
    decision_path = decision_path.resolve()
    record = _load_json_object(decision_path)
    _exact_keys(record, DECISION_TOP_KEYS, context="interim decision")
    if record.get("schema_version") != protocol.INTERIM_DECISION_SCHEMA:
        raise InterimAnalysisError("Interim decision schema version drifted.")
    registry_path, results_path = _paths_from_decision(record)
    expected = build_decision_payload(registry_path, results_path)
    if record != expected:
        raise InterimAnalysisError(
            "Interim decision does not exactly equal independent recomputation from bound evidence."
        )
    if (
        expected_registry_content_sha256 is not None
        and record["formal_interim_registry_content_sha256"] != expected_registry_content_sha256
    ):
        raise InterimAnalysisError("Interim decision is bound to a different formal registry.")
    if expected_selection_record_file_sha256 is not None and dict(
        record["selection_record_file_sha256"]
    ) != dict(expected_selection_record_file_sha256):
        raise InterimAnalysisError("Interim decision is bound to different selection records.")
    if require_continue and record.get("decision") != "continue":
        raise InterimAnalysisError("Interim futility decision is stop; extension is not authorized.")
    return record


def _atomic_write_or_validate(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        if _load_json_object(path) != dict(payload):
            raise InterimAnalysisError(f"Refusing to overwrite a different decision artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise InterimAnalysisError(f"Decision artifact appeared concurrently: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def analyze_and_write_decision(
    formal_registry_path: Path,
    formal_results_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    payload = build_decision_payload(formal_registry_path, formal_results_path)
    formal_bundle_root = formal_registry_path.resolve().parent
    formal_output_root = formal_results_path.resolve().parent
    output = output_path.resolve()
    if _inside(output, formal_bundle_root) or _inside(output, formal_output_root):
        raise InterimAnalysisError(
            "Decision output must be outside generated bundles and running/formal output roots."
        )
    _atomic_write_or_validate(output, payload)
    validate_decision_artifact(output)
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-registry", type=Path, required=True)
    parser.add_argument("--formal-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    payload = analyze_and_write_decision(args.formal_registry, args.formal_results, args.output)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "triggered_condition_ids": payload["analysis"]["triggered_condition_ids"],
                "output": str(args.output.resolve()),
                "file_sha256": protocol.file_sha256(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
