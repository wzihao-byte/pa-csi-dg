"""Fail-closed checker for generated SupCon + TPP protocol bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

try:  # Support module and direct-script execution.
    from . import protocol, select_lambda
except ImportError:  # pragma: no cover
    import protocol  # type: ignore
    import select_lambda  # type: ignore


BUNDLE_MANIFEST_NAME = "bundle_manifest.json"


class ProtocolValidationError(ValueError):
    pass


def load_json_object(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ProtocolValidationError(f"Expected a JSON object: {path}")
    return payload


def flatten_config(value: object, prefix: str = "") -> Dict[str, object]:
    if isinstance(value, Mapping):
        flattened: Dict[str, object] = {}
        for key in sorted(value, key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_config(value[key], path))
        return flattened
    return {prefix: value}


def differing_paths(configs: Mapping[str, Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    if not configs:
        raise ProtocolValidationError("At least one config is required.")
    flattened = {name: flatten_config(config) for name, config in configs.items()}
    all_paths: Set[str] = set().union(*(set(values) for values in flattened.values()))
    missing = object()
    differences: Dict[str, Dict[str, object]] = {}
    for path in sorted(all_paths):
        values = {name: flat.get(path, missing) for name, flat in flattened.items()}
        canonical = {
            "__MISSING__" if value is missing else json.dumps(value, sort_keys=True, default=str)
            for value in values.values()
        }
        if len(canonical) > 1:
            differences[path] = {
                name: "__MISSING__" if value is missing else value for name, value in values.items()
            }
    return differences


def assert_only_allowed_differences(
    configs: Mapping[str, Mapping[str, object]],
    allowed_paths: Iterable[str],
    *,
    context: str,
) -> Dict[str, Dict[str, object]]:
    allowed = set(allowed_paths)
    differences = differing_paths(configs)
    illegal = {path: values for path, values in differences.items() if path not in allowed}
    if illegal:
        details = "; ".join(f"{path}={values}" for path, values in illegal.items())
        raise ProtocolValidationError(f"{context} contains non-whitelisted drift: {details}")
    return differences


TEMPLATE_ALLOWED_DIFFERENCES = frozenset(
    {
        "experiment_name",
        "protocol.arm",
        "protocol.supcon_enabled",
        "protocol.temporal_factor",
        "model.joint_temporal.factor",
        "losses.lambda_supcon",
        "losses.supcon_enabled",
    }
)


def _read_path(config: Mapping[str, object], path: str) -> object:
    current: object = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ProtocolValidationError(f"Missing required path: {path}")
        current = current[part]
    return current


def _assert(config: Mapping[str, object], path: str, expected: object) -> None:
    actual = _read_path(config, path)
    if actual != expected:
        raise ProtocolValidationError(f"{path} must be {expected!r}; received {actual!r}")


def _labels() -> Dict[str, dict]:
    return {
        str(label): {
            "name": spec.name,
            "support_per_environment": spec.support_per_environment,
            "support_per_group": spec.support_per_group,
        }
        for label, spec in protocol.LABELS.items()
    }


def _global_basis() -> dict:
    return {
        "input_normalization": {
            "name": "layer_norm",
            "epsilon": protocol.GLOBAL_BASIS_CALIBRATION["layer_norm_epsilon"],
            "affine": protocol.GLOBAL_BASIS_CALIBRATION["layer_norm_affine"],
            "clip_min": protocol.GLOBAL_BASIS_CALIBRATION["clip_min"],
            "clip_max": protocol.GLOBAL_BASIS_CALIBRATION["clip_max"],
        },
        "basis": [dict(item) for item in protocol.GLOBAL_BASIS],
        "calibration": dict(protocol.GLOBAL_BASIS_CALIBRATION),
    }


def validate_common_config(config: Mapping[str, object]) -> None:
    """Absolutely lock critical fields, including common drift in every config."""

    expected = {
        "mode": "dg_loeo",
        "data.format": "env_npy_triplets",
        "data.unwrap_phase": False,
        "data.strict_group_protocol": True,
        "split.strict_group_protocol": True,
        "split.inner_validation": "inferred_group_block_holdout",
        "split.subject_block_size": protocol.GROUP_BLOCK_SIZE,
        "split.groups_per_env": protocol.GROUPS_PER_ENVIRONMENT,
        "split.train_groups_per_source_env": protocol.TRAIN_GROUPS_PER_SOURCE_ENVIRONMENT,
        "split.validation_groups_per_source_env": protocol.VALIDATION_GROUPS_PER_SOURCE_ENVIRONMENT,
        "split.validation_group_ids": {
            environment: list(groups) for environment, groups in protocol.VALIDATION_GROUP_IDS.items()
        },
        "split.split_seed": protocol.SPLIT_SEED,
        "training.epochs": protocol.EPOCHS,
        "training.batch_size": protocol.BATCH_SIZE,
        "training.samples_per_source_domain_per_batch": protocol.SAMPLES_PER_SOURCE_DOMAIN_PER_BATCH,
        "training.optimizer_steps_per_epoch": protocol.OPTIMIZER_STEPS_PER_EPOCH,
        "training.total_optimizer_steps": protocol.TOTAL_OPTIMIZER_STEPS,
        "training.lr": 0.001,
        "training.optimizer": "adam",
        "training.lr_scheduler": {
            "name": "exponential_decay",
            "update": "step",
            "decay_rate": 0.9,
            "decay_steps": 10000,
            "staircase": False,
        },
        "training.weight_decay": 0.0,
        "training.patience": protocol.EPOCHS,
        "training.early_stopping": False,
        "training.selection_metric": protocol.CHECKPOINT_METRIC,
        "training.selection_tie_break": "earlier_epoch",
        "training.sampler": protocol.SAMPLER,
        "training.domain_balanced_batch": True,
        "training.grad_clip_norm": 0.0,
        "losses.contrastive_loss_type": "pairwise_supcon",
        "losses.temperature": protocol.SUPCON_TEMPERATURE,
        "losses.linear_warmup_steps": protocol.LINEAR_WARMUP_STEPS,
        "losses.include_same_domain_same_class": True,
        "losses.supcon_positive_mode": "all_views",
        "losses.supcon_view_scope": protocol.SUPCON_VIEW_SCOPE,
        "model.name": "supcon_tpp_joint",
        "model.joint_temporal.branch": "amplitude_only",
        "model.joint_temporal.levels": list(protocol.TPP_LEVELS),
        "model.joint_temporal.pool_type": protocol.TPP_POOL_TYPE,
        "model.joint_temporal.preserve_global": protocol.TPP_PRESERVE_GLOBAL,
        "model.joint_temporal.residual_scale": protocol.TPP_RESIDUAL_SCALE,
        "model.joint_temporal.global_basis": _global_basis(),
        "protocol.name": protocol.PROTOCOL_NAME,
        "protocol.version": protocol.PROTOCOL_VERSION,
        "protocol.lock_sha256": protocol.protocol_lock_sha256(),
        "protocol.phase_unwrap": False,
        "protocol.strict_group_protocol": True,
        "protocol.labels": _labels(),
        "protocol.screen_thresholds": dict(protocol.SCREEN_THRESHOLDS),
        "protocol.success_thresholds": dict(protocol.SUCCESS_THRESHOLDS),
        "protocol.futility_thresholds": dict(protocol.FUTILITY_THRESHOLDS),
        "calibration.uses_target_data": False,
        "calibration.initialization_seed": protocol.GLOBAL_BASIS_CALIBRATION["initialization_seed"],
    }
    for path, value in expected.items():
        _assert(config, path, value)

    flat = flatten_config(config)
    forbidden = {
        "data.phase_unwrap_axis",
        "split.fallback_inner_validation",
        "split.subject_fold",
        "split.split_seed_provenance",
        "losses.lambda_supcon_warmup_epochs",
        "losses.lambda_supcon_warmup_steps",
    }
    present = sorted(forbidden & set(flat))
    if present:
        raise ProtocolValidationError(f"Forbidden stale schema fields: {present}")
    model = _read_path(config, "model")
    if isinstance(model, Mapping) and "antenna_layout" in model:
        raise ProtocolValidationError("model.antenna_layout is forbidden for global-only SupCon.")


def validate_arm(config: Mapping[str, object], arm_code: str, expected_lambda: Optional[float]) -> None:
    if arm_code not in protocol.ARMS:
        raise ProtocolValidationError(f"Unknown arm: {arm_code}")
    arm = protocol.ARMS[arm_code]
    _assert(config, "protocol.arm", arm_code)
    _assert(config, "protocol.temporal_factor", arm.temporal_factor)
    _assert(config, "protocol.supcon_enabled", arm.supcon_enabled)
    _assert(config, "model.joint_temporal.factor", arm.temporal_factor)
    _assert(config, "losses.supcon_enabled", arm.supcon_enabled)
    actual = _read_path(config, "losses.lambda_supcon")
    if expected_lambda is None:
        expected = None if arm.supcon_enabled else 0.0
    else:
        expected = expected_lambda if arm.supcon_enabled else 0.0
    if actual != expected:
        raise ProtocolValidationError(f"{arm_code} lambda must be {expected!r}; received {actual!r}")


def validate_fold_data(config: Mapping[str, object], *, target_env: str, evaluate_target: bool) -> None:
    sources = list(protocol.source_environments(target_env))
    _assert(config, "target_env", target_env)
    _assert(config, "source_envs", sources)
    _assert(config, "protocol.target_env", target_env)
    _assert(config, "protocol.source_envs", sources)
    _assert(config, "data.loaded_envs", sources)
    _assert(config, "data.source_only", True)
    _assert(config, "evaluation.evaluate_target", evaluate_target)
    _assert(config, "target_data.target_env", target_env)
    _assert(config, "target_data.load_for_evaluation", evaluate_target)
    env_files = _read_path(config, "data.env_files")
    target_files = _read_path(config, "target_data.env_files")
    if not isinstance(env_files, Mapping) or set(env_files) != set(sources):
        raise ProtocolValidationError("Source loader data.env_files must contain exactly the two source environments.")
    if target_env in env_files:
        raise ProtocolValidationError("Target environment leaked into source loader data.env_files.")
    if not isinstance(target_files, Mapping) or set(target_files) != {"amp", "phase", "label"}:
        raise ProtocolValidationError("Target file declaration must be isolated in target_data.env_files.")


def check_templates(templates: Mapping[str, Mapping[str, object]]) -> dict:
    if set(templates) != set(protocol.ARMS):
        raise ProtocolValidationError("Templates must be exactly Y00/Y01/Y10/Y11.")
    names: List[str] = []
    for arm_code, config in templates.items():
        validate_common_config(config)
        validate_arm(config, arm_code, None)
        _assert(config, "protocol.stage", "template")
        _assert(config, "protocol.resolved", False)
        _assert(config, "protocol.job_id", None)
        _assert(config, "data.loaded_envs", [])
        _assert(config, "data.env_files", {})
        _assert(config, "target_data.env_files", {})
        names.append(str(_read_path(config, "experiment_name")))
    if len(names) != len(set(names)):
        raise ProtocolValidationError("Template experiment names must be unique.")
    differences = assert_only_allowed_differences(
        templates, TEMPLATE_ALLOWED_DIFFERENCES, context="arm templates"
    )
    return {"template_count": 4, "difference_paths": sorted(differences)}


def _validate_registry_header(registry: Mapping[str, object], stage: str, expected_count: int) -> List[Mapping[str, object]]:
    expected = {
        "schema_version": protocol.REGISTRY_SCHEMA,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "stage": stage,
        "expected_job_count": expected_count,
    }
    for key, value in expected.items():
        if registry.get(key) != value:
            raise ProtocolValidationError(f"Registry {key} must be {value!r}.")
    expected_keys = set(expected) | {"jobs"}
    if set(registry) != expected_keys:
        raise ProtocolValidationError(
            f"Registry top-level schema drifted; expected {sorted(expected_keys)}, received {sorted(registry)}."
        )
    jobs = registry.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != expected_count or not all(isinstance(job, Mapping) for job in jobs):
        raise ProtocolValidationError(f"{stage} registry must contain exactly {expected_count} job objects.")
    ids = [str(job.get("job_id")) for job in jobs]
    if len(ids) != len(set(ids)):
        raise ProtocolValidationError(f"{stage} registry job IDs must be unique.")
    return jobs


def _validate_job_metadata(job: Mapping[str, object], config: Mapping[str, object], stage: str) -> None:
    job_id = str(job.get("job_id"))
    target_env = str(job.get("target_env"))
    arm_code = str(job.get("arm"))
    seed = int(job.get("seed"))
    value = float(job.get("lambda_supcon"))
    sources = list(protocol.source_environments(target_env))
    validate_common_config(config)
    validate_arm(config, arm_code, value)
    validate_fold_data(config, target_env=target_env, evaluate_target=stage != "screen")
    _assert(config, "experiment_name", job_id)
    _assert(config, "protocol.job_id", job_id)
    _assert(config, "protocol.stage", stage)
    _assert(config, "protocol.resolved", True)
    _assert(config, "protocol.seed", seed)
    _assert(config, "seed_list", [seed])
    if job.get("stage") != stage or job.get("source_envs") != sources:
        raise ProtocolValidationError(f"Top-level registry metadata mismatch for {job_id}.")
    if "config_content_sha256" in job and job["config_content_sha256"] != protocol.canonical_sha256(config):
        raise ProtocolValidationError(f"Config content hash mismatch for {job_id}.")


def check_screen_job_registry(jobs_or_registry: object) -> dict:
    if isinstance(jobs_or_registry, Mapping):
        jobs = _validate_registry_header(jobs_or_registry, "screen", protocol.BUDGETS["screen_runs"])
    elif isinstance(jobs_or_registry, Sequence):
        jobs = jobs_or_registry
        if len(jobs) != protocol.BUDGETS["screen_runs"]:
            raise ProtocolValidationError("Expected exactly 18 screen jobs.")
    else:
        raise ProtocolValidationError("Screen registry must be an object or job sequence.")
    counts = {environment: 0 for environment in protocol.ENVIRONMENTS}
    seen = set()
    for job in jobs:
        if not isinstance(job, Mapping):
            raise ProtocolValidationError("Screen jobs must be objects.")
        target = str(job.get("target_env"))
        arm = str(job.get("arm"))
        value = float(job.get("lambda_supcon"))
        seed = int(job.get("seed"))
        if seed != protocol.SCREEN_SEEDS.get(target) or job.get("screen_seed") != seed:
            raise ProtocolValidationError(f"Screen seed mismatch for {target}.")
        expected_values = protocol.LAMBDA_CANDIDATES if protocol.ARMS[arm].supcon_enabled else (0.0,)
        if value not in expected_values:
            raise ProtocolValidationError(f"Invalid screen lambda {value} for {arm}.")
        cell = (target, arm, value)
        if cell in seen:
            raise ProtocolValidationError(f"Duplicate screen cell: {cell}")
        seen.add(cell)
        config = job.get("config")
        if not isinstance(config, Mapping):
            raise ProtocolValidationError(f"Screen job {job.get('job_id')} has no nested config.")
        _validate_job_metadata(job, config, "screen")
        _assert(config, "protocol.selected_lambda", value)
        expected_job_id = (
            f"screen__target_{target}__arm_{arm}__lam{round(value * 1000):04d}__seed_{seed}"
        )
        if job.get("job_id") != expected_job_id:
            raise ProtocolValidationError(f"Screen job ID drifted: expected {expected_job_id}.")
        counts[target] += 1
    if counts != {"E1": 6, "E2": 6, "E3": 6}:
        raise ProtocolValidationError(f"Screen jobs per target drifted: {counts}")
    return {"screen_job_count": len(jobs), "jobs_per_target": counts}


def check_formal_job_registry(
    jobs_or_registry: object,
    *,
    stage: str,
    selections: Mapping[str, select_lambda.SelectionArtifact],
    decision: Optional[Mapping[str, object]] = None,
    decision_file_sha256: Optional[str] = None,
) -> dict:
    expected_count = protocol.BUDGETS["formal_interim_jobs" if stage == "formal_interim" else "formal_extension_jobs"]
    if isinstance(jobs_or_registry, Mapping):
        jobs = _validate_registry_header(jobs_or_registry, stage, expected_count)
    elif isinstance(jobs_or_registry, Sequence):
        jobs = jobs_or_registry
        if len(jobs) != expected_count:
            raise ProtocolValidationError(f"Expected {expected_count} {stage} jobs.")
    else:
        raise ProtocolValidationError("Formal registry must be an object or job sequence.")
    if set(selections) != set(protocol.ENVIRONMENTS):
        raise ProtocolValidationError("Formal checking requires three selection artifacts.")
    if stage == "formal_extension" and decision is None:
        raise ProtocolValidationError("Extension jobs require an interim decision.")

    seen = set()
    counts = {environment: 0 for environment in protocol.ENVIRONMENTS}
    for job in jobs:
        if not isinstance(job, Mapping):
            raise ProtocolValidationError("Formal jobs must be objects.")
        target = str(job.get("target_env"))
        arm = str(job.get("arm"))
        seed = int(job.get("seed"))
        artifact = selections[target]
        expected_seeds = protocol.formal_seeds(target, stage)
        if seed not in expected_seeds:
            raise ProtocolValidationError(f"Seed {seed} is not registered for {target}/{stage}.")
        cell = (target, arm, seed)
        if cell in seen:
            raise ProtocolValidationError(f"Duplicate formal cell: {cell}")
        seen.add(cell)
        config = job.get("config")
        if not isinstance(config, Mapping):
            raise ProtocolValidationError(f"Formal job {job.get('job_id')} has no nested config.")
        _validate_job_metadata(job, config, stage)
        expected_job_id = f"{stage}__target_{target}__arm_{arm}__seed_{seed}"
        if job.get("job_id") != expected_job_id:
            raise ProtocolValidationError(f"Formal job ID drifted: expected {expected_job_id}.")
        expected_loss_lambda = artifact.selected_lambda if protocol.ARMS[arm].supcon_enabled else 0.0
        if (
            job.get("selected_lambda") != artifact.selected_lambda
            or job.get("lambda_supcon") != expected_loss_lambda
        ):
            raise ProtocolValidationError(f"Formal top-level lambda binding mismatch for {job.get('job_id')}.")
        _assert(config, "protocol.selected_lambda", artifact.selected_lambda)
        selection_ref = _read_path(config, "protocol.selection_record")
        if not isinstance(selection_ref, Mapping):
            raise ProtocolValidationError("Formal config selection_record must be an object.")
        expected_ref = {
            "target_env": target,
            "selected_lambda": artifact.selected_lambda,
            "path": f"selection_records/selection_{target}.json",
            "file_sha256": artifact.file_sha256,
            "screen_registry_file_sha256": artifact.record["screen_registry"]["file_sha256"],  # type: ignore[index]
            "screen_results_file_sha256": artifact.record["screen_results"]["file_sha256"],  # type: ignore[index]
        }
        if dict(selection_ref) != expected_ref or job.get("selection_record_file_sha256") != artifact.file_sha256:
            raise ProtocolValidationError(f"Selection-record binding mismatch for {job.get('job_id')}.")
        interim_ref = _read_path(config, "protocol.interim_decision")
        if stage == "formal_interim":
            if interim_ref is not None or job.get("interim_decision_file_sha256") is not None:
                raise ProtocolValidationError("Interim jobs must not reference an interim decision.")
        else:
            if not isinstance(interim_ref, Mapping) or decision is None:
                raise ProtocolValidationError("Extension config lacks interim decision reference.")
            expected_interim_ref = {
                "path": "interim_decision.json",
                "file_sha256": decision_file_sha256,
                "decision": "continue",
                "formal_interim_registry_content_sha256": decision[
                    "formal_interim_registry_content_sha256"
                ],
            }
            if dict(interim_ref) != expected_interim_ref:
                raise ProtocolValidationError("Extension nested interim-decision reference mismatch.")
            if job.get("interim_decision_file_sha256") != decision_file_sha256:
                raise ProtocolValidationError("Extension top-level interim-decision hash mismatch.")
        counts[target] += 1
    expected_per_target = len(protocol.ARMS) * len(protocol.formal_seeds("E1", stage))
    if any(count != expected_per_target for count in counts.values()):
        raise ProtocolValidationError(f"Formal jobs per target drifted: {counts}")
    return {"stage": stage, "job_count": len(jobs), "jobs_per_target": counts}


def check_protocol_lock(path: Path) -> dict:
    payload = load_json_object(path)
    expected = protocol.protocol_lock_payload()
    if payload != expected:
        raise ProtocolValidationError("protocol_lock.json does not exactly match machine constants.")
    return {"content_sha256": protocol.canonical_sha256(payload)}


def check_bundle_manifest(bundle_dir: Path) -> dict:
    manifest_path = bundle_dir / BUNDLE_MANIFEST_NAME
    manifest = load_json_object(manifest_path)
    expected_header = {
        "schema_version": protocol.BUNDLE_MANIFEST_SCHEMA,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
    }
    for key, value in expected_header.items():
        if manifest.get(key) != value:
            raise ProtocolValidationError(f"Bundle manifest {key} mismatch.")
    recorded = manifest.get("files")
    if not isinstance(recorded, Mapping):
        raise ProtocolValidationError("Bundle manifest files must be an object.")
    actual_paths = {
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_file() and path.name != BUNDLE_MANIFEST_NAME
    }
    if actual_paths != set(recorded):
        raise ProtocolValidationError(
            f"Stale/missing bundle files; extra={sorted(actual_paths-set(recorded))}, missing={sorted(set(recorded)-actual_paths)}"
        )
    for relative, expected_hash in recorded.items():
        if protocol.file_sha256(bundle_dir / relative) != expected_hash:
            raise ProtocolValidationError(f"Bundle file hash mismatch: {relative}")
    return {"generation_mode": manifest.get("generation_mode"), "file_count": len(recorded)}


def _load_registry_with_configs(bundle_dir: Path, filename: str) -> dict:
    registry = load_json_object(bundle_dir / filename)
    jobs = registry.get("jobs")
    if not isinstance(jobs, list):
        raise ProtocolValidationError(f"{filename}.jobs must be a list.")
    config_dirs = {
        "screen_job_registry.json": "screen_configs",
        "formal_interim_job_registry.json": "formal_interim_configs",
        "formal_extension_job_registry.json": "formal_extension_configs",
    }
    if filename not in config_dirs:
        raise ProtocolValidationError(f"Unknown registry filename: {filename}")
    bundle_root = bundle_dir.resolve()
    hydrated: List[dict] = []
    for row in jobs:
        if not isinstance(row, Mapping):
            raise ProtocolValidationError(f"{filename} row must be an object.")
        expected_relative = f"{config_dirs[filename]}/{row.get('job_id')}.json"
        if row.get("config_path") != expected_relative:
            raise ProtocolValidationError(f"Registry config_path mismatch for {row.get('job_id')}.")
        config_path = (bundle_dir / expected_relative).resolve()
        if bundle_root not in config_path.parents:
            raise ProtocolValidationError(f"Registry config path escapes the bundle: {expected_relative}")
        config = load_json_object(config_path)
        if row.get("config_file_sha256") != protocol.file_sha256(config_path):
            raise ProtocolValidationError(f"Config file hash mismatch: {config_path}")
        hydrated.append(dict(row) | {"config": config})
    return dict(registry) | {"jobs": hydrated}


def _load_selection_artifacts(bundle_dir: Path) -> Dict[str, select_lambda.SelectionArtifact]:
    paths = [bundle_dir / "selection_records" / f"selection_{environment}.json" for environment in protocol.ENVIRONMENTS]
    try:
        return select_lambda.load_selection_record_files(paths)
    except select_lambda.SelectionValidationError as exc:
        raise ProtocolValidationError(str(exc)) from exc


def _validate_decision(
    bundle_dir: Path,
    selections: Mapping[str, select_lambda.SelectionArtifact],
    interim_registry: Mapping[str, object],
) -> Mapping[str, object]:
    path = bundle_dir / "interim_decision.json"
    try:
        try:
            from . import analyze_interim
        except ImportError:  # pragma: no cover - direct-script execution
            import analyze_interim  # type: ignore

        return analyze_interim.validate_decision_artifact(
            path,
            expected_registry_content_sha256=protocol.canonical_sha256(interim_registry),
            expected_selection_record_file_sha256={
                environment: selections[environment].file_sha256
                for environment in protocol.ENVIRONMENTS
            },
            require_continue=True,
        )
    except (ImportError, ValueError, RuntimeError) as exc:
        raise ProtocolValidationError(
            f"Interim decision failed independent evidence recomputation: {exc}"
        ) from exc


def check_generation_summary(bundle_dir: Path, generation_mode: str) -> dict:
    raw_screen_registry = load_json_object(bundle_dir / "screen_job_registry.json")
    expected = {
        "generation_mode": generation_mode,
        "template_count": len(protocol.ARMS),
        "screen_job_count": protocol.BUDGETS["screen_runs"],
        "formal_interim_job_count": (
            protocol.BUDGETS["formal_interim_jobs"]
            if generation_mode in {"formal_interim", "formal_extension"}
            else 0
        ),
        "formal_extension_job_count": (
            protocol.BUDGETS["formal_extension_jobs"] if generation_mode == "formal_extension" else 0
        ),
        "budgets": dict(protocol.BUDGETS),
        "screen_registry_content_sha256": protocol.canonical_sha256(raw_screen_registry),
    }
    summary = load_json_object(bundle_dir / "generation_summary.json")
    if summary != expected:
        raise ProtocolValidationError("generation_summary.json is inconsistent with the locked bundle contents.")
    return summary


def check_budget_constants() -> dict:
    protocol.validate_protocol_constants()
    expected = {
        "screen_runs": 18,
        "formal_interim_jobs": 36,
        "formal_extension_jobs": 24,
        "formal_full_jobs": 60,
        "internal_stop_path_full_runs": 54,
        "internal_complete_path_full_runs": 78,
    }
    for key, value in expected.items():
        if protocol.BUDGETS[key] != value:
            raise ProtocolValidationError(f"Budget {key} must be {value}.")
    return dict(protocol.BUDGETS)


def check_bundle(bundle_dir: Path) -> dict:
    manifest_summary = check_bundle_manifest(bundle_dir)
    lock_summary = check_protocol_lock(bundle_dir / "protocol_lock.json")
    templates = {
        path.stem: load_json_object(path) for path in sorted((bundle_dir / "templates").glob("*.json"))
    }
    template_summary = check_templates(templates)
    screen_registry = _load_registry_with_configs(bundle_dir, "screen_job_registry.json")
    screen_summary = check_screen_job_registry(screen_registry)
    mode = manifest_summary["generation_mode"]
    generation_summary = check_generation_summary(bundle_dir, str(mode))
    summary = {
        "manifest": manifest_summary,
        "protocol_lock": lock_summary,
        "templates": template_summary,
        "screen": screen_summary,
        "generation_summary": generation_summary,
        "budgets": check_budget_constants(),
    }
    if mode in {"formal_interim", "formal_extension"}:
        selections = _load_selection_artifacts(bundle_dir)
        screen_registry_file_sha256 = protocol.file_sha256(bundle_dir / "screen_job_registry.json")
        screen_registry_content_sha256 = protocol.canonical_sha256(
            load_json_object(bundle_dir / "screen_job_registry.json")
        )
        for target, artifact in selections.items():
            bound = artifact.record["screen_registry"]
            assert isinstance(bound, Mapping)
            if (
                bound.get("file_sha256") != screen_registry_file_sha256
                or bound.get("content_sha256") != screen_registry_content_sha256
            ):
                raise ProtocolValidationError(
                    f"Selection record for {target} is not bound to this bundle's screen registry."
                )
        interim_registry = _load_registry_with_configs(bundle_dir, "formal_interim_job_registry.json")
        summary["formal_interim"] = check_formal_job_registry(
            interim_registry, stage="formal_interim", selections=selections
        )
        if mode == "formal_extension":
            # Decision binds the on-disk, non-hydrated registry content.
            raw_interim = load_json_object(bundle_dir / "formal_interim_job_registry.json")
            decision = _validate_decision(bundle_dir, selections, raw_interim)
            decision_file_sha256 = protocol.file_sha256(bundle_dir / "interim_decision.json")
            extension_registry = _load_registry_with_configs(bundle_dir, "formal_extension_job_registry.json")
            summary["formal_extension"] = check_formal_job_registry(
                extension_registry,
                stage="formal_extension",
                selections=selections,
                decision=decision,
                decision_file_sha256=decision_file_sha256,
            )
    elif mode != "screen":
        raise ProtocolValidationError(f"Unknown bundle generation_mode: {mode}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(check_bundle(args.bundle_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
