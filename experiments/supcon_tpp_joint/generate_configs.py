"""Generate fail-closed protocol artifacts; this module never starts training."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

try:  # Support module and direct-script execution.
    from . import analyze_interim, protocol, select_lambda
except ImportError:  # pragma: no cover
    import analyze_interim  # type: ignore
    import protocol  # type: ignore
    import select_lambda  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
JOINT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "pa_csi_dg_supcon_paper_aligned.json"
DEFAULT_OUTPUT_DIR = JOINT_ROOT / "generated_protocol"
BUNDLE_MANIFEST_NAME = "bundle_manifest.json"


class StaleOutputError(RuntimeError):
    pass


@dataclass(frozen=True)
class InterimDecisionArtifact:
    record: Mapping[str, object]
    path: Path
    file_sha256: str


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_or_copy_base(base_config: Optional[Mapping[str, object]]) -> dict:
    return load_json(DEFAULT_BASE_CONFIG) if base_config is None else copy.deepcopy(dict(base_config))


def _environment_catalog(base_config: Optional[Mapping[str, object]]) -> Dict[str, dict]:
    base = _load_or_copy_base(base_config)
    data = base.get("data")
    env_files = data.get("env_files") if isinstance(data, Mapping) else None
    if not isinstance(env_files, Mapping) or set(env_files) != set(protocol.ENVIRONMENTS):
        raise ValueError("Base config data.env_files must contain exactly E1/E2/E3.")
    catalog: Dict[str, dict] = {}
    for environment in protocol.ENVIRONMENTS:
        files = env_files[environment]
        if not isinstance(files, Mapping) or set(files) != {"amp", "phase", "label"}:
            raise ValueError(f"{environment} file declaration must contain amp/phase/label.")
        catalog[environment] = copy.deepcopy(dict(files))
    return catalog


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


def build_common_config(base_config: Optional[Mapping[str, object]] = None) -> dict:
    """Build a strict unresolved template with no environment available to a loader."""

    base = _load_or_copy_base(base_config)
    old_training = dict(base.get("training", {}))
    model = dict(base.get("model", {}))
    model.pop("antenna_layout", None)
    model["name"] = "supcon_tpp_joint"
    model["joint_temporal"] = {
        "factor": "__ARM_SPECIFIC__",
        "branch": "amplitude_only",
        "levels": list(protocol.TPP_LEVELS),
        "pool_type": protocol.TPP_POOL_TYPE,
        "preserve_global": protocol.TPP_PRESERVE_GLOBAL,
        "residual_scale": protocol.TPP_RESIDUAL_SCALE,
        "global_basis": _global_basis(),
    }

    return {
        "experiment_name": "__UNRESOLVED__",
        "mode": "dg_loeo",
        "device": base.get("device", "cuda"),
        "output_root": "outputs_supcon_tpp_joint",
        "target_env": "__FOLD_SPECIFIC__",
        "source_envs": [],
        "seed_list": [],
        "data": {
            "format": "env_npy_triplets",
            "unwrap_phase": False,
            "strict_group_protocol": True,
            "source_only": True,
            "loaded_envs": [],
            "env_files": {},
        },
        "target_data": {
            "target_env": "__FOLD_SPECIFIC__",
            "load_for_evaluation": False,
            "env_files": {},
        },
        "split": {
            "strict_group_protocol": True,
            "inner_validation": protocol.INNER_VALIDATION,
            "subject_block_size": protocol.GROUP_BLOCK_SIZE,
            "groups_per_env": protocol.GROUPS_PER_ENVIRONMENT,
            "train_groups_per_source_env": protocol.TRAIN_GROUPS_PER_SOURCE_ENVIRONMENT,
            "validation_groups_per_source_env": protocol.VALIDATION_GROUPS_PER_SOURCE_ENVIRONMENT,
            "validation_group_ids": {
                environment: list(groups) for environment, groups in protocol.VALIDATION_GROUP_IDS.items()
            },
            "split_seed": protocol.SPLIT_SEED,
        },
        "model": model,
        "training": {
            "epochs": protocol.EPOCHS,
            "batch_size": protocol.BATCH_SIZE,
            "samples_per_source_domain_per_batch": protocol.SAMPLES_PER_SOURCE_DOMAIN_PER_BATCH,
            "optimizer_steps_per_epoch": protocol.OPTIMIZER_STEPS_PER_EPOCH,
            "total_optimizer_steps": protocol.TOTAL_OPTIMIZER_STEPS,
            "lr": 0.001,
            "optimizer": "adam",
            "lr_scheduler": {
                "name": "exponential_decay",
                "update": "step",
                "decay_rate": 0.9,
                "decay_steps": 10000,
                "staircase": False,
            },
            "weight_decay": 0.0,
            "patience": protocol.EPOCHS,
            "early_stopping": False,
            "selection_metric": protocol.CHECKPOINT_METRIC,
            "selection_tie_break": "earlier_epoch",
            "sampler": protocol.SAMPLER,
            "num_workers": int(old_training.get("num_workers", 0)),
            "domain_balanced_batch": True,
            "grad_clip_norm": 0.0,
        },
        "losses": {
            "contrastive_loss_type": "pairwise_supcon",
            "temperature": protocol.SUPCON_TEMPERATURE,
            "lambda_supcon": None,
            "linear_warmup_steps": protocol.LINEAR_WARMUP_STEPS,
            "include_same_domain_same_class": True,
            "supcon_positive_mode": "all_views",
            "supcon_view_scope": protocol.SUPCON_VIEW_SCOPE,
            "supcon_enabled": None,
        },
        "evaluation": {
            "evaluate_target": False,
            "save_confusion_matrix": True,
            "save_per_sample_predictions": True,
        },
        "protocol": {
            "name": protocol.PROTOCOL_NAME,
            "version": protocol.PROTOCOL_VERSION,
            "lock_sha256": protocol.protocol_lock_sha256(),
            "stage": "template",
            "resolved": False,
            "job_id": None,
            "arm": "__ARM_SPECIFIC__",
            "temporal_factor": "__ARM_SPECIFIC__",
            "supcon_enabled": None,
            "target_env": "__FOLD_SPECIFIC__",
            "source_envs": [],
            "seed": None,
            "selected_lambda": None,
            "selection_record": None,
            "interim_decision": None,
            "phase_unwrap": False,
            "strict_group_protocol": True,
            "labels": _labels(),
            "screen_thresholds": dict(protocol.SCREEN_THRESHOLDS),
            "success_thresholds": dict(protocol.SUCCESS_THRESHOLDS),
            "futility_thresholds": dict(protocol.FUTILITY_THRESHOLDS),
        },
        "calibration": {
            "artifact_id": "__FOLD_SPECIFIC__",
            "uses_target_data": False,
            "initialization_seed": protocol.GLOBAL_BASIS_CALIBRATION["initialization_seed"],
        },
    }


def _apply_arm(config: MutableMapping[str, object], arm_code: str, value: Optional[float]) -> None:
    if arm_code not in protocol.ARMS:
        raise ValueError(f"Unknown arm: {arm_code}")
    arm = protocol.ARMS[arm_code]
    if arm.supcon_enabled:
        if value is not None and value <= 0:
            raise ValueError(f"{arm_code} requires positive lambda or unresolved None.")
        loss_value = value
    else:
        if value not in (None, 0, 0.0):
            raise ValueError(f"{arm_code} must use lambda=0.")
        loss_value = 0.0
    config["protocol"]["arm"] = arm_code  # type: ignore[index]
    config["protocol"]["temporal_factor"] = arm.temporal_factor  # type: ignore[index]
    config["protocol"]["supcon_enabled"] = arm.supcon_enabled  # type: ignore[index]
    config["model"]["joint_temporal"]["factor"] = arm.temporal_factor  # type: ignore[index]
    config["losses"]["supcon_enabled"] = arm.supcon_enabled  # type: ignore[index]
    config["losses"]["lambda_supcon"] = loss_value  # type: ignore[index]


def _bind_fold_data(
    config: MutableMapping[str, object],
    catalog: Mapping[str, Mapping[str, object]],
    target_env: str,
    *,
    evaluate_target: bool,
) -> None:
    sources = protocol.source_environments(target_env)
    config["target_env"] = target_env
    config["source_envs"] = list(sources)
    # The training loader is source-only in every stage.  Formal target files are
    # consumed lazily after checkpoint selection via the separate target_data block.
    config["data"]["source_only"] = True  # type: ignore[index]
    config["data"]["loaded_envs"] = list(sources)  # type: ignore[index]
    config["data"]["env_files"] = {source: copy.deepcopy(catalog[source]) for source in sources}  # type: ignore[index]
    config["target_data"] = {
        "target_env": target_env,
        "load_for_evaluation": evaluate_target,
        "env_files": copy.deepcopy(catalog[target_env]),
    }
    config["evaluation"]["evaluate_target"] = evaluate_target  # type: ignore[index]
    config["protocol"]["target_env"] = target_env  # type: ignore[index]
    config["protocol"]["source_envs"] = list(sources)  # type: ignore[index]
    config["calibration"]["artifact_id"] = f"global_basis_calibration_{target_env}"  # type: ignore[index]


def build_arm_templates(base_config: Optional[Mapping[str, object]] = None) -> Dict[str, dict]:
    common = build_common_config(base_config)
    templates: Dict[str, dict] = {}
    for arm_code in protocol.ARMS:
        config = copy.deepcopy(common)
        _apply_arm(config, arm_code, None)
        config["experiment_name"] = f"{protocol.PROTOCOL_NAME}_template_{arm_code}"
        templates[arm_code] = config
    return templates


def _lambda_token(value: float) -> str:
    return f"{round(value * 1000):04d}"


def _screen_values(arm_code: str) -> Sequence[float]:
    return protocol.LAMBDA_CANDIDATES if protocol.ARMS[arm_code].supcon_enabled else (0.0,)


def build_screen_job_registry(base_config: Optional[Mapping[str, object]] = None) -> List[dict]:
    common = build_common_config(base_config)
    catalog = _environment_catalog(base_config)
    jobs: List[dict] = []
    for target_env in protocol.ENVIRONMENTS:
        seed = protocol.SCREEN_SEEDS[target_env]
        for arm_code in protocol.ARMS:
            for value in _screen_values(arm_code):
                job_id = f"screen__target_{target_env}__arm_{arm_code}__lam{_lambda_token(value)}__seed_{seed}"
                config = copy.deepcopy(common)
                _apply_arm(config, arm_code, value)
                _bind_fold_data(config, catalog, target_env, evaluate_target=False)
                config["experiment_name"] = job_id
                config["seed_list"] = [seed]
                config["protocol"].update(
                    {
                        "stage": "screen",
                        "resolved": True,
                        "job_id": job_id,
                        "seed": seed,
                        "selected_lambda": value,
                    }
                )
                jobs.append(
                    {
                        "job_id": job_id,
                        "stage": "screen",
                        "target_env": target_env,
                        "source_envs": list(protocol.source_environments(target_env)),
                        "screen_seed": seed,
                        "seed": seed,
                        "arm": arm_code,
                        "lambda_supcon": value,
                        "config": config,
                    }
                )
    return jobs


def _selection_reference(artifact: select_lambda.SelectionArtifact) -> dict:
    return {
        "target_env": artifact.target_env,
        "selected_lambda": artifact.selected_lambda,
        "path": f"selection_records/selection_{artifact.target_env}.json",
        "file_sha256": artifact.file_sha256,
        "screen_registry_file_sha256": artifact.record["screen_registry"]["file_sha256"],  # type: ignore[index]
        "screen_results_file_sha256": artifact.record["screen_results"]["file_sha256"],  # type: ignore[index]
    }


def _build_formal_jobs(
    selections: Mapping[str, select_lambda.SelectionArtifact],
    stage: str,
    base_config: Optional[Mapping[str, object]],
    decision: Optional[InterimDecisionArtifact] = None,
) -> List[dict]:
    if set(selections) != set(protocol.ENVIRONMENTS):
        raise ValueError("Formal generation requires three validated selection records.")
    if stage == "formal_extension" and decision is None:
        raise ValueError("formal_extension requires a validated interim decision artifact.")
    if stage == "formal_interim" and decision is not None:
        raise ValueError("formal_interim must not reference an interim decision.")

    common = build_common_config(base_config)
    catalog = _environment_catalog(base_config)
    jobs: List[dict] = []
    for target_env in protocol.ENVIRONMENTS:
        artifact = selections[target_env]
        selection_ref = _selection_reference(artifact)
        for seed in protocol.formal_seeds(target_env, stage):
            for arm_code, arm in protocol.ARMS.items():
                job_id = f"{stage}__target_{target_env}__arm_{arm_code}__seed_{seed}"
                config = copy.deepcopy(common)
                loss_lambda = artifact.selected_lambda if arm.supcon_enabled else 0.0
                _apply_arm(config, arm_code, loss_lambda)
                _bind_fold_data(config, catalog, target_env, evaluate_target=True)
                config["experiment_name"] = job_id
                config["seed_list"] = [seed]
                interim_ref = None
                if decision is not None:
                    interim_ref = {
                        "path": "interim_decision.json",
                        "file_sha256": decision.file_sha256,
                        "decision": decision.record["decision"],
                        "formal_interim_registry_content_sha256": decision.record[
                            "formal_interim_registry_content_sha256"
                        ],
                    }
                config["protocol"].update(
                    {
                        "stage": stage,
                        "resolved": True,
                        "job_id": job_id,
                        "seed": seed,
                        "selected_lambda": artifact.selected_lambda,
                        "selection_record": selection_ref,
                        "interim_decision": interim_ref,
                    }
                )
                jobs.append(
                    {
                        "job_id": job_id,
                        "stage": stage,
                        "target_env": target_env,
                        "source_envs": list(protocol.source_environments(target_env)),
                        "seed": seed,
                        "arm": arm_code,
                        "lambda_supcon": loss_lambda,
                        "selected_lambda": artifact.selected_lambda,
                        "selection_record_file_sha256": artifact.file_sha256,
                        "interim_decision_file_sha256": decision.file_sha256 if decision else None,
                        "config": config,
                    }
                )
    return jobs


def build_formal_interim_job_registry(
    selections: Mapping[str, select_lambda.SelectionArtifact],
    base_config: Optional[Mapping[str, object]] = None,
) -> List[dict]:
    return _build_formal_jobs(selections, "formal_interim", base_config)


def build_formal_extension_job_registry(
    selections: Mapping[str, select_lambda.SelectionArtifact],
    decision: InterimDecisionArtifact,
    base_config: Optional[Mapping[str, object]] = None,
) -> List[dict]:
    return _build_formal_jobs(selections, "formal_extension", base_config, decision)


def _registry_payload(stage: str, jobs: Sequence[Mapping[str, object]], config_dir: str) -> dict:
    rows: List[dict] = []
    for job in jobs:
        config = job["config"]
        assert isinstance(config, Mapping)
        row = {key: value for key, value in job.items() if key != "config"}
        row["config_path"] = f"{config_dir}/{job['job_id']}.json"
        row["config_content_sha256"] = protocol.canonical_sha256(config)
        rows.append(row)
    return {
        "schema_version": protocol.REGISTRY_SCHEMA,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "stage": stage,
        "expected_job_count": len(jobs),
        "jobs": rows,
    }


def validate_interim_decision(
    decision_path: Path,
    selections: Mapping[str, select_lambda.SelectionArtifact],
    *,
    interim_registry_content_sha256: str,
) -> InterimDecisionArtifact:
    if not decision_path.is_file():
        raise ValueError(f"Interim decision artifact does not exist: {decision_path}")
    selection_hashes = {
        environment: selections[environment].file_sha256 for environment in protocol.ENVIRONMENTS
    }
    try:
        record = analyze_interim.validate_decision_artifact(
            decision_path,
            expected_registry_content_sha256=interim_registry_content_sha256,
            expected_selection_record_file_sha256=selection_hashes,
            require_continue=True,
        )
    except analyze_interim.InterimAnalysisError as exc:
        raise ValueError(f"Invalid interim decision: {exc}") from exc
    return InterimDecisionArtifact(record=record, path=decision_path.resolve(), file_sha256=protocol.file_sha256(decision_path))


def _safe_prepare_output(output_dir: Path, *, clean: bool) -> None:
    output = output_dir.resolve()
    if output in {REPO_ROOT.resolve(), JOINT_ROOT.resolve(), Path(output.anchor).resolve()}:
        raise StaleOutputError(f"Refusing unsafe output directory: {output}")
    if not output.exists():
        output.mkdir(parents=True)
        return
    if not output.is_dir():
        raise StaleOutputError(f"Output path is not a directory: {output}")
    existing = list(output.iterdir())
    if not existing:
        return
    if not clean:
        raise StaleOutputError(
            f"Output directory is non-empty: {output}. Re-run with --clean; stale artifacts are never reused."
        )

    manifest_path = output / BUNDLE_MANIFEST_NAME
    is_default = output == DEFAULT_OUTPUT_DIR.resolve()
    marker_valid = False
    if manifest_path.is_file():
        try:
            manifest = load_json(manifest_path)
            marker_valid = manifest.get("protocol_name") == protocol.PROTOCOL_NAME
        except (OSError, ValueError, json.JSONDecodeError):
            marker_valid = False
    if not (is_default or marker_valid):
        raise StaleOutputError(
            "Refusing to clean an unmarked custom directory. Remove it manually or first generate a valid bundle there."
        )
    shutil.rmtree(output)
    output.mkdir(parents=True)


def _write_registry(output_dir: Path, filename: str, config_dir: str, stage: str, jobs: Sequence[dict]) -> dict:
    for job in jobs:
        write_json(output_dir / config_dir / f"{job['job_id']}.json", job["config"])
    payload = _registry_payload(stage, jobs, config_dir)
    for row in payload["jobs"]:
        row["config_file_sha256"] = protocol.file_sha256(output_dir / str(row["config_path"]))
    write_json(output_dir / filename, payload)
    return payload


def _write_bundle_manifest(output_dir: Path, mode: str) -> dict:
    files = {
        path.relative_to(output_dir).as_posix(): protocol.file_sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != BUNDLE_MANIFEST_NAME
    }
    manifest = {
        "schema_version": protocol.BUNDLE_MANIFEST_SCHEMA,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "generation_mode": mode,
        "files": files,
    }
    write_json(output_dir / BUNDLE_MANIFEST_NAME, manifest)
    return manifest


def write_generation_bundle(
    output_dir: Path,
    *,
    base_config: Optional[Mapping[str, object]] = None,
    selection_record_paths: Optional[Sequence[Path]] = None,
    interim_decision_path: Optional[Path] = None,
    clean: bool = False,
) -> dict:
    """Write a screen, interim, or extension bundle with no stale-file reuse."""

    if interim_decision_path is not None and not selection_record_paths:
        raise ValueError("An interim decision requires all three selection records.")
    selections = (
        select_lambda.load_selection_record_files(selection_record_paths)
        if selection_record_paths is not None
        else None
    )
    # Read external inputs before an explicitly requested clean in case the user
    # accidentally placed an input under the output directory.
    if selections:
        output_resolved = output_dir.resolve()
        if any(output_resolved in artifact.path.parents for artifact in selections.values()):
            raise StaleOutputError("Selection records must live outside the generated bundle directory.")
    if interim_decision_path is not None and output_dir.resolve() in interim_decision_path.resolve().parents:
        raise StaleOutputError("Interim decision must live outside the generated bundle directory.")

    _safe_prepare_output(output_dir, clean=clean)
    write_json(output_dir / "protocol_lock.json", protocol.protocol_lock_payload())

    templates = build_arm_templates(base_config)
    for arm_code, config in templates.items():
        write_json(output_dir / "templates" / f"{arm_code}.json", config)
    screen_jobs = build_screen_job_registry(base_config)
    screen_registry = _write_registry(
        output_dir,
        "screen_job_registry.json",
        "screen_configs",
        "screen",
        screen_jobs,
    )

    mode = "screen"
    interim_count = 0
    extension_count = 0
    if selections is not None:
        mode = "formal_interim"
        generated_screen_registry_sha = protocol.file_sha256(output_dir / "screen_job_registry.json")
        for target_env, artifact in selections.items():
            bound_sha = artifact.record["screen_registry"]["file_sha256"]  # type: ignore[index]
            if bound_sha != generated_screen_registry_sha:
                raise ValueError(
                    f"Selection record for {target_env} is bound to a different screen registry."
                )
        (output_dir / "selection_records").mkdir(parents=True, exist_ok=True)
        for target_env, artifact in selections.items():
            shutil.copyfile(artifact.path, output_dir / "selection_records" / f"selection_{target_env}.json")
        interim_jobs = build_formal_interim_job_registry(selections, base_config)
        interim_registry = _write_registry(
            output_dir,
            "formal_interim_job_registry.json",
            "formal_interim_configs",
            "formal_interim",
            interim_jobs,
        )
        interim_count = len(interim_jobs)

        if interim_decision_path is not None:
            mode = "formal_extension"
            interim_registry_hash = protocol.canonical_sha256(interim_registry)
            decision = validate_interim_decision(
                interim_decision_path,
                selections,
                interim_registry_content_sha256=interim_registry_hash,
            )
            shutil.copyfile(decision.path, output_dir / "interim_decision.json")
            extension_jobs = build_formal_extension_job_registry(selections, decision, base_config)
            _write_registry(
                output_dir,
                "formal_extension_job_registry.json",
                "formal_extension_configs",
                "formal_extension",
                extension_jobs,
            )
            extension_count = len(extension_jobs)

    summary = {
        "generation_mode": mode,
        "template_count": len(templates),
        "screen_job_count": len(screen_jobs),
        "formal_interim_job_count": interim_count,
        "formal_extension_job_count": extension_count,
        "budgets": dict(protocol.BUDGETS),
        "screen_registry_content_sha256": protocol.canonical_sha256(screen_registry),
    }
    write_json(output_dir / "generation_summary.json", summary)
    _write_bundle_manifest(output_dir, mode)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--selection-records",
        nargs=3,
        type=Path,
        metavar=("E1_RECORD", "E2_RECORD", "E3_RECORD"),
        help="Three passed, hash-bound selection records; required for formal_interim generation.",
    )
    parser.add_argument(
        "--interim-decision",
        type=Path,
        help="A validated decision=continue artifact; requires --selection-records and enables extension generation.",
    )
    parser.add_argument("--clean", action="store_true", help="Delete only a safe prior generated bundle before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_generation_bundle(
        args.output_dir,
        base_config=load_json(args.base_config),
        selection_record_paths=args.selection_records,
        interim_decision_path=args.interim_decision,
        clean=args.clean,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
