"""Generate the append-only 36-job H11 L1248 residual-scale bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
REFERENCE_EXPERIMENT = REPO_ROOT / "experiments" / "tpp_level8_ablation"
REFERENCE_BUNDLE = REFERENCE_EXPERIMENT / "generated_l1248_v1"
REFERENCE_RESULTS = REPO_ROOT / "outputs_tpp_level8_ablation" / "l1248_v1" / "level8_results.json"
SMOKE_PATH = REPO_ROOT / "outputs_tpp_residual_scale_ablation" / "audit" / "gpu_smoke.json"
DEFAULT_OUTPUT = HERE / "generated_h11_l1248_v1"
TARGETS = ("E1", "E2", "E3")
SEEDS = (42, 52, 62)
SCALES = (0.125, 0.5, 0.75, 1.0)
REFERENCE_SCALE = 0.25
EXPECTED_PARAMETERS = 4_786_294
ALLOWED_CONFIG_CHANGES = {
    "experiment_name",
    "job_id",
    "model.tpp_pyramid_scale",
    "factorial.classifier_representation",
    "protocol.version",
}
SCRIPT_NAMES = (
    "generate_bundle.py",
    "train_factorial.py",
    "run_scale.py",
    "analyze_scale.py",
    "gpu_smoke.py",
    "pipeline.py",
)


class BundleError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def scale_tag(scale: float) -> str:
    return f"{int(round(scale * 1000)):04d}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BundleError(f"Missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BundleError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


def changed_paths(reference: Any, candidate: Any, prefix: str = "") -> set[str]:
    if isinstance(reference, dict) and isinstance(candidate, dict):
        result: set[str] = set()
        for key in sorted(set(reference) | set(candidate)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in reference or key not in candidate:
                result.add(path)
            else:
                result |= changed_paths(reference[key], candidate[key], path)
        return result
    return {prefix} if reference != candidate else set()


def verify_reference() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = read_json(REFERENCE_BUNDLE / "bundle_manifest.json")
    protocol = read_json(REFERENCE_BUNDLE / "protocol_lock.json")
    registry = read_json(REFERENCE_BUNDLE / "level8_job_registry.json")
    results = read_json(REFERENCE_RESULTS)
    if sha256_file(REFERENCE_BUNDLE / "protocol_lock.json") != manifest["protocol_lock_sha256"]:
        raise BundleError("Reference protocol lock hash mismatch")
    if sha256_file(REFERENCE_BUNDLE / "level8_job_registry.json") != manifest["registry_sha256"]:
        raise BundleError("Reference registry hash mismatch")
    if sha256_file(REFERENCE_BUNDLE / "reference_registry.json") != manifest["reference_registry_sha256"]:
        raise BundleError("Reference registry provenance mismatch")
    if sha256_file(REFERENCE_BUNDLE / "execution_lock.json") != manifest["execution_lock_sha256"]:
        raise BundleError("Reference execution lock hash mismatch")
    if registry.get("job_count") != 9 or manifest.get("job_count") != 9:
        raise BundleError("Reference L1248 bundle is not the sealed nine-job matrix")
    if results.get("status") != "complete" or results.get("job_count") != 9:
        raise BundleError("Reference L1248 results are incomplete")
    if results.get("bundle_manifest_sha256") != sha256_file(REFERENCE_BUNDLE / "bundle_manifest.json"):
        raise BundleError("Reference result/bundle binding mismatch")
    smoke = read_json(SMOKE_PATH)
    if smoke.get("status") != "pass" or tuple(smoke.get("scales", ())) != SCALES:
        raise BundleError("Residual-scale GPU smoke did not pass for all registered scales")
    for row in smoke.get("records", []):
        if int(row.get("parameter_count", -1)) != EXPECTED_PARAMETERS:
            raise BundleError("GPU smoke parameter count drifted")
    return manifest, protocol, registry, results


def generate(output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise BundleError(f"Refusing to overwrite existing bundle: {output}")
    reference_manifest, reference_protocol, reference_registry, reference_results = verify_reference()
    for name in SCRIPT_NAMES:
        if not (HERE / name).is_file():
            raise BundleError(f"Missing experiment script: {name}")

    registry_lookup = {
        (item["target_env"], int(item["seed"])): item for item in reference_registry["jobs"]
    }
    result_lookup = {
        (row["target_env"], int(row["seed"])): row for row in reference_results["rows"]
    }
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        reference_rows: list[dict[str, Any]] = []
        jobs: list[dict[str, Any]] = []
        alignment_rows: list[dict[str, Any]] = []
        for target in TARGETS:
            for seed in SEEDS:
                source_item = registry_lookup[(target, seed)]
                source_config_path = REFERENCE_BUNDLE / source_item["config_path"]
                reference_config = read_json(source_config_path)
                model = reference_config["model"]
                if model.get("tpp_levels") != [1, 2, 4, 8]:
                    raise BundleError(f"Reference levels drifted: {target}/seed_{seed}")
                if float(model.get("tpp_pyramid_scale")) != REFERENCE_SCALE:
                    raise BundleError(f"Reference scale drifted: {target}/seed_{seed}")
                if float(model.get("channel_tpp_pyramid_scale", REFERENCE_SCALE)) != REFERENCE_SCALE:
                    raise BundleError(f"Reference channel scale drifted: {target}/seed_{seed}")
                if reference_config["factorial"]["arm"] != "H11":
                    raise BundleError("Reference arm is not H11")
                result_row = result_lookup[(target, seed)]
                attempt = Path(result_row["attempt_dir"]).resolve()
                completion_path = attempt / "attempt_completed.json"
                seal_path = attempt / "attempt_seal.json"
                completion = read_json(completion_path)
                seal = read_json(seal_path)
                if completion.get("status") != "success":
                    raise BundleError(f"Reference attempt is not successful: {attempt}")
                if seal.get("completion_manifest_sha256") != sha256_file(completion_path):
                    raise BundleError(f"Reference completion seal mismatch: {attempt}")
                run_dir = attempt / "payload" / "run"
                schedule = read_json(run_dir / "sampler_schedule.json")
                split = read_json(run_dir / "split_manifest.json")
                metrics = read_json(run_dir / "metrics.json")
                reference_id = f"reference_h11_l1248_scale0250__target_{target}__seed_{seed}"
                reference_rows.append(
                    {
                        "reference_id": reference_id,
                        "target_env": target,
                        "seed": seed,
                        "arm": "H11",
                        "scale": REFERENCE_SCALE,
                        "source_config_path": str(source_config_path.resolve()),
                        "source_config_file_sha256": sha256_file(source_config_path),
                        "source_config_payload_sha256": payload_sha256(reference_config),
                        "attempt_dir": str(attempt),
                        "completion_manifest_sha256": sha256_file(completion_path),
                        "attempt_seal_sha256": sha256_file(seal_path),
                        "metrics_sha256": sha256_file(run_dir / "metrics.json"),
                        "checkpoint_sha256": metrics["checkpoint_sha256"],
                        "target_predictions_sha256": metrics["target_predictions_file_sha256"],
                        "sampler_schedule_sha256": schedule["global_sample_schedule_sha256"],
                        "split_hashes": {
                            "train": split["train"]["global_sample_id_sha256"],
                            "val": split["val"]["global_sample_id_sha256"],
                            "test": split["test"]["global_sample_id_sha256"],
                        },
                        "test_metrics": copy.deepcopy(metrics["test_metrics"]),
                        "per_class_metrics": copy.deepcopy(metrics["per_class_metrics"]),
                    }
                )

                for scale in SCALES:
                    tag = scale_tag(scale)
                    candidate = copy.deepcopy(reference_config)
                    job_id = f"h11_l1248_scale{tag}__target_{target}__seed_{seed}"
                    candidate["job_id"] = job_id
                    candidate["experiment_name"] = job_id
                    candidate["model"]["tpp_pyramid_scale"] = scale
                    candidate["protocol"]["version"] = "tpp-residual-scale-ablation/v1"
                    candidate["factorial"]["classifier_representation"] = f"dual_stream_residual{tag}_tpp"
                    differences = changed_paths(reference_config, candidate)
                    if differences != ALLOWED_CONFIG_CHANGES:
                        raise BundleError(
                            f"Config alignment failure {target}/seed_{seed}/scale_{scale}: {sorted(differences)}"
                        )
                    if float(candidate["model"].get("channel_tpp_pyramid_scale", scale)) != scale:
                        raise BundleError("Candidate channel path does not inherit the registered scale")
                    relative = Path("configs") / f"{job_id}.json"
                    candidate_path = staging / relative
                    write_json(candidate_path, candidate)
                    alignment_rows.append(
                        {
                            "job_id": job_id,
                            "reference_id": reference_id,
                            "scale": scale,
                            "changed_paths": sorted(differences),
                        }
                    )
                    jobs.append(
                        {
                            "job_id": job_id,
                            "target_env": target,
                            "source_envs": list(candidate["protocol"]["source_envs"]),
                            "seed": seed,
                            "arm": "H11",
                            "scale": scale,
                            "scale_tag": tag,
                            "reference_id": reference_id,
                            "config_path": relative.as_posix(),
                            "config_file_sha256": sha256_file(candidate_path),
                            "config_payload_sha256": payload_sha256(candidate),
                        }
                    )

        reference_registry_path = staging / "reference_registry.json"
        write_json(
            reference_registry_path,
            {
                "schema_version": "tpp-residual-scale-reference-registry/v1",
                "reference_bundle_manifest_sha256": sha256_file(REFERENCE_BUNDLE / "bundle_manifest.json"),
                "reference_results_sha256": sha256_file(REFERENCE_RESULTS),
                "rows": reference_rows,
                "row_count": len(reference_rows),
            },
        )
        protocol_lock = {
            "schema_version": "tpp-residual-scale-bundle/v1",
            "created_at_utc": utc_now(),
            "protocol_name": "h11_l1248_residual_scale_sensitivity",
            "protocol_version": "1.0.0",
            "question": "H11 L1248 sensitivity to the residual coefficient h=g+alpha*t",
            "post_hoc_exploratory": True,
            "targets": list(TARGETS),
            "seeds": list(SEEDS),
            "scales": list(SCALES),
            "reference_scale": REFERENCE_SCALE,
            "arm_order": ["H11"],
            "job_order": "target_then_seed_then_scale",
            "job_count": len(jobs),
            "reference": {
                "name": "sealed H11 L1248 residual scale 0.25",
                "bundle": str(REFERENCE_BUNDLE.resolve()),
                "bundle_manifest_sha256": sha256_file(REFERENCE_BUNDLE / "bundle_manifest.json"),
                "protocol_lock_sha256": reference_manifest["protocol_lock_sha256"],
                "results": str(REFERENCE_RESULTS.resolve()),
                "results_sha256": sha256_file(REFERENCE_RESULTS),
            },
            "allowed_config_changes": sorted(ALLOWED_CONFIG_CHANGES),
            "alignment_audit": alignment_rows,
            "carrier": copy.deepcopy(reference_protocol["carrier"]),
            "supcon": copy.deepcopy(reference_protocol["supcon"]),
            "effective_tpp": {
                "levels": [1, 2, 4, 8],
                "channel_levels": [1, 2, 4, 8],
                "pool_type": "max",
                "preserve_global": True,
                "candidate_scales": list(SCALES),
                "reference_scale": REFERENCE_SCALE,
            },
            "model_parameters": {"all_scales": EXPECTED_PARAMETERS},
            "checkpoint_rule": "source_val_accuracy_last_tied_best",
            "delayed_target_evaluation": True,
            "target_prediction_count": 3000,
            "pairing": {
                "split_hashes_must_match_reference": True,
                "sampler_schedule_must_match_reference": True,
            },
            "analysis": {
                "primary_metric": "target_macro_f1",
                "paired_reference": REFERENCE_SCALE,
                "bootstrap_seed": 20260809,
                "bootstrap_replicates": 100000,
                "exact_two_sided_sign_flip": True,
                "multiple_comparison_adjustment": "Holm across four scale-vs-0.25 contrasts",
                "no_confirmatory_best_scale_selection": True,
            },
            "provenance": copy.deepcopy(reference_protocol["provenance"]),
        }
        protocol_path = staging / "protocol_lock.json"
        write_json(protocol_path, protocol_lock)
        registry_path = staging / "scale_job_registry.json"
        write_json(
            registry_path,
            {
                "schema_version": "tpp-residual-scale-bundle/v1",
                "protocol_lock_sha256": sha256_file(protocol_path),
                "reference_registry_sha256": sha256_file(reference_registry_path),
                "job_order": "target_then_seed_then_scale",
                "jobs": jobs,
                "job_count": len(jobs),
            },
        )
        scripts = {
            name: {"path": str((HERE / name).resolve()), "sha256": sha256_file(HERE / name)}
            for name in SCRIPT_NAMES
        }
        legacy = REPO_ROOT / "experiments" / "supcon_tpp_legacy_reproduction"
        scripts["base_train_factorial.py"] = {
            "path": str((legacy / "train_factorial.py").resolve()),
            "sha256": sha256_file(legacy / "train_factorial.py"),
        }
        scripts["base_run_factorial.py"] = {
            "path": str((legacy / "run_factorial.py").resolve()),
            "sha256": sha256_file(legacy / "run_factorial.py"),
        }
        tpp_models = REPO_ROOT / "experiments" / "tpp_isolated" / "tpp_models.py"
        scripts["tpp_models.py"] = {"path": str(tpp_models.resolve()), "sha256": sha256_file(tpp_models)}
        execution_path = staging / "execution_lock.json"
        write_json(
            execution_path,
            {
                "schema_version": "tpp-residual-scale-bundle/v1",
                "created_at_utc": utc_now(),
                "protocol_lock_sha256": sha256_file(protocol_path),
                "registry_sha256": sha256_file(registry_path),
                "reference_registry_sha256": sha256_file(reference_registry_path),
                "scripts": scripts,
                "required_gate_artifacts": {
                    "reference_l1248_results": {
                        "path": str(REFERENCE_RESULTS.resolve()),
                        "sha256": sha256_file(REFERENCE_RESULTS),
                        "required_status": "complete",
                    },
                    "residual_scale_gpu_smoke": {
                        "path": str(SMOKE_PATH.resolve()),
                        "sha256": sha256_file(SMOKE_PATH),
                        "required_status": "pass",
                    },
                },
                "constraints": {
                    "attempts_append_only": True,
                    "single_runner_os_lock": True,
                    "cuda_device": "cuda:0",
                    "paired_reference_barrier_required": True,
                    "target_open_only_after_checkpoint_freeze": True,
                    "no_target_metric_tuning": True,
                    "no_automatic_restart_or_skip_after_failure": True,
                },
            },
        )
        manifest_path = staging / "bundle_manifest.json"
        write_json(
            manifest_path,
            {
                "schema_version": "tpp-residual-scale-bundle/v1",
                "protocol_lock_sha256": sha256_file(protocol_path),
                "registry_sha256": sha256_file(registry_path),
                "reference_registry_sha256": sha256_file(reference_registry_path),
                "execution_lock_sha256": sha256_file(execution_path),
                "job_count": len(jobs),
            },
        )
        if len(jobs) != 36 or len(reference_rows) != 9:
            raise BundleError("Expected exactly 36 candidate jobs and nine reference rows")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "generated",
                "output": str(output),
                "candidate_jobs": len(jobs),
                "reference_jobs": len(reference_rows),
                "scales": list(SCALES),
                "allowed_config_changes": sorted(ALLOWED_CONFIG_CHANGES),
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the locked H11 L1248 residual-scale bundle")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
