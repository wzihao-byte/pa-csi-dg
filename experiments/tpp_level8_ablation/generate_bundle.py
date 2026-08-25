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
LEGACY_EXPERIMENT = REPO_ROOT / "experiments" / "supcon_tpp_legacy_reproduction"
REFERENCE_BUNDLE = LEGACY_EXPERIMENT / "generated_factorial_s32_v1"
REFERENCE_OUTPUT = REPO_ROOT / "outputs_supcon_tpp_legacy_reproduction" / "factorial_s32"
REFERENCE_RESULTS = REFERENCE_OUTPUT / "factorial_results.json"
REFERENCE_AUDIT = (
    REPO_ROOT
    / "outputs_supcon_tpp_legacy_reproduction"
    / "audit"
    / "factorial_s32_completion.json"
)
SMOKE_PATH = REPO_ROOT / "outputs_tpp_level8_ablation" / "audit" / "gpu_smoke.json"
DEFAULT_OUTPUT = HERE / "generated_l1248_v1"
TARGETS = ("E1", "E2", "E3")
SEEDS = (42, 52, 62)
LEVELS = [1, 2, 4, 8]
EXPECTED_PARAMETERS = 4_786_294
ALLOWED_CONFIG_CHANGES = {
    "experiment_name",
    "job_id",
    "model.tpp_levels",
    "protocol.expected_parameter_count",
    "protocol.version",
}
SCRIPT_NAMES = (
    "generate_bundle.py",
    "train_factorial.py",
    "run_level8.py",
    "analyze_level8.py",
    "gpu_smoke.py",
)


class BundleError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if reference != candidate:
        return {prefix}
    return set()


def verify_reference_bundle() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = read_json(REFERENCE_BUNDLE / "bundle_manifest.json")
    protocol = read_json(REFERENCE_BUNDLE / "protocol_lock.json")
    registry = read_json(REFERENCE_BUNDLE / "factorial_job_registry.json")
    if sha256_file(REFERENCE_BUNDLE / "protocol_lock.json") != manifest["protocol_lock_sha256"]:
        raise BundleError("Reference protocol lock hash mismatch")
    if sha256_file(REFERENCE_BUNDLE / "factorial_job_registry.json") != manifest["registry_sha256"]:
        raise BundleError("Reference registry hash mismatch")
    if sha256_file(REFERENCE_BUNDLE / "execution_lock.json") != manifest["execution_lock_sha256"]:
        raise BundleError("Reference execution lock hash mismatch")
    if manifest.get("job_count") != 36 or registry.get("job_count") != 36:
        raise BundleError("Reference Factorial-S32 is not the complete 36-job bundle")
    audit = read_json(REFERENCE_AUDIT)
    if audit.get("status") != "passed" or audit.get("job_count") != 36:
        raise BundleError("Reference factorial completion audit did not pass")
    return manifest, protocol, registry


def effective_tpp(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tpp_levels": list(model.get("tpp_levels", [1, 2, 4])),
        "channel_tpp_levels": list(model.get("channel_tpp_levels", model.get("tpp_levels", [1, 2, 4]))),
        "tpp_pool_type": str(model.get("tpp_pool_type", "max")),
        "channel_tpp_pool_type": str(model.get("channel_tpp_pool_type", model.get("tpp_pool_type", "max"))),
        "tpp_preserve_global": bool(model.get("tpp_preserve_global", False)),
        "channel_tpp_preserve_global": bool(model.get("channel_tpp_preserve_global", model.get("tpp_preserve_global", False))),
        "tpp_pyramid_scale": float(model.get("tpp_pyramid_scale", 1.0)),
        "channel_tpp_pyramid_scale": float(model.get("channel_tpp_pyramid_scale", model.get("tpp_pyramid_scale", 1.0))),
    }


def generate(output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise BundleError(f"Refusing to overwrite existing bundle: {output}")
    reference_manifest, reference_protocol, reference_registry = verify_reference_bundle()
    smoke = read_json(SMOKE_PATH)
    if smoke.get("status") != "pass" or smoke.get("h11", {}).get("parameter_count") != EXPECTED_PARAMETERS:
        raise BundleError("Level-8 GPU smoke did not pass")
    if smoke.get("effective_tpp", {}).get("tpp_levels") != LEVELS:
        raise BundleError("GPU smoke level mismatch")
    for name in SCRIPT_NAMES:
        if not (HERE / name).is_file():
            raise BundleError(f"Missing experiment script: {name}")

    reference_results = read_json(REFERENCE_RESULTS)
    result_lookup = {
        (row["target_env"], int(row["seed"]), row["arm"]): row
        for row in reference_results["rows"]
    }
    registry_lookup = {
        (item["target_env"], int(item["seed"]), item["arm"]): item
        for item in reference_registry["jobs"]
    }
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        reference_rows = []
        jobs = []
        alignment_rows = []
        for target in TARGETS:
            for seed in SEEDS:
                key = (target, seed, "H11")
                source_item = registry_lookup[key]
                source_config_path = REFERENCE_BUNDLE / source_item["config_path"]
                reference_config = read_json(source_config_path)
                if effective_tpp(reference_config["model"]) != {
                    "tpp_levels": [1, 2, 4],
                    "channel_tpp_levels": [1, 2, 4],
                    "tpp_pool_type": "max",
                    "channel_tpp_pool_type": "max",
                    "tpp_preserve_global": True,
                    "channel_tpp_preserve_global": True,
                    "tpp_pyramid_scale": 0.25,
                    "channel_tpp_pyramid_scale": 0.25,
                }:
                    raise BundleError(f"Reference H11 TPP drift: {target}/seed_{seed}")

                result_row = result_lookup[key]
                attempt = Path(result_row["attempt_dir"]).resolve()
                completion_path = attempt / "attempt_completed.json"
                seal_path = attempt / "attempt_seal.json"
                completion = read_json(completion_path)
                seal = read_json(seal_path)
                if seal["completion_manifest_sha256"] != sha256_file(completion_path):
                    raise BundleError(f"Reference completion seal mismatch: {attempt}")
                run_dir = attempt / "payload" / "run"
                schedule = read_json(run_dir / "sampler_schedule.json")
                split = read_json(run_dir / "split_manifest.json")
                metrics = read_json(run_dir / "metrics.json")
                reference_id = f"reference_h11_l124__target_{target}__seed_{seed}"
                reference_rows.append({
                    "reference_id": reference_id,
                    "target_env": target,
                    "seed": seed,
                    "arm": "H11",
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
                })

                candidate = copy.deepcopy(reference_config)
                job_id = f"tpp_l1248__target_{target}__seed_{seed}__arm_H11"
                candidate["job_id"] = job_id
                candidate["experiment_name"] = job_id
                candidate["model"]["tpp_levels"] = copy.deepcopy(LEVELS)
                candidate["model"].pop("channel_tpp_levels", None)
                candidate["protocol"]["version"] = "tpp-level8-ablation/v1"
                candidate["protocol"]["expected_parameter_count"] = EXPECTED_PARAMETERS
                differences = changed_paths(reference_config, candidate)
                if differences != ALLOWED_CONFIG_CHANGES:
                    raise BundleError(
                        f"Config alignment failure {target}/seed_{seed}: {sorted(differences)}"
                    )
                if effective_tpp(candidate["model"])["channel_tpp_levels"] != LEVELS:
                    raise BundleError("Candidate channel path does not inherit level 8")
                relative = Path("configs") / f"{job_id}.json"
                candidate_path = staging / relative
                write_json(candidate_path, candidate)
                alignment_rows.append({
                    "job_id": job_id,
                    "reference_id": reference_id,
                    "changed_paths": sorted(differences),
                    "reference_effective_tpp": effective_tpp(reference_config["model"]),
                    "candidate_effective_tpp": effective_tpp(candidate["model"]),
                })
                jobs.append({
                    "job_id": job_id,
                    "target_env": target,
                    "source_envs": list(candidate["protocol"]["source_envs"]),
                    "seed": seed,
                    "arm": "H11",
                    "reference_id": reference_id,
                    "config_path": relative.as_posix(),
                    "config_file_sha256": sha256_file(candidate_path),
                    "config_payload_sha256": payload_sha256(candidate),
                })

        reference_registry_path = staging / "reference_registry.json"
        write_json(reference_registry_path, {
            "schema_version": "tpp-level8-reference-registry/v1",
            "reference_bundle_manifest_sha256": sha256_file(REFERENCE_BUNDLE / "bundle_manifest.json"),
            "reference_results_sha256": sha256_file(REFERENCE_RESULTS),
            "reference_completion_audit_sha256": sha256_file(REFERENCE_AUDIT),
            "rows": reference_rows,
            "row_count": len(reference_rows),
        })
        protocol_lock = {
            "schema_version": "tpp-level8-bundle/v1",
            "created_at_utc": utc_now(),
            "protocol_name": "h11_tpp_level8_ablation",
            "protocol_version": "1.0.0",
            "question": "effect of adding TPP level 8 on the locked H11 carrier",
            "targets": list(TARGETS),
            "seeds": list(SEEDS),
            "arm_order": ["H11"],
            "job_count": 9,
            "reference": {
                "name": "Factorial-S32 H11 L124",
                "bundle": str(REFERENCE_BUNDLE.resolve()),
                "bundle_manifest_sha256": sha256_file(REFERENCE_BUNDLE / "bundle_manifest.json"),
                "protocol_lock_sha256": reference_manifest["protocol_lock_sha256"],
                "results": str(REFERENCE_RESULTS.resolve()),
                "results_sha256": sha256_file(REFERENCE_RESULTS),
                "completion_audit": str(REFERENCE_AUDIT.resolve()),
                "completion_audit_sha256": sha256_file(REFERENCE_AUDIT),
            },
            "allowed_config_changes": sorted(ALLOWED_CONFIG_CHANGES),
            "alignment_audit": alignment_rows,
            "carrier": copy.deepcopy(reference_protocol["carrier"]),
            "supcon": copy.deepcopy(reference_protocol["supcon"]),
            "reference_effective_tpp": alignment_rows[0]["reference_effective_tpp"],
            "candidate_effective_tpp": alignment_rows[0]["candidate_effective_tpp"],
            "model_parameters": {
                "reference_l124": 3_721_334,
                "candidate_l1248": EXPECTED_PARAMETERS,
                "increment": EXPECTED_PARAMETERS - 3_721_334,
            },
            "checkpoint_rule": "source_val_accuracy_last_tied_best",
            "delayed_target_evaluation": True,
            "target_prediction_count": 3000,
            "cross_protocol_pairing": {
                "split_hashes_must_match": True,
                "sampler_schedule_sha256_must_match": True,
            },
            "provenance": copy.deepcopy(reference_protocol["provenance"]),
        }
        protocol_path = staging / "protocol_lock.json"
        write_json(protocol_path, protocol_lock)
        registry_path = staging / "level8_job_registry.json"
        write_json(registry_path, {
            "schema_version": "tpp-level8-bundle/v1",
            "protocol_lock_sha256": sha256_file(protocol_path),
            "reference_registry_sha256": sha256_file(reference_registry_path),
            "job_order": "target_then_seed",
            "jobs": jobs,
            "job_count": len(jobs),
        })
        scripts = {
            name: {"path": str((HERE / name).resolve()), "sha256": sha256_file(HERE / name)}
            for name in SCRIPT_NAMES
        }
        scripts["base_train_factorial.py"] = {
            "path": str((LEGACY_EXPERIMENT / "train_factorial.py").resolve()),
            "sha256": sha256_file(LEGACY_EXPERIMENT / "train_factorial.py"),
        }
        scripts["base_run_factorial.py"] = {
            "path": str((LEGACY_EXPERIMENT / "run_factorial.py").resolve()),
            "sha256": sha256_file(LEGACY_EXPERIMENT / "run_factorial.py"),
        }
        tpp_models = REPO_ROOT / "experiments" / "tpp_isolated" / "tpp_models.py"
        scripts["tpp_models.py"] = {"path": str(tpp_models.resolve()), "sha256": sha256_file(tpp_models)}
        execution_path = staging / "execution_lock.json"
        write_json(execution_path, {
            "schema_version": "tpp-level8-bundle/v1",
            "created_at_utc": utc_now(),
            "protocol_lock_sha256": sha256_file(protocol_path),
            "registry_sha256": sha256_file(registry_path),
            "reference_registry_sha256": sha256_file(reference_registry_path),
            "scripts": scripts,
            "required_gate_artifacts": {
                "reference_factorial_completion": {
                    "path": str(REFERENCE_AUDIT.resolve()),
                    "sha256": sha256_file(REFERENCE_AUDIT),
                    "required_status": "passed",
                },
                "level8_gpu_smoke": {
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
            },
        })
        manifest_path = staging / "bundle_manifest.json"
        write_json(manifest_path, {
            "schema_version": "tpp-level8-bundle/v1",
            "protocol_lock_sha256": sha256_file(protocol_path),
            "registry_sha256": sha256_file(registry_path),
            "reference_registry_sha256": sha256_file(reference_registry_path),
            "execution_lock_sha256": sha256_file(execution_path),
            "job_count": len(jobs),
        })
        if len(jobs) != 9 or len(reference_rows) != 9:
            raise BundleError("Expected exactly nine candidate and reference rows")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "status": "generated",
        "output": str(output),
        "candidate_jobs": 9,
        "reference_jobs": 9,
        "allowed_config_changes": sorted(ALLOWED_CONFIG_CHANGES),
    }, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the locked H11 L1248 ablation bundle")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
