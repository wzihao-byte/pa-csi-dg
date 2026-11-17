from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import shutil
import sys
import uuid
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
H11_EXPERIMENT = REPO_ROOT / "experiments" / "tpp_level8_ablation"
SOURCE = H11_EXPERIMENT / "generate_bundle.py"
REFERENCE_EXPERIMENT = REPO_ROOT / "experiments" / "supcon_tpp_legacy_reproduction"
REFERENCE_BUNDLE = REFERENCE_EXPERIMENT / "generated_factorial_s32_v1"
REFERENCE_OUTPUT = REPO_ROOT / "outputs_supcon_tpp_legacy_reproduction" / "factorial_s32"
REFERENCE_RESULTS = REFERENCE_OUTPUT / "factorial_results.json"
REFERENCE_AUDIT = REPO_ROOT / "outputs_supcon_tpp_legacy_reproduction" / "audit" / "factorial_s32_completion.json"
SMOKE_PATH = REPO_ROOT / "outputs_tpp_level8_ablation" / "audit" / "gpu_smoke.json"
DEFAULT_OUTPUT = HERE / "generated_l1248_h10_v1"
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
LOCAL_SCRIPTS = ("generate_bundle.py", "train_factorial.py", "run_level8.py", "analyze_level8.py")


def load_common():
    spec = importlib.util.spec_from_file_location("level8_h10_generate_common", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generator utilities: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


common = load_common()
BundleError = common.BundleError


def generate(output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise BundleError(f"Refusing to overwrite existing bundle: {output}")
    reference_manifest, reference_protocol, reference_registry = common.verify_reference_bundle()
    smoke = common.read_json(SMOKE_PATH)
    if smoke.get("status") != "pass" or smoke.get("h11", {}).get("parameter_count") != EXPECTED_PARAMETERS:
        raise BundleError("Level-8 architecture GPU smoke did not pass")
    if smoke.get("effective_tpp", {}).get("tpp_levels") != LEVELS:
        raise BundleError("GPU smoke level mismatch")
    for name in LOCAL_SCRIPTS:
        if not (HERE / name).is_file():
            raise BundleError(f"Missing experiment script: {name}")

    reference_results = common.read_json(REFERENCE_RESULTS)
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
                key = (target, seed, "H10")
                source_item = registry_lookup[key]
                source_config_path = REFERENCE_BUNDLE / source_item["config_path"]
                reference_config = common.read_json(source_config_path)
                expected_reference_tpp = {
                    "tpp_levels": [1, 2, 4],
                    "channel_tpp_levels": [1, 2, 4],
                    "tpp_pool_type": "max",
                    "channel_tpp_pool_type": "max",
                    "tpp_preserve_global": True,
                    "channel_tpp_preserve_global": True,
                    "tpp_pyramid_scale": 0.25,
                    "channel_tpp_pyramid_scale": 0.25,
                }
                if common.effective_tpp(reference_config["model"]) != expected_reference_tpp:
                    raise BundleError(f"Reference H10 TPP drift: {target}/seed_{seed}")
                if reference_config["factorial"]["arm"] != "H10":
                    raise BundleError(f"Reference arm drift: {target}/seed_{seed}")
                if reference_config["factorial"]["supcon_on"] is not False:
                    raise BundleError(f"Reference SupCon drift: {target}/seed_{seed}")

                result_row = result_lookup[key]
                attempt = Path(result_row["attempt_dir"]).resolve()
                completion_path = attempt / "attempt_completed.json"
                seal_path = attempt / "attempt_seal.json"
                seal = common.read_json(seal_path)
                if seal["completion_manifest_sha256"] != common.sha256_file(completion_path):
                    raise BundleError(f"Reference completion seal mismatch: {attempt}")
                run_dir = attempt / "payload" / "run"
                schedule = common.read_json(run_dir / "sampler_schedule.json")
                split = common.read_json(run_dir / "split_manifest.json")
                metrics = common.read_json(run_dir / "metrics.json")
                reference_id = f"reference_h10_l124__target_{target}__seed_{seed}"
                reference_rows.append({
                    "reference_id": reference_id,
                    "target_env": target,
                    "seed": seed,
                    "arm": "H10",
                    "source_config_path": str(source_config_path.resolve()),
                    "source_config_file_sha256": common.sha256_file(source_config_path),
                    "source_config_payload_sha256": common.payload_sha256(reference_config),
                    "attempt_dir": str(attempt),
                    "completion_manifest_sha256": common.sha256_file(completion_path),
                    "attempt_seal_sha256": common.sha256_file(seal_path),
                    "metrics_sha256": common.sha256_file(run_dir / "metrics.json"),
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
                job_id = f"tpp_l1248__target_{target}__seed_{seed}__arm_H10"
                candidate["job_id"] = job_id
                candidate["experiment_name"] = job_id
                candidate["model"]["tpp_levels"] = copy.deepcopy(LEVELS)
                candidate["model"].pop("channel_tpp_levels", None)
                candidate["protocol"]["version"] = "tpp-level8-h10-ablation/v1"
                candidate["protocol"]["expected_parameter_count"] = EXPECTED_PARAMETERS
                differences = common.changed_paths(reference_config, candidate)
                if differences != ALLOWED_CONFIG_CHANGES:
                    raise BundleError(f"Config alignment failure {target}/seed_{seed}: {sorted(differences)}")
                if common.effective_tpp(candidate["model"])["channel_tpp_levels"] != LEVELS:
                    raise BundleError("Candidate channel path does not inherit level 8")
                relative = Path("configs") / f"{job_id}.json"
                candidate_path = staging / relative
                common.write_json(candidate_path, candidate)
                alignment_rows.append({
                    "job_id": job_id,
                    "reference_id": reference_id,
                    "changed_paths": sorted(differences),
                    "reference_effective_tpp": common.effective_tpp(reference_config["model"]),
                    "candidate_effective_tpp": common.effective_tpp(candidate["model"]),
                })
                jobs.append({
                    "job_id": job_id,
                    "target_env": target,
                    "source_envs": list(candidate["protocol"]["source_envs"]),
                    "seed": seed,
                    "arm": "H10",
                    "reference_id": reference_id,
                    "config_path": relative.as_posix(),
                    "config_file_sha256": common.sha256_file(candidate_path),
                    "config_payload_sha256": common.payload_sha256(candidate),
                })

        reference_registry_path = staging / "reference_registry.json"
        common.write_json(reference_registry_path, {
            "schema_version": "tpp-level8-h10-reference-registry/v1",
            "reference_bundle_manifest_sha256": common.sha256_file(REFERENCE_BUNDLE / "bundle_manifest.json"),
            "reference_results_sha256": common.sha256_file(REFERENCE_RESULTS),
            "reference_completion_audit_sha256": common.sha256_file(REFERENCE_AUDIT),
            "rows": reference_rows,
            "row_count": len(reference_rows),
        })
        protocol_lock = {
            "schema_version": "tpp-level8-h10-bundle/v1",
            "created_at_utc": common.utc_now(),
            "protocol_name": "h10_tpp_level8_ablation",
            "protocol_version": "1.0.0",
            "question": "effect of adding TPP level 8 on the locked H10 carrier",
            "targets": list(TARGETS),
            "seeds": list(SEEDS),
            "arm_order": ["H10"],
            "job_count": 9,
            "reference": {
                "name": "Factorial-S32 H10 L124",
                "bundle": str(REFERENCE_BUNDLE.resolve()),
                "bundle_manifest_sha256": common.sha256_file(REFERENCE_BUNDLE / "bundle_manifest.json"),
                "protocol_lock_sha256": reference_manifest["protocol_lock_sha256"],
                "results": str(REFERENCE_RESULTS.resolve()),
                "results_sha256": common.sha256_file(REFERENCE_RESULTS),
                "completion_audit": str(REFERENCE_AUDIT.resolve()),
                "completion_audit_sha256": common.sha256_file(REFERENCE_AUDIT),
            },
            "allowed_config_changes": sorted(ALLOWED_CONFIG_CHANGES),
            "alignment_audit": alignment_rows,
            "carrier": copy.deepcopy(reference_protocol["carrier"]),
            "supcon": {"enabled": False, "lambda": 0.0, "views": 1},
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
        common.write_json(protocol_path, protocol_lock)
        registry_path = staging / "level8_job_registry.json"
        common.write_json(registry_path, {
            "schema_version": "tpp-level8-h10-bundle/v1",
            "protocol_lock_sha256": common.sha256_file(protocol_path),
            "reference_registry_sha256": common.sha256_file(reference_registry_path),
            "job_order": "target_then_seed",
            "jobs": jobs,
            "job_count": len(jobs),
        })

        scripts = {
            name: {"path": str((HERE / name).resolve()), "sha256": common.sha256_file(HERE / name)}
            for name in LOCAL_SCRIPTS
        }
        dependency_paths = {
            "h11_level8_runner.py": H11_EXPERIMENT / "run_level8.py",
            "h11_level8_analysis.py": H11_EXPERIMENT / "analyze_level8.py",
            "base_train_factorial.py": REFERENCE_EXPERIMENT / "train_factorial.py",
            "base_run_factorial.py": REFERENCE_EXPERIMENT / "run_factorial.py",
            "tpp_models.py": REPO_ROOT / "experiments" / "tpp_isolated" / "tpp_models.py",
        }
        for name, path in dependency_paths.items():
            scripts[name] = {"path": str(path.resolve()), "sha256": common.sha256_file(path)}
        execution_path = staging / "execution_lock.json"
        common.write_json(execution_path, {
            "schema_version": "tpp-level8-h10-bundle/v1",
            "created_at_utc": common.utc_now(),
            "protocol_lock_sha256": common.sha256_file(protocol_path),
            "registry_sha256": common.sha256_file(registry_path),
            "reference_registry_sha256": common.sha256_file(reference_registry_path),
            "scripts": scripts,
            "required_gate_artifacts": {
                "reference_factorial_completion": {
                    "path": str(REFERENCE_AUDIT.resolve()),
                    "sha256": common.sha256_file(REFERENCE_AUDIT),
                    "required_status": "passed",
                },
                "level8_gpu_smoke": {
                    "path": str(SMOKE_PATH.resolve()),
                    "sha256": common.sha256_file(SMOKE_PATH),
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
        common.write_json(manifest_path, {
            "schema_version": "tpp-level8-h10-bundle/v1",
            "protocol_lock_sha256": common.sha256_file(protocol_path),
            "registry_sha256": common.sha256_file(registry_path),
            "reference_registry_sha256": common.sha256_file(reference_registry_path),
            "execution_lock_sha256": common.sha256_file(execution_path),
            "job_count": len(jobs),
        })
        if len(jobs) != 9 or len(reference_rows) != 9:
            raise BundleError("Expected exactly nine candidate and reference rows")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(common.json.dumps({
        "status": "generated",
        "output": str(output),
        "candidate_jobs": 9,
        "reference_jobs": 9,
        "arm": "H10",
        "allowed_config_changes": sorted(ALLOWED_CONFIG_CHANGES),
    }, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the locked H10 L1248 ablation bundle")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()

