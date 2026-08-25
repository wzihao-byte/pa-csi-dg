from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import protocol


SCHEMA_VERSION = "factorial-s32-bundle/v1"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "generated_factorial_s32_v1"
ARMS = {
    "H00": {"tpp_on": False, "supcon_on": False, "lambda_supcon": 0.0},
    "H01": {"tpp_on": False, "supcon_on": True, "lambda_supcon": 0.05},
    "H10": {"tpp_on": True, "supcon_on": False, "lambda_supcon": 0.0},
    "H11": {"tpp_on": True, "supcon_on": True, "lambda_supcon": 0.05},
}
SEEDS = (42, 52, 62)
SCRIPT_NAMES = (
    "generate_factorial.py",
    "train_factorial.py",
    "run_factorial.py",
)
LEGACY_MODULE_NAMES = (
    "adaptive_kmargin.py",
    "dg_dataset.py",
    "dg_losses.py",
    "dg_models.py",
    "train_dg.py",
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
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise BundleError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def git_output(arguments: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def verify_clean_worktree() -> dict[str, Any]:
    root = protocol.DEFAULT_CLEAN_WORKTREE.resolve()
    if not root.is_dir():
        raise BundleError(f"Missing clean legacy worktree: {root}")
    commit = git_output(["rev-parse", "HEAD"], root)
    if commit != protocol.LEGACY_COMMIT:
        raise BundleError(f"Legacy worktree commit mismatch: {commit}")
    if git_output(["status", "--porcelain"], root):
        raise BundleError(f"Legacy worktree is dirty: {root}")
    modules = {}
    for name in LEGACY_MODULE_NAMES:
        path = root / name
        modules[name.removesuffix(".py")] = {"path": str(path), "sha256": sha256_file(path)}
    return {"path": str(root), "commit": commit, "modules": modules}


def file_record(env_name: str, kind: str) -> dict[str, Any]:
    record = protocol.DATA_FILES[env_name][kind]
    return {
        "path": str(Path(record["path"]).resolve()),
        "sha256": str(record["sha256"]),
        "shape": list(record["shape"]),
        "dtype": str(record["dtype"]),
    }


def env_data_record(env_name: str) -> dict[str, str]:
    return {
        "amp": str(Path(protocol.DATA_FILES[env_name]["amp"]["path"]).resolve()),
        "phase": str(Path(protocol.DATA_FILES[env_name]["phase"]["path"]).resolve()),
        "label": str(Path(protocol.DATA_FILES[env_name]["label"]["path"]).resolve()),
    }


def build_config(
    *,
    arm: str,
    target: str,
    seed: int,
    supcon_template: Mapping[str, Any],
    residual_template: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    definition = ARMS[arm]
    sources = [env for env in protocol.ENV_ORDER if env != target]
    config = copy.deepcopy(dict(supcon_template))
    job_id = f"factorial_s32__target_{target}__seed_{seed}__arm_{arm}"
    config["job_id"] = job_id
    config["experiment_name"] = job_id
    config["target_env"] = target
    config["seed_list"] = [seed]
    config["device"] = "cuda:0"
    config.pop("output_root", None)
    config["output_dir"] = "__ATTEMPT_OUTPUT_DIR__"
    config["data"] = {
        "format": "env_npy_triplets",
        "unwrap_phase": True,
        "phase_unwrap_axis": -1,
        "env_files": {env: env_data_record(env) for env in sources},
    }
    config["source_file_records"] = {
        env: {kind: file_record(env, kind) for kind in ("amp", "phase", "label")}
        for env in sources
    }
    config["target_data"] = {
        "target_env": target,
        "data": {
            "format": "env_npy_triplets",
            "unwrap_phase": True,
            "phase_unwrap_axis": -1,
            "env_files": {target: env_data_record(target)},
        },
        "file_records": {
            kind: file_record(target, kind) for kind in ("amp", "phase", "label")
        },
    }
    config["model"] = copy.deepcopy(
        residual_template["model"] if definition["tpp_on"] else supcon_template["model"]
    )
    config["training"] = copy.deepcopy(supcon_template["training"])
    config["losses"] = copy.deepcopy(supcon_template["losses"])
    config["losses"]["lambda_supcon"] = float(definition["lambda_supcon"])
    config["factorial"] = {
        "name": "Factorial-S32",
        "arm": arm,
        "tpp_on": bool(definition["tpp_on"]),
        "supcon_on": bool(definition["supcon_on"]),
        "classifier_representation": (
            "dual_stream_residual025_tpp" if definition["tpp_on"] else "historical_global_pooling"
        ),
        "supcon_views": 4 if definition["supcon_on"] else 1,
        "lambda_supcon": float(definition["lambda_supcon"]),
        "fully_coupled_tpp_views": bool(definition["tpp_on"] and definition["supcon_on"]),
    }
    config["protocol"] = {
        "version": "factorial-s32/v1",
        "environment_order": list(protocol.ENV_ORDER),
        "target_env": target,
        "source_envs": sources,
        "seed": seed,
        "expected_split_hashes": copy.deepcopy(protocol.RECONSTRUCTED_SPLIT_HASHES[(target, seed)]),
        "optimizer_steps_per_epoch": 150,
        "epochs": 200,
        "total_optimizer_steps": 30000,
        "target_prediction_count": 3000,
        "target_access": "only_after_source_val_checkpoint_frozen_and_strict_reloaded",
        "checkpoint_rule": "source_val_accuracy_last_tied_best",
        "sampler_schedule_must_match_within_target_seed_quartet": True,
    }
    config["provenance"] = copy.deepcopy(dict(provenance))
    return config


def generate(output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise BundleError(f"Refusing to overwrite existing bundle: {output}")
    experiment_root = Path(__file__).resolve().parent
    for name in SCRIPT_NAMES:
        if not (experiment_root / name).is_file():
            raise BundleError(f"Required Factorial-S32 script is missing: {experiment_root / name}")

    clean = verify_clean_worktree()
    tpp_path = Path(protocol.UNVERSIONED_TPP_SNAPSHOT["tpp_models.py"]["path"]).resolve()
    if sha256_file(tpp_path) != protocol.UNVERSIONED_TPP_SNAPSHOT["tpp_models.py"]["sha256"]:
        raise BundleError("TPP model snapshot hash mismatch")
    anchor_audit = protocol.OUTPUT_ROOT / "audit" / "anchor_matrix_completion.json"
    gpu_smoke = protocol.OUTPUT_ROOT / "audit" / "gpu_smoke.json"
    for required in (anchor_audit, gpu_smoke):
        if not required.is_file():
            raise BundleError(f"Missing required gate artifact: {required}")
    if read_json(anchor_audit).get("status") != "passed":
        raise BundleError("Anchor completion audit did not pass")
    if read_json(gpu_smoke).get("status") != "pass":
        raise BundleError("H11 GPU smoke did not pass")

    generated_v1 = protocol.GENERATED_ROOT
    supcon_template = read_json(generated_v1 / "anchor_configs" / "anchor_supcon.json")
    residual_template = read_json(generated_v1 / "anchor_configs" / "anchor_tpp_residual025.json")
    provenance = {
        "clean_worktree": clean["path"],
        "legacy_commit": clean["commit"],
        "legacy_modules": clean["modules"],
        "tpp_snapshot_root": str(protocol.REPO_ROOT.resolve()),
        "tpp_models_sha256": sha256_file(tpp_path),
    }

    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        protocol_lock = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "protocol_name": "supcon_tpp_legacy_reproduction",
            "factorial_name": "Factorial-S32",
            "factorial_version": "1.0.0",
            "arms": copy.deepcopy(ARMS),
            "targets": list(protocol.TARGETS),
            "seeds": list(SEEDS),
            "job_count": 36,
            "carrier": {
                "batch_size": 32,
                "sampler": "domain_class_balanced",
                "epochs": 200,
                "optimizer_steps_per_epoch": 150,
                "total_optimizer_steps": 30000,
                "optimizer": "adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "scheduler": copy.deepcopy(supcon_template["training"]["lr_scheduler"]),
            },
            "supcon": {
                "temperature": 0.2,
                "lambda": 0.05,
                "warmup_epochs": 0,
                "positive_mode": "global_class_instance_views",
                "include_same_domain_same_class": True,
                "views": 4,
            },
            "tpp": {
                key: copy.deepcopy(value)
                for key, value in residual_template["model"].items()
                if key.startswith("tpp_") or key.startswith("channel_tpp_")
            },
            "model_parameters": {
                "pa_csi_dg_lite": 2_788_918,
                "isolated_pa_csi_dg_lite_tpp": 3_721_334,
            },
            "delayed_target_evaluation": True,
            "target_prediction_count": 3000,
            "success_criteria": copy.deepcopy(protocol.FACTORIAL_SUCCESS),
            "expected_split_hashes": {
                f"{target}/seed_{seed}": copy.deepcopy(protocol.RECONSTRUCTED_SPLIT_HASHES[(target, seed)])
                for target in protocol.TARGETS
                for seed in SEEDS
            },
            "data_file_records": {
                env: {kind: file_record(env, kind) for kind in ("amp", "phase", "label")}
                for env in protocol.ENV_ORDER
            },
            "provenance": provenance,
            "required_gate_artifacts": {
                "anchor_matrix_completion": {"path": str(anchor_audit), "sha256": sha256_file(anchor_audit)},
                "h11_gpu_smoke": {"path": str(gpu_smoke), "sha256": sha256_file(gpu_smoke)},
            },
            "performance_gate_policy": "historical reproduction tolerance is advisory-only by user authorization 2026-07-23",
        }
        protocol_path = staging / "protocol_lock.json"
        write_json(protocol_path, protocol_lock)

        jobs: list[dict[str, Any]] = []
        for target in protocol.TARGETS:
            for seed in SEEDS:
                for arm in ARMS:
                    config = build_config(
                        arm=arm,
                        target=target,
                        seed=seed,
                        supcon_template=supcon_template,
                        residual_template=residual_template,
                        provenance=provenance,
                    )
                    relative = Path("configs") / f"{config['job_id']}.json"
                    config_path = staging / relative
                    write_json(config_path, config)
                    jobs.append(
                        {
                            "job_id": config["job_id"],
                            "target_env": target,
                            "source_envs": list(config["protocol"]["source_envs"]),
                            "seed": seed,
                            "arm": arm,
                            "config_path": relative.as_posix(),
                            "config_file_sha256": sha256_file(config_path),
                            "config_payload_sha256": payload_sha256(config),
                        }
                    )
        registry = {
            "schema_version": SCHEMA_VERSION,
            "protocol_lock": "protocol_lock.json",
            "protocol_lock_sha256": sha256_file(protocol_path),
            "job_order": "target_then_seed_then_H00_H01_H10_H11",
            "quartet_barrier": True,
            "jobs": jobs,
            "job_count": len(jobs),
        }
        registry_path = staging / "factorial_job_registry.json"
        write_json(registry_path, registry)

        scripts = {
            name: {"path": str((experiment_root / name).resolve()), "sha256": sha256_file(experiment_root / name)}
            for name in SCRIPT_NAMES
        }
        execution_lock = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "scripts": scripts,
            "protocol_lock_sha256": sha256_file(protocol_path),
            "registry_sha256": sha256_file(registry_path),
            "required_gate_artifacts": protocol_lock["required_gate_artifacts"],
            "constraints": {
                "attempts_append_only": True,
                "single_runner_os_lock": True,
                "cuda_device": "cuda:0",
                "quartet_pairing_required": True,
                "target_open_only_after_checkpoint_freeze": True,
                "no_target_metric_tuning": True,
            },
        }
        execution_path = staging / "execution_lock.json"
        write_json(execution_path, execution_lock)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol_lock_sha256": sha256_file(protocol_path),
            "registry_sha256": sha256_file(registry_path),
            "execution_lock_sha256": sha256_file(execution_path),
            "job_count": len(jobs),
        }
        write_json(staging / "bundle_manifest.json", manifest)
        if len(jobs) != 36:
            raise BundleError(f"Expected 36 jobs, generated {len(jobs)}")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {"status": "generated", "output": str(output), "jobs": 36},
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the locked Factorial-S32 bundle.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args().output)
