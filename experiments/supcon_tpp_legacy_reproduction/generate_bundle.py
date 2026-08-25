from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import protocol


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def data_config() -> Dict[str, Any]:
    env_files: Dict[str, Any] = {}
    for env_name in protocol.ENV_ORDER:
        records = protocol.DATA_FILES[env_name]
        env_files[env_name] = {
            "amp": str(records["amp"]["path"]).replace("\\", "/"),
            "phase": str(records["phase"]["path"]).replace("\\", "/"),
            "label": str(records["label"]["path"]).replace("\\", "/"),
        }
    return {
        "format": "env_npy_triplets",
        "unwrap_phase": True,
        "phase_unwrap_axis": -1,
        "env_files": env_files,
    }


def training_config(*, batch_size: int, sampler: str) -> Dict[str, Any]:
    value = copy.deepcopy(protocol.COMMON_OPTIMIZER)
    value["batch_size"] = batch_size
    value["sampler"] = sampler
    return value


def losses_config(lambda_supcon: float) -> Dict[str, Any]:
    if lambda_supcon == 0.0:
        # Match the historical CE/TPP resolved-config schema exactly.  The
        # omitted SupCon-only fields are not consulted when lambda is zero.
        return {
            "temperature": 0.2,
            "lambda_supcon": 0.0,
            "include_same_domain_same_class": True,
        }
    value = copy.deepcopy(protocol.COMMON_LOSSES)
    value["lambda_supcon"] = lambda_supcon
    return value


def config(
    *,
    experiment_name: str,
    model_name: str,
    output_subdir: str,
    batch_size: int,
    sampler: str,
    lambda_supcon: float,
    device: str,
    tpp: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    model = {"name": model_name, **copy.deepcopy(protocol.COMMON_MODEL)}
    if tpp:
        model.update(copy.deepcopy(dict(tpp)))
    return {
        "experiment_name": experiment_name,
        "mode": "dg_loeo",
        "target_env": "all",
        "device": device,
        "output_root": str((protocol.OUTPUT_ROOT / output_subdir).resolve()).replace("\\", "/"),
        "seed_list": list(protocol.SEEDS),
        "data": data_config(),
        "split": {
            "val_ratio": 0.2,
            "test_ratio": 0.2,
        },
        "model": model,
        "training": training_config(batch_size=batch_size, sampler=sampler),
        "losses": losses_config(lambda_supcon),
        "evaluation": {"save_confusion_matrix": True},
    }


def serialise_reference_cells() -> Dict[str, Any]:
    return {
        f"{target}/seed_{seed}": dict(metrics)
        for (target, seed), metrics in sorted(protocol.SUPCON_REFERENCE_CELLS.items())
    }


def serialise_split_hashes() -> Dict[str, Any]:
    return {
        f"{target}/seed_{seed}": dict(hashes)
        for (target, seed), hashes in sorted(protocol.RECONSTRUCTED_SPLIT_HASHES.items())
    }


def main() -> None:
    generated = protocol.GENERATED_ROOT
    experiment_root = Path(__file__).resolve().parent
    configs = {
        "anchor_ce": config(
            experiment_name="legacy_anchor_ce_b08_none",
            model_name="pa_csi_dg_lite",
            output_subdir="anchor_ce",
            batch_size=8,
            sampler="none",
            lambda_supcon=0.0,
            device="cuda",
        ),
        "anchor_supcon": config(
            experiment_name="legacy_anchor_supcon_lam0050_bs32_gci",
            model_name="pa_csi_dg_lite",
            output_subdir="anchor_supcon",
            batch_size=32,
            sampler="domain_class_balanced",
            lambda_supcon=0.05,
            device="cuda:0",
        ),
        "anchor_tpp_l124": config(
            experiment_name="legacy_anchor_tpp_l124_b08_none",
            model_name="isolated_pa_csi_dg_lite_tpp",
            output_subdir="anchor_tpp_l124",
            batch_size=8,
            sampler="none",
            lambda_supcon=0.0,
            device="cuda",
            tpp={
                "tpp_levels": [1, 2, 4],
            },
        ),
        "anchor_tpp_residual025": config(
            experiment_name="legacy_anchor_tpp_residual025_l124_b08_none",
            model_name="isolated_pa_csi_dg_lite_tpp",
            output_subdir="anchor_tpp_residual025",
            batch_size=8,
            sampler="none",
            lambda_supcon=0.0,
            device="cuda",
            tpp=protocol.TPP_RESIDUAL025_HISTORICAL_FIELDS,
        ),
    }

    config_records = []
    for config_id, payload in configs.items():
        path = generated / "anchor_configs" / f"{config_id}.json"
        save_json(path, payload)
        config_records.append(
            {
                "config_id": config_id,
                "path": str(path.resolve()).replace("\\", "/"),
                # Keep the byte-level file binding separate from the semantic
                # (canonical JSON) binding.  Pretty-printed JSON and canonical
                # JSON intentionally have different hashes.
                "file_sha256": sha256_file(path),
                "canonical_payload_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
                "runner": "train_dg.py" if payload["model"]["name"] == "pa_csi_dg_lite" else "train_tpp_isolated.py",
                "exact_historical_carrier": True,
                "jobs": len(protocol.TARGETS) * len(protocol.SEEDS),
            }
        )

    execution_scripts = (
        "protocol.py",
        "generate_bundle.py",
        "preflight.py",
        "run_anchors.py",
        "gpu_smoke.py",
        "checkpoint_replay.py",
    )
    gate_paths = {
        "gpu_smoke": protocol.OUTPUT_ROOT / "audit" / "gpu_smoke.json",
        "checkpoint_replay": protocol.OUTPUT_ROOT / "audit" / "checkpoint_replay.json",
    }
    for script_name in execution_scripts:
        if not (experiment_root / script_name).is_file():
            raise FileNotFoundError(f"Execution script is missing: {experiment_root / script_name}")
    gate_payloads = {}
    for gate_name, gate_path in gate_paths.items():
        if not gate_path.is_file():
            raise FileNotFoundError(f"Required gate artifact is missing: {gate_path}")
        gate_payloads[gate_name] = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate_payloads["gpu_smoke"].get("status") != "pass":
        raise RuntimeError("gpu_smoke gate has not passed")
    if gate_payloads["checkpoint_replay"].get("status") != "passed":
        raise RuntimeError("checkpoint_replay gate has not passed")

    execution_lock = {
        "schema_version": 1,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "scripts": {
            script_name: {
                "path": str((experiment_root / script_name).resolve()).replace("\\", "/"),
                "sha256": sha256_file(experiment_root / script_name),
            }
            for script_name in execution_scripts
        },
        "required_gate_artifacts": {
            gate_name: {
                "path": str(gate_path.resolve()).replace("\\", "/"),
                "sha256": sha256_file(gate_path),
                "required_status": "pass" if gate_name == "gpu_smoke" else "passed",
            }
            for gate_name, gate_path in gate_paths.items()
        },
        "constraints": {
            "training_may_start_only_after_all_gates_pass": True,
            "runner_must_match_self_hash": True,
            "attempts_are_append_only": True,
        },
    }
    save_json(generated / "execution_lock.json", execution_lock)

    lock = {
        "schema_version": 1,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "legacy_commit": protocol.LEGACY_COMMIT,
        "execution_lock": {
            "path": str((generated / "execution_lock.json").resolve()).replace("\\", "/"),
            "sha256": sha256_file(generated / "execution_lock.json"),
        },
        "clean_worktree": str(protocol.DEFAULT_CLEAN_WORKTREE).replace("\\", "/"),
        "environment_order": list(protocol.ENV_ORDER),
        "targets": list(protocol.TARGETS),
        "seeds": list(protocol.SEEDS),
        "data_files": {
            env: {
                kind: {
                    **{key: value for key, value in record.items() if key != "path"},
                    "path": str(record["path"]).replace("\\", "/"),
                }
                for kind, record in records.items()
            }
            for env, records in protocol.DATA_FILES.items()
        },
        "label_counts": {str(key): value for key, value in protocol.LABEL_COUNTS.items()},
        "label_names": {str(key): value for key, value in protocol.LABEL_NAMES.items()},
        "expected_parameters": protocol.EXPECTED_PARAMETERS,
        "reference_aggregates": protocol.REFERENCE_AGGREGATES,
        "reference_provenance": protocol.REFERENCE_PROVENANCE,
        "unversioned_tpp_snapshot": protocol.UNVERSIONED_TPP_SNAPSHOT,
        "legacy_resolved_config_references": protocol.LEGACY_RESOLVED_CONFIG_REFERENCES,
        "expanded_effective_tpp_residual025": protocol.TPP_RESIDUAL025,
        "supcon_reference_cells": serialise_reference_cells(),
        "reconstructed_split_hashes": serialise_split_hashes(),
        "reproduction_tolerance": protocol.REPRODUCTION_TOLERANCE,
        "factorial_success": protocol.FACTORIAL_SUCCESS,
        "anchor_configs": config_records,
        "semantic_constraints": {
            "anchor_supcon": "batch32, domain_class_balanced, 30000 optimizer steps",
            "anchor_ce_tpp": "batch8, sampler none, 120000 optimizer steps",
            "factorial_name": "Factorial-S32",
            "factorial_tpp_interpretation": "legacy TPP architecture under the SupCon carrier; not an exact Anchor-T replay",
            "historical_phase_only": "unwrap_phase=true on flattened feature axis -1 is retained only for compatibility replay",
            "target_use": "target metrics may validate a preregistered replay but may not choose parameters",
        },
    }
    save_json(generated / "protocol_lock.json", lock)

    registry = {
        "schema_version": 1,
        "protocol_lock": str((generated / "protocol_lock.json").resolve()).replace("\\", "/"),
        "protocol_lock_sha256": sha256_file(generated / "protocol_lock.json"),
        "anchor_configs": config_records,
        "total_anchor_jobs": sum(record["jobs"] for record in config_records),
        "stage_order": [
            "anchor_preflight",
            "anchor_seed42_sentinel",
            "anchor_full_matrix",
            "factorial_s32_generation",
        ],
    }
    save_json(generated / "anchor_registry.json", registry)

    manifest_files = [
        generated / "protocol_lock.json",
        generated / "anchor_registry.json",
        generated / "execution_lock.json",
    ]
    manifest_files.extend(generated / "anchor_configs" / f"{config_id}.json" for config_id in configs)
    save_json(
        generated / "bundle_manifest.json",
        {
            "schema_version": 1,
            "files": {
                str(path.relative_to(generated)).replace("\\", "/"): sha256_file(path)
                for path in manifest_files
            },
        },
    )


if __name__ == "__main__":
    main()
