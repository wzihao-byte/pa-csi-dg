from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import sklearn
import torch
from sklearn.model_selection import StratifiedShuffleSplit

import protocol


class AuditFailure(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_hash(values: Iterable[int]) -> str:
    array = np.asarray(list(values), dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def run(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def assert_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise AuditFailure(f"{context}: expected {expected!r}, got {actual!r}")


def verify_clean_worktree(path: Path) -> Dict[str, Any]:
    if not path.is_dir():
        raise AuditFailure(f"Clean worktree does not exist: {path}")
    head = run(["git", "rev-parse", "HEAD"], path)
    status = run(["git", "status", "--short"], path)
    assert_equal(head, protocol.LEGACY_COMMIT, "clean worktree commit")
    assert_equal(status, "", "clean worktree status")
    module_probe = run(
        [
            sys.executable,
            "-c",
            "import dg_dataset,dg_losses,dg_models,train_dg;"
            "print('|'.join([dg_dataset.__file__,dg_losses.__file__,dg_models.__file__,train_dg.__file__]))",
        ],
        path,
    )
    module_paths = [Path(value).resolve() for value in module_probe.split("|")]
    if any(path.resolve() not in module_path.parents for module_path in module_paths):
        raise AuditFailure(f"A legacy module resolved outside the clean worktree: {module_paths}")
    train_source = (path / "train_dg.py").read_text(encoding="utf-8")
    deterministic_tokens = (
        "random.seed(seed)",
        "np.random.seed(seed)",
        "torch.manual_seed(seed)",
        "torch.backends.cudnn.deterministic = True",
        "torch.backends.cudnn.benchmark = False",
    )
    deterministic_seed_setup = all(token in train_source for token in deterministic_tokens)
    checkpoint_tie_rule_verified = "current_val_score >= best_val_score" in train_source
    assert_equal(deterministic_seed_setup, True, "legacy deterministic seed setup")
    assert_equal(checkpoint_tie_rule_verified, True, "legacy checkpoint late-tie rule")
    return {
        "path": str(path.resolve()),
        "head": head,
        "status_clean": True,
        "module_paths": [str(value) for value in module_paths],
        "tracked_code_sha256": {
            name: sha256_file(path / name)
            for name in (
                "train_dg.py",
                "dg_dataset.py",
                "dg_models.py",
                "dg_losses.py",
                "adaptive_kmargin.py",
            )
        },
        "determinism": {
            "python_numpy_torch_seeded": deterministic_seed_setup,
            "checkpoint_tie_rule": "late_tie_ge",
            "checkpoint_tie_rule_verified": checkpoint_tie_rule_verified,
        },
    }


def verify_bundle(generated_root: Path) -> Dict[str, Any]:
    manifest_path = generated_root / "bundle_manifest.json"
    lock_path = generated_root / "protocol_lock.json"
    registry_path = generated_root / "anchor_registry.json"
    execution_lock_path = generated_root / "execution_lock.json"
    for required in (manifest_path, lock_path, registry_path, execution_lock_path):
        if not required.is_file():
            raise AuditFailure(f"Missing generated bundle file: {required}")

    manifest = read_json(manifest_path)
    expected_members = {
        "protocol_lock.json",
        "anchor_registry.json",
        "execution_lock.json",
        "anchor_configs/anchor_ce.json",
        "anchor_configs/anchor_supcon.json",
        "anchor_configs/anchor_tpp_l124.json",
        "anchor_configs/anchor_tpp_residual025.json",
    }
    assert_equal(set(manifest["files"]), expected_members, "bundle manifest members")
    file_hashes: Dict[str, str] = {}
    for relative, expected in manifest["files"].items():
        path = generated_root / relative
        if not path.is_file():
            raise AuditFailure(f"Bundle manifest member is missing: {path}")
        actual = sha256_file(path)
        assert_equal(actual, expected, f"bundle file hash {relative}")
        file_hashes[relative] = actual

    lock = read_json(lock_path)
    registry = read_json(registry_path)
    execution_lock = read_json(execution_lock_path)
    assert_equal(lock["protocol_name"], protocol.PROTOCOL_NAME, "protocol name")
    assert_equal(lock["protocol_version"], protocol.PROTOCOL_VERSION, "protocol version")
    assert_equal(lock["legacy_commit"], protocol.LEGACY_COMMIT, "legacy commit in lock")
    assert_equal(execution_lock["protocol_name"], protocol.PROTOCOL_NAME, "execution protocol name")
    assert_equal(execution_lock["protocol_version"], protocol.PROTOCOL_VERSION, "execution protocol version")
    assert_equal(Path(lock["execution_lock"]["path"]).resolve(), execution_lock_path.resolve(), "execution lock path")
    assert_equal(lock["execution_lock"]["sha256"], sha256_file(execution_lock_path), "execution lock hash")
    assert_equal(registry["total_anchor_jobs"], 36, "anchor job count")
    lock_hash = sha256_file(lock_path)
    assert_equal(Path(registry["protocol_lock"]).resolve(), lock_path.resolve(), "registry protocol lock path")
    assert_equal(registry["protocol_lock_sha256"], lock_hash, "registry protocol lock hash")

    registry_records = registry["anchor_configs"]
    lock_records = lock["anchor_configs"]
    assert_equal(registry_records, lock_records, "registry/lock config records")
    expected_ids = {
        "anchor_ce",
        "anchor_supcon",
        "anchor_tpp_l124",
        "anchor_tpp_residual025",
    }
    assert_equal({item["config_id"] for item in registry_records}, expected_ids, "anchor config ids")
    assert_equal(len(registry_records), len(expected_ids), "unique anchor config count")
    for item in registry_records:
        config_id = item["config_id"]
        expected_path = (generated_root / "anchor_configs" / f"{config_id}.json").resolve()
        actual_path = Path(item["path"]).resolve()
        assert_equal(actual_path, expected_path, f"registry config path {config_id}")
        payload = read_json(actual_path)
        assert_equal(sha256_file(actual_path), item["file_sha256"], f"registry file hash {config_id}")
        canonical_hash = hashlib.sha256(canonical_json(payload)).hexdigest()
        assert_equal(canonical_hash, item["canonical_payload_sha256"], f"registry payload hash {config_id}")

    experiment_root = Path(__file__).resolve().parent
    expected_scripts = {
        "protocol.py",
        "generate_bundle.py",
        "preflight.py",
        "run_anchors.py",
        "gpu_smoke.py",
        "checkpoint_replay.py",
    }
    assert_equal(set(execution_lock["scripts"]), expected_scripts, "execution script set")
    for script_name, record in execution_lock["scripts"].items():
        expected_path = (experiment_root / script_name).resolve()
        actual_path = Path(record["path"]).resolve()
        assert_equal(actual_path, expected_path, f"execution script path {script_name}")
        assert_equal(sha256_file(actual_path), record["sha256"], f"execution script hash {script_name}")

    expected_gates = {
        "gpu_smoke": (protocol.OUTPUT_ROOT / "audit" / "gpu_smoke.json", "pass"),
        "checkpoint_replay": (protocol.OUTPUT_ROOT / "audit" / "checkpoint_replay.json", "passed"),
    }
    assert_equal(set(execution_lock["required_gate_artifacts"]), set(expected_gates), "execution gate set")
    for gate_name, (expected_path, expected_status) in expected_gates.items():
        record = execution_lock["required_gate_artifacts"][gate_name]
        actual_path = Path(record["path"]).resolve()
        assert_equal(actual_path, expected_path.resolve(), f"gate path {gate_name}")
        assert_equal(sha256_file(actual_path), record["sha256"], f"gate hash {gate_name}")
        assert_equal(record["required_status"], expected_status, f"gate locked status {gate_name}")
        gate_payload = read_json(actual_path)
        assert_equal(gate_payload["status"], expected_status, f"gate actual status {gate_name}")
    gpu_gate = read_json(expected_gates["gpu_smoke"][0])
    assert_equal(gpu_gate["real_data_opened"], False, "GPU gate real-data isolation")
    assert_equal(gpu_gate["formal_training_started"], False, "GPU gate training isolation")
    assert_equal(gpu_gate["h11_batch32_v4"]["backward"]["completed"], True, "H11 backward gate")
    checkpoint_gate = read_json(expected_gates["checkpoint_replay"][0])
    assert_equal(
        checkpoint_gate["all_checkpoint_hashes_and_strict_loads_passed_before_target_open"],
        True,
        "checkpoint target-open gate",
    )
    assert_equal(checkpoint_gate["summary"]["passed"], 4, "checkpoint replay pass count")
    assert_equal(checkpoint_gate["summary"]["training_started"], False, "checkpoint replay training isolation")
    return {
        "protocol_lock_sha256": lock_hash,
        "anchor_registry_sha256": sha256_file(registry_path),
        "manifest_sha256": sha256_file(manifest_path),
        "execution_lock_sha256": sha256_file(execution_lock_path),
        "members": file_hashes,
        "lock": lock,
        "registry": registry,
        "execution_lock": execution_lock,
    }


def verify_data(full_hash: bool) -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    hash_tasks: list[tuple[str, str, Path, str]] = []
    for env_name in protocol.ENV_ORDER:
        records[env_name] = {}
        for kind in ("amp", "phase", "label"):
            expected = protocol.DATA_FILES[env_name][kind]
            path = Path(expected["path"])
            if not path.is_file():
                raise AuditFailure(f"Missing data file: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            assert_equal(tuple(array.shape), tuple(expected["shape"]), f"{env_name}/{kind} shape")
            assert_equal(str(array.dtype), expected["dtype"], f"{env_name}/{kind} dtype")
            item: Dict[str, Any] = {
                "path": str(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "size_bytes": path.stat().st_size,
            }
            if kind == "label":
                labels, counts = np.unique(np.asarray(array), return_counts=True)
                actual_counts = {int(label): int(count) for label, count in zip(labels, counts)}
                assert_equal(actual_counts, protocol.LABEL_COUNTS, f"{env_name} label counts")
                item["label_counts"] = {str(key): value for key, value in actual_counts.items()}
            records[env_name][kind] = item
            if full_hash:
                hash_tasks.append((env_name, kind, path, expected["sha256"]))

    if full_hash:
        def calculate(task: tuple[str, str, Path, str]) -> tuple[str, str, str, str]:
            env_name, kind, path, expected = task
            return env_name, kind, sha256_file(path), expected

        with ThreadPoolExecutor(max_workers=3) as executor:
            for env_name, kind, actual, expected in executor.map(calculate, hash_tasks):
                assert_equal(actual, expected, f"{env_name}/{kind} sha256")
                records[env_name][kind]["sha256"] = actual
    else:
        for env_name in protocol.ENV_ORDER:
            for kind in ("amp", "phase", "label"):
                records[env_name][kind]["sha256"] = "not_recomputed_quick_mode"
    return {"full_hash": full_hash, "files": records}


def safe_stratified_split(
    indices: np.ndarray,
    labels: np.ndarray,
    ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
    train_positions, validation_positions = next(splitter.split(np.zeros(len(indices)), labels))
    return np.sort(indices[train_positions]), np.sort(indices[validation_positions])


def verify_reconstructed_splits() -> Dict[str, Any]:
    labels_by_env = {
        env_name: np.load(protocol.DATA_FILES[env_name]["label"]["path"], allow_pickle=False)
        for env_name in protocol.ENV_ORDER
    }
    offsets = {env_name: index * 3000 for index, env_name in enumerate(protocol.ENV_ORDER)}
    records: Dict[str, Any] = {}
    for target in protocol.TARGETS:
        target_index = protocol.ENV_ORDER.index(target)
        test_indices = np.arange(offsets[target], offsets[target] + 3000, dtype=np.int64)
        for seed in protocol.SEEDS:
            train_parts = []
            val_parts = []
            for env_id, env_name in enumerate(protocol.ENV_ORDER):
                if env_name == target:
                    continue
                env_indices = np.arange(offsets[env_name], offsets[env_name] + 3000, dtype=np.int64)
                env_train, env_val = safe_stratified_split(
                    env_indices,
                    labels_by_env[env_name],
                    0.2,
                    seed + env_id + 1,
                )
                train_parts.append(env_train)
                val_parts.append(env_val)
            train_indices = np.sort(np.concatenate(train_parts))
            val_indices = np.sort(np.concatenate(val_parts))
            hashes = {
                "train": index_hash(train_indices),
                "val": index_hash(val_indices),
                "test": index_hash(test_indices),
            }
            expected = protocol.RECONSTRUCTED_SPLIT_HASHES[(target, seed)]
            assert_equal(hashes, expected, f"reconstructed split {target}/seed_{seed}")
            assert_equal((len(train_indices), len(val_indices), len(test_indices)), (4800, 1200, 3000), "split sizes")
            records[f"{target}/seed_{seed}"] = {
                **hashes,
                "counts": {"train": 4800, "val": 1200, "test": 3000},
                "target_env_index": target_index,
            }
    return records


def verify_snapshot_hashes() -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    for name, expected in protocol.UNVERSIONED_TPP_SNAPSHOT.items():
        path = Path(expected["path"])
        if not path.is_file():
            raise AuditFailure(f"Missing unversioned TPP snapshot member: {path}")
        actual = sha256_file(path)
        assert_equal(actual, expected["sha256"], f"TPP snapshot {name}")
        records[name] = {"path": str(path), "sha256": actual}
    for reference_name in ("anchor_supcon",):
        record = protocol.REFERENCE_PROVENANCE[reference_name]
        for path_key, hash_key in (
            ("per_run_csv", "per_run_csv_sha256"),
            ("aggregate_csv", "aggregate_csv_sha256"),
        ):
            path = Path(record[path_key])
            actual = sha256_file(path)
            assert_equal(actual, record[hash_key], f"reference file {path.name}")
            records[path.name] = {"path": str(path), "sha256": actual}
    return records


def comparable_anchor_config(payload: Mapping[str, Any]) -> Dict[str, Any]:
    value = copy.deepcopy(dict(payload))
    for key in ("experiment_name", "target_env", "output_root", "seed_list", "device_actual"):
        value.pop(key, None)
    return value


def verify_configs(bundle: Mapping[str, Any], clean_worktree: Path) -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    tpp_dir = protocol.REPO_ROOT / "experiments" / "tpp_isolated"
    for item in bundle["registry"]["anchor_configs"]:
        config_id = item["config_id"]
        path = Path(item["path"])
        config = read_json(path)
        assert_equal(config["mode"], "dg_loeo", f"{config_id} mode")
        assert_equal(config["target_env"], "all", f"{config_id} target")
        assert_equal(config["seed_list"], list(protocol.SEEDS), f"{config_id} seeds")
        assert_equal(list(config["data"]["env_files"]), list(protocol.ENV_ORDER), f"{config_id} env order")
        assert_equal(config["data"]["unwrap_phase"], True, f"{config_id} unwrap")
        assert_equal(config["data"]["phase_unwrap_axis"], -1, f"{config_id} unwrap axis")
        assert_equal(config["split"], {"val_ratio": 0.2, "test_ratio": 0.2}, f"{config_id} split")
        assert_equal(config["training"]["epochs"], 200, f"{config_id} epochs")
        assert_equal(config["training"]["selection_metric"], "accuracy", f"{config_id} selection")
        assert_equal(config["training"]["patience"], 200, f"{config_id} patience")

        legacy_reference = protocol.LEGACY_RESOLVED_CONFIG_REFERENCES[config_id]
        legacy_path = Path(legacy_reference["path"])
        legacy_hash = sha256_file(legacy_path)
        assert_equal(legacy_hash, legacy_reference["sha256"], f"legacy resolved config hash {config_id}")
        legacy_config = read_json(legacy_path)
        assert_equal(
            comparable_anchor_config(config),
            comparable_anchor_config(legacy_config),
            f"historical resolved config semantics {config_id}",
        )

        model_name = config["model"]["name"]
        if config_id == "anchor_supcon":
            assert_equal((config["training"]["batch_size"], config["training"]["sampler"]), (32, "domain_class_balanced"), "SupCon carrier")
            assert_equal(config["losses"]["lambda_supcon"], 0.05, "SupCon lambda")
            assert_equal(config["losses"]["lambda_supcon_warmup_epochs"], 0, "SupCon warmup")
            assert_equal(config["losses"]["supcon_positive_mode"], "global_class_instance_views", "SupCon positive mode")
            steps_per_epoch = 150
        else:
            assert_equal((config["training"]["batch_size"], config["training"]["sampler"]), (8, "none"), f"{config_id} carrier")
            assert_equal(config["losses"]["lambda_supcon"], 0.0, f"{config_id} lambda")
            steps_per_epoch = 600

        if model_name == "isolated_pa_csi_dg_lite_tpp":
            model_import = (
                "import json,sys;from pathlib import Path;"
                f"sys.path.insert(0,{str(tpp_dir)!r});sys.path.insert(0,{str(clean_worktree)!r});"
                "from tpp_models import IsolatedPaCsiDGLiteTPP;"
                f"c=json.loads(Path({str(path)!r}).read_text());"
                "m=IsolatedPaCsiDGLiteTPP(90,6,c['model']);"
                "print(sum(p.numel() for p in m.parameters()))"
            )
        else:
            model_import = (
                "import json,sys;from pathlib import Path;"
                f"sys.path.insert(0,{str(clean_worktree)!r});"
                "from dg_models import PaCsiDGLite;"
                f"c=json.loads(Path({str(path)!r}).read_text());"
                "m=PaCsiDGLite(90,6,c['model']);"
                "print(sum(p.numel() for p in m.parameters()))"
            )
        parameters = int(run([sys.executable, "-c", model_import], clean_worktree))
        assert_equal(parameters, protocol.EXPECTED_PARAMETERS[model_name], f"{config_id} parameters")
        records[config_id] = {
            "path": str(path),
            "file_sha256": sha256_file(path),
            "model_name": model_name,
            "parameters": parameters,
            "batch_size": config["training"]["batch_size"],
            "sampler": config["training"]["sampler"],
            "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": 200 * steps_per_epoch,
            "sample_exposures": 200 * 4800,
            "initial_lr": config["training"]["lr"],
            "lr_after_final_step": config["training"]["lr"]
            * config["training"]["lr_scheduler"]["decay_rate"]
            ** ((200 * steps_per_epoch) / config["training"]["lr_scheduler"]["decay_steps"]),
            "supcon_views": 4 if config["losses"]["lambda_supcon"] > 0 else 1,
            "legacy_resolved_config": str(legacy_path),
            "legacy_resolved_config_sha256": legacy_hash,
            "historical_fields_equal": True,
        }
    return records


def runtime_record() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed preflight for legacy reproduction anchors.")
    parser.add_argument("--clean-worktree", type=Path, default=protocol.DEFAULT_CLEAN_WORKTREE)
    parser.add_argument("--quick", action="store_true", help="Skip recomputing the nine large data SHA256 values.")
    parser.add_argument(
        "--output",
        type=Path,
        default=protocol.OUTPUT_ROOT / "audit" / "anchor_preflight.json",
    )
    args = parser.parse_args()

    audit: Dict[str, Any] = {
        "schema_version": 1,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "status": "running",
        "runtime": runtime_record(),
    }
    try:
        audit["clean_worktree"] = verify_clean_worktree(args.clean_worktree.resolve())
        bundle = verify_bundle(protocol.GENERATED_ROOT)
        audit["bundle"] = {key: value for key, value in bundle.items() if key not in {"lock", "registry"}}
        audit["snapshot_hashes"] = verify_snapshot_hashes()
        audit["data"] = verify_data(full_hash=not args.quick)
        audit["reconstructed_splits"] = verify_reconstructed_splits()
        audit["configs"] = verify_configs(bundle, args.clean_worktree.resolve())
        audit["status"] = "passed"
    except Exception as error:
        audit["status"] = "failed"
        audit["error_type"] = type(error).__name__
        audit["error"] = str(error)
        save_json(args.output.resolve(), audit)
        raise
    save_json(args.output.resolve(), audit)
    print(json.dumps({"status": audit["status"], "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
