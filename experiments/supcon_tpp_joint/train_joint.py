"""Unified trainer for the locked SupCon x temporal-factor experiment.

Every arm uses this entry point.  Source arrays are loaded first and the outer
target is deliberately absent from the training dataset.  When target
evaluation is authorized, its arrays are opened only after the best
source-validation checkpoint has been selected.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dg_dataset import build_split_manifest, ensure_time_feature_layout, load_dataset, resolve_path
from dg_losses import DomainAwareSupConLoss, supcon_positive_pair_diagnostics
from experiments.supcon_tpp_joint import analyze_interim, protocol, screen_health, select_lambda
from experiments.supcon_tpp_joint.joint_models import (
    CALIBRATION_INITIALIZATION_SEED,
    CalibrationProvenance,
    GlobalBasisCalibrationArtifact,
    JointPaCsiDGLite,
    audit_joint_model_parameters,
    build_joint_arm_models,
    build_joint_model,
    calibrate_global_basis_control,
)
from experiments.supcon_tpp_joint.runner_utils import (
    DeterministicDomainClassBatchSampler,
    assert_phase_unwrap_disabled,
    canonical_json_hash,
    ce_supcon_gradient_diagnostics,
    count_trainable_parameters,
    filter_data_config_to_environments,
    fixed_label_metrics,
    sha256_int_sequence,
    trainable_parameter_hash,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GRADIENT_DIAGNOSTIC_INTERVAL = 5
CALIBRATION_MICROBATCH_SIZE = 16
SMOKE_SAMPLES_PER_SOURCE = 8
SMOKE_EPOCHS = 10
CALIBRATION_LOCK_TIMEOUT_SECONDS = 2 * 60 * 60
CALIBRATION_LOCK_POLL_SECONDS = 0.1
CALIBRATION_BINDING_SCHEMA_VERSION = 2


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, received {value!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, help="Smoke-only override; formal configs remain locked at 200.")
    parser.add_argument("--device", type=str)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evaluate-target", type=parse_bool)
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument("--calibration-only", action="store_true")
    return parser.parse_args()


def set_model_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but CUDA is unavailable.")
    return torch.device(requested)


def resolve_single_seed(config: Mapping[str, Any], override: Optional[int]) -> int:
    seeds = [int(seed) for seed in config.get("seed_list", [])]
    if len(seeds) != 1:
        raise ValueError(
            "A single runner invocation requires --seed unless config.seed_list contains exactly one value."
        )
    locked_seed = seeds[0]
    if override is not None and int(override) != locked_seed:
        raise ValueError(
            f"--seed cannot change a resolved protocol job: expected {locked_seed}, received {override}."
        )
    return locked_seed


def resolve_runtime_config(raw: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = copy.deepcopy(dict(raw))
    if args.epochs is not None:
        stage = str(raw.get("protocol", {}).get("stage", ""))
        if stage != "screen":
            raise ValueError("--epochs is permitted only for the registered Stage-0 screen smoke gate.")
        if int(args.epochs) != SMOKE_EPOCHS:
            raise ValueError(f"The Stage-0 smoke override is locked to {SMOKE_EPOCHS} epochs.")
        config.setdefault("training", {})["epochs"] = int(args.epochs)
        config.setdefault("protocol", {})["runtime_kind"] = "smoke_override"
    if args.device is not None:
        config["device"] = args.device
    if args.evaluate_target is not None:
        config.setdefault("evaluation", {})["evaluate_target"] = bool(args.evaluate_target)
    return config


def validate_runtime_config(raw: Mapping[str, Any], resolved: Mapping[str, Any]) -> None:
    assert_phase_unwrap_disabled(raw, resolved)
    protocol_config = resolved.get("protocol", {})
    if not isinstance(protocol_config, Mapping) or not bool(protocol_config.get("resolved")):
        raise ValueError("Only protocol-resolved screen/formal configs may enter the joint runner.")
    if protocol_config.get("name") != protocol.PROTOCOL_NAME:
        raise ValueError("Config protocol.name does not match the joint protocol.")
    if protocol_config.get("version") != protocol.PROTOCOL_VERSION:
        raise ValueError("Config protocol.version does not match the executing code.")
    if protocol_config.get("lock_sha256") != protocol.protocol_lock_sha256():
        raise ValueError("Config protocol lock hash does not match the executing protocol.")
    arm = str(protocol_config.get("arm"))
    if arm not in protocol.ARMS:
        raise ValueError(f"Unknown factorial arm {arm!r}.")
    target_env = str(resolved.get("target_env"))
    if target_env not in protocol.ENVIRONMENTS:
        raise ValueError(f"Unknown outer target {target_env!r}.")
    expected_sources = list(protocol.source_environments(target_env))
    if list(resolved.get("source_envs", [])) != expected_sources:
        raise ValueError(
            f"source_envs must be {expected_sources} for target {target_env}; "
            f"received {resolved.get('source_envs')}."
        )
    seeds = [int(value) for value in resolved.get("seed_list", [])]
    if len(seeds) != 1 or int(protocol_config.get("seed", -1)) != seeds[0]:
        raise ValueError("Resolved protocol.seed and the single seed_list entry must match exactly.")

    training = resolved.get("training", {})
    if int(training.get("batch_size", -1)) != protocol.BATCH_SIZE:
        raise ValueError("Joint protocol requires batch_size=16.")
    if int(training.get("optimizer_steps_per_epoch", -1)) != protocol.OPTIMIZER_STEPS_PER_EPOCH:
        raise ValueError("Joint protocol requires exactly 300 optimizer steps per epoch.")
    if str(training.get("sampler")) != protocol.SAMPLER:
        raise ValueError("Joint protocol requires the domain_class_balanced sampler.")
    if int(training.get("num_workers", 0)) != 0:
        raise ValueError("Stage-0 deterministic gate currently requires training.num_workers=0.")
    locked_training = {
        "optimizer": "adam",
        "lr": 0.001,
        "weight_decay": 0.0,
        "early_stopping": False,
        "selection_metric": protocol.CHECKPOINT_METRIC,
        "selection_tie_break": "earlier_epoch",
        "domain_balanced_batch": True,
        "samples_per_source_domain_per_batch": protocol.SAMPLES_PER_SOURCE_DOMAIN_PER_BATCH,
        "grad_clip_norm": 0.0,
    }
    for field, expected in locked_training.items():
        if training.get(field) != expected:
            raise ValueError(f"training.{field} must remain locked to {expected!r}.")
    scheduler = training.get("lr_scheduler", {})
    expected_scheduler = {
        "name": "exponential_decay",
        "decay_rate": 0.9,
        "decay_steps": 10_000,
        "staircase": False,
        "update": "step",
    }
    if not isinstance(scheduler, Mapping) or any(
        scheduler.get(field) != expected for field, expected in expected_scheduler.items()
    ):
        raise ValueError("training.lr_scheduler drifted from the locked step-wise exponential schedule.")

    losses = resolved.get("losses", {})
    if str(losses.get("supcon_view_scope")) != protocol.SUPCON_VIEW_SCOPE:
        raise ValueError("Only true global_only SupCon is permitted.")
    if float(losses.get("temperature", -1.0)) != protocol.SUPCON_TEMPERATURE:
        raise ValueError("SupCon temperature drifted from 0.2.")
    if int(losses.get("linear_warmup_steps", -1)) != protocol.LINEAR_WARMUP_STEPS:
        raise ValueError("SupCon warmup must remain locked to 6000 optimizer steps.")
    enabled = bool(protocol.ARMS[arm].supcon_enabled)
    if bool(losses.get("supcon_enabled")) != enabled:
        raise ValueError("Arm and losses.supcon_enabled disagree.")
    lambda_supcon = float(losses.get("lambda_supcon", 0.0))
    if enabled and lambda_supcon not in protocol.LAMBDA_CANDIDATES:
        raise ValueError(f"SupCon arm lambda must be one of {protocol.LAMBDA_CANDIDATES}.")
    if not enabled and lambda_supcon != 0.0:
        raise ValueError("CE-only arms require lambda_supcon=0.")

    stage = str(protocol_config.get("stage"))
    evaluate_target = bool(resolved.get("evaluation", {}).get("evaluate_target", False))
    if stage == "screen":
        if evaluate_target:
            raise ValueError("Screen jobs must be target-disabled.")
        if float(protocol_config.get("selected_lambda", -1.0)) != lambda_supcon:
            raise ValueError("Screen protocol.selected_lambda must equal losses.lambda_supcon.")
        runtime_kind = protocol_config.get("runtime_kind")
        expected_epochs = SMOKE_EPOCHS if runtime_kind == "smoke_override" else protocol.EPOCHS
        if int(training.get("epochs", -1)) != expected_epochs:
            raise ValueError(f"Screen epochs must be {expected_epochs} for runtime kind {runtime_kind!r}.")
    elif stage in {"formal_interim", "formal_extension"}:
        if not evaluate_target:
            raise ValueError("Formal jobs require deferred target evaluation.")
        selection_record = protocol_config.get("selection_record")
        if not isinstance(selection_record, Mapping) or not selection_record.get("file_sha256"):
            raise ValueError("Formal jobs require a hash-bound lambda selection record.")
        selected_lambda = float(protocol_config.get("selected_lambda", -1.0))
        if selected_lambda not in protocol.LAMBDA_CANDIDATES:
            raise ValueError("Formal protocol.selected_lambda is not a screened candidate.")
        if float(selection_record.get("selected_lambda", -1.0)) != selected_lambda:
            raise ValueError("Formal selection reference and protocol.selected_lambda disagree.")
        expected_loss_lambda = selected_lambda if enabled else 0.0
        if lambda_supcon != expected_loss_lambda:
            raise ValueError("Formal losses.lambda_supcon is inconsistent with the selected lambda and arm.")
        if int(training.get("epochs", -1)) != protocol.EPOCHS:
            raise ValueError("Formal jobs are locked to 200 epochs and cannot be overridden.")
        if int(training.get("total_optimizer_steps", -1)) != protocol.TOTAL_OPTIMIZER_STEPS:
            raise ValueError("Formal jobs are locked to exactly 60000 optimizer steps.")
        if stage == "formal_extension":
            interim = protocol_config.get("interim_decision")
            if not isinstance(interim, Mapping) or interim.get("decision") != "continue":
                raise ValueError("Formal extension requires a hash-bound continue decision.")
    else:
        raise ValueError(f"Unsupported executable protocol stage {stage!r}.")


def validate_bound_runtime_artifacts(config: Mapping[str, Any], config_path: Path) -> None:
    """Verify formal selection/decision files at execution time, not only generation time."""

    protocol_config = config["protocol"]
    stage = str(protocol_config["stage"])
    if stage == "screen":
        return
    bundle_root = config_path.resolve().parent.parent
    target_env = str(config["target_env"])
    selection_ref = protocol_config.get("selection_record")
    if not isinstance(selection_ref, Mapping):
        raise ValueError("Formal selection_record reference is missing.")
    expected_relative = Path("selection_records") / f"selection_{target_env}.json"
    if Path(str(selection_ref.get("path", ""))).as_posix() != expected_relative.as_posix():
        raise ValueError("Formal selection record path is not the canonical bundle-relative path.")
    selection_path = (bundle_root / expected_relative).resolve()
    try:
        selection_path.relative_to(bundle_root)
    except ValueError as exc:
        raise ValueError("Formal selection record escapes the protocol bundle.") from exc
    if not selection_path.is_file():
        raise FileNotFoundError(f"Formal selection record is absent: {selection_path}")
    file_digest = protocol.file_sha256(selection_path)
    if file_digest != selection_ref.get("file_sha256"):
        raise ValueError("Formal selection record file hash does not match its config binding.")
    selection_payload = load_json(selection_path)
    select_lambda.validate_selection_record(selection_payload, expected_target=target_env)
    selected_lambda = float(protocol_config["selected_lambda"])
    if float(selection_payload["selected_lambda"]) != selected_lambda:
        raise ValueError("Formal selected lambda does not match the validated selection artifact.")
    if selection_ref.get("screen_registry_file_sha256") != selection_payload["screen_registry"]["file_sha256"]:
        raise ValueError("Formal selection reference has the wrong screen registry hash.")
    if selection_ref.get("screen_results_file_sha256") != selection_payload["screen_results"]["file_sha256"]:
        raise ValueError("Formal selection reference has the wrong screen results hash.")

    if stage == "formal_extension":
        decision_ref = protocol_config.get("interim_decision")
        if not isinstance(decision_ref, Mapping) or decision_ref.get("path") != "interim_decision.json":
            raise ValueError("Formal extension has no canonical interim decision reference.")
        decision_path = (bundle_root / "interim_decision.json").resolve()
        if not decision_path.is_file() or protocol.file_sha256(decision_path) != decision_ref.get("file_sha256"):
            raise ValueError("Formal extension interim decision file/hash is invalid.")
        decision = load_json(decision_path)
        required = {
            "protocol_name": protocol.PROTOCOL_NAME,
            "protocol_version": protocol.PROTOCOL_VERSION,
            "protocol_lock_sha256": protocol.protocol_lock_sha256(),
            "decision": "continue",
        }
        if any(decision.get(field) != value for field, value in required.items()):
            raise ValueError("Formal extension interim decision content is invalid.")
        if decision_ref.get("formal_interim_registry_content_sha256") != decision.get(
            "formal_interim_registry_content_sha256"
        ):
            raise ValueError("Formal extension decision is bound to the wrong interim registry.")
        try:
            analyze_interim.validate_decision_artifact(
                decision_path,
                expected_registry_content_sha256=str(
                    decision_ref["formal_interim_registry_content_sha256"]
                ),
                require_continue=True,
            )
        except analyze_interim.InterimAnalysisError as exc:
            raise ValueError(
                f"Formal extension interim evidence failed independent runtime recomputation: {exc}"
            ) from exc


def resolve_source_data_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    source_envs = list(config["source_envs"])
    data_config = config.get("data", {})
    if not isinstance(data_config, Mapping):
        raise ValueError("Config data must be a mapping.")
    env_files = data_config.get("env_files", {})
    if not isinstance(env_files, Mapping):
        raise ValueError("data.env_files must be a mapping.")
    if set(env_files) == set(source_envs):
        filtered = copy.deepcopy(dict(data_config))
        filtered["env_files"] = {env: copy.deepcopy(env_files[env]) for env in source_envs}
    else:
        filtered = filter_data_config_to_environments(data_config, source_envs)
    if set(filtered["env_files"]) != set(source_envs):
        raise AssertionError("Source-only filter retained an unauthorized environment.")
    filtered["strict_group_protocol"] = True
    filtered["unwrap_phase"] = False
    return filtered


def resolve_target_data_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    target_env = str(config["target_env"])
    data_config = config.get("data", {})
    target_data = config.get("target_data", {})
    if not isinstance(data_config, Mapping) or not isinstance(target_data, Mapping):
        raise ValueError("Deferred target evaluation requires an isolated top-level target_data mapping.")
    source_files = data_config.get("env_files", {})
    if not isinstance(source_files, Mapping) or target_env in source_files:
        raise ValueError("The outer target must be absent from data.env_files at every stage.")
    if target_data.get("target_env") != target_env:
        raise ValueError("target_data.target_env does not match the outer target.")
    if target_data.get("load_for_evaluation") is not True:
        raise ValueError("Formal target_data.load_for_evaluation must be true.")
    deferred_files = target_data.get("env_files")
    if not isinstance(deferred_files, Mapping) or not {"amp", "phase", "label"}.issubset(deferred_files):
        raise ValueError(f"Deferred target_data.env_files is invalid for {target_env}.")
    target_group = deferred_files
    source_paths = {
        str(Path(str(path_value)).resolve()).lower()
        for group in source_files.values()
        if isinstance(group, Mapping)
        for path_value in group.values()
    }
    target_paths = {str(Path(str(path_value)).resolve()).lower() for path_value in target_group.values()}
    if source_paths & target_paths:
        raise ValueError("Source and deferred-target file paths overlap.")
    result = {
        key: copy.deepcopy(value)
        for key, value in data_config.items()
        if key not in {"env_files", "target_data", "loaded_envs"}
    }
    result.update(
        {
            "format": "env_npy_triplets",
            "env_files": {target_env: copy.deepcopy(target_group)},
            "strict_group_protocol": True,
            "unwrap_phase": False,
        }
    )
    return result


def resolve_group_mode(dataset: Any, split_config: Mapping[str, Any]) -> str:
    requested = str(split_config.get("inner_validation", "inferred_group_block_holdout"))
    if requested not in {"subject_grouped_holdout", "inferred_group_block_holdout"}:
        raise ValueError(f"Unsupported joint grouped mode {requested!r}.")
    if requested == "subject_grouped_holdout":
        if not bool(dataset.provenance.get("subject_sidecars_complete", False)):
            raise ValueError(
                "Resolved config requests subject_grouped_holdout, but verified bound sidecars are absent. "
                "Regenerate the protocol with inferred_group_block_holdout."
            )
    return requested


def build_source_only_split(
    dataset: Any,
    *,
    target_env: str,
    split_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Call the data-layer source-only API, with a fail-closed import error."""

    try:
        from dg_dataset import build_source_only_group_split
    except ImportError as exc:  # pragma: no cover - exercised only during partial installs.
        raise RuntimeError("dg_dataset.build_source_only_group_split is required.") from exc

    mode = resolve_group_mode(dataset, split_config)
    validation_groups = split_config.get("validation_group_ids")
    split_seed = int(split_config.get("split_seed", protocol.SPLIT_SEED))
    split = build_source_only_group_split(
        dataset,
        target_env=target_env,
        inner_validation=mode,
        validation_group_ids=validation_groups,
        split_seed=split_seed,
    )
    if split.get("test_indices"):
        raise AssertionError("Source-only split unexpectedly contains target indices.")
    return split


def _stable_sample_maps(dataset: Any) -> Tuple[Dict[int, int], Dict[int, str]]:
    sample_to_index: Dict[int, int] = {}
    index_to_key: Dict[int, str] = {}
    for env_id, env_name in enumerate(dataset.env_names):
        env_indices = np.where(dataset.env_ids == env_id)[0].tolist()
        for local_index, dataset_index in enumerate(env_indices):
            raw_sample_id = int(dataset.sample_ids[dataset_index])
            if raw_sample_id in sample_to_index:
                raise ValueError(f"Duplicate sample_id {raw_sample_id} in loaded dataset.")
            sample_to_index[raw_sample_id] = int(dataset_index)
            index_to_key[int(dataset_index)] = f"{env_name}:{local_index:04d}"
    return sample_to_index, index_to_key


def phase_input_audit(
    dataset: Any,
    data_config: Mapping[str, Any],
    *,
    base_dir: Path,
) -> Dict[str, Any]:
    """Prove that loader output equals float32 raw phase with no unwrap."""

    if data_config.get("unwrap_phase") is not False:
        raise ValueError("Phase audit requires unwrap_phase=false.")
    records: Dict[str, Any] = {}
    for env_id, env_name in enumerate(dataset.env_names):
        phase_path = resolve_path(base_dir, str(data_config["env_files"][env_name]["phase"]))
        raw = np.load(phase_path, mmap_mode="r", allow_pickle=True)
        expected = ensure_time_feature_layout(np.asarray(raw[:1]))[0]
        dataset_index = int(np.where(dataset.env_ids == env_id)[0][0])
        actual = np.asarray(dataset.phase[dataset_index])
        exact = bool(np.array_equal(expected, actual))
        max_abs_difference = float(np.max(np.abs(expected - actual)))
        if not exact:
            raise ValueError(
                f"Phase dataloader audit failed for {env_name}: max_abs_difference={max_abs_difference}."
            )
        records[env_name] = {
            "phase_path": str(phase_path),
            "raw_shape": list(raw.shape),
            "loaded_shape": list(actual.shape),
            "unwrap_applied": False,
            "exact_float32_match": exact,
            "max_abs_difference": max_abs_difference,
            "first_sample_sha256": hashlib.sha256(actual.tobytes(order="C")).hexdigest(),
        }
    return {
        "loaded_environments": list(dataset.env_names),
        "unwrap_phase": False,
        "records": records,
    }


def source_file_hashes(dataset: Any) -> Dict[str, str]:
    records = dataset.provenance.get("data_files", {})
    flattened: Dict[str, str] = {}
    for env_name in dataset.env_names:
        env_record = records.get(env_name, {})
        for kind in ("amp", "phase", "label"):
            digest = str(env_record.get(kind, {}).get("sha256", ""))
            if len(digest) != 64:
                raise ValueError(f"Missing locked SHA256 for {env_name}.{kind}.")
            flattened[f"{env_name}.{kind}"] = digest
    return flattened


def select_calibration_indices(
    dataset: Any,
    train_indices: Sequence[int],
) -> Tuple[List[int], List[int], Tuple[str, ...], Tuple[str, ...]]:
    train_set = {int(index) for index in train_indices}
    _, index_to_key = _stable_sample_maps(dataset)
    calibration_indices: List[int] = []
    smoke_indices: List[int] = []
    for env_id, env_name in enumerate(dataset.env_names):
        candidates = sorted(
            (
                index
                for index in train_set
                if int(dataset.env_ids[index]) == env_id
            ),
            key=lambda index: int(dataset.sample_ids[index]),
        )
        required = int(protocol.GLOBAL_BASIS_CALIBRATION["samples_per_source_environment"])
        if len(candidates) < required + SMOKE_SAMPLES_PER_SOURCE:
            raise ValueError(f"Source {env_name} lacks enough disjoint calibration/smoke samples.")
        calibration_indices.extend(candidates[:required])
        smoke_indices.extend(candidates[required : required + SMOKE_SAMPLES_PER_SOURCE])
    calibration_ids = tuple(index_to_key[index] for index in calibration_indices)
    smoke_ids = tuple(index_to_key[index] for index in smoke_indices)
    if len(set(calibration_ids)) != len(calibration_ids) or set(calibration_ids) & set(smoke_ids):
        raise AssertionError("Calibration and smoke IDs are not unique/disjoint.")
    return calibration_indices, smoke_indices, calibration_ids, smoke_ids


def _amplitude_microbatches(
    dataset: Any,
    indices: Sequence[int],
    device: torch.device,
) -> Iterable[torch.Tensor]:
    for start in range(0, len(indices), CALIBRATION_MICROBATCH_SIZE):
        selected = [int(index) for index in indices[start : start + CALIBRATION_MICROBATCH_SIZE]]
        amplitude = torch.from_numpy(np.asarray(dataset.amplitude[selected])).float().to(device)
        yield amplitude


def default_calibration_path(config: Mapping[str, Any], output_dir: Path) -> Path:
    artifact_id = str(config.get("calibration", {}).get("artifact_id", ""))
    if not artifact_id or artifact_id.startswith("__"):
        raise ValueError("Resolved config requires a concrete calibration.artifact_id.")
    output_root = Path(str(config.get("output_root", "outputs_supcon_tpp_joint")))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    # One artifact per outer fold is shared by every arm and model seed.
    return output_root / "calibration" / f"{artifact_id}.json"


@contextmanager
def exclusive_calibration_lock(
    lock_path: Path,
    *,
    timeout_seconds: float = CALIBRATION_LOCK_TIMEOUT_SECONDS,
    poll_seconds: float = CALIBRATION_LOCK_POLL_SECONDS,
) -> Iterable[None]:
    """Hold an artifact-scoped advisory lock that is released on process exit.

    A persistent lock file is intentional.  The operating system owns the
    actual byte lock, so an abnormal process exit releases it without requiring
    unsafe stale-file deletion by another worker.
    """

    if timeout_seconds <= 0.0 or poll_seconds <= 0.0:
        raise ValueError("Calibration lock timeout and poll interval must be positive.")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    acquired = False
    started = time.monotonic()
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())

        if os.name == "nt":
            import msvcrt

            while not acquired:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                except OSError as exc:
                    if time.monotonic() - started >= timeout_seconds:
                        raise TimeoutError(
                            f"Timed out waiting for calibration artifact lock {lock_path}."
                        ) from exc
                    time.sleep(poll_seconds)
        else:
            import fcntl

            while not acquired:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError as exc:
                    if time.monotonic() - started >= timeout_seconds:
                        raise TimeoutError(
                            f"Timed out waiting for calibration artifact lock {lock_path}."
                        ) from exc
                    time.sleep(poll_seconds)
        yield
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def calibration_code_sha256() -> Dict[str, str]:
    """Hash every repository source file that can change calibration bytes."""

    relative_paths = (
        Path("dg_dataset.py"),
        Path("dg_models.py"),
        Path("experiments/supcon_tpp_joint/joint_models.py"),
        Path("experiments/supcon_tpp_joint/protocol.py"),
        Path("experiments/supcon_tpp_joint/runner_utils.py"),
        Path("experiments/supcon_tpp_joint/train_joint.py"),
        Path("experiments/tpp_isolated/tpp_models.py"),
    )
    hashes: Dict[str, str] = {}
    for relative_path in relative_paths:
        source_path = REPO_ROOT / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Calibration code dependency is absent: {source_path}")
        hashes[relative_path.as_posix()] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return hashes


def expected_calibration_binding(
    calibration_model_config: Mapping[str, Any],
    provenance: CalibrationProvenance,
    *,
    initialization_sha256: str,
) -> Dict[str, Any]:
    if len(initialization_sha256) != 64:
        raise ValueError("Calibration initialization SHA256 must contain 64 hexadecimal characters.")
    return {
        "binding_schema_version": CALIBRATION_BINDING_SCHEMA_VERSION,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "calibration_initialization_seed": CALIBRATION_INITIALIZATION_SEED,
        "calibration_model_initialization_sha256": initialization_sha256,
        "calibration_model_config_sha256": canonical_json_hash(calibration_model_config),
        "calibration_code_sha256": calibration_code_sha256(),
        "source_index_sha256": provenance.source_index_sha256,
        "source_file_sha256": dict(provenance.source_file_sha256),
    }


def calibration_rebuild_guidance(artifact_path: Path, binding_path: Path) -> str:
    return (
        "Stop all jobs using this outer fold, then move or delete the artifact and binding together "
        f"({artifact_path}, {binding_path}) and rerun --calibration-only. "
        "Leave the persistent .lock file in place; it is not evidence of a live owner."
    )


def load_validated_calibration_pair(
    artifact_path: Path,
    binding_path: Path,
    provenance: CalibrationProvenance,
    expected_binding: Mapping[str, Any],
) -> Optional[GlobalBasisCalibrationArtifact]:
    """Load only a complete, hash-consistent artifact/binding pair."""

    artifact_exists = artifact_path.is_file()
    binding_exists = binding_path.is_file()
    if not artifact_exists and not binding_exists:
        return None
    if artifact_exists != binding_exists:
        raise ValueError(
            "Incomplete calibration artifact pair. "
            + calibration_rebuild_guidance(artifact_path, binding_path)
        )
    artifact = GlobalBasisCalibrationArtifact.load_json(artifact_path)
    if artifact.provenance != provenance:
        raise ValueError(
            f"Existing calibration artifact {artifact_path} is not bound to this exact source split. "
            + calibration_rebuild_guidance(artifact_path, binding_path)
        )
    binding = load_json(binding_path)
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if binding.get("artifact_sha256") != artifact_sha256:
        raise ValueError(
            "Calibration binding artifact hash does not match the artifact bytes. "
            + calibration_rebuild_guidance(artifact_path, binding_path)
        )
    if binding.get("artifact_path") != str(artifact_path.resolve()):
        raise ValueError(
            "Calibration binding points to a different artifact path. "
            + calibration_rebuild_guidance(artifact_path, binding_path)
        )
    commit_id = str(binding.get("pair_commit_id", ""))
    if len(commit_id) != 32 or any(character not in "0123456789abcdef" for character in commit_id):
        raise ValueError(
            "Calibration binding has no valid pair commit identifier. "
            + calibration_rebuild_guidance(artifact_path, binding_path)
        )
    if {key: binding.get(key) for key in expected_binding} != dict(expected_binding):
        raise ValueError(
            "Calibration artifact binding does not match code/model/source protocol. "
            + calibration_rebuild_guidance(artifact_path, binding_path)
        )
    return artifact


def _write_fsynced_unique_temp(destination: Path, payload: bytes, commit_id: str) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.{commit_id}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory sync; Windows does not expose a portable equivalent."""

    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def commit_calibration_pair(
    artifact_path: Path,
    binding_path: Path,
    artifact_payload: Mapping[str, Any],
    expected_binding: Mapping[str, Any],
) -> Dict[str, Any]:
    """Stage unique files, then publish artifact first and its commit marker second.

    Readers require the binding sidecar and verify its artifact hash while
    holding the same lock.  Therefore a crash between replacements cannot be
    mistaken for a valid pair.  A normal second-replace failure rolls the new
    artifact back when its bytes still match this transaction.
    """

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists() or binding_path.exists():
        raise FileExistsError("Refusing to overwrite an existing calibration artifact pair.")
    commit_id = uuid.uuid4().hex
    artifact_bytes = (
        json.dumps(dict(artifact_payload), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    binding_payload = {
        **dict(expected_binding),
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": artifact_sha256,
        "pair_commit_id": commit_id,
    }
    binding_bytes = (
        json.dumps(binding_payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    artifact_temporary = _write_fsynced_unique_temp(artifact_path, artifact_bytes, commit_id)
    binding_temporary: Optional[Path] = None
    artifact_published = False
    binding_published = False
    try:
        binding_temporary = _write_fsynced_unique_temp(binding_path, binding_bytes, commit_id)
        os.replace(artifact_temporary, artifact_path)
        artifact_published = True
        _fsync_directory(artifact_path.parent)
        os.replace(binding_temporary, binding_path)
        binding_published = True
        _fsync_directory(binding_path.parent)
    except BaseException:
        if artifact_published and not binding_published and artifact_path.is_file():
            if hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact_sha256:
                artifact_path.unlink()
                _fsync_directory(artifact_path.parent)
        raise
    finally:
        artifact_temporary.unlink(missing_ok=True)
        if binding_temporary is not None:
            binding_temporary.unlink(missing_ok=True)
    return binding_payload


def ensure_calibration_artifact(
    dataset: Any,
    split: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: torch.device,
    artifact_path: Path,
) -> GlobalBasisCalibrationArtifact:
    calibration_indices, smoke_indices, calibration_ids, smoke_ids = select_calibration_indices(
        dataset,
        split["train_indices"],
    )
    provenance = CalibrationProvenance(
        outer_target=str(config["target_env"]),
        source_environments=tuple(str(value) for value in config["source_envs"]),
        calibration_sample_ids=calibration_ids,
        smoke_sample_ids=smoke_ids,
        source_file_sha256=source_file_hashes(dataset),
        source_index_sha256=sha256_int_sequence(split["train_indices"]),
    )
    calibration_model_config = copy.deepcopy(dict(config["model"]))
    calibration_model_config["joint_temporal"] = copy.deepcopy(
        dict(calibration_model_config["joint_temporal"])
    )
    calibration_model_config["joint_temporal"]["factor"] = protocol.ARMS["Y00"].temporal_factor
    model = build_joint_model(
        "Y00",
        input_dim=dataset.input_dim,
        num_classes=len(protocol.LABELS),
        model_config=calibration_model_config,
        initialization_seed=CALIBRATION_INITIALIZATION_SEED,
    )
    initialization_sha256 = trainable_parameter_hash(model)
    expected_binding = expected_calibration_binding(
        calibration_model_config,
        provenance,
        initialization_sha256=initialization_sha256,
    )
    binding_path = artifact_path.with_suffix(artifact_path.suffix + ".binding.json")
    lock_path = artifact_path.with_suffix(artifact_path.suffix + ".lock")
    try:
        with exclusive_calibration_lock(lock_path):
            # Always recheck after acquiring the process lock: another worker
            # may have completed the exact artifact while this one was waiting.
            existing = load_validated_calibration_pair(
                artifact_path,
                binding_path,
                provenance,
                expected_binding,
            )
            if existing is not None:
                return existing

            model = model.to(device)
            _, index_to_key = _stable_sample_maps(dataset)
            source_train_allowlist = tuple(
                index_to_key[int(index)] for index in split["train_indices"]
            )
            artifact = calibrate_global_basis_control(
                model,
                _amplitude_microbatches(dataset, calibration_indices, device),
                _amplitude_microbatches(dataset, smoke_indices, device),
                provenance,
                allowed_calibration_sample_ids=source_train_allowlist,
                expected_calibration_sample_ids=calibration_ids,
                allowed_smoke_sample_ids=source_train_allowlist,
                expected_smoke_sample_ids=smoke_ids,
            )
            commit_calibration_pair(
                artifact_path,
                binding_path,
                artifact.to_dict(),
                expected_binding,
            )
            committed = load_validated_calibration_pair(
                artifact_path,
                binding_path,
                provenance,
                expected_binding,
            )
            if committed is None:
                raise AssertionError("Calibration pair disappeared before post-commit validation.")
            return committed
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def audit_batch_schedule(
    dataset: Any,
    sampler: DeterministicDomainClassBatchSampler,
) -> Dict[str, Any]:
    sampler.set_epoch(0)
    schedule = list(iter(sampler))
    if len(schedule) != protocol.OPTIMIZER_STEPS_PER_EPOCH:
        raise ValueError(
            f"Sampler yielded {len(schedule)} batches; expected {protocol.OPTIMIZER_STEPS_PER_EPOCH}."
        )
    _, index_to_key = _stable_sample_maps(dataset)
    covered_anchors = 0
    total_anchors = 0
    for batch in schedule:
        if len(batch) != protocol.BATCH_SIZE:
            raise ValueError(f"Sampler yielded batch size {len(batch)} instead of 16.")
        batch_labels = [int(dataset.labels[index]) for index in batch]
        batch_domains = [int(dataset.env_ids[index]) for index in batch]
        for domain in sampler.domains:
            positions = [offset for offset, value in enumerate(batch_domains) if value == domain]
            if len(positions) != protocol.SAMPLES_PER_SOURCE_DOMAIN_PER_BATCH:
                raise ValueError(f"Domain {domain} does not contribute exactly eight samples.")
            classes = {batch_labels[offset] for offset in positions}
            if classes != set(protocol.LABELS):
                raise ValueError(f"Domain {domain} batch class coverage is {sorted(classes)}, expected 0..5.")
        for offset, (label, domain) in enumerate(zip(batch_labels, batch_domains)):
            total_anchors += 1
            has_cross_domain_positive = any(
                other != offset
                and batch_labels[other] == label
                and batch_domains[other] != domain
                for other in range(len(batch))
            )
            covered_anchors += int(has_cross_domain_positive)
    coverage = covered_anchors / max(1, total_anchors)
    if coverage < float(protocol.SCREEN_THRESHOLDS["minimum_positive_anchor_fraction"]):
        raise ValueError(f"Cross-domain positive anchor coverage {coverage:.6f} is below the gate.")
    return {
        "sampler_seed": sampler.seed,
        "epoch": 0,
        "num_batches": len(schedule),
        "batch_size": protocol.BATCH_SIZE,
        "samples_per_source_domain": protocol.SAMPLES_PER_SOURCE_DOMAIN_PER_BATCH,
        "schedule_sha256": sampler.schedule_hash(),
        "first_batch_sample_ids": [index_to_key[int(index)] for index in schedule[0]],
        "cross_domain_positive_anchor_fraction": coverage,
    }


def build_loaders(
    dataset: Any,
    split: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    sampler_seed: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, DeterministicDomainClassBatchSampler]:
    training = config["training"]
    sampler = DeterministicDomainClassBatchSampler(
        dataset.indices_by_env(split["train_indices"]),
        dataset.labels,
        batch_size=int(training["batch_size"]),
        num_batches=int(training["optimizer_steps_per_epoch"]),
        seed=int(sampler_seed),
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(dataset, [int(index) for index in split["val_indices"]]),
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, sampler


def effective_lambda_at_step(target_lambda: float, completed_steps: int, warmup_steps: int) -> float:
    """Linear ramp used for the *next* optimizer step.

    Step 1 receives ``target / warmup_steps`` and step ``warmup_steps`` first
    reaches the target.  A CE arm always returns zero.
    """

    if target_lambda <= 0.0:
        return 0.0
    if warmup_steps <= 0:
        return float(target_lambda)
    next_step = int(completed_steps) + 1
    return float(target_lambda) * min(1.0, next_step / float(warmup_steps))


def _finite_gradient_l2_norm(model: torch.nn.Module) -> float:
    """Return global gradient L2 while checking finiteness in one traversal.

    Squaring makes any NaN/Inf contaminate the scalar sum, so this replaces the
    former elementwise finite pass without adding a second parameter traversal.
    The single ``item`` is also the only host synchronization in the audit.
    """

    squared_norm: Optional[torch.Tensor] = None
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        gradient_float = gradient.float()
        local_squared_norm = torch.sum(gradient_float * gradient_float)
        squared_norm = (
            local_squared_norm
            if squared_norm is None
            else squared_norm + local_squared_norm
        )
    if squared_norm is None:
        return 0.0
    squared_norm_value = float(squared_norm.item())
    if not math.isfinite(squared_norm_value):
        raise FloatingPointError("Non-finite gradient or global gradient norm.")
    return math.sqrt(max(0.0, squared_norm_value))


def train_one_epoch(
    model: JointPaCsiDGLite,
    loader: DataLoader,
    sampler: DeterministicDomainClassBatchSampler,
    optimizer: torch.optim.Optimizer,
    contrastive_loss: DomainAwareSupConLoss,
    *,
    epoch: int,
    device: torch.device,
    target_lambda: float,
    warmup_steps: int,
    global_step: int,
    base_lr: float,
    decay_rate: float,
    decay_steps: int,
) -> Tuple[Dict[str, Any], int]:
    model.train()
    sampler.set_epoch(epoch - 1)
    schedule_hash = sampler.schedule_hash()
    total_loss = 0.0
    total_ce = 0.0
    total_supcon = 0.0
    total_weighted_supcon = 0.0
    total_lambda = 0.0
    all_labels: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []
    positive_anchors = 0.0
    total_anchors = 0.0
    gradient_l2_norms: List[float] = []
    shape_debug: Optional[Dict[str, List[int]]] = None

    for batch_id, (amplitude, phase, labels, domains, sample_ids) in enumerate(loader):
        if batch_id >= protocol.OPTIMIZER_STEPS_PER_EPOCH:
            raise RuntimeError("Training loader exceeded the locked 300 steps.")
        amplitude = amplitude.to(device, non_blocking=True)
        phase = phase.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        domains = domains.to(device, non_blocking=True)
        sample_ids = sample_ids.to(device, non_blocking=True)
        current_lambda = effective_lambda_at_step(target_lambda, global_step, warmup_steps)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(amplitude, phase)
        ce_loss = F.cross_entropy(outputs["logits"], labels)
        views = outputs["supcon_projection_views"]
        if views.ndim != 3 or views.shape[1] != 1:
            raise ValueError(f"global_only SupCon must have V=1, received {tuple(views.shape)}.")
        if current_lambda > 0.0:
            supcon_loss = contrastive_loss(
                views,
                labels=labels,
                domains=domains,
                sample_ids=sample_ids,
            )
            diagnostics = supcon_positive_pair_diagnostics(
                views,
                labels=labels,
                domains=domains,
                sample_ids=sample_ids,
                include_same_domain_same_class=True,
                positive_mode="all_views",
            )
            batch_anchors = float(diagnostics["supcon_num_anchors"])
            positive_anchors += batch_anchors * float(diagnostics["supcon_frac_anchors_with_positive"])
            total_anchors += batch_anchors
        else:
            supcon_loss = ce_loss.new_zeros(())
        loss = ce_loss + current_lambda * supcon_loss
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch}, batch={batch_id}.")
        loss.backward()
        try:
            gradient_l2_norms.append(_finite_gradient_l2_norm(model))
        except FloatingPointError as exc:
            raise FloatingPointError(
                f"Non-finite gradient at epoch={epoch}, batch={batch_id}."
            ) from exc
        optimizer.step()
        global_step += 1
        next_lr = float(base_lr) * float(decay_rate) ** (global_step / float(decay_steps))
        for group in optimizer.param_groups:
            group["lr"] = next_lr

        predictions = outputs["logits"].argmax(dim=1)
        all_labels.append(labels.detach().cpu().numpy())
        all_predictions.append(predictions.detach().cpu().numpy())
        total_loss += float(loss.detach().item())
        total_ce += float(ce_loss.detach().item())
        total_supcon += float(supcon_loss.detach().item())
        total_weighted_supcon += current_lambda * float(supcon_loss.detach().item())
        total_lambda += current_lambda
        if shape_debug is None:
            shape_debug = {
                "amplitude": list(amplitude.shape),
                "phase": list(phase.shape),
                "fused_feature": list(outputs["fused_feature"].shape),
                "projection_views": list(views.shape),
            }

    if len(all_labels) != protocol.OPTIMIZER_STEPS_PER_EPOCH:
        raise RuntimeError(
            f"Epoch executed {len(all_labels)} steps instead of {protocol.OPTIMIZER_STEPS_PER_EPOCH}."
        )
    labels_np = np.concatenate(all_labels)
    predictions_np = np.concatenate(all_predictions)
    gradient_l2_array = np.asarray(gradient_l2_norms, dtype=np.float64)
    metrics = fixed_label_metrics(labels_np, predictions_np, num_classes=len(protocol.LABELS))
    metrics.update(
        {
            "loss_total": total_loss / len(all_labels),
            "loss_ce": total_ce / len(all_labels),
            "loss_supcon": total_supcon / len(all_labels),
            "loss_supcon_weighted": total_weighted_supcon / len(all_labels),
            "mean_effective_lambda": total_lambda / len(all_labels),
            "last_effective_lambda": effective_lambda_at_step(
                target_lambda,
                max(0, global_step - 1),
                warmup_steps,
            ),
            "positive_anchor_fraction": (
                positive_anchors / total_anchors if total_anchors > 0 else None
            ),
            "schedule_sha256": schedule_hash,
            "global_step": global_step,
            "learning_rate_next_step": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm_l2": {
                "count": int(gradient_l2_array.size),
                "minimum": float(np.min(gradient_l2_array)),
                "median": float(np.median(gradient_l2_array)),
                "p95": float(np.percentile(gradient_l2_array, 95)),
                "maximum": float(np.max(gradient_l2_array)),
                "mean": float(np.mean(gradient_l2_array)),
            },
            "shape_debug": shape_debug or {},
        }
    )
    return metrics, global_step


@torch.no_grad()
def evaluate_model(
    model: JointPaCsiDGLite,
    loader: DataLoader,
    dataset: Any,
    *,
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    sample_to_index, index_to_key = _stable_sample_maps(dataset)
    verified_by_env = {
        str(env_name): bool(dataset.provenance.get("sidecar_status", {}).get(env_name, {}).get("verified"))
        for env_name in dataset.env_names
    }
    all_labels: List[int] = []
    all_predictions: List[int] = []
    records: List[Dict[str, Any]] = []
    total_ce = 0.0
    batch_count = 0
    for amplitude, phase, labels, domains, sample_ids in loader:
        amplitude = amplitude.to(device, non_blocking=True)
        phase = phase.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        outputs = model(amplitude, phase)
        probabilities = torch.softmax(outputs["logits"], dim=1)
        predictions = probabilities.argmax(dim=1)
        total_ce += float(F.cross_entropy(outputs["logits"], labels_device).item())
        batch_count += 1

        labels_list = labels.tolist()
        predictions_list = predictions.cpu().tolist()
        probability_list = probabilities.cpu().tolist()
        domains_list = domains.tolist()
        sample_ids_list = sample_ids.tolist()
        all_labels.extend(int(value) for value in labels_list)
        all_predictions.extend(int(value) for value in predictions_list)
        for true, pred, probs, domain_id, raw_sample_id in zip(
            labels_list,
            predictions_list,
            probability_list,
            domains_list,
            sample_ids_list,
        ):
            dataset_index = sample_to_index[int(raw_sample_id)]
            env_name = dataset.env_name(int(domain_id))
            raw_subject = dataset.subject_ids[dataset_index]
            subject_id = (
                str(raw_subject)
                if verified_by_env[env_name] and raw_subject is not None
                else None
            )
            records.append(
                {
                    "sample_id": index_to_key[dataset_index],
                    "group_id": str(dataset.group_ids[dataset_index]),
                    "subject_id": subject_id,
                    "environment": env_name,
                    "true": int(true),
                    "pred": int(pred),
                    "probability": [float(value) for value in probs],
                }
            )
    metrics = fixed_label_metrics(all_labels, all_predictions, num_classes=len(protocol.LABELS))
    metrics["loss_ce"] = total_ce / max(1, batch_count)
    return metrics, records


def write_prediction_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def diagnostic_batch(
    dataset: Any,
    sampler: DeterministicDomainClassBatchSampler,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sampler.set_epoch(0)
    indices = next(iter(sampler))
    amplitude = torch.from_numpy(np.asarray(dataset.amplitude[indices])).float().to(device)
    phase = torch.from_numpy(np.asarray(dataset.phase[indices])).float().to(device)
    labels = torch.from_numpy(np.asarray(dataset.labels[indices])).long().to(device)
    domains = torch.from_numpy(np.asarray(dataset.env_ids[indices])).long().to(device)
    sample_ids = torch.from_numpy(np.asarray(dataset.sample_ids[indices])).long().to(device)
    return amplitude, phase, labels, domains, sample_ids


def run_gradient_diagnostic(
    model: JointPaCsiDGLite,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    contrastive_loss: DomainAwareSupConLoss,
    *,
    lambda_supcon: float,
) -> Dict[str, Any]:
    was_training = model.training
    model.eval()
    amplitude, phase, labels, domains, sample_ids = batch
    outputs = model(amplitude, phase)
    ce_loss = F.cross_entropy(outputs["logits"], labels)
    supcon_loss = contrastive_loss(
        outputs["supcon_projection_views"],
        labels=labels,
        domains=domains,
        sample_ids=sample_ids,
    )
    report = ce_supcon_gradient_diagnostics(
        model,
        ce_loss,
        supcon_loss,
        lambda_supcon=lambda_supcon,
    )
    model.train(was_training)
    return {
        "lambda_supcon": float(lambda_supcon),
        "ce_loss": float(ce_loss.detach().item()),
        "supcon_loss": float(supcon_loss.detach().item()),
        "blocks": report,
    }


def _cpu_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def train_model(
    model: JointPaCsiDGLite,
    dataset: Any,
    split: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> Tuple[Dict[str, Any], DataLoader]:
    train_loader, val_loader, sampler = build_loaders(
        dataset,
        split,
        config,
        sampler_seed=seed,
        device=device,
    )
    batch_audit = audit_batch_schedule(dataset, sampler)
    write_json(output_dir / "batch_schedule_audit.json", batch_audit)

    training = config["training"]
    losses = config["losses"]
    base_lr = float(training["lr"])
    weight_decay = float(training.get("weight_decay", 0.0))
    if str(training.get("optimizer", "adam")).lower() != "adam":
        raise ValueError("Locked joint protocol requires Adam.")
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler_config = training.get("lr_scheduler", {})
    if str(scheduler_config.get("name")) != "exponential_decay":
        raise ValueError("Locked protocol requires step exponential decay.")
    decay_rate = float(scheduler_config.get("decay_rate"))
    decay_steps = int(scheduler_config.get("decay_steps"))
    if decay_rate != 0.9 or decay_steps != 10_000:
        raise ValueError("Scheduler must be 0.9 decay per 10,000 steps.")

    target_lambda = float(losses.get("lambda_supcon", 0.0))
    warmup_steps = int(losses.get("linear_warmup_steps", losses.get("lambda_supcon_warmup_steps", 0)))
    contrastive_loss = DomainAwareSupConLoss(
        temperature=float(losses["temperature"]),
        include_same_domain_same_class=True,
        positive_mode="all_views",
    )
    fixed_diagnostic_batch = (
        diagnostic_batch(dataset, sampler, device) if target_lambda > 0.0 else None
    )

    epochs = int(training["epochs"])
    best_score = -math.inf
    best_epoch: Optional[int] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, Any]] = []
    global_step = 0
    epoch_seconds: List[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        train_metrics, global_step = train_one_epoch(
            model,
            train_loader,
            sampler,
            optimizer,
            contrastive_loss,
            epoch=epoch,
            device=device,
            target_lambda=target_lambda,
            warmup_steps=warmup_steps,
            global_step=global_step,
            base_lr=base_lr,
            decay_rate=decay_rate,
            decay_steps=decay_steps,
        )
        val_metrics, _ = evaluate_model(model, val_loader, dataset, device=device)
        gradient_report = None
        if fixed_diagnostic_batch is not None and epoch % GRADIENT_DIAGNOSTIC_INTERVAL == 0:
            gradient_report = run_gradient_diagnostic(
                model,
                fixed_diagnostic_batch,
                contrastive_loss,
                lambda_supcon=effective_lambda_at_step(
                    target_lambda,
                    max(0, global_step - 1),
                    warmup_steps,
                ),
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        epoch_seconds.append(elapsed)

        current_score = float(val_metrics[protocol.CHECKPOINT_METRIC])
        # Strict > preserves the earlier checkpoint on an exact tie.
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            torch.save(best_state, output_dir / "best_model.pt")
        record: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "seconds": elapsed,
            "train": train_metrics,
            "val": val_metrics,
            "is_best": best_epoch == epoch,
        }
        if gradient_report is not None:
            record["gradient_diagnostic"] = gradient_report
        history.append(record)
        write_json(output_dir / "history.json", history)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_f1": train_metrics["f1_macro"],
                    "val_f1": val_metrics["f1_macro"],
                    "best_epoch": best_epoch,
                    "seconds": elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if best_state is None or best_epoch is None:
        raise RuntimeError("Training produced no checkpoint.")
    if global_step != epochs * protocol.OPTIMIZER_STEPS_PER_EPOCH:
        raise RuntimeError("Total optimizer-step budget does not match epochs x 300.")
    model.load_state_dict(best_state)
    resource_report = {
        "epochs": epochs,
        "optimizer_steps": global_step,
        "best_epoch": best_epoch,
        "best_source_val_f1_macro": best_score,
        "seconds_per_epoch": epoch_seconds,
        "mean_seconds_per_epoch": float(np.mean(epoch_seconds)),
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
        "batch_schedule_audit": batch_audit,
        "history": history,
    }
    return resource_report, val_loader


def git_metadata() -> Dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def resolve_output_dir(config: Mapping[str, Any], seed: int, override: Optional[Path]) -> Path:
    if override is not None:
        return override.resolve()
    output_root = Path(str(config.get("output_root", "outputs_supcon_tpp_joint")))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    return (
        output_root
        / str(config["experiment_name"])
        / f"target_{config['target_env']}"
        / f"seed_{seed}"
    )


def target_manifest(dataset: Any, target_env: str) -> Dict[str, Any]:
    indices = list(range(len(dataset)))
    pseudo_split = {
        "mode": "deferred_target_inference",
        "inner_validation": None,
        "target_env": target_env,
        "source_envs": [],
        "train_envs": [],
        "val_envs": [],
        "train_indices": [],
        "val_indices": [],
        "test_indices": indices,
    }
    return build_split_manifest(dataset, pseudo_split)


def _optional_identifier(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    normalized = str(value).strip()
    return normalized if normalized and normalized.lower() not in {"none", "nan", "null"} else None


def _manifest_identifier_sets(
    dataset: Any,
    indices: Sequence[int],
) -> Tuple[set[str], set[str], set[str]]:
    sample_keys: set[str] = set()
    group_ids: set[str] = set()
    subject_ids: set[str] = set()
    for raw_index in indices:
        index = int(raw_index)
        env_name = str(dataset.env_names[int(dataset.env_ids[index])])
        sample_keys.add(f"{env_name}:{int(dataset.sample_ids[index]):04d}")
        group_id = _optional_identifier(dataset.group_ids[index])
        subject_id = _optional_identifier(dataset.subject_ids[index])
        if group_id is not None:
            group_ids.add(group_id)
        if subject_id is not None:
            subject_ids.add(subject_id)
    return sample_keys, group_ids, subject_ids


def combine_deferred_target_manifest(
    source_manifest: Mapping[str, Any],
    source_dataset: Any,
    source_split: Mapping[str, Any],
    deferred_manifest: Mapping[str, Any],
    target_dataset: Any,
    target_env: str,
) -> Dict[str, Any]:
    """Build a self-consistent post-checkpoint manifest with namespaced identifiers."""

    combined = copy.deepcopy(dict(source_manifest))
    train_indices = [int(value) for value in source_split["train_indices"]]
    val_indices = [int(value) for value in source_split["val_indices"]]
    target_indices = list(range(len(target_dataset.sample_ids)))
    train_sets = _manifest_identifier_sets(source_dataset, train_indices)
    val_sets = _manifest_identifier_sets(source_dataset, val_indices)
    test_sets = _manifest_identifier_sets(target_dataset, target_indices)
    split_sets = {"train": train_sets, "val": val_sets, "test": test_sets}

    def pairwise(position: int) -> Dict[str, List[str]]:
        values = {name: sets[position] for name, sets in split_sets.items()}
        return {
            "train_val": sorted(values["train"] & values["val"]),
            "train_test": sorted(values["train"] & values["test"]),
            "val_test": sorted(values["val"] & values["test"]),
        }

    sample_overlap = pairwise(0)
    group_overlap = pairwise(1)
    subject_overlap = pairwise(2)
    group_checked = all(
        bool(manifest.get(subset, {}).get("group_ids_complete", False))
        for manifest, subset in (
            (source_manifest, "train"),
            (source_manifest, "val"),
            (deferred_manifest, "test"),
        )
    )
    subject_checked = all(
        bool(manifest.get(subset, {}).get("subject_ids_complete", False))
        for manifest, subset in (
            (source_manifest, "train"),
            (source_manifest, "val"),
            (deferred_manifest, "test"),
        )
    )
    namespaced_ids = {
        name: sorted(sets[0])
        for name, sets in split_sets.items()
    }
    combined.update(
        {
            "mode": "dg_loeo_deferred_target_evaluation",
            "target_env": target_env,
            "target_not_loaded": False,
            "target_absent_during_training": True,
            "target_data_loaded": True,
            "target_loaded_after_checkpoint": True,
            "target_statistics_available": True,
            "dataset_size": int(source_manifest["dataset_size"]) + int(deferred_manifest["dataset_size"]),
            "loaded_environments": list(source_dataset.env_names) + [target_env],
            "test": copy.deepcopy(deferred_manifest["test"]),
            "sample_id_namespace": "environment:local_sample_id",
            "index_sha256": canonical_json_hash(namespaced_ids),
            "source_split_manifest_sha256": canonical_json_hash(source_manifest),
            "target_evaluation_manifest_sha256": canonical_json_hash(deferred_manifest),
            "overlap_audit": {
                "sample_ids": sample_overlap,
                "group_ids": group_overlap,
                "group_ids_checked": group_checked,
                "subject_ids": subject_overlap,
                "subject_ids_checked": subject_checked,
                "all_clear": not any(sample_overlap.values())
                and (not group_checked or not any(group_overlap.values()))
                and (not subject_checked or not any(subject_overlap.values())),
            },
        }
    )
    for subset_name, sample_keys in namespaced_ids.items():
        combined[subset_name]["sample_keys_sha256"] = canonical_json_hash(sample_keys)
    for field in ("data_files", "file_sha256", "sidecar_status"):
        merged = copy.deepcopy(dict(source_manifest.get(field, {})))
        merged.update(copy.deepcopy(dict(deferred_manifest.get(field, {}))))
        combined[field] = merged
    combined["subject_sidecars_complete"] = bool(
        source_manifest.get("subject_sidecars_complete", False)
        and deferred_manifest.get("subject_sidecars_complete", False)
    )
    if not combined["overlap_audit"]["all_clear"]:
        raise ValueError("Combined source/target manifest contains identifier overlap.")
    return combined


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    raw_config = load_json(config_path)
    config = resolve_runtime_config(raw_config, args)
    validate_runtime_config(raw_config, config)
    validate_bound_runtime_artifacts(config, config_path)
    seed = resolve_single_seed(config, args.seed)
    set_model_seed(seed)
    device = choose_device(str(config.get("device", "cuda:0")))
    output_dir = resolve_output_dir(config, seed, args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Run directory {output_dir} is non-empty; refusing to mix stale artifacts.")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_data_config = resolve_source_data_config(config)
    target_env = str(config["target_env"])
    source_envs = list(config["source_envs"])
    if target_env in source_data_config["env_files"]:
        raise AssertionError("Outer target survived the source-only filter.")
    source_dataset = load_dataset(source_data_config, config_path.parent)
    if list(source_dataset.env_names) != source_envs:
        raise ValueError(
            f"Loaded source order {source_dataset.env_names} does not match locked order {source_envs}."
        )
    if source_dataset.num_classes != len(protocol.LABELS):
        raise ValueError("All six classes must be present in source-only data.")
    split = build_source_only_split(
        source_dataset,
        target_env=target_env,
        split_config=config["split"],
    )
    split_manifest = build_split_manifest(source_dataset, split)
    split_manifest["target_data_loaded"] = False
    split_manifest["target_statistics_available"] = False
    split_manifest["loaded_environments"] = list(source_dataset.env_names)
    write_json(output_dir / "split_manifest.json", split_manifest)

    phase_audit = phase_input_audit(source_dataset, source_data_config, base_dir=config_path.parent)
    write_json(output_dir / "phase_input_audit.json", phase_audit)
    config.setdefault("runtime", {}).update(
        {
            "seed": seed,
            "device": str(device),
            "loaded_training_environments": list(source_dataset.env_names),
            "target_loaded_during_training": False,
            "actual_inner_validation": split["inner_validation"],
        }
    )
    write_json(output_dir / "resolved_config.json", config)
    write_json(
        output_dir / "run_invocation.json",
        {
            "argv": sys.argv,
            "config_path": str(config_path),
            "raw_config_sha256": canonical_json_hash(raw_config),
            "git_at_start": git_metadata(),
        },
    )

    calibration_path = (
        args.calibration_artifact.resolve()
        if args.calibration_artifact is not None
        else default_calibration_path(config, output_dir)
    )
    artifact = ensure_calibration_artifact(
        source_dataset,
        split,
        config,
        device=device,
        artifact_path=calibration_path,
    )
    calibration_hash = hashlib.sha256(calibration_path.read_bytes()).hexdigest()

    arm = str(config["protocol"]["arm"])
    audit_models = build_joint_arm_models(
        input_dim=source_dataset.input_dim,
        num_classes=len(protocol.LABELS),
        model_config=config["model"],
        initialization_seed=seed,
        calibration_artifact=artifact,
    )
    parameter_audit = audit_joint_model_parameters(
        audit_models,
        strict=True,
        check_initialization=True,
    )
    initialization_hashes = {
        code: trainable_parameter_hash(model) for code, model in audit_models.items()
    }
    if len(set(initialization_hashes.values())) != 1:
        raise ValueError("Four-arm trainable initialization hashes differ.")
    model = audit_models.pop(arm).to(device)
    del audit_models
    parameter_report = {
        **parameter_audit,
        "trainable_parameter_count": count_trainable_parameters(model),
        "initialization_hashes": initialization_hashes,
        "calibration_artifact": str(calibration_path),
        "calibration_sha256": calibration_hash,
    }
    write_json(output_dir / "parameter_audit.json", parameter_report)

    stage0_gate = {
        "target_env": target_env,
        "source_envs": source_envs,
        "target_opened": False,
        "split_manifest_sha256": canonical_json_hash(split_manifest),
        "phase_audit_sha256": canonical_json_hash(phase_audit),
        "calibration_sha256": calibration_hash,
        "parameter_audit": parameter_report,
    }
    write_json(output_dir / "stage0_gate.json", stage0_gate)
    if args.calibration_only:
        print(json.dumps(stage0_gate, indent=2, sort_keys=True))
        return

    resource_report, val_loader = train_model(
        model,
        source_dataset,
        split,
        config,
        seed=seed,
        device=device,
        output_dir=output_dir,
    )
    source_val_metrics, source_val_records = evaluate_model(
        model,
        val_loader,
        source_dataset,
        device=device,
    )
    write_prediction_jsonl(output_dir / "source_val_predictions.jsonl", source_val_records)
    write_json(output_dir / "source_val_metrics.json", source_val_metrics)

    evaluate_target = bool(config.get("evaluation", {}).get("evaluate_target", False))
    target_metrics: Optional[Dict[str, Any]] = None
    if evaluate_target:
        # This is the first point at which target paths are resolved or opened.
        target_data_config = resolve_target_data_config(config)
        target_dataset = load_dataset(target_data_config, config_path.parent)
        if list(target_dataset.env_names) != [target_env]:
            raise ValueError("Deferred target loader opened an unexpected environment.")
        target_phase_audit = phase_input_audit(
            target_dataset,
            target_data_config,
            base_dir=config_path.parent,
        )
        write_json(output_dir / "target_phase_input_audit.json", target_phase_audit)
        target_loader = DataLoader(
            target_dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        target_metrics, target_records = evaluate_model(
            model,
            target_loader,
            target_dataset,
            device=device,
        )
        write_prediction_jsonl(output_dir / "target_predictions.jsonl", target_records)
        write_json(output_dir / "target_metrics.json", target_metrics)
        deferred_manifest = target_manifest(target_dataset, target_env)
        write_json(output_dir / "source_split_manifest.json", split_manifest)
        write_json(output_dir / "target_evaluation_manifest.json", deferred_manifest)
        split_manifest = combine_deferred_target_manifest(
            split_manifest,
            source_dataset,
            split,
            deferred_manifest,
            target_dataset,
            target_env,
        )
        write_json(output_dir / "split_manifest.json", split_manifest)
    else:
        forbidden = [path.name for path in output_dir.iterdir() if path.name.startswith("target_")]
        if forbidden:
            raise AssertionError(f"Target-disabled run emitted forbidden artifacts: {forbidden}")

    history = resource_report["history"]
    observed_positive_fractions = [
        float(epoch["train"]["positive_anchor_fraction"])
        for epoch in history
        if epoch["train"].get("positive_anchor_fraction") is not None
    ]
    screen_summary = screen_health.build_screen_metrics(
        history,
        best_source_val_sit_stand_f1=float(source_val_metrics["per_class"]["3"]["f1"]),
        positive_anchor_fraction=(
            min(observed_positive_fractions)
            if observed_positive_fractions
            else float(resource_report["batch_schedule_audit"]["cross_domain_positive_anchor_fraction"])
        ),
    )

    result = {
        "protocol": dict(config["protocol"]),
        "seed": seed,
        "arm": arm,
        "target_env": target_env,
        "source_envs": source_envs,
        "evaluate_target": evaluate_target,
        "source_val_metrics": source_val_metrics,
        "target_metrics": target_metrics,
        "resources": resource_report,
        "parameter_report": parameter_report,
        "calibration_artifact": str(calibration_path),
        "calibration_sha256": calibration_hash,
        "screen_metrics": screen_summary,
        "git": git_metadata(),
    }
    write_json(output_dir / "metrics.json", result)
    print(json.dumps({"output_dir": str(output_dir), "completed": True}, sort_keys=True))


if __name__ == "__main__":
    main()
