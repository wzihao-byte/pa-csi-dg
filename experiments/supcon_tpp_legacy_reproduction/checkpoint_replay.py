from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


SCHEMA_VERSION = 1
PROTOCOL_NAME = "supcon_tpp_legacy_checkpoint_replay"
LEGACY_COMMIT = "7b8ec0947846daef65c6c6f37b912f9eeb5efcd6"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEAN_WORKTREE = Path(
    os.environ.get(
        "PA_CSI_LEGACY_WORKTREE",
        REPO_ROOT.parent / f"{REPO_ROOT.name}-legacy-replay-7b8ec09",
    )
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs_supcon_tpp_legacy_reproduction" / "audit" / "checkpoint_replay.json"
TPP_MODEL_SOURCE = REPO_ROOT / "experiments" / "tpp_isolated" / "tpp_models.py"
TPP_MODEL_SOURCE_SHA256 = "fc6aeb3858763c282b892f8c8e91ac31425d93df4343d83bb33cd1c6749ebccf"
METRIC_ATOL = 1e-12
EXPECTED_TARGET_SAMPLES = 3000
EXPECTED_CLASSES = 6
EXPECTED_INPUT_DIM = 90


class ReplayFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class ReferenceSpec:
    reference_id: str
    run_dir: Path
    target_env: str
    seed: int
    model_kind: str
    model_name: str
    batch_size: int
    checkpoint_sha256: str
    metrics_sha256: str
    config_sha256: str
    confusion_sha256: str
    expected_state_keys: int
    expected_parameters: int
    expected_best_val_accuracy: float
    expected_accuracy: float
    expected_precision_macro: float
    expected_recall_macro: float
    expected_f1_macro: float

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / "best_model.pt"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"

    @property
    def config_path(self) -> Path:
        return self.run_dir / "resolved_config.json"

    @property
    def confusion_path(self) -> Path:
        return self.run_dir / "confusion_matrix.npy"


REFERENCES = (
    ReferenceSpec(
        reference_id="supcon_E1_seed42",
        run_dir=REPO_ROOT
        / "outputs_supcon_final_eval"
        / "dg_loeo"
        / "pa_csi_dg_supcon_final_lam0050_warm000_tau020_bs32_gci"
        / "target_E1"
        / "seed_42",
        target_env="E1",
        seed=42,
        model_kind="legacy",
        model_name="pa_csi_dg_lite",
        batch_size=32,
        checkpoint_sha256="cb1fe6dad78bc34fd397b1e31e7b49c21e917b6be8fa0980d1414188d9309889",
        metrics_sha256="a961a8b6e005982fda153b2ab215f376711356396f649e0c6f6c75967f233c88",
        config_sha256="c4bdb683047301c06b4c12d689e606c9392f4e83a59e7acd4438c4565c20f823",
        confusion_sha256="d3a9c37c8608b3e5a7dc05f34070555146df26601c0eab007aba4485de80d178",
        expected_state_keys=358,
        expected_parameters=2_788_918,
        expected_best_val_accuracy=0.9983333333333333,
        expected_accuracy=0.8966666666666666,
        expected_precision_macro=0.9058351109402151,
        expected_recall_macro=0.8762500000000001,
        expected_f1_macro=0.8632369330841257,
    ),
    ReferenceSpec(
        reference_id="ce_E1_seed42",
        run_dir=REPO_ROOT
        / "outputs_anchor_ce_baseline"
        / "dg_loeo"
        / "anchor_ce_baseline"
        / "target_E1"
        / "seed_42",
        target_env="E1",
        seed=42,
        model_kind="legacy",
        model_name="pa_csi_dg_lite",
        batch_size=8,
        checkpoint_sha256="aeba89b0274199dd04587c31d50d67c51e360fda09160804942237868ff8874f",
        metrics_sha256="56859dcbef268447850d0b2f47c74535764814a24cf8cb44ccc3a401a78a29e5",
        config_sha256="c4a725c129a2abf91477af174d0e2a8489724622394a20421c494ac7f747c377",
        confusion_sha256="fb951416ef7f4cbc9ed08fc5b1899cfb78940aab038163363b546316a47ef701",
        expected_state_keys=358,
        expected_parameters=2_788_918,
        expected_best_val_accuracy=0.9991666666666666,
        expected_accuracy=0.8526666666666667,
        expected_precision_macro=0.8886247559604753,
        expected_recall_macro=0.8341666666666666,
        expected_f1_macro=0.8257570109034975,
    ),
    ReferenceSpec(
        reference_id="tpp_l124_E1_seed42",
        run_dir=REPO_ROOT
        / "outputs_tpp_isolated"
        / "dg_loeo"
        / "tpp_l124"
        / "target_E1"
        / "seed_42",
        target_env="E1",
        seed=42,
        model_kind="tpp_snapshot",
        model_name="isolated_pa_csi_dg_lite_tpp",
        batch_size=8,
        checkpoint_sha256="569aa34e0105dcc4eee17e62d8ad0b8766b5edcc4ae0f5301ece7f4e4cafd7cc",
        metrics_sha256="161a44e9e581a9064d444a4a8bcf2b46a4e9c4106a15a1776f1711db7442a1bd",
        config_sha256="cca978e6482f9dc2ffe75e0f9bae6df37046c2944a978ba9d0882aee31a25c1d",
        confusion_sha256="5b3821917e1edf7ed952a5cafb86bce0d51bb59d42647ac582f137bc059cfe6f",
        expected_state_keys=366,
        expected_parameters=3_721_334,
        expected_best_val_accuracy=0.9833333333333333,
        expected_accuracy=0.8396666666666667,
        expected_precision_macro=0.8504048297646879,
        expected_recall_macro=0.8147222222222222,
        expected_f1_macro=0.8072287933303489,
    ),
    ReferenceSpec(
        reference_id="tpp_residual025_E1_seed52",
        run_dir=REPO_ROOT
        / "outputs_tpp_c3_recovery"
        / "dg_loeo"
        / "tpp_c3_residual025_l124"
        / "target_E1"
        / "seed_52",
        target_env="E1",
        seed=52,
        model_kind="tpp_snapshot",
        model_name="isolated_pa_csi_dg_lite_tpp",
        batch_size=8,
        checkpoint_sha256="98c096fd1e54d703229c4bad611eac856a4f59fd3f32af28033dc7cfca13327e",
        metrics_sha256="18ac6be1b38a7f0cb232f1ae42146254027732be36311e6cddbb91521f76c781",
        config_sha256="68e6da532aea0a56b31ea8cfcaf57a7388a13158efa78209e29dce43501c112a",
        confusion_sha256="65b7a9baad525d239175032c834750b8e4e432a0e3d879ff76d64dbeeb7164cd",
        expected_state_keys=366,
        expected_parameters=3_721_334,
        expected_best_val_accuracy=0.9991666666666666,
        expected_accuracy=0.9103333333333333,
        expected_precision_macro=0.9120851913997119,
        expected_recall_macro=0.8901388888888889,
        expected_f1_macro=0.8901276960839671,
    ),
)


TARGET_DATA_SHA256 = {
    "E1": {
        "amp": "26e4700ac8b9f0b1af3feca99d05d8502251c0274b3ce8718873179416aaa74a",
        "phase": "14f0837b28188b3cce9bea02ec1f19fddf0ee2f1abba36ade0b2531faa3ece02",
        "label": "619e56cc7768625b0f2cdff8f973501b3b09ea2bf9300a045470e7a99c7a8b85",
    }
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_int64(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayFailure(message)


def require_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise ReplayFailure(f"{context}: expected {expected!r}, got {actual!r}")


def require_close(actual: float, expected: float, context: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=METRIC_ATOL):
        raise ReplayFailure(
            f"{context}: expected {expected:.17g}, got {actual:.17g}, "
            f"absolute error={abs(actual - expected):.17g}, tolerance={METRIC_ATOL}"
        )


def run_git(arguments: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def verify_clean_worktree(path: Path) -> dict[str, Any]:
    require(path.is_dir(), f"Clean legacy worktree does not exist: {path}")
    head = run_git(["rev-parse", "HEAD"], path)
    status = run_git(["status", "--short"], path)
    require_equal(head, LEGACY_COMMIT, "legacy worktree HEAD")
    require_equal(status, "", "legacy worktree status")
    files = {}
    for name in ("dg_dataset.py", "dg_models.py", "mcat.py", "position_encoding.py"):
        source = path / name
        require(source.is_file(), f"Missing legacy source: {source}")
        files[name] = {"path": str(source.resolve()), "sha256": sha256_file(source)}
    return {
        "path": str(path.resolve()),
        "head": head,
        "status_clean": True,
        "source_files": files,
    }


def import_replay_modules(clean_worktree: Path) -> tuple[Any, Any, Any, dict[str, str]]:
    legacy_path = str(clean_worktree.resolve())
    for module_name in ("dg_dataset", "dg_models", "mcat", "position_encoding"):
        existing = sys.modules.get(module_name)
        if existing is not None:
            module_file = Path(getattr(existing, "__file__", "")).resolve()
            require(
                clean_worktree.resolve() in module_file.parents,
                f"Legacy module {module_name} was already imported from outside the clean worktree: {module_file}",
            )

    sys.path.insert(0, legacy_path)
    try:
        legacy_dataset = importlib.import_module("dg_dataset")
        legacy_models = importlib.import_module("dg_models")
    finally:
        if sys.path[0] == legacy_path:
            sys.path.pop(0)

    module_paths: dict[str, str] = {}
    # At 7b8ec09 the legacy model is self-contained in dg_models.py; mcat.py and
    # position_encoding.py are repository siblings, not runtime dependencies.
    # Record their hashes above, but only require modules that this replay
    # actually imports to have resolved from the clean worktree.
    for module_name in ("dg_dataset", "dg_models"):
        module = sys.modules.get(module_name)
        require(module is not None, f"Expected legacy module was not imported: {module_name}")
        module_path = Path(getattr(module, "__file__", "")).resolve()
        require(
            clean_worktree.resolve() in module_path.parents,
            f"Legacy module {module_name} resolved outside the clean worktree: {module_path}",
        )
        module_paths[module_name] = str(module_path)

    require(TPP_MODEL_SOURCE.is_file(), f"Missing locked TPP model source: {TPP_MODEL_SOURCE}")
    require_equal(sha256_file(TPP_MODEL_SOURCE), TPP_MODEL_SOURCE_SHA256, "TPP model source SHA256")
    specification = importlib.util.spec_from_file_location("_locked_legacy_tpp_models", TPP_MODEL_SOURCE)
    require(specification is not None and specification.loader is not None, "Could not create TPP module spec")
    tpp_models = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(tpp_models)
    module_paths["tpp_models"] = str(TPP_MODEL_SOURCE.resolve())
    return legacy_dataset, legacy_models, tpp_models, module_paths


def expected_metrics(spec: ReferenceSpec) -> dict[str, float]:
    return {
        "accuracy": spec.expected_accuracy,
        "precision_macro": spec.expected_precision_macro,
        "recall_macro": spec.expected_recall_macro,
        "f1_macro": spec.expected_f1_macro,
    }


def verify_reference_files(spec: ReferenceSpec) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    locked_files = {
        "checkpoint": (spec.checkpoint_path, spec.checkpoint_sha256),
        "metrics": (spec.metrics_path, spec.metrics_sha256),
        "resolved_config": (spec.config_path, spec.config_sha256),
        "confusion_matrix": (spec.confusion_path, spec.confusion_sha256),
    }
    file_records: dict[str, Any] = {}
    for name, (path, expected_sha) in locked_files.items():
        require(path.is_file(), f"Missing {spec.reference_id} {name}: {path}")
        actual_sha = sha256_file(path)
        require_equal(actual_sha, expected_sha, f"{spec.reference_id} {name} SHA256")
        file_records[name] = {
            "path": str(path.resolve()),
            "sha256": actual_sha,
            "size_bytes": path.stat().st_size,
        }

    config = read_json(spec.config_path)
    metrics = read_json(spec.metrics_path)
    historical_confusion = np.load(spec.confusion_path, allow_pickle=False)

    require_equal(config.get("mode"), "dg_loeo", f"{spec.reference_id} mode")
    require_equal(config.get("target_env"), spec.target_env, f"{spec.reference_id} config target")
    require_equal(config.get("seed_list"), [spec.seed], f"{spec.reference_id} config seed")
    require_equal(config.get("model", {}).get("name"), spec.model_name, f"{spec.reference_id} model name")
    require_equal(config.get("training", {}).get("batch_size"), spec.batch_size, f"{spec.reference_id} batch")
    require_equal(config.get("training", {}).get("selection_metric"), "accuracy", f"{spec.reference_id} selection")
    require_equal(config.get("data", {}).get("unwrap_phase"), True, f"{spec.reference_id} unwrap")
    require_equal(config.get("data", {}).get("phase_unwrap_axis"), -1, f"{spec.reference_id} unwrap axis")

    require_equal(metrics.get("target_env"), spec.target_env, f"{spec.reference_id} metrics target")
    require_equal(metrics.get("seed"), spec.seed, f"{spec.reference_id} metrics seed")
    require_equal(metrics.get("test_size"), EXPECTED_TARGET_SAMPLES, f"{spec.reference_id} test size")
    require_close(
        float(metrics.get("best_val_accuracy")),
        spec.expected_best_val_accuracy,
        f"{spec.reference_id} historical best-val accuracy",
    )
    historical_test = metrics.get("test_metrics", {})
    for metric_name, value in expected_metrics(spec).items():
        require_close(
            float(historical_test.get(metric_name)),
            value,
            f"{spec.reference_id} locked historical {metric_name}",
        )

    require_equal(tuple(historical_confusion.shape), (EXPECTED_CLASSES, EXPECTED_CLASSES), f"{spec.reference_id} confusion shape")
    require_equal(int(historical_confusion.sum()), EXPECTED_TARGET_SAMPLES, f"{spec.reference_id} confusion total")
    return config, {
        "files": file_records,
        "historical_metrics": {
            "best_val_accuracy": float(metrics["best_val_accuracy"]),
            **{name: float(historical_test[name]) for name in expected_metrics(spec)},
        },
        "historical_confusion_matrix": historical_confusion.astype(np.int64).tolist(),
    }, historical_confusion.astype(np.int64, copy=False)


def build_and_freeze_model(
    spec: ReferenceSpec,
    config: Mapping[str, Any],
    legacy_models: Any,
    tpp_models: Any,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if spec.model_kind == "legacy":
        model = legacy_models.PaCsiDGLite(
            input_dim=EXPECTED_INPUT_DIM,
            num_classes=EXPECTED_CLASSES,
            model_config=config["model"],
        )
        constructor = "legacy@7b8ec094.dg_models.PaCsiDGLite"
    elif spec.model_kind == "tpp_snapshot":
        model = tpp_models.IsolatedPaCsiDGLiteTPP(
            input_dim=EXPECTED_INPUT_DIM,
            num_classes=EXPECTED_CLASSES,
            model_config=config["model"],
        )
        constructor = "sha256-locked-tpp_models.IsolatedPaCsiDGLiteTPP"
    else:
        raise ReplayFailure(f"Unsupported model kind for {spec.reference_id}: {spec.model_kind}")

    state = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=True)
    require(isinstance(state, OrderedDict), f"{spec.reference_id} checkpoint is not an OrderedDict")
    require(
        all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()),
        f"{spec.reference_id} checkpoint is not a direct state_dict[str, Tensor]",
    )
    require(not any(key.startswith("module.") for key in state), f"{spec.reference_id} has DataParallel prefixes")
    require_equal(len(state), spec.expected_state_keys, f"{spec.reference_id} state key count")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    require_equal(parameter_count, spec.expected_parameters, f"{spec.reference_id} parameter count")
    incompatible = model.load_state_dict(state, strict=True)
    require_equal(list(incompatible.missing_keys), [], f"{spec.reference_id} missing state keys")
    require_equal(list(incompatible.unexpected_keys), [], f"{spec.reference_id} unexpected state keys")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "constructor": constructor,
        "checkpoint_format": "collections.OrderedDict[str, Tensor]",
        "strict_state_load": True,
        "state_key_count": len(state),
        "state_tensor_numel": int(sum(value.numel() for value in state.values())),
        "parameter_count": parameter_count,
        "buffer_count": int(sum(buffer.numel() for buffer in model.buffers())),
        "module_prefix": False,
        "trainable_after_freeze": int(sum(parameter.requires_grad for parameter in model.parameters())),
    }


def target_data_config(config: Mapping[str, Any], target_env: str) -> dict[str, Any]:
    data = config.get("data", {})
    env_files = data.get("env_files", {})
    require(target_env in env_files, f"Target {target_env} is absent from the locked resolved config")
    return {
        "format": data.get("format"),
        "unwrap_phase": data.get("unwrap_phase"),
        "phase_unwrap_axis": data.get("phase_unwrap_axis"),
        "env_files": {target_env: dict(env_files[target_env])},
    }


def verify_and_load_target(
    target_env: str,
    configs: Mapping[str, Mapping[str, Any]],
    legacy_dataset: Any,
) -> tuple[Any, dict[str, Any]]:
    reference_configs = [target_data_config(config, target_env) for config in configs.values()]
    canonical = json.dumps(reference_configs[0], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    for index, item in enumerate(reference_configs[1:], start=1):
        require_equal(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            canonical,
            f"target data configuration agreement #{index}",
        )

    locked_hashes = TARGET_DATA_SHA256.get(target_env)
    require(locked_hashes is not None, f"No hard-coded target data hashes for {target_env}")
    file_records: dict[str, Any] = {}
    for kind, path_text in reference_configs[0]["env_files"][target_env].items():
        path = Path(path_text)
        require(path.is_file(), f"Missing locked target {target_env}/{kind}: {path}")
        actual_hash = sha256_file(path)
        require_equal(actual_hash, locked_hashes[kind], f"target {target_env}/{kind} SHA256")
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = (EXPECTED_TARGET_SAMPLES,) if kind == "label" else (EXPECTED_TARGET_SAMPLES, 850, EXPECTED_INPUT_DIM)
        expected_dtype = "int32" if kind == "label" else "float64"
        require_equal(tuple(raw.shape), expected_shape, f"target {target_env}/{kind} shape")
        require_equal(str(raw.dtype), expected_dtype, f"target {target_env}/{kind} dtype")
        file_records[kind] = {
            "path": str(path.resolve()),
            "sha256": actual_hash,
            "raw_shape": list(raw.shape),
            "raw_dtype": str(raw.dtype),
            "size_bytes": path.stat().st_size,
        }

    dataset = legacy_dataset.load_dataset(reference_configs[0], REPO_ROOT)
    require_equal(len(dataset), EXPECTED_TARGET_SAMPLES, f"target {target_env} dataset size")
    require_equal(dataset.input_dim, EXPECTED_INPUT_DIM, f"target {target_env} input dim")
    require_equal(dataset.time_steps, 850, f"target {target_env} time steps")
    require_equal(dataset.num_classes, EXPECTED_CLASSES, f"target {target_env} classes")
    require_equal(dataset.env_names, [target_env], f"target {target_env} environment order")
    require_equal(str(dataset.amplitude.dtype), "float32", f"target {target_env} loaded amplitude dtype")
    require_equal(str(dataset.phase.dtype), "float32", f"target {target_env} loaded phase dtype")
    require_equal(str(dataset.labels.dtype), "int64", f"target {target_env} loaded labels dtype")
    label_counts = Counter(int(value) for value in dataset.labels.tolist())
    require_equal(dict(sorted(label_counts.items())), {0: 1200, 1: 400, 2: 400, 3: 400, 4: 400, 5: 200}, "target label counts")
    require_equal(dataset.sample_ids.tolist(), list(range(EXPECTED_TARGET_SAMPLES)), "target ordered sample ids")
    return dataset, {
        "target_env": target_env,
        "purpose": "audit replay of four preregistered historical checkpoints only",
        "selection_or_tuning_permitted": False,
        "opened_after_all_checkpoint_hashes_and_strict_loads_passed": True,
        "samples": len(dataset),
        "loaded_shape": {
            "amplitude": list(dataset.amplitude.shape),
            "phase": list(dataset.phase.shape),
            "labels": list(dataset.labels.shape),
        },
        "loaded_dtype": {
            "amplitude": str(dataset.amplitude.dtype),
            "phase": str(dataset.phase.dtype),
            "labels": str(dataset.labels.dtype),
        },
        "preprocessing": {"unwrap_phase": True, "phase_unwrap_axis": -1, "normalization": None},
        "files": file_records,
    }


def replay_metrics(model: torch.nn.Module, dataset: Any, batch_size: int, device: torch.device) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    labels_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    sample_id_parts: list[np.ndarray] = []
    env_id_parts: list[np.ndarray] = []
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        for amplitude, phase, labels, env_ids, sample_ids in loader:
            outputs = model(amplitude.to(device), phase.to(device))
            logits = outputs.get("logits") if isinstance(outputs, Mapping) else None
            require(torch.is_tensor(logits), "Model output does not contain tensor logits")
            require_equal(tuple(logits.shape), (len(labels), EXPECTED_CLASSES), "replay logits shape")
            predictions = logits.argmax(dim=1)
            labels_parts.append(labels.numpy().astype(np.int64, copy=False))
            prediction_parts.append(predictions.cpu().numpy().astype(np.int64, copy=False))
            sample_id_parts.append(sample_ids.numpy().astype(np.int64, copy=False))
            env_id_parts.append(env_ids.numpy().astype(np.int64, copy=False))
    model.to("cpu")

    labels = np.concatenate(labels_parts)
    predictions = np.concatenate(prediction_parts)
    sample_ids = np.concatenate(sample_id_parts)
    env_ids = np.concatenate(env_id_parts)
    require_equal(len(labels), EXPECTED_TARGET_SAMPLES, "replay prediction count")
    require_equal(sample_ids.tolist(), list(range(EXPECTED_TARGET_SAMPLES)), "replay sample order")
    require_equal(np.unique(env_ids).tolist(), [0], "replay target environment ids")

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }
    confusion = confusion_matrix(labels, predictions, labels=np.arange(EXPECTED_CLASSES, dtype=np.int64))
    per_class_precision = precision_score(labels, predictions, labels=np.arange(EXPECTED_CLASSES), average=None, zero_division=0)
    per_class_recall = recall_score(labels, predictions, labels=np.arange(EXPECTED_CLASSES), average=None, zero_division=0)
    per_class_f1 = f1_score(labels, predictions, labels=np.arange(EXPECTED_CLASSES), average=None, zero_division=0)
    support = np.bincount(labels, minlength=EXPECTED_CLASSES)
    return {
        "metrics": metrics,
        "confusion_matrix": confusion.astype(np.int64).tolist(),
        "prediction_count": int(len(predictions)),
        "ordered_labels_sha256": sha256_int64(labels),
        "ordered_predictions_sha256": sha256_int64(predictions),
        "ordered_sample_ids_sha256": sha256_int64(sample_ids),
        "per_class": {
            str(index): {
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
            }
            for index in range(EXPECTED_CLASSES)
        },
    }


def compare_replay(
    spec: ReferenceSpec,
    replay: Mapping[str, Any],
    historical_confusion: np.ndarray,
) -> dict[str, Any]:
    differences: dict[str, float] = {}
    for metric_name, expected in expected_metrics(spec).items():
        actual = float(replay["metrics"][metric_name])
        require_close(actual, expected, f"{spec.reference_id} replay {metric_name}")
        differences[metric_name] = actual - expected
    replay_confusion = np.asarray(replay["confusion_matrix"], dtype=np.int64)
    require_equal(tuple(replay_confusion.shape), (EXPECTED_CLASSES, EXPECTED_CLASSES), "replay confusion shape")
    if not np.array_equal(replay_confusion, historical_confusion):
        delta = replay_confusion - historical_confusion
        raise ReplayFailure(
            f"{spec.reference_id} confusion matrix mismatch; absolute delta sum={int(np.abs(delta).sum())}, "
            f"maximum absolute cell delta={int(np.abs(delta).max())}"
        )
    return {
        "passed": True,
        "metric_atol": METRIC_ATOL,
        "metric_differences": differences,
        "confusion_exact_match": True,
    }


def runtime_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        require(torch.cuda.is_available(), f"CUDA replay requested but CUDA is unavailable: {value}")
        index = 0 if device.index is None else device.index
        require(index < torch.cuda.device_count(), f"CUDA device index is unavailable: {value}")
        torch.cuda.set_device(index)
    return device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed target replay for four preregistered historical checkpoints. No training is performed."
    )
    parser.add_argument("--clean-worktree", type=Path, default=DEFAULT_CLEAN_WORKTREE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    device = resolve_device(args.device)
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "legacy_commit": LEGACY_COMMIT,
        "status": "running",
        "started_at_utc": utc_now(),
        "target_use_policy": {
            "purpose": "replay validation of four preregistered historical checkpoints",
            "selection": False,
            "tuning": False,
            "training": False,
            "checkpoint_paths_fixed_before_target_open": True,
        },
        "metric_atol": METRIC_ATOL,
        "runtime": runtime_record(device),
    }
    try:
        clean_worktree = args.clean_worktree.resolve()
        audit["clean_worktree"] = verify_clean_worktree(clean_worktree)
        legacy_dataset, legacy_models, tpp_models, module_paths = import_replay_modules(clean_worktree)
        audit["module_paths"] = module_paths
        audit["tpp_snapshot"] = {
            "path": str(TPP_MODEL_SOURCE.resolve()),
            "sha256": sha256_file(TPP_MODEL_SOURCE),
        }

        configs: dict[str, Mapping[str, Any]] = {}
        models: dict[str, torch.nn.Module] = {}
        historical_confusions: dict[str, np.ndarray] = {}
        reference_records: dict[str, Any] = {}
        # No target file is opened before every reference hash and strict state load passes.
        for spec in REFERENCES:
            config, reference_record, historical_confusion = verify_reference_files(spec)
            model, model_record = build_and_freeze_model(spec, config, legacy_models, tpp_models)
            configs[spec.reference_id] = config
            models[spec.reference_id] = model
            historical_confusions[spec.reference_id] = historical_confusion
            reference_records[spec.reference_id] = {
                "target_env": spec.target_env,
                "seed": spec.seed,
                "model_kind": spec.model_kind,
                "batch_size": spec.batch_size,
                **reference_record,
                "model": model_record,
            }
        audit["references"] = reference_records
        audit["all_checkpoint_hashes_and_strict_loads_passed_before_target_open"] = True

        targets = sorted({spec.target_env for spec in REFERENCES})
        require_equal(targets, ["E1"], "fixed replay target set")
        dataset, target_record = verify_and_load_target("E1", configs, legacy_dataset)
        audit["target_data"] = target_record

        for spec in REFERENCES:
            replay = replay_metrics(models[spec.reference_id], dataset, spec.batch_size, device)
            comparison = compare_replay(spec, replay, historical_confusions[spec.reference_id])
            audit["references"][spec.reference_id]["replay"] = replay
            audit["references"][spec.reference_id]["comparison"] = comparison
            if device.type == "cuda":
                torch.cuda.empty_cache()

        audit["status"] = "passed"
        audit["completed_at_utc"] = utc_now()
        audit["summary"] = {
            "references": len(REFERENCES),
            "passed": len(REFERENCES),
            "failed": 0,
            "target_samples_per_reference": EXPECTED_TARGET_SAMPLES,
            "all_confusion_matrices_exact": True,
            "training_started": False,
            "target_used_for_selection_or_tuning": False,
        }
    except Exception as error:
        audit["status"] = "failed"
        audit["completed_at_utc"] = utc_now()
        audit["error_type"] = type(error).__name__
        audit["error"] = str(error)
        atomic_write_json(output, audit)
        raise

    atomic_write_json(output, audit)
    print(json.dumps({"status": "passed", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
