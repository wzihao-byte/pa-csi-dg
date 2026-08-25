from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import struct
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = "factorial-s32-train/v1"
ENV_ORDER = ("E1", "E2", "E3")
NUM_CLASSES = 6
INPUT_DIM = 90
EXPECTED_PARAMETERS = {
    "pa_csi_dg_lite": 2_788_918,
    "isolated_pa_csi_dg_lite_tpp": 3_721_334,
}


class FactorialTrainingError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_hash(values: Sequence[int] | np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise FactorialTrainingError(f"Expected JSON object: {path}")
    return payload


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
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


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_confusion_csv(path: Path, confusion: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(confusion.astype(np.int64).tolist())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def assert_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise FactorialTrainingError(f"{context}: expected {expected!r}, got {actual!r}")


def verify_hash(path: Path, expected: str, context: str) -> str:
    if not path.is_file():
        raise FactorialTrainingError(f"Missing {context}: {path}")
    actual = sha256_file(path)
    assert_equal(actual, expected, f"{context} SHA-256")
    return actual


def configure_imports(config: Mapping[str, Any]) -> dict[str, Any]:
    provenance = config["provenance"]
    clean_root = Path(provenance["clean_worktree"]).resolve()
    snapshot_root = Path(provenance["tpp_snapshot_root"]).resolve()
    legacy_modules = provenance["legacy_modules"]
    for module_name, record in legacy_modules.items():
        verify_hash(Path(record["path"]), str(record["sha256"]), f"legacy module {module_name}")

    clean_text = str(clean_root)
    sys.path[:] = [item for item in sys.path if str(Path(item or ".").resolve()) != clean_text]
    sys.path.insert(0, clean_text)
    shared_names = ("adaptive_kmargin", "dg_dataset", "dg_losses", "dg_models", "train_dg")
    for name in shared_names:
        sys.modules.pop(name, None)
    modules = {name: importlib.import_module(name) for name in shared_names}
    for name, module in modules.items():
        actual = Path(module.__file__).resolve()
        if clean_root not in actual.parents:
            raise FactorialTrainingError(f"{name} escaped clean worktree: {actual}")

    tpp_path = snapshot_root / "experiments" / "tpp_isolated" / "tpp_models.py"
    verify_hash(tpp_path, str(provenance["tpp_models_sha256"]), "TPP model snapshot")
    spec = importlib.util.spec_from_file_location("factorial_tpp_models", tpp_path)
    if spec is None or spec.loader is None:
        raise FactorialTrainingError(f"Cannot construct TPP module spec: {tpp_path}")
    tpp_models = importlib.util.module_from_spec(spec)
    sys.modules["factorial_tpp_models"] = tpp_models
    spec.loader.exec_module(tpp_models)
    modules["tpp_models"] = tpp_models
    modules["clean_root"] = clean_root
    return modules


def model_class(config: Mapping[str, Any], modules: Mapping[str, Any]):
    name = str(config["model"]["name"])
    if name == "pa_csi_dg_lite":
        return modules["dg_models"].PaCsiDGLite
    if name == "isolated_pa_csi_dg_lite_tpp":
        return modules["tpp_models"].IsolatedPaCsiDGLiteTPP
    raise FactorialTrainingError(f"Unsupported Factorial-S32 model: {name}")


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def load_state_strict(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, Mapping):
        raise FactorialTrainingError(f"Checkpoint is not a state dictionary: {path}")
    return state


def verify_source_files(config: Mapping[str, Any]) -> dict[str, str]:
    records: dict[str, str] = {}
    for env_name, files in config["source_file_records"].items():
        for kind, record in files.items():
            path = Path(record["path"])
            records[f"{env_name}/{kind}"] = verify_hash(
                path, str(record["sha256"]), f"source file {env_name}/{kind}"
            )
    return records


def remap_global_sample_ids(dataset) -> None:
    mapped = np.empty(len(dataset), dtype=np.int64)
    for local_env_id, env_name in enumerate(dataset.env_names):
        original_env_id = ENV_ORDER.index(env_name)
        indices = np.flatnonzero(dataset.env_ids == local_env_id)
        mapped[indices] = original_env_id * 3000 + np.arange(len(indices), dtype=np.int64)
    dataset.sample_ids = mapped


def source_split(dataset, seed: int, target_env: str, dg_dataset) -> dict[str, Any]:
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for local_env_id, env_name in enumerate(dataset.env_names):
        original_env_id = ENV_ORDER.index(env_name)
        env_indices = np.flatnonzero(dataset.env_ids == local_env_id).astype(np.int64)
        train_indices, val_indices = dg_dataset.safe_stratified_split(
            env_indices,
            dataset.labels[env_indices],
            test_ratio=0.2,
            seed=seed + original_env_id + 1,
        )
        train_parts.append(train_indices)
        val_parts.append(val_indices)
    train = np.sort(np.concatenate(train_parts))
    val = np.sort(np.concatenate(val_parts))
    return {
        "mode": "dg_loeo",
        "target_env": target_env,
        "source_envs": list(dataset.env_names),
        "train_indices": train.tolist(),
        "val_indices": val.tolist(),
        # Historical run_single_experiment always evaluates a test loader.
        # Reusing source validation here keeps target files unopened; this
        # diagnostic is discarded after the checkpoint is frozen.
        "test_indices": val.tolist(),
    }


def source_manifest(dataset, split: Mapping[str, Any], target_env: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mode": "dg_loeo_source_only_delayed_target",
        "target_env": target_env,
        "source_envs": list(split["source_envs"]),
        "dataset_size_source_only": len(dataset),
    }
    for name in ("train", "val"):
        indices = np.asarray(split[f"{name}_indices"], dtype=np.int64)
        global_ids = dataset.sample_ids[indices]
        result[name] = {
            "count": int(len(indices)),
            "env_counts": dataset.env_counts(indices),
            "label_counts": dataset.label_counts(indices),
            "global_sample_id_sha256": index_hash(global_ids),
        }
    result["test"] = {"count": 0, "status": "target_not_opened_during_source_training"}
    return result


class SamplerRecorder:
    def __init__(self, base_class, sample_ids: np.ndarray):
        self.base_class = base_class
        self.sample_ids = sample_ids
        self.instance = None

    def build_class(self):
        owner = self

        class RecordingSampler(owner.base_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.schedule_digest = hashlib.sha256()
                self.epochs_seen = 0
                self.batches_seen = 0
                self.samples_seen = 0
                owner.instance = self

            def __iter__(self):
                self.epochs_seen += 1
                self.schedule_digest.update(struct.pack("<Q", self.epochs_seen))
                for batch in super().__iter__():
                    global_ids = owner.sample_ids[np.asarray(batch, dtype=np.int64)]
                    self.schedule_digest.update(np.asarray(global_ids, dtype="<i8").tobytes())
                    self.batches_seen += 1
                    self.samples_seen += len(batch)
                    yield batch

        return RecordingSampler


def confusion_metrics(confusion: np.ndarray) -> tuple[dict[str, float], list[dict[str, Any]]]:
    confusion = confusion.astype(np.float64, copy=False)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(true_positive, predicted, out=np.zeros(NUM_CLASSES), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros(NUM_CLASSES), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(NUM_CLASSES), where=(precision + recall) > 0)
    accuracy = float(true_positive.sum() / max(1.0, confusion.sum()))
    metrics = {
        "accuracy": accuracy,
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
    }
    per_class = [
        {
            "class_id": class_id,
            "support": int(support[class_id]),
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
        }
        for class_id in range(NUM_CLASSES)
    ]
    return metrics, per_class


def evaluate_target(model, dataset, device: torch.device) -> tuple[list[dict[str, Any]], np.ndarray]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    rows: list[dict[str, Any]] = []
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    expected_index = 0
    model.eval()
    with torch.no_grad():
        for amplitude, phase, labels, _domains, sample_ids in loader:
            outputs = model(amplitude.to(device), phase.to(device))
            probabilities = torch.softmax(outputs["logits"], dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            labels_np = labels.numpy()
            sample_ids_np = sample_ids.numpy()
            for offset in range(len(labels_np)):
                sample_index = int(sample_ids_np[offset])
                assert_equal(sample_index, expected_index, "ordered target sample id")
                label = int(labels_np[offset])
                prediction = int(predictions[offset])
                probability_row = [float(value) for value in probabilities[offset]]
                if not np.all(np.isfinite(probability_row)):
                    raise FactorialTrainingError("Non-finite target probability")
                rows.append(
                    {
                        "index": expected_index,
                        "label": label,
                        "prediction": prediction,
                        "probabilities": probability_row,
                    }
                )
                confusion[label, prediction] += 1
                expected_index += 1
    assert_equal(expected_index, 3000, "target prediction count")
    return rows, confusion


def train(config_path: Path) -> None:
    config = read_json(config_path)
    modules = configure_imports(config)
    dg_dataset = modules["dg_dataset"]
    train_dg = modules["train_dg"]
    target_env = str(config["target_env"])
    seed = int(config["seed_list"][0])
    run_dir = Path(config["output_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        allowed = {"runner.stdout.log", "runner.stderr.log"}
        unexpected = [path.name for path in run_dir.iterdir() if path.name not in allowed]
        if unexpected:
            raise FactorialTrainingError(f"Run directory is not empty: {unexpected}")

    assert_equal(config["training"]["epochs"], 200, "epochs")
    assert_equal(config["training"]["batch_size"], 32, "batch size")
    assert_equal(config["training"]["sampler"], "domain_class_balanced", "sampler")
    assert_equal(config["protocol"]["optimizer_steps_per_epoch"], 150, "steps per epoch")
    assert_equal(config["protocol"]["total_optimizer_steps"], 30000, "total steps")
    source_hashes = verify_source_files(config)
    source_dataset = dg_dataset.load_dataset(config["data"], modules["clean_root"])
    remap_global_sample_ids(source_dataset)

    split = source_split(source_dataset, seed, target_env, dg_dataset)
    split_record = source_manifest(source_dataset, split, target_env)
    expected_hashes = config["protocol"]["expected_split_hashes"]
    assert_equal(split_record["train"]["global_sample_id_sha256"], expected_hashes["train"], "train split hash")
    assert_equal(split_record["val"]["global_sample_id_sha256"], expected_hashes["val"], "validation split hash")
    assert_equal(split_record["train"]["count"], 4800, "train count")
    assert_equal(split_record["val"]["count"], 1200, "validation count")

    selected_model_class = model_class(config, modules)
    train_dg.PaCsiDGLite = selected_model_class
    original_split_builder = train_dg.build_split_indices
    original_manifest_builder = train_dg.build_split_manifest
    original_sampler = train_dg.DomainClassBalancedBatchSampler
    recorder = SamplerRecorder(original_sampler, source_dataset.sample_ids)
    train_dg.DomainClassBalancedBatchSampler = recorder.build_class()
    train_dg.build_split_indices = lambda *args, **kwargs: copy.deepcopy(split)
    train_dg.build_split_manifest = lambda *_args, **_kwargs: copy.deepcopy(split_record)

    device = train_dg.choose_device(str(config["device"]))
    resolved = copy.deepcopy(config)
    resolved["device_actual"] = str(device)
    atomic_write_json(run_dir / "resolved_config.json", resolved)
    try:
        source_result = train_dg.run_single_experiment(
            dataset=source_dataset,
            config=resolved,
            seed=seed,
            target_env=target_env,
            output_dir=run_dir,
            device=device,
        )
    finally:
        train_dg.build_split_indices = original_split_builder
        train_dg.build_split_manifest = original_manifest_builder
        train_dg.DomainClassBalancedBatchSampler = original_sampler

    if recorder.instance is None:
        raise FactorialTrainingError("Domain-class sampler recorder was not instantiated")
    schedule = {
        "schema_version": "factorial-s32-sampler-schedule/v1",
        "epochs_seen": int(recorder.instance.epochs_seen),
        "batches_seen": int(recorder.instance.batches_seen),
        "samples_seen": int(recorder.instance.samples_seen),
        "global_sample_schedule_sha256": recorder.instance.schedule_digest.hexdigest(),
    }
    assert_equal(schedule["epochs_seen"], 200, "sampler epochs")
    assert_equal(schedule["batches_seen"], 30000, "sampler batches")
    assert_equal(schedule["samples_seen"], 960000, "sampler samples")
    atomic_write_json(run_dir / "sampler_schedule.json", schedule)

    history = source_result["history"]
    assert_equal(len(history), 200, "history epochs")
    if bool(config["factorial"]["supcon_on"]):
        coverage = [float(item["train"]["supcon_frac_anchors_with_positive"]) for item in history]
        if min(coverage) < 1.0:
            raise FactorialTrainingError(f"SupCon positive-anchor coverage below 100%: {min(coverage)}")

    checkpoint_path = run_dir / "best_model.pt"
    checkpoint_sha256 = sha256_file(checkpoint_path)
    fresh_model = selected_model_class(
        input_dim=source_dataset.input_dim,
        num_classes=source_dataset.num_classes,
        model_config=config["model"],
    ).to(device)
    expected_parameter_count = EXPECTED_PARAMETERS[str(config["model"]["name"])]
    assert_equal(parameter_count(fresh_model), expected_parameter_count, "model parameter count")
    state = load_state_strict(checkpoint_path)
    fresh_model.load_state_dict(state, strict=True)
    freeze_record = {
        "schema_version": "factorial-s32-checkpoint-freeze/v1",
        "status": "frozen_and_strict_reloaded",
        "created_at_utc": utc_now(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "strict_load": True,
        "parameter_count": expected_parameter_count,
        "best_val_score": float(source_result["best_val_score"]),
        "target_data_opened": False,
    }
    atomic_write_json(run_dir / "checkpoint_frozen.json", freeze_record)
    freeze_manifest_sha256 = sha256_file(run_dir / "checkpoint_frozen.json")

    # The first target-file read in this process happens below, after the
    # checkpoint freeze manifest exists and a fresh model has strict-loaded it.
    target_access_started = utc_now()
    target_hashes: dict[str, str] = {}
    for kind, record in config["target_data"]["file_records"].items():
        path = Path(record["path"])
        target_hashes[kind] = verify_hash(path, str(record["sha256"]), f"target file {target_env}/{kind}")
    target_dataset = dg_dataset.load_dataset(config["target_data"]["data"], modules["clean_root"])
    assert_equal(list(target_dataset.env_names), [target_env], "target-only environment")
    assert_equal(len(target_dataset), 3000, "target dataset size")

    del source_dataset
    gc.collect()
    rows, confusion = evaluate_target(fresh_model, target_dataset, device)
    metrics, per_class = confusion_metrics(confusion)
    predictions_path = run_dir / "target_predictions.jsonl"
    atomic_write_jsonl(predictions_path, rows)
    atomic_save_npy(run_dir / "confusion_matrix.npy", confusion)
    atomic_write_confusion_csv(run_dir / "confusion_matrix.csv", confusion)

    final_split_manifest = copy.deepcopy(split_record)
    final_split_manifest["test"] = {
        "count": 3000,
        "env_counts": {target_env: 3000},
        "global_sample_id_sha256": str(expected_hashes["test"]),
        "ordered_predictions": True,
    }
    atomic_write_json(run_dir / "split_manifest.json", final_split_manifest)

    result = {
        "schema_version": SCHEMA_VERSION,
        "job_id": config["job_id"],
        "arm": config["factorial"]["arm"],
        "target_env": target_env,
        "source_envs": list(config["protocol"]["source_envs"]),
        "seed": seed,
        "selection_metric": "accuracy",
        "best_val_score": float(source_result["best_val_score"]),
        "best_val_accuracy": float(source_result["best_val_accuracy"]),
        "train_size": 4800,
        "val_size": 1200,
        "test_size": 3000,
        "epochs": 200,
        "optimizer_steps_per_epoch": 150,
        "total_optimizer_steps": 30000,
        "sampler_mode": "domain_class_balanced",
        "sampler_schedule_sha256": schedule["global_sample_schedule_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "target_prediction_count": 3000,
        "target_predictions_file_sha256": sha256_file(predictions_path),
        "test_metrics": metrics,
        "per_class_metrics": per_class,
        "history": history,
        "source_file_sha256": source_hashes,
        "target_file_sha256": target_hashes,
        "factorial": copy.deepcopy(config["factorial"]),
    }
    atomic_write_json(run_dir / "metrics.json", result)
    target_access = {
        "schema_version": "factorial-s32-target-access/v1",
        "checkpoint_frozen_manifest": str(run_dir / "checkpoint_frozen.json"),
        "checkpoint_frozen_manifest_sha256": freeze_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_strict_reloaded_before_target_access": True,
        "target_access_started_at_utc": target_access_started,
        "target_access_completed_at_utc": utc_now(),
        "target_file_sha256": target_hashes,
        "target_prediction_count": 3000,
        "target_predictions_file_sha256": sha256_file(predictions_path),
    }
    atomic_write_json(run_dir / "target_access_audit.json", target_access)
    print(
        json.dumps(
            {
                "status": "success",
                "job_id": config["job_id"],
                "checkpoint_sha256": checkpoint_sha256,
                "target_prediction_count": 3000,
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def validate_checkpoint(config_path: Path, checkpoint_path: Path) -> None:
    config = read_json(config_path)
    modules = configure_imports(config)
    cls = model_class(config, modules)
    model = cls(input_dim=INPUT_DIM, num_classes=NUM_CLASSES, model_config=config["model"])
    expected = EXPECTED_PARAMETERS[str(config["model"]["name"])]
    assert_equal(parameter_count(model), expected, "checkpoint validation parameter count")
    model.load_state_dict(load_state_strict(checkpoint_path), strict=True)
    print(
        json.dumps(
            {
                "status": "passed",
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "parameter_count": expected,
                "strict_load": True,
                "target_files_opened": False,
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked Factorial-S32 training adapter.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if args.validate_checkpoint is not None:
        validate_checkpoint(config_path, args.validate_checkpoint.resolve())
    else:
        train(config_path)


if __name__ == "__main__":
    main()
