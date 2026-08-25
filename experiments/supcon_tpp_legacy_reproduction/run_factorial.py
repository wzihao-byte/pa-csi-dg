from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "factorial-s32-runner/v1"
COMPLETION_VERSION = "factorial-s32-attempt-completion/v1"
SEAL_VERSION = "factorial-s32-attempt-seal/v1"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_BUNDLE = EXPERIMENT_ROOT / "generated_factorial_s32_v1"
OUTPUT_ROOT = REPO_ROOT / "outputs_supcon_tpp_legacy_reproduction" / "factorial_s32"
ARM_ORDER = ("H00", "H01", "H10", "H11")
ATTEMPT_REQUIRED_FILES = (
    "job_config.json",
    "attempt_started.json",
    "runner.stdout.log",
    "runner.stderr.log",
)
RUN_REQUIRED_FILES = (
    "resolved_config.json",
    "split_manifest.json",
    "sampler_schedule.json",
    "best_model.pt",
    "checkpoint_frozen.json",
    "target_access_audit.json",
    "target_predictions.jsonl",
    "metrics.json",
    "confusion_matrix.npy",
    "confusion_matrix.csv",
)


class RunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Job:
    job_id: str
    target: str
    seed: int
    arm: str
    config_path: Path
    config_file_sha256: str
    config_payload_sha256: str

    @property
    def root(self) -> Path:
        return OUTPUT_ROOT / "runs" / self.target / f"seed_{self.seed}" / self.arm


@dataclass(frozen=True)
class Scan:
    action: str
    next_attempt: int
    successful_attempt: Path | None
    prior: tuple[Mapping[str, Any], ...]


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
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RunnerError(f"Expected JSON object: {path}")
    return value


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
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


def assert_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise RunnerError(f"{context}: expected {expected!r}, got {actual!r}")


def validate_finite(value: Any, context: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, np.integer)):
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise RunnerError(f"Non-finite value at {context}: {value}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            validate_finite(child, f"{context}.{key}")
        return
    if isinstance(value, Sequence):
        for index, child in enumerate(value):
            validate_finite(child, f"{context}[{index}]")


def git_output(arguments: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def verify_bundle(bundle_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[Job]]:
    bundle_root = bundle_root.resolve()
    required = (
        "bundle_manifest.json",
        "protocol_lock.json",
        "factorial_job_registry.json",
        "execution_lock.json",
    )
    for name in required:
        if not (bundle_root / name).is_file():
            raise RunnerError(f"Bundle member missing: {bundle_root / name}")
    manifest = read_json(bundle_root / "bundle_manifest.json")
    protocol = read_json(bundle_root / "protocol_lock.json")
    registry = read_json(bundle_root / "factorial_job_registry.json")
    execution = read_json(bundle_root / "execution_lock.json")
    assert_equal(sha256_file(bundle_root / "protocol_lock.json"), manifest["protocol_lock_sha256"], "manifest protocol lock")
    assert_equal(sha256_file(bundle_root / "factorial_job_registry.json"), manifest["registry_sha256"], "manifest registry")
    assert_equal(sha256_file(bundle_root / "execution_lock.json"), manifest["execution_lock_sha256"], "manifest execution lock")
    assert_equal(manifest["job_count"], 36, "manifest job count")
    assert_equal(registry["protocol_lock_sha256"], manifest["protocol_lock_sha256"], "registry protocol binding")
    assert_equal(execution["protocol_lock_sha256"], manifest["protocol_lock_sha256"], "execution protocol binding")
    assert_equal(execution["registry_sha256"], manifest["registry_sha256"], "execution registry binding")

    for name, record in execution["scripts"].items():
        path = Path(record["path"]).resolve()
        assert_equal(sha256_file(path), record["sha256"], f"execution script {name}")
    for name, record in execution["required_gate_artifacts"].items():
        path = Path(record["path"]).resolve()
        assert_equal(sha256_file(path), record["sha256"], f"gate artifact {name}")
    anchor_gate = read_json(Path(execution["required_gate_artifacts"]["anchor_matrix_completion"]["path"]))
    assert_equal(anchor_gate["status"], "passed", "anchor gate status")
    smoke_gate = read_json(Path(execution["required_gate_artifacts"]["h11_gpu_smoke"]["path"]))
    assert_equal(smoke_gate["status"], "pass", "H11 smoke status")

    clean_root = Path(protocol["provenance"]["clean_worktree"]).resolve()
    assert_equal(git_output(["rev-parse", "HEAD"], clean_root), protocol["provenance"]["legacy_commit"], "legacy commit")
    assert_equal(git_output(["status", "--porcelain"], clean_root), "", "clean legacy worktree")
    for name, record in protocol["provenance"]["legacy_modules"].items():
        assert_equal(sha256_file(Path(record["path"])), record["sha256"], f"legacy module {name}")
    tpp_path = Path(protocol["provenance"]["tpp_snapshot_root"]) / "experiments" / "tpp_isolated" / "tpp_models.py"
    assert_equal(sha256_file(tpp_path), protocol["provenance"]["tpp_models_sha256"], "TPP model snapshot")

    jobs: list[Job] = []
    observed_order: list[tuple[str, int, str]] = []
    for item in registry["jobs"]:
        config_path = bundle_root / item["config_path"]
        config = read_json(config_path)
        assert_equal(sha256_file(config_path), item["config_file_sha256"], f"config file {item['job_id']}")
        assert_equal(payload_sha256(config), item["config_payload_sha256"], f"config payload {item['job_id']}")
        assert_equal(config["job_id"], item["job_id"], "config job id")
        assert_equal(config["target_env"], item["target_env"], "config target")
        assert_equal(config["seed_list"], [item["seed"]], "config seed")
        assert_equal(config["factorial"]["arm"], item["arm"], "config arm")
        assert_equal(config["training"]["epochs"], 200, "config epochs")
        assert_equal(config["training"]["batch_size"], 32, "config batch size")
        assert_equal(config["training"]["sampler"], "domain_class_balanced", "config sampler")
        assert_equal(config["protocol"]["total_optimizer_steps"], 30000, "config steps")
        assert_equal(list(config["data"]["env_files"]), list(item["source_envs"]), "source-only config")
        if item["target_env"] in config["data"]["env_files"]:
            raise RunnerError(f"Target leaked into source config: {item['job_id']}")
        assert_equal(list(config["target_data"]["data"]["env_files"]), [item["target_env"]], "target-only config")
        jobs.append(
            Job(
                job_id=str(item["job_id"]),
                target=str(item["target_env"]),
                seed=int(item["seed"]),
                arm=str(item["arm"]),
                config_path=config_path,
                config_file_sha256=str(item["config_file_sha256"]),
                config_payload_sha256=str(item["config_payload_sha256"]),
            )
        )
        observed_order.append((str(item["target_env"]), int(item["seed"]), str(item["arm"])))
    expected_order = [
        (target, seed, arm)
        for target in protocol["targets"]
        for seed in protocol["seeds"]
        for arm in ARM_ORDER
    ]
    assert_equal(observed_order, expected_order, "36-job registry order")
    assert_equal(len(jobs), 36, "registry job count")
    return manifest, protocol, execution, jobs


def derived_config(job: Job, attempt: Path) -> dict[str, Any]:
    config = read_json(job.config_path)
    config["output_dir"] = str((attempt / "payload" / "run").resolve())
    return config


def tree_records(root: Path, excluded: Iterable[str] = ()) -> dict[str, dict[str, Any]]:
    excluded_set = set(excluded)
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded_set:
            relative = path.relative_to(root).as_posix()
            records[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return records


def checkpoint_probe(config_path: Path, checkpoint: Path, python: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(python), str(EXPERIMENT_ROOT / "train_factorial.py"), "--config", str(config_path), "--validate-checkpoint", str(checkpoint)],
        cwd=EXPERIMENT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RunnerError("Checkpoint probe emitted no output")
    value = json.loads(lines[-1])
    assert_equal(value["status"], "passed", "checkpoint probe status")
    assert_equal(value["target_files_opened"], False, "checkpoint probe target access")
    return value


def recompute_predictions(path: Path) -> tuple[np.ndarray, int]:
    confusion = np.zeros((6, 6), dtype=np.int64)
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            assert_equal(int(row["index"]), count, "ordered prediction index")
            label = int(row["label"])
            prediction = int(row["prediction"])
            probabilities = np.asarray(row["probabilities"], dtype=np.float64)
            if probabilities.shape != (6,) or not np.all(np.isfinite(probabilities)):
                raise RunnerError(f"Malformed probability row {count}")
            if abs(float(probabilities.sum()) - 1.0) > 1e-5:
                raise RunnerError(f"Probability row does not sum to one: {count}")
            if int(probabilities.argmax()) != prediction:
                raise RunnerError(f"Prediction/probability mismatch at row {count}")
            confusion[label, prediction] += 1
            count += 1
    assert_equal(count, 3000, "prediction count")
    return confusion, count


def confusion_metrics(confusion: np.ndarray) -> dict[str, float]:
    values = confusion.astype(np.float64)
    support = values.sum(axis=1)
    predicted = values.sum(axis=0)
    diagonal = np.diag(values)
    precision = np.divide(diagonal, predicted, out=np.zeros(6), where=predicted > 0)
    recall = np.divide(diagonal, support, out=np.zeros(6), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(6), where=(precision + recall) > 0)
    return {
        "accuracy": float(diagonal.sum() / values.sum()),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
    }


def validate_run(job: Job, attempt: Path, python: Path) -> dict[str, Any]:
    for name in ATTEMPT_REQUIRED_FILES:
        if not (attempt / name).is_file():
            raise RunnerError(f"Missing attempt artifact: {attempt / name}")
    run_dir = attempt / "payload" / "run"
    for name in RUN_REQUIRED_FILES:
        if not (run_dir / name).is_file():
            raise RunnerError(f"Missing run artifact: {run_dir / name}")
    config = read_json(attempt / "job_config.json")
    assert_equal(config, derived_config(job, attempt), "derived attempt config")
    resolved = read_json(run_dir / "resolved_config.json")
    device_actual = resolved.pop("device_actual", None)
    assert_equal(resolved, config, "resolved config")
    assert_equal(device_actual, "cuda:0", "actual CUDA device")

    metrics = read_json(run_dir / "metrics.json")
    validate_finite(metrics, "metrics")
    assert_equal(metrics["job_id"], job.job_id, "metrics job")
    assert_equal(metrics["arm"], job.arm, "metrics arm")
    assert_equal(metrics["target_env"], job.target, "metrics target")
    assert_equal(int(metrics["seed"]), job.seed, "metrics seed")
    assert_equal((metrics["train_size"], metrics["val_size"], metrics["test_size"]), (4800, 1200, 3000), "data sizes")
    assert_equal((metrics["epochs"], metrics["optimizer_steps_per_epoch"], metrics["total_optimizer_steps"]), (200, 150, 30000), "training schedule")
    assert_equal(len(metrics["history"]), 200, "history length")
    assert_equal([row["epoch"] for row in metrics["history"]], list(range(1, 201)), "history epochs")
    if job.arm in {"H01", "H11"}:
        coverage = [float(row["train"]["supcon_frac_anchors_with_positive"]) for row in metrics["history"]]
        if min(coverage) < 1.0:
            raise RunnerError(f"Positive-anchor coverage below 100% for {job.job_id}")

    schedule = read_json(run_dir / "sampler_schedule.json")
    assert_equal((schedule["epochs_seen"], schedule["batches_seen"], schedule["samples_seen"]), (200, 30000, 960000), "sampler schedule")
    assert_equal(schedule["global_sample_schedule_sha256"], metrics["sampler_schedule_sha256"], "sampler hash")
    split = read_json(run_dir / "split_manifest.json")
    expected_split = config["protocol"]["expected_split_hashes"]
    assert_equal(split["train"]["global_sample_id_sha256"], expected_split["train"], "train split hash")
    assert_equal(split["val"]["global_sample_id_sha256"], expected_split["val"], "val split hash")
    assert_equal(split["test"]["global_sample_id_sha256"], expected_split["test"], "test split hash")
    assert_equal((split["train"]["count"], split["val"]["count"], split["test"]["count"]), (4800, 1200, 3000), "split counts")

    checkpoint = run_dir / "best_model.pt"
    freeze = read_json(run_dir / "checkpoint_frozen.json")
    checkpoint_hash = sha256_file(checkpoint)
    assert_equal(freeze["status"], "frozen_and_strict_reloaded", "checkpoint freeze status")
    assert_equal(freeze["checkpoint_sha256"], checkpoint_hash, "checkpoint freeze hash")
    assert_equal(freeze["target_data_opened"], False, "freeze target access")
    access = read_json(run_dir / "target_access_audit.json")
    assert_equal(access["checkpoint_frozen_manifest_sha256"], sha256_file(run_dir / "checkpoint_frozen.json"), "target access freeze binding")
    assert_equal(access["checkpoint_sha256"], checkpoint_hash, "target access checkpoint binding")
    assert_equal(access["checkpoint_strict_reloaded_before_target_access"], True, "delayed target order")
    predictions_path = run_dir / "target_predictions.jsonl"
    assert_equal(access["target_predictions_file_sha256"], sha256_file(predictions_path), "prediction access hash")
    assert_equal(metrics["target_predictions_file_sha256"], sha256_file(predictions_path), "prediction metrics hash")

    derived_confusion, prediction_count = recompute_predictions(predictions_path)
    stored_confusion = np.load(run_dir / "confusion_matrix.npy", allow_pickle=False)
    if stored_confusion.shape != (6, 6) or not np.array_equal(stored_confusion, derived_confusion):
        raise RunnerError(f"Confusion matrix mismatch for {job.job_id}")
    with (run_dir / "confusion_matrix.csv").open("r", encoding="utf-8", newline="") as handle:
        csv_confusion = np.asarray([[int(value) for value in row] for row in csv.reader(handle)], dtype=np.int64)
    if not np.array_equal(csv_confusion, derived_confusion):
        raise RunnerError(f"CSV confusion mismatch for {job.job_id}")
    derived_metrics = confusion_metrics(derived_confusion)
    for name, expected in derived_metrics.items():
        if abs(float(metrics["test_metrics"][name]) - expected) > 1e-12:
            raise RunnerError(f"Metric mismatch {job.job_id}/{name}")

    probe = checkpoint_probe(attempt / "job_config.json", checkpoint, python)
    assert_equal(probe["checkpoint_sha256"], checkpoint_hash, "checkpoint probe hash")
    return {
        "run_dir": str(run_dir),
        "checkpoint_sha256": checkpoint_hash,
        "parameter_count": int(probe["parameter_count"]),
        "sampler_schedule_sha256": schedule["global_sample_schedule_sha256"],
        "split_hashes": {
            "train": split["train"]["global_sample_id_sha256"],
            "val": split["val"]["global_sample_id_sha256"],
            "test": split["test"]["global_sample_id_sha256"],
        },
        "target_prediction_count": prediction_count,
        "target_predictions_sha256": sha256_file(predictions_path),
        "confusion_sha256": sha256_file(run_dir / "confusion_matrix.npy"),
        "metrics_sha256": sha256_file(run_dir / "metrics.json"),
    }


def validate_completion(job: Job, attempt: Path, python: Path) -> dict[str, Any]:
    completion_path = attempt / "attempt_completed.json"
    seal_path = attempt / "attempt_seal.json"
    if completion_path.exists() != seal_path.exists():
        raise RunnerError(f"Partial completion/seal pair: {attempt}")
    completion = read_json(completion_path)
    seal = read_json(seal_path)
    assert_equal(completion["schema_version"], COMPLETION_VERSION, "completion schema")
    assert_equal(completion["status"], "success", "completion status")
    assert_equal(completion["job_id"], job.job_id, "completion job")
    assert_equal(seal["schema_version"], SEAL_VERSION, "seal schema")
    assert_equal(seal["completion_manifest_sha256"], sha256_file(completion_path), "completion seal")
    actual = tree_records(attempt, excluded=("attempt_completed.json", "attempt_seal.json"))
    assert_equal(actual, completion["artifacts"], "sealed artifact tree")
    health = validate_run(job, attempt, python)
    assert_equal(payload_sha256(health), completion["health_sha256"], "completion health")
    return health


def attempt_dirs(root: Path) -> list[tuple[int, Path]]:
    if not root.exists():
        return []
    records: list[tuple[int, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("attempt_"):
            raise RunnerError(f"Unexpected job-root member: {child}")
        try:
            number = int(child.name.removeprefix("attempt_"))
        except ValueError as exc:
            raise RunnerError(f"Malformed attempt directory: {child}") from exc
        records.append((number, child))
    return records


def scan(job: Job, python: Path) -> Scan:
    prior: list[Mapping[str, Any]] = []
    successes: list[Path] = []
    attempts = attempt_dirs(job.root)
    for number, attempt in attempts:
        completed = (attempt / "attempt_completed.json").exists() or (attempt / "attempt_seal.json").exists()
        failed = (attempt / "attempt_failed.json").exists()
        started = (attempt / "attempt_started.json").exists()
        if completed:
            health = validate_completion(job, attempt, python)
            successes.append(attempt)
            prior.append({"attempt": number, "status": "success", "path": str(attempt), "health": health})
        elif failed:
            if not started:
                raise RunnerError(f"Failed attempt lacks start record: {attempt}")
            prior.append({"attempt": number, "status": "failed", "path": str(attempt)})
        elif started:
            prior.append({"attempt": number, "status": "interrupted", "path": str(attempt)})
        else:
            raise RunnerError(f"Attempt lacks lifecycle record: {attempt}")
    if len(successes) > 1:
        raise RunnerError(f"Multiple successful attempts for {job.job_id}")
    return Scan(
        action="skip" if successes else "run",
        next_attempt=max((number for number, _ in attempts), default=0) + 1,
        successful_attempt=successes[0] if successes else None,
        prior=tuple(prior),
    )


class RunnerLock:
    def __init__(self, invocation_id: str):
        self.path = OUTPUT_ROOT / ".factorial_runner.lock"
        self.invocation_id = invocation_id
        self.descriptor: int | None = None

    def __enter__(self):
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            self.descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError as exc:
            raise RunnerError(f"Factorial runner lock already exists: {self.path}") from exc
        with os.fdopen(self.descriptor, "w", encoding="utf-8", closefd=False) as handle:
            json.dump({"schema_version": SCHEMA_VERSION, "invocation_id": self.invocation_id, "pid": os.getpid(), "created_at_utc": utc_now()}, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.descriptor is not None:
            os.close(self.descriptor)
        try:
            payload = read_json(self.path)
            if payload.get("invocation_id") == self.invocation_id:
                self.path.unlink()
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass


def execute_job(job: Job, attempt_number: int, python: Path) -> tuple[Path, dict[str, Any]]:
    attempt = job.root / f"attempt_{attempt_number:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    config = derived_config(job, attempt)
    config_path = attempt / "job_config.json"
    write_json_exclusive(config_path, config)
    write_json_exclusive(
        attempt / "attempt_started.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "started",
            "created_at_utc": utc_now(),
            "job_id": job.job_id,
            "target": job.target,
            "seed": job.seed,
            "arm": job.arm,
            "attempt": attempt_number,
            "source_config_file_sha256": job.config_file_sha256,
            "source_config_payload_sha256": job.config_payload_sha256,
        },
    )
    stdout_path = attempt / "runner.stdout.log"
    stderr_path = attempt / "runner.stderr.log"
    command = [str(python), str(EXPERIMENT_ROOT / "train_factorial.py"), "--config", str(config_path)]
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            completed = subprocess.run(command, cwd=EXPERIMENT_ROOT, stdout=stdout, stderr=stderr, check=False)
        if completed.returncode != 0:
            raise RunnerError(f"Training subprocess returned {completed.returncode}")
        health = validate_run(job, attempt, python)
        completion = {
            "schema_version": COMPLETION_VERSION,
            "status": "success",
            "created_at_utc": utc_now(),
            "job_id": job.job_id,
            "target": job.target,
            "seed": job.seed,
            "arm": job.arm,
            "attempt": attempt_number,
            "returncode": 0,
            "source_config_file_sha256": job.config_file_sha256,
            "source_config_payload_sha256": job.config_payload_sha256,
            "health_sha256": payload_sha256(health),
            "artifacts": tree_records(attempt, excluded=("attempt_completed.json", "attempt_seal.json")),
        }
        write_json_exclusive(attempt / "attempt_completed.json", completion)
        write_json_exclusive(
            attempt / "attempt_seal.json",
            {"schema_version": SEAL_VERSION, "created_at_utc": utc_now(), "completion_manifest_sha256": sha256_file(attempt / "attempt_completed.json")},
        )
        return attempt, health
    except BaseException as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "created_at_utc": utc_now(),
            "job_id": job.job_id,
            "attempt": attempt_number,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "command": command,
        }
        try:
            write_json_exclusive(attempt / "attempt_failed.json", failure)
        except FileExistsError:
            pass
        raise


def quartet_barrier(jobs: Sequence[Job], python: Path, target: str, seed: int) -> dict[str, Any]:
    quartet = [job for job in jobs if job.target == target and job.seed == seed]
    assert_equal([job.arm for job in quartet], list(ARM_ORDER), "quartet arm order")
    records: list[dict[str, Any]] = []
    for job in quartet:
        state = scan(job, python)
        if state.successful_attempt is None:
            raise RunnerError(f"Quartet barrier missing success: {job.job_id}")
        health = next(item["health"] for item in state.prior if item["status"] == "success")
        records.append({"job_id": job.job_id, "arm": job.arm, "attempt": str(state.successful_attempt), "health": health})
    assert_equal(len({item["health"]["sampler_schedule_sha256"] for item in records}), 1, "quartet sampler schedule")
    assert_equal(len({payload_sha256(item["health"]["split_hashes"]) for item in records}), 1, "quartet split hashes")
    return {
        "target": target,
        "seed": seed,
        "status": "passed",
        "sampler_schedule_sha256": records[0]["health"]["sampler_schedule_sha256"],
        "split_hashes": records[0]["health"]["split_hashes"],
        "arms": records,
    }


def finalize_results(jobs: Sequence[Job], python: Path, bundle_root: Path) -> Path:
    rows: list[dict[str, Any]] = []
    barriers: list[dict[str, Any]] = []
    for target in sorted({job.target for job in jobs}):
        for seed in sorted({job.seed for job in jobs if job.target == target}):
            barriers.append(quartet_barrier(jobs, python, target, seed))
    for job in jobs:
        state = scan(job, python)
        if state.successful_attempt is None:
            raise RunnerError(f"Cannot finalize missing job: {job.job_id}")
        attempt = state.successful_attempt
        metrics_path = attempt / "payload" / "run" / "metrics.json"
        metrics = read_json(metrics_path)
        rows.append(
            {
                "job_id": job.job_id,
                "target_env": job.target,
                "seed": job.seed,
                "arm": job.arm,
                "attempt_dir": str(attempt),
                "completion_manifest_sha256": sha256_file(attempt / "attempt_completed.json"),
                "metrics_file_sha256": sha256_file(metrics_path),
                "checkpoint_sha256": metrics["checkpoint_sha256"],
                "target_predictions_file_sha256": metrics["target_predictions_file_sha256"],
                "target_prediction_count": metrics["target_prediction_count"],
                "test_metrics": metrics["test_metrics"],
                "per_class_metrics": metrics["per_class_metrics"],
            }
        )
    result = {
        "schema_version": "factorial-s32-results/v1",
        "status": "complete",
        "created_at_utc": utc_now(),
        "job_count": len(rows),
        "bundle_manifest_sha256": sha256_file(bundle_root / "bundle_manifest.json"),
        "quartet_barriers": barriers,
        "rows": rows,
        "analysis_status": "not_yet_run",
        "no_target_metric_tuning": True,
    }
    output = OUTPUT_ROOT / "factorial_results.json"
    atomic_write_json(output, result)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append-only, fail-closed Factorial-S32 runner.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = args.python.resolve()
    if not python.is_file():
        raise SystemExit(f"Python executable missing: {python}")
    bundle_root = args.bundle.resolve()
    manifest, protocol, execution, jobs = verify_bundle(bundle_root)
    plans: list[dict[str, Any]] = []
    for job in jobs:
        state = scan(job, python)
        plans.append(
            {
                "job_id": job.job_id,
                "target_env": job.target,
                "seed": job.seed,
                "arm": job.arm,
                "action": state.action,
                "next_attempt": state.next_attempt,
                "successful_attempt": str(state.successful_attempt) if state.successful_attempt else None,
                "prior": list(state.prior),
            }
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": "dry_run" if args.dry_run else "execute",
        "bundle_manifest_sha256": sha256_file(bundle_root / "bundle_manifest.json"),
        "planned_jobs": len(plans),
        "jobs_to_run": sum(item["action"] == "run" for item in plans),
        "jobs_to_skip": sum(item["action"] == "skip" for item in plans),
        "jobs": plans,
    }
    if args.dry_run:
        if args.summary_only:
            compact = {key: summary[key] for key in ("schema_version", "mode", "bundle_manifest_sha256", "planned_jobs", "jobs_to_run", "jobs_to_skip")}
            compact["jobs"] = [{key: item[key] for key in ("job_id", "target_env", "seed", "arm", "action", "next_attempt")} for item in plans]
            print(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return

    invocation_id = uuid.uuid4().hex
    with RunnerLock(invocation_id):
        invocation = OUTPUT_ROOT / "runner_invocations" / invocation_id
        invocation.mkdir(parents=True, exist_ok=False)
        write_json_exclusive(invocation / "plan.json", summary)
        results: list[dict[str, Any]] = []
        current_block: tuple[str, int] | None = None
        for job, plan in zip(jobs, plans):
            block = (job.target, job.seed)
            if current_block is not None and block != current_block:
                barrier = quartet_barrier(jobs, python, current_block[0], current_block[1])
                write_json_exclusive(invocation / f"barrier_{current_block[0]}_seed_{current_block[1]}.json", barrier)
            current_block = block
            if plan["action"] == "skip":
                results.append({"job_id": job.job_id, "status": "skipped_healthy_success", "attempt": plan["successful_attempt"]})
                continue
            attempt, health = execute_job(job, int(plan["next_attempt"]), python)
            results.append({"job_id": job.job_id, "status": "success", "attempt": str(attempt), "health_sha256": payload_sha256(health)})
        if current_block is not None:
            barrier = quartet_barrier(jobs, python, current_block[0], current_block[1])
            write_json_exclusive(invocation / f"barrier_{current_block[0]}_seed_{current_block[1]}.json", barrier)
        write_json_exclusive(invocation / "results.json", {"schema_version": SCHEMA_VERSION, "status": "complete", "results": results})
        final = finalize_results(jobs, python, bundle_root)
        write_json_exclusive(invocation / "completed.json", {"schema_version": SCHEMA_VERSION, "status": "complete", "factorial_results": str(final), "factorial_results_sha256": sha256_file(final)})
    print(json.dumps({"status": "complete", "jobs": 36, "factorial_results": str(final)}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
