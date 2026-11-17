"""Safely run and resume the locked formal interim or extension matrix.

The executor is intentionally serial and fail closed.  It never removes an
attempt directory, skips a job only after revalidating every formal artifact,
and crosses a target/seed barrier only when all four factorial arms have an
identical source split, deferred-target manifest, calibration, initialization,
and batch schedule.  A target fold is completed before the next fold begins.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch

try:  # Support module and direct-script execution.
    from . import analyze_interim, check_protocol, protocol, select_lambda
    from .runner_utils import fixed_label_metrics
except ImportError:  # pragma: no cover
    import analyze_interim  # type: ignore
    import check_protocol  # type: ignore
    import protocol  # type: ignore
    import select_lambda  # type: ignore
    from runner_utils import fixed_label_metrics  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_MODULE = "experiments.supcon_tpp_joint.train_joint"
FORMAL_RESULTS_SCHEMA = "supcon-tpp-formal-results/v1"
ATTEMPT_MANIFEST_SCHEMA = "supcon-tpp-formal-attempt-manifest/v1"
EXECUTION_LOCK_SCHEMA = "supcon-tpp-formal-execution-lock/v1"

EXPECTED_JOB_KEYS = {
    "job_id",
    "stage",
    "target_env",
    "source_envs",
    "seed",
    "arm",
    "lambda_supcon",
    "selected_lambda",
    "selection_record_file_sha256",
    "interim_decision_file_sha256",
    "config_path",
    "config_content_sha256",
    "config_file_sha256",
}

RUN_REGISTRY_FIELDS = [
    "job_id",
    "stage",
    "target_env",
    "source_envs",
    "seed",
    "arm",
    "lambda_supcon",
    "selected_lambda",
    "selection_record_file_sha256",
    "interim_decision_file_sha256",
    "config_path",
    "config_content_sha256",
    "config_file_sha256",
    "formal_registry_file_sha256",
    "formal_registry_content_sha256",
    "protocol_lock_file_sha256",
    "bundle_manifest_file_sha256",
    "execution_lock_file_sha256",
    "status",
    "attempt",
    "output_dir",
    "command",
    "stdout_log",
    "stderr_log",
    "started_at_utc",
    "completed_at_utc",
    "elapsed_seconds",
    "return_code",
    "metrics_path",
    "metrics_file_sha256",
    "target_metrics_file_sha256",
    "target_predictions_file_sha256",
    "best_model_file_sha256",
    "split_manifest_file_sha256",
    "attempt_manifest_file_sha256",
    "error",
]
IMMUTABLE_RUN_FIELDS = RUN_REGISTRY_FIELDS[: RUN_REGISTRY_FIELDS.index("status")]
VALID_RUN_STATUSES = {"planned", "running", "completed", "failed"}


class FormalExecutionError(RuntimeError):
    """A formal input, output, or recovery attempt failed closed."""


@dataclass(frozen=True)
class FormalJob:
    job_id: str
    stage: str
    target_env: str
    source_envs: tuple[str, ...]
    seed: int
    arm: str
    lambda_supcon: float
    selected_lambda: float
    selection_record_file_sha256: str
    interim_decision_file_sha256: Optional[str]
    config_path: Path
    config_content_sha256: str
    config_file_sha256: str
    calibration_artifact_id: str
    config: Mapping[str, Any]


@dataclass(frozen=True)
class FormalPlan:
    stage: str
    bundle_dir: Path
    registry_path: Path
    registry_file_sha256: str
    registry_content_sha256: str
    protocol_lock_file_sha256: str
    bundle_manifest_file_sha256: str
    registry: Mapping[str, Any]
    jobs: tuple[FormalJob, ...]


CommandRunner = Callable[..., Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def exclusive_runner_lock(lock_path: Path):
    """Hold one nonblocking OS byte lock for the whole executor invocation.

    The lock file intentionally persists.  Ownership is the OS lock, which is
    released automatically on abnormal process exit, so recovery never deletes
    a stale sentinel and cannot race a still-live runner.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except (OSError, BlockingIOError) as exc:
            raise FormalExecutionError(
                f"Another formal runner already owns output root {lock_path.parent}."
            ) from exc
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


def _load_json_object(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalExecutionError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FormalExecutionError(f"Expected a JSON object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def training_code_sha256() -> Dict[str, str]:
    """Hash the executable training surface frozen across the formal matrix."""

    relative_paths = (
        Path("dg_dataset.py"),
        Path("dg_losses.py"),
        Path("dg_models.py"),
        Path("experiments/tpp_isolated/tpp_models.py"),
        Path("experiments/supcon_tpp_joint/joint_models.py"),
        Path("experiments/supcon_tpp_joint/analyze_interim.py"),
        Path("experiments/supcon_tpp_joint/check_protocol.py"),
        Path("experiments/supcon_tpp_joint/generate_configs.py"),
        Path("experiments/supcon_tpp_joint/protocol.py"),
        Path("experiments/supcon_tpp_joint/run_formal.py"),
        Path("experiments/supcon_tpp_joint/runner_utils.py"),
        Path("experiments/supcon_tpp_joint/screen_health.py"),
        Path("experiments/supcon_tpp_joint/select_lambda.py"),
        Path("experiments/supcon_tpp_joint/train_joint.py"),
    )
    result: Dict[str, str] = {}
    for relative in relative_paths:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FormalExecutionError(f"Training-code dependency is absent: {path}")
        result[relative.as_posix()] = _file_sha256(path)
    return result


def execution_lock_payload(
    plan: FormalPlan,
    *,
    device: str,
    formal_output_root: Path,
) -> Dict[str, Any]:
    runtime: Dict[str, Any] = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": importlib.metadata.version("scipy"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "requested_device": device,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index
        index = torch.cuda.current_device() if index is None else index
        runtime["cuda_device_index"] = int(index)
        runtime["cuda_device_name"] = torch.cuda.get_device_name(index)
        runtime["cuda_device_capability"] = list(torch.cuda.get_device_capability(index))
    return {
        "schema_version": EXECUTION_LOCK_SCHEMA,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "stage": plan.stage,
        "formal_registry_file_sha256": plan.registry_file_sha256,
        "formal_registry_content_sha256": plan.registry_content_sha256,
        "formal_output_root": str(formal_output_root.resolve()),
        "training_code_sha256": training_code_sha256(),
        "runtime": runtime,
    }


def _selection_artifacts(bundle_dir: Path) -> Dict[str, select_lambda.SelectionArtifact]:
    paths = [
        bundle_dir / "selection_records" / f"selection_{target}.json"
        for target in protocol.ENVIRONMENTS
    ]
    try:
        return select_lambda.load_selection_record_files(paths)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FormalExecutionError(f"Formal selection records failed validation: {exc}") from exc


def _validate_job_file(
    raw_job: Mapping[str, Any],
    *,
    stage: str,
    bundle_dir: Path,
) -> tuple[FormalJob, Dict[str, Any]]:
    job_id = str(raw_job.get("job_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
        raise FormalExecutionError(f"Unsafe or empty formal job_id: {job_id!r}")
    if set(raw_job) != EXPECTED_JOB_KEYS:
        raise FormalExecutionError(
            f"Registry row {job_id} schema drifted; expected {sorted(EXPECTED_JOB_KEYS)}, "
            f"received {sorted(raw_job)}."
        )
    if raw_job.get("stage") != stage:
        raise FormalExecutionError(f"Registry row {job_id} has the wrong formal stage.")

    relative_config = Path(str(raw_job["config_path"]))
    expected_relative = Path(f"{stage}_configs") / f"{job_id}.json"
    if relative_config.is_absolute() or relative_config.as_posix() != expected_relative.as_posix():
        raise FormalExecutionError(f"Registry config_path is not canonical for {job_id}.")
    config_path = (bundle_dir / relative_config).resolve()
    if not _inside(config_path, bundle_dir) or not config_path.is_file():
        raise FormalExecutionError(f"Missing or out-of-bundle config for {job_id}: {config_path}")
    expected_file_hash = str(raw_job["config_file_sha256"])
    if _file_sha256(config_path) != expected_file_hash:
        raise FormalExecutionError(f"Config file hash mismatch for {job_id}.")
    config = _load_json_object(config_path)
    expected_content_hash = str(raw_job["config_content_sha256"])
    if protocol.canonical_sha256(config) != expected_content_hash:
        raise FormalExecutionError(f"Config content hash mismatch for {job_id}.")

    target_env = str(raw_job["target_env"])
    selection_hash = str(raw_job["selection_record_file_sha256"])
    selection_path = bundle_dir / "selection_records" / f"selection_{target_env}.json"
    if not selection_path.is_file() or _file_sha256(selection_path) != selection_hash:
        raise FormalExecutionError(f"Selection record/hash mismatch for {job_id}.")
    decision_hash_raw = raw_job.get("interim_decision_file_sha256")
    decision_hash = None if decision_hash_raw is None else str(decision_hash_raw)
    if stage == "formal_interim" and decision_hash is not None:
        raise FormalExecutionError("Interim jobs must not bind an interim decision.")
    if stage == "formal_extension":
        decision_path = bundle_dir / "interim_decision.json"
        if not decision_hash or not decision_path.is_file() or _file_sha256(decision_path) != decision_hash:
            raise FormalExecutionError(f"Interim decision/hash mismatch for extension job {job_id}.")

    calibration = config.get("calibration")
    evaluation = config.get("evaluation")
    training = config.get("training")
    if not isinstance(calibration, Mapping) or bool(calibration.get("uses_target_data", True)):
        raise FormalExecutionError(f"Config {job_id} has no source-only calibration binding.")
    artifact_id = str(calibration.get("artifact_id", ""))
    if not artifact_id:
        raise FormalExecutionError(f"Config {job_id} has an empty calibration artifact ID.")
    if not isinstance(evaluation, Mapping) or evaluation.get("evaluate_target") is not True:
        raise FormalExecutionError(f"Formal config {job_id} must authorize deferred target evaluation.")
    if (
        not isinstance(training, Mapping)
        or int(training.get("epochs", -1)) != protocol.EPOCHS
        or int(training.get("optimizer_steps_per_epoch", -1)) != protocol.OPTIMIZER_STEPS_PER_EPOCH
        or int(training.get("total_optimizer_steps", -1)) != protocol.TOTAL_OPTIMIZER_STEPS
    ):
        raise FormalExecutionError(f"Formal config {job_id} does not preserve the 200/300/60000 budget.")

    job = FormalJob(
        job_id=job_id,
        stage=stage,
        target_env=target_env,
        source_envs=tuple(map(str, raw_job["source_envs"])),
        seed=int(raw_job["seed"]),
        arm=str(raw_job["arm"]),
        lambda_supcon=float(raw_job["lambda_supcon"]),
        selected_lambda=float(raw_job["selected_lambda"]),
        selection_record_file_sha256=selection_hash,
        interim_decision_file_sha256=decision_hash,
        config_path=config_path,
        config_content_sha256=expected_content_hash,
        config_file_sha256=expected_file_hash,
        calibration_artifact_id=artifact_id,
        config=config,
    )
    return job, dict(raw_job) | {"config": config}


def load_formal_plan(registry_path: Path) -> FormalPlan:
    """Validate a complete generated bundle and return a barrier-ordered plan."""

    resolved_registry = registry_path.resolve()
    bundle_dir = resolved_registry.parent.resolve()
    registry = _load_json_object(resolved_registry)
    stage = str(registry.get("stage", ""))
    if stage not in {"formal_interim", "formal_extension"}:
        raise FormalExecutionError(f"Unsupported formal registry stage: {stage!r}")
    expected_name = f"{stage}_job_registry.json"
    if resolved_registry.name != expected_name:
        raise FormalExecutionError(f"Formal registry filename must be {expected_name}.")
    try:
        bundle_summary = check_protocol.check_bundle(bundle_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FormalExecutionError(f"Generated formal bundle failed the full protocol checker: {exc}") from exc
    if stage not in bundle_summary:
        raise FormalExecutionError(f"Bundle does not authorize registry stage {stage}.")

    selections = _selection_artifacts(bundle_dir)
    decision: Optional[Mapping[str, Any]] = None
    decision_hash: Optional[str] = None
    if stage == "formal_extension":
        decision_path = bundle_dir / "interim_decision.json"
        interim_registry = _load_json_object(bundle_dir / "formal_interim_job_registry.json")
        try:
            decision = analyze_interim.validate_decision_artifact(
                decision_path,
                expected_registry_content_sha256=protocol.canonical_sha256(interim_registry),
                expected_selection_record_file_sha256={
                    target: selections[target].file_sha256 for target in protocol.ENVIRONMENTS
                },
                require_continue=True,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise FormalExecutionError(
                "Formal extension is blocked until the locked analyzer independently reproduces a "
                f"continue decision from all 36 formal outputs: {exc}"
            ) from exc
        decision_hash = _file_sha256(decision_path)

    raw_jobs = registry.get("jobs")
    expected_count = protocol.BUDGETS[
        "formal_interim_jobs" if stage == "formal_interim" else "formal_extension_jobs"
    ]
    if not isinstance(raw_jobs, list) or len(raw_jobs) != expected_count:
        raise FormalExecutionError(f"{stage} registry must contain exactly {expected_count} jobs.")
    jobs: List[FormalJob] = []
    hydrated: List[dict] = []
    for raw_job in raw_jobs:
        if not isinstance(raw_job, Mapping):
            raise FormalExecutionError("Every formal registry row must be an object.")
        job, hydrated_row = _validate_job_file(raw_job, stage=stage, bundle_dir=bundle_dir)
        jobs.append(job)
        hydrated.append(hydrated_row)
    try:
        check_protocol.check_formal_job_registry(
            dict(registry) | {"jobs": hydrated},
            stage=stage,
            selections=selections,
            decision=decision,
            decision_file_sha256=decision_hash,
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise FormalExecutionError(f"Formal registry/config protocol validation failed: {exc}") from exc

    arm_order = {arm: index for index, arm in enumerate(protocol.ARMS)}
    ordered: List[FormalJob] = []
    for target in protocol.ENVIRONMENTS:
        expected_seeds = protocol.formal_seeds(target, stage)
        fold_jobs = [job for job in jobs if job.target_env == target]
        if {job.calibration_artifact_id for job in fold_jobs} != {f"global_basis_calibration_{target}"}:
            raise FormalExecutionError(f"Formal calibration ID drifted inside target fold {target}.")
        if {job.selection_record_file_sha256 for job in fold_jobs} != {selections[target].file_sha256}:
            raise FormalExecutionError(f"Selection-record binding drifted inside target fold {target}.")
        for seed in expected_seeds:
            block = [job for job in fold_jobs if job.seed == seed]
            if {job.arm for job in block} != set(protocol.ARMS) or len(block) != len(protocol.ARMS):
                raise FormalExecutionError(f"Formal block {target}/seed={seed} is not a complete four-arm block.")
            ordered.extend(sorted(block, key=lambda job: arm_order[job.arm]))

    lock_path = bundle_dir / "protocol_lock.json"
    manifest_path = bundle_dir / "bundle_manifest.json"
    return FormalPlan(
        stage=stage,
        bundle_dir=bundle_dir,
        registry_path=resolved_registry,
        registry_file_sha256=_file_sha256(resolved_registry),
        registry_content_sha256=protocol.canonical_sha256(registry),
        protocol_lock_file_sha256=_file_sha256(lock_path),
        bundle_manifest_file_sha256=_file_sha256(manifest_path),
        registry=registry,
        jobs=tuple(ordered),
    )


def _revalidate_launch_binding(
    plan: FormalPlan,
    job: FormalJob,
    *,
    execution_lock_path: Path,
    execution_lock_file_sha256: str,
    device: str,
) -> None:
    """Cheap per-launch recheck of every external binding used by this job."""

    try:
        checked = check_protocol.check_bundle(plan.bundle_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FormalExecutionError(
            f"Generated formal bundle failed its pre-launch checker before {job.job_id}: {exc}"
        ) from exc
    if plan.stage not in checked:
        raise FormalExecutionError(f"Bundle no longer authorizes {plan.stage} before {job.job_id}.")
    fixed = {
        plan.registry_path: plan.registry_file_sha256,
        plan.bundle_dir / "protocol_lock.json": plan.protocol_lock_file_sha256,
        plan.bundle_dir / "bundle_manifest.json": plan.bundle_manifest_file_sha256,
        execution_lock_path: execution_lock_file_sha256,
        job.config_path: job.config_file_sha256,
        plan.bundle_dir / "selection_records" / f"selection_{job.target_env}.json": (
            job.selection_record_file_sha256
        ),
    }
    if job.interim_decision_file_sha256 is not None:
        fixed[plan.bundle_dir / "interim_decision.json"] = job.interim_decision_file_sha256
    for path, expected_hash in fixed.items():
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise FormalExecutionError(f"Hash-bound formal input changed before {job.job_id}: {path}")
    if _load_json_object(execution_lock_path) != execution_lock_payload(
        plan,
        device=device,
        formal_output_root=execution_lock_path.parent,
    ):
        raise FormalExecutionError(
            f"Training code/runtime fingerprint changed before {job.job_id}; formal execution is frozen."
        )
    if protocol.canonical_sha256(_load_json_object(job.config_path)) != job.config_content_sha256:
        raise FormalExecutionError(f"Formal config content changed before {job.job_id}.")


def _base_run_row(
    job: FormalJob,
    plan: FormalPlan,
    execution_lock_file_sha256: str,
) -> Dict[str, str]:
    return {
        "job_id": job.job_id,
        "stage": job.stage,
        "target_env": job.target_env,
        "source_envs": json.dumps(job.source_envs, separators=(",", ":")),
        "seed": str(job.seed),
        "arm": job.arm,
        "lambda_supcon": format(job.lambda_supcon, ".17g"),
        "selected_lambda": format(job.selected_lambda, ".17g"),
        "selection_record_file_sha256": job.selection_record_file_sha256,
        "interim_decision_file_sha256": job.interim_decision_file_sha256 or "",
        "config_path": str(job.config_path),
        "config_content_sha256": job.config_content_sha256,
        "config_file_sha256": job.config_file_sha256,
        "formal_registry_file_sha256": plan.registry_file_sha256,
        "formal_registry_content_sha256": plan.registry_content_sha256,
        "protocol_lock_file_sha256": plan.protocol_lock_file_sha256,
        "bundle_manifest_file_sha256": plan.bundle_manifest_file_sha256,
        "execution_lock_file_sha256": execution_lock_file_sha256,
        "status": "planned",
        "attempt": "0",
        "output_dir": "",
        "command": "",
        "stdout_log": "",
        "stderr_log": "",
        "started_at_utc": "",
        "completed_at_utc": "",
        "elapsed_seconds": "",
        "return_code": "",
        "metrics_path": "",
        "metrics_file_sha256": "",
        "target_metrics_file_sha256": "",
        "target_predictions_file_sha256": "",
        "best_model_file_sha256": "",
        "split_manifest_file_sha256": "",
        "attempt_manifest_file_sha256": "",
        "error": "",
    }


def _load_run_rows(
    path: Path,
    plan: FormalPlan,
    execution_lock_file_sha256: str,
) -> Dict[str, Dict[str, str]]:
    expected = {
        job.job_id: _base_run_row(job, plan, execution_lock_file_sha256)
        for job in plan.jobs
    }
    if not path.is_file():
        return expected
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RUN_REGISTRY_FIELDS:
            raise FormalExecutionError("run_registry.csv schema drifted.")
        observed: Dict[str, Dict[str, str]] = {}
        for raw in reader:
            row = {key: str(raw.get(key, "")) for key in RUN_REGISTRY_FIELDS}
            job_id = row["job_id"]
            if job_id in observed or job_id not in expected:
                raise FormalExecutionError(f"Duplicate or unknown formal run row: {job_id}")
            for key in IMMUTABLE_RUN_FIELDS:
                if row[key] != expected[job_id][key]:
                    raise FormalExecutionError(f"run_registry.csv binding mismatch for {job_id}.{key}.")
            if row["status"] not in VALID_RUN_STATUSES:
                raise FormalExecutionError(f"Invalid run status for {job_id}: {row['status']!r}")
            try:
                attempt = int(row["attempt"])
            except ValueError as exc:
                raise FormalExecutionError(f"Invalid attempt counter for {job_id}.") from exc
            if attempt < 0:
                raise FormalExecutionError(f"Negative attempt counter for {job_id}.")
            output_root = path.resolve().parent
            for location_field in ("output_dir", "stdout_log", "stderr_log"):
                location = row.get(location_field)
                if location and not _inside(Path(location).resolve(), output_root):
                    raise FormalExecutionError(
                        f"run_registry.csv {location_field} escapes the formal output root for {job_id}."
                    )
            observed[job_id] = row
    if set(observed) != set(expected):
        raise FormalExecutionError("run_registry.csv does not cover the exact formal job set.")
    return observed


def _write_run_rows(path: Path, rows: Mapping[str, Mapping[str, str]], jobs: Sequence[FormalJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_REGISTRY_FIELDS)
        writer.writeheader()
        for job in jobs:
            writer.writerow({key: rows[job.job_id].get(key, "") for key in RUN_REGISTRY_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _attempt_output_dir(output_root: Path, job: FormalJob, attempt: int) -> Path:
    return output_root / "runs" / job.target_env / f"seed_{job.seed}" / job.job_id / f"attempt_{attempt:03d}"


def _attempt_log_paths(output_root: Path, job: FormalJob, attempt: int) -> tuple[Path, Path]:
    log_dir = output_root / "logs" / job.target_env / f"seed_{job.seed}" / job.job_id
    return log_dir / f"attempt_{attempt:03d}.stdout.log", log_dir / f"attempt_{attempt:03d}.stderr.log"


def build_train_command(job: FormalJob, output_dir: Path, *, device: str) -> List[str]:
    return [
        sys.executable,
        "-m",
        TRAIN_MODULE,
        "--config",
        str(job.config_path),
        "--seed",
        str(job.seed),
        "--evaluate-target",
        "true",
        "--device",
        device,
        "--output-dir",
        str(output_dir),
    ]


def _prepare_attempt(
    row: MutableMapping[str, str],
    job: FormalJob,
    output_root: Path,
    *,
    device: str,
) -> tuple[Path, List[str], Path, Path]:
    reuse = row["status"] == "planned" and row["output_dir"] and int(row["attempt"]) > 0
    if reuse:
        attempt = int(row["attempt"])
        run_output = Path(row["output_dir"]).resolve()
        if run_output.exists() and any(run_output.iterdir()):
            reuse = False
    if not reuse:
        attempt = int(row["attempt"] or 0) + 1
        run_output = _attempt_output_dir(output_root, job, attempt).resolve()
        while run_output.exists() and any(run_output.iterdir()):
            attempt += 1
            run_output = _attempt_output_dir(output_root, job, attempt).resolve()
    command = build_train_command(job, run_output, device=device)
    stdout_log, stderr_log = _attempt_log_paths(output_root, job, attempt)
    row.update(
        {
            "status": "planned",
            "attempt": str(attempt),
            "output_dir": str(run_output),
            "command": subprocess.list2cmdline(command),
            "stdout_log": str(stdout_log.resolve()),
            "stderr_log": str(stderr_log.resolve()),
            "started_at_utc": "",
            "completed_at_utc": "",
            "elapsed_seconds": "",
            "return_code": "",
            "metrics_path": "",
            "metrics_file_sha256": "",
            "target_metrics_file_sha256": "",
            "target_predictions_file_sha256": "",
            "best_model_file_sha256": "",
            "split_manifest_file_sha256": "",
            "attempt_manifest_file_sha256": "",
            "error": "",
        }
    )
    return run_output, command, stdout_log, stderr_log


def run_logged_command(
    command: List[str],
    *,
    cwd: Path,
    check: bool,
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.CompletedProcess[Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        return subprocess.run(command, cwd=cwd, check=check, stdout=stdout, stderr=stderr)


def _require_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FormalExecutionError(f"Required formal artifact is absent: {path}")
    return _load_json_object(path)


def _validate_prediction_file(path: Path, target_env: str) -> Dict[str, Any]:
    seen: set[str] = set()
    identities: List[list[Any]] = []
    labels: List[int] = []
    predictions: List[int] = []
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise ValueError("prediction row is not an object")
                sample_id = str(record.get("sample_id", ""))
                group_id = str(record.get("group_id", ""))
                probability = record.get("probability")
                true = record.get("true")
                pred = record.get("pred")
                valid_class_ids = set(protocol.LABELS)
                valid_probabilities = (
                    isinstance(probability, list)
                    and len(probability) == len(protocol.LABELS)
                    and all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        and float(value) >= 0.0
                        for value in probability
                    )
                )
                argmax = (
                    max(range(len(probability)), key=lambda index: float(probability[index]))
                    if valid_probabilities
                    else None
                )
                if (
                    record.get("environment") != target_env
                    or not sample_id.startswith(f"{target_env}:")
                    or not group_id
                    or not group_id.startswith(f"{target_env}:")
                    or sample_id in seen
                    or not isinstance(true, int)
                    or isinstance(true, bool)
                    or true not in valid_class_ids
                    or not isinstance(pred, int)
                    or isinstance(pred, bool)
                    or pred not in valid_class_ids
                    or not valid_probabilities
                    or abs(sum(float(value) for value in probability) - 1.0) > 1e-5
                    or pred != argmax
                ):
                    raise ValueError("prediction provenance/probability validation failed")
                seen.add(sample_id)
                identities.append([sample_id, int(true), group_id])
                labels.append(int(true))
                predictions.append(int(pred))
                count += 1
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise FormalExecutionError(f"Invalid target prediction file {path}: {exc}") from exc
    expected = sum(spec.support_per_environment for spec in protocol.LABELS.values())
    if count != expected:
        raise FormalExecutionError(f"Target predictions contain {count} rows; expected {expected}.")
    return {
        "count": count,
        "ordered_sample_true_group_sha256": protocol.canonical_sha256(identities),
        "recomputed_metrics": fixed_label_metrics(
            labels,
            predictions,
            num_classes=len(protocol.LABELS),
        ),
    }


def _assert_recomputed_target_metrics(
    recomputed: Mapping[str, Any],
    recorded: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Compare all prediction-derived target metrics, excluding logits-only CE."""

    def compare(expected: Any, actual: Any, path: str) -> None:
        if isinstance(expected, Mapping):
            if not isinstance(actual, Mapping):
                raise FormalExecutionError(f"{context}.{path} is not a metric object.")
            for key, value in expected.items():
                if key not in actual:
                    raise FormalExecutionError(f"{context}.{path}.{key} is absent.")
                compare(value, actual[key], f"{path}.{key}" if path else str(key))
            return
        if isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                raise FormalExecutionError(f"{context}.{path} shape drifted.")
            for index, value in enumerate(expected):
                compare(value, actual[index], f"{path}[{index}]")
            return
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if (
                not isinstance(actual, (int, float))
                or isinstance(actual, bool)
                or not math.isfinite(float(actual))
                or abs(float(actual) - float(expected)) > 1e-12
            ):
                raise FormalExecutionError(f"{context}.{path} disagrees with target predictions.")
            return
        if actual != expected:
            raise FormalExecutionError(f"{context}.{path} disagrees with target predictions.")

    compare(recomputed, recorded, "")


def _commit_or_validate_attempt_manifest(
    path: Path,
    payload: Mapping[str, Any],
    *,
    create: bool,
) -> str:
    if create:
        _write_or_validate_json(path, payload)
    elif not path.is_file() or _load_json_object(path) != dict(payload):
        raise FormalExecutionError(f"Attempt manifest is absent or drifted: {path}")
    return _file_sha256(path)


def _validate_metrics(
    job: FormalJob,
    output_dir: Path,
    *,
    plan: FormalPlan,
    execution_lock_file_sha256: str,
    create_attempt_manifest: bool,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    metrics_path = output_dir / "metrics.json"
    metrics = _require_json(metrics_path)
    expected = {
        "target_env": job.target_env,
        "source_envs": list(job.source_envs),
        "arm": job.arm,
        "seed": job.seed,
        "evaluate_target": True,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise FormalExecutionError(f"Metrics provenance mismatch for {job.job_id}.{key}.")
    if metrics.get("protocol") != job.config.get("protocol"):
        raise FormalExecutionError(f"Metrics protocol binding mismatch for {job.job_id}.")
    resources = metrics.get("resources")
    if not isinstance(resources, Mapping):
        raise FormalExecutionError(f"Metrics for {job.job_id} have no resource report.")
    history = resources.get("history")
    if (
        int(resources.get("epochs", -1)) != protocol.EPOCHS
        or int(resources.get("optimizer_steps", -1)) != protocol.TOTAL_OPTIMIZER_STEPS
        or not isinstance(history, list)
        or len(history) != protocol.EPOCHS
        or not isinstance(history[-1], Mapping)
        or int(history[-1].get("global_step", -1)) != protocol.TOTAL_OPTIMIZER_STEPS
    ):
        raise FormalExecutionError(f"Formal budget/history mismatch for {job.job_id}.")
    best_epoch = int(resources.get("best_epoch", -1))
    if not 1 <= best_epoch <= protocol.EPOCHS:
        raise FormalExecutionError(f"Invalid best source-validation epoch for {job.job_id}.")

    invocation = _require_json(output_dir / "run_invocation.json")
    if (
        invocation.get("raw_config_sha256") != job.config_content_sha256
        or Path(str(invocation.get("config_path", ""))).resolve() != job.config_path
    ):
        raise FormalExecutionError(f"Run/config invocation mismatch for {job.job_id}.")
    resolved_config = _require_json(output_dir / "resolved_config.json")
    if resolved_config.get("protocol") != job.config.get("protocol"):
        raise FormalExecutionError(f"Resolved config protocol drifted for {job.job_id}.")

    source_manifest = _require_json(output_dir / "source_split_manifest.json")
    final_manifest = _require_json(output_dir / "split_manifest.json")
    target_manifest = _require_json(output_dir / "target_evaluation_manifest.json")
    stage0_gate = _require_json(output_dir / "stage0_gate.json")
    if (
        stage0_gate.get("target_opened") is not False
        or stage0_gate.get("target_env") != job.target_env
        or stage0_gate.get("source_envs") != list(job.source_envs)
        or source_manifest.get("target_data_loaded") is not False
        or source_manifest.get("target_statistics_available") is not False
        or source_manifest.get("target_not_loaded") is not True
        or job.target_env in source_manifest.get("loaded_environments", [])
        or final_manifest.get("target_absent_during_training") is not True
        or final_manifest.get("target_loaded_after_checkpoint") is not True
        or final_manifest.get("target_data_loaded") is not True
        or final_manifest.get("loaded_environments") != [*job.source_envs, job.target_env]
        or final_manifest.get("overlap_audit", {}).get("all_clear") is not True
        or target_manifest.get("target_env") != job.target_env
        or int(target_manifest.get("test", {}).get("count", -1))
        != sum(spec.support_per_environment for spec in protocol.LABELS.values())
        or int(final_manifest.get("test", {}).get("count", -1))
        != sum(spec.support_per_environment for spec in protocol.LABELS.values())
    ):
        raise FormalExecutionError(f"Deferred-target manifest gate failed for {job.job_id}.")

    source_phase = _require_json(output_dir / "phase_input_audit.json")
    target_phase = _require_json(output_dir / "target_phase_input_audit.json")
    if (
        source_phase.get("unwrap_phase") is not False
        or source_phase.get("loaded_environments") != list(job.source_envs)
        or set(source_phase.get("records", {})) != set(job.source_envs)
        or target_phase.get("unwrap_phase") is not False
        or target_phase.get("loaded_environments") != [job.target_env]
        or set(target_phase.get("records", {})) != {job.target_env}
    ):
        raise FormalExecutionError(f"Source/target phase audit failed for {job.job_id}.")

    source_metrics = _require_json(output_dir / "source_val_metrics.json")
    target_metrics = _require_json(output_dir / "target_metrics.json")
    if metrics.get("source_val_metrics") != source_metrics or metrics.get("target_metrics") != target_metrics:
        raise FormalExecutionError(f"Standalone metrics drifted from metrics.json for {job.job_id}.")
    target_predictions = output_dir / "target_predictions.jsonl"
    prediction_evidence = _validate_prediction_file(target_predictions, job.target_env)
    _assert_recomputed_target_metrics(
        prediction_evidence["recomputed_metrics"],
        target_metrics,
        context=f"target_metrics[{job.job_id}]",
    )

    best_model = output_dir / "best_model.pt"
    if not best_model.is_file() or best_model.stat().st_size <= 0:
        raise FormalExecutionError(f"Best checkpoint is absent for {job.job_id}.")
    calibration_path = Path(str(metrics.get("calibration_artifact", ""))).resolve()
    calibration_hash = str(metrics.get("calibration_sha256", ""))
    if not calibration_path.is_file() or _file_sha256(calibration_path) != calibration_hash:
        raise FormalExecutionError(f"Calibration artifact/hash mismatch for {job.job_id}.")
    calibration = _require_json(calibration_path)
    provenance = calibration.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("outer_target") != job.target_env
        or provenance.get("source_environments") != list(job.source_envs)
        or any(str(sample).startswith(f"{job.target_env}:") for sample in provenance.get("calibration_sample_ids", []))
    ):
        raise FormalExecutionError(f"Calibration provenance used unauthorized target data for {job.job_id}.")
    binding_path = calibration_path.with_suffix(calibration_path.suffix + ".binding.json")
    binding = _require_json(binding_path)
    if (
        binding.get("artifact_sha256") != calibration_hash
        or binding.get("protocol_lock_sha256") != protocol.protocol_lock_sha256()
        or any(str(key).startswith(f"{job.target_env}.") for key in binding.get("source_file_sha256", {}))
    ):
        raise FormalExecutionError(f"Calibration binding gate failed for {job.job_id}.")
    parameter = metrics.get("parameter_report")
    if (
        not isinstance(parameter, Mapping)
        or parameter.get("calibration_sha256") != calibration_hash
        or int(parameter.get("trainable_parameter_count", -1)) <= 0
        or not isinstance(parameter.get("initialization_hashes"), Mapping)
    ):
        raise FormalExecutionError(f"Parameter/calibration audit failed for {job.job_id}.")
    if _require_json(output_dir / "parameter_audit.json") != parameter:
        raise FormalExecutionError(f"Standalone parameter audit drifted for {job.job_id}.")
    if _require_json(output_dir / "batch_schedule_audit.json") != resources.get(
        "batch_schedule_audit"
    ):
        raise FormalExecutionError(f"Standalone batch schedule audit drifted for {job.job_id}.")
    if (
        stage0_gate.get("calibration_sha256") != calibration_hash
        or stage0_gate.get("parameter_audit") != parameter
    ):
        raise FormalExecutionError(f"Stage-0 gate is not bound to calibration/parameters for {job.job_id}.")

    required_artifacts = (
        "metrics.json",
        "resolved_config.json",
        "run_invocation.json",
        "stage0_gate.json",
        "parameter_audit.json",
        "batch_schedule_audit.json",
        "history.json",
        "best_model.pt",
        "source_val_metrics.json",
        "source_val_predictions.jsonl",
        "source_split_manifest.json",
        "target_metrics.json",
        "target_predictions.jsonl",
        "target_evaluation_manifest.json",
        "split_manifest.json",
        "phase_input_audit.json",
        "target_phase_input_audit.json",
    )
    artifact_file_sha256: Dict[str, str] = {}
    for relative in required_artifacts:
        artifact_path = output_dir / relative
        if not artifact_path.is_file():
            raise FormalExecutionError(f"Required formal artifact is absent: {artifact_path}")
        artifact_file_sha256[relative] = _file_sha256(artifact_path)
    try:
        with (output_dir / "history.json").open("r", encoding="utf-8") as handle:
            standalone_history = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalExecutionError(f"Cannot validate standalone history for {job.job_id}: {exc}") from exc
    if standalone_history != history:
        raise FormalExecutionError(f"Standalone history drifted from metrics resources for {job.job_id}.")

    hashes: Dict[str, Any] = {
        "metrics_file_sha256": _file_sha256(metrics_path),
        "target_metrics_file_sha256": _file_sha256(output_dir / "target_metrics.json"),
        "target_predictions_file_sha256": _file_sha256(target_predictions),
        "best_model_file_sha256": _file_sha256(best_model),
        "split_manifest_file_sha256": _file_sha256(output_dir / "split_manifest.json"),
        "source_split_manifest_sha256": protocol.canonical_sha256(source_manifest),
        "target_evaluation_manifest_sha256": protocol.canonical_sha256(target_manifest),
        "source_phase_audit_sha256": protocol.canonical_sha256(source_phase),
        "target_phase_audit_sha256": protocol.canonical_sha256(target_phase),
        "calibration_sha256": calibration_hash,
        "target_prediction_identity_sha256": prediction_evidence[
            "ordered_sample_true_group_sha256"
        ],
        "artifact_file_sha256": artifact_file_sha256,
    }
    attempt_manifest = {
        "schema_version": ATTEMPT_MANIFEST_SCHEMA,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "stage": job.stage,
        "job_id": job.job_id,
        "target_env": job.target_env,
        "source_envs": list(job.source_envs),
        "seed": job.seed,
        "arm": job.arm,
        "selected_lambda": job.selected_lambda,
        "lambda_supcon": job.lambda_supcon,
        "config_path": str(job.config_path),
        "config_content_sha256": job.config_content_sha256,
        "config_file_sha256": job.config_file_sha256,
        "selection_record_file_sha256": job.selection_record_file_sha256,
        "interim_decision_file_sha256": job.interim_decision_file_sha256,
        "formal_registry_file_sha256": plan.registry_file_sha256,
        "formal_registry_content_sha256": plan.registry_content_sha256,
        "execution_lock_file_sha256": execution_lock_file_sha256,
        "artifact_file_sha256": artifact_file_sha256,
        "target_metrics": target_metrics,
    }
    attempt_manifest_path = output_dir / "attempt_manifest.json"
    hashes["attempt_manifest_file_sha256"] = _commit_or_validate_attempt_manifest(
        attempt_manifest_path,
        attempt_manifest,
        create=create_attempt_manifest,
    )
    return metrics, hashes


def _completed_metrics(
    row: Mapping[str, str],
    job: FormalJob,
    *,
    plan: FormalPlan,
    execution_lock_file_sha256: str,
    require_recorded_hash: bool,
    create_attempt_manifest: bool = False,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if not row.get("output_dir"):
        raise FormalExecutionError(f"Completed row {job.job_id} has no output directory.")
    expected_metrics_path = Path(row["output_dir"]).resolve() / "metrics.json"
    if row.get("metrics_path") and Path(row["metrics_path"]).resolve() != expected_metrics_path:
        raise FormalExecutionError(f"Completed row {job.job_id} has a non-canonical metrics path.")
    metrics, hashes = _validate_metrics(
        job,
        Path(row["output_dir"]),
        plan=plan,
        execution_lock_file_sha256=execution_lock_file_sha256,
        create_attempt_manifest=create_attempt_manifest,
    )
    if require_recorded_hash:
        for field in (
            "metrics_file_sha256",
            "target_metrics_file_sha256",
            "target_predictions_file_sha256",
            "best_model_file_sha256",
            "split_manifest_file_sha256",
            "attempt_manifest_file_sha256",
        ):
            if row.get(field) != hashes[field]:
                raise FormalExecutionError(f"Recorded {field} changed for completed job {job.job_id}.")
    return metrics, hashes


def _recover_or_validate_rows(
    rows: MutableMapping[str, MutableMapping[str, str]],
    plan: FormalPlan,
    execution_lock_file_sha256: str,
) -> None:
    for job in plan.jobs:
        row = rows[job.job_id]
        if row["status"] == "completed":
            _completed_metrics(
                row,
                job,
                plan=plan,
                execution_lock_file_sha256=execution_lock_file_sha256,
                require_recorded_hash=True,
            )
        elif row["status"] == "running" and row.get("output_dir"):
            try:
                _, hashes = _completed_metrics(
                    row,
                    job,
                    plan=plan,
                    execution_lock_file_sha256=execution_lock_file_sha256,
                    require_recorded_hash=False,
                    create_attempt_manifest=True,
                )
            except FormalExecutionError:
                continue
            row.update(
                {
                    "status": "completed",
                    "return_code": "0",
                    "completed_at_utc": _utc_now(),
                    "metrics_path": str(Path(row["output_dir"]) / "metrics.json"),
                    **{key: hashes[key] for key in RUN_REGISTRY_FIELDS if key in hashes},
                    "error": "",
                }
            )


def _block_evidence(
    jobs: Sequence[FormalJob],
    rows: Mapping[str, Mapping[str, str]],
    *,
    plan: FormalPlan,
    execution_lock_file_sha256: str,
) -> Dict[str, Dict[str, Any]]:
    evidence: Dict[str, Dict[str, Any]] = {}
    for job in jobs:
        if rows[job.job_id]["status"] != "completed":
            raise FormalExecutionError(f"Formal block is incomplete at {job.job_id}.")
        metrics, hashes = _completed_metrics(
            rows[job.job_id],
            job,
            plan=plan,
            execution_lock_file_sha256=execution_lock_file_sha256,
            require_recorded_hash=True,
        )
        batch = metrics["resources"]["batch_schedule_audit"]
        parameter = metrics["parameter_report"]
        evidence[job.job_id] = {
            **hashes,
            "batch_schedule_sha256": batch["schedule_sha256"],
            "first_batch_sample_ids": batch["first_batch_sample_ids"],
            "trainable_parameter_count": parameter["trainable_parameter_count"],
            "initialization_hashes": parameter["initialization_hashes"],
        }
    return evidence


def _assert_same(evidence: Mapping[str, Mapping[str, Any]], fields: Sequence[str], context: str) -> None:
    for field in fields:
        values = {json.dumps(row[field], sort_keys=True) for row in evidence.values()}
        if len(values) != 1:
            raise FormalExecutionError(f"{context} disagrees across completed jobs for {field}.")


def _validate_seed_block(
    target: str,
    seed: int,
    jobs: Sequence[FormalJob],
    rows: Mapping[str, Mapping[str, str]],
    *,
    plan: FormalPlan,
    execution_lock_file_sha256: str,
) -> None:
    if {job.arm for job in jobs} != set(protocol.ARMS):
        raise FormalExecutionError(f"{target}/seed={seed} is not a complete four-arm block.")
    evidence = _block_evidence(
        jobs,
        rows,
        plan=plan,
        execution_lock_file_sha256=execution_lock_file_sha256,
    )
    _assert_same(
        evidence,
        (
            "calibration_sha256",
            "source_split_manifest_sha256",
            "target_evaluation_manifest_sha256",
            "source_phase_audit_sha256",
            "target_phase_audit_sha256",
            "target_prediction_identity_sha256",
            "batch_schedule_sha256",
            "first_batch_sample_ids",
            "trainable_parameter_count",
            "initialization_hashes",
        ),
        f"Four-arm barrier {target}/seed={seed}",
    )


def _validate_fold_barrier(
    target: str,
    jobs: Sequence[FormalJob],
    rows: Mapping[str, Mapping[str, str]],
    *,
    plan: FormalPlan,
    execution_lock_file_sha256: str,
) -> None:
    evidence = _block_evidence(
        jobs,
        rows,
        plan=plan,
        execution_lock_file_sha256=execution_lock_file_sha256,
    )
    _assert_same(
        evidence,
        (
            "calibration_sha256",
            "source_split_manifest_sha256",
            "target_evaluation_manifest_sha256",
            "source_phase_audit_sha256",
            "target_phase_audit_sha256",
            "target_prediction_identity_sha256",
            "trainable_parameter_count",
        ),
        f"Target-fold barrier {target}",
    )


def build_formal_results(
    plan: FormalPlan,
    rows: Mapping[str, Mapping[str, str]],
    run_registry_path: Path,
    *,
    execution_lock_file_sha256: str,
) -> Dict[str, Any]:
    result_rows: List[dict] = []
    for job in plan.jobs:
        metrics, hashes = _completed_metrics(
            rows[job.job_id],
            job,
            plan=plan,
            execution_lock_file_sha256=execution_lock_file_sha256,
            require_recorded_hash=True,
        )
        result_rows.append(
            {
                "job_id": job.job_id,
                "stage": job.stage,
                "target_env": job.target_env,
                "seed": job.seed,
                "arm": job.arm,
                "selected_lambda": job.selected_lambda,
                "lambda_supcon": job.lambda_supcon,
                "config_content_sha256": job.config_content_sha256,
                "selection_record_file_sha256": job.selection_record_file_sha256,
                "interim_decision_file_sha256": job.interim_decision_file_sha256,
                "metrics_file_sha256": hashes["metrics_file_sha256"],
                "target_metrics_file_sha256": hashes["target_metrics_file_sha256"],
                "target_predictions_file_sha256": hashes["target_predictions_file_sha256"],
                "best_model_file_sha256": hashes["best_model_file_sha256"],
                "attempt_manifest_file_sha256": hashes["attempt_manifest_file_sha256"],
                "target_metrics": metrics["target_metrics"],
                "best_source_val_f1_macro": metrics["resources"]["best_source_val_f1_macro"],
                "best_epoch": metrics["resources"]["best_epoch"],
            }
        )
    return {
        "schema_version": FORMAL_RESULTS_SCHEMA,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "stage": plan.stage,
        "formal_registry_file_sha256": plan.registry_file_sha256,
        "formal_registry_content_sha256": plan.registry_content_sha256,
        "execution_lock_file_sha256": execution_lock_file_sha256,
        "run_registry_file_sha256": _file_sha256(run_registry_path),
        "completed_job_count": len(result_rows),
        "results": result_rows,
    }


def _write_or_validate_json(path: Path, payload: object) -> None:
    if path.exists():
        if _load_json_object(path) != payload:
            raise FormalExecutionError(f"Existing final artifact drifted: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _execute_formal_locked(
    registry_path: Path,
    output_dir: Path,
    *,
    device: str,
    dry_run: bool,
    command_runner: CommandRunner = run_logged_command,
) -> Dict[str, Any]:
    plan = load_formal_plan(registry_path)
    output_root = output_dir.resolve()
    if output_root in {REPO_ROOT.resolve(), plan.bundle_dir} or _inside(output_root, plan.bundle_dir):
        raise FormalExecutionError(f"Unsafe formal output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    execution_lock_path = output_root / "execution_lock.json"
    locked_execution = execution_lock_payload(
        plan,
        device=device,
        formal_output_root=output_root,
    )
    _write_or_validate_json(execution_lock_path, locked_execution)
    execution_lock_file_sha256 = _file_sha256(execution_lock_path)
    csv_path = output_root / "run_registry.csv"
    rows = _load_run_rows(csv_path, plan, execution_lock_file_sha256)
    _recover_or_validate_rows(rows, plan, execution_lock_file_sha256)

    if dry_run:
        for job in plan.jobs:
            if rows[job.job_id]["status"] != "completed":
                _prepare_attempt(rows[job.job_id], job, output_root, device=device)
        _write_run_rows(csv_path, rows, plan.jobs)
        return {
            "mode": "dry_run",
            "stage": plan.stage,
            "job_count": len(plan.jobs),
            "completed_job_count": sum(row["status"] == "completed" for row in rows.values()),
            "run_registry": str(csv_path),
            "formal_results": None,
        }

    for target in protocol.ENVIRONMENTS:
        fold_jobs = [job for job in plan.jobs if job.target_env == target]
        for seed in protocol.formal_seeds(target, plan.stage):
            block_jobs = [job for job in fold_jobs if job.seed == seed]
            for job in block_jobs:
                row = rows[job.job_id]
                if row["status"] == "completed":
                    continue
                _revalidate_launch_binding(
                    plan,
                    job,
                    execution_lock_path=execution_lock_path,
                    execution_lock_file_sha256=execution_lock_file_sha256,
                    device=device,
                )
                run_output, command, stdout_log, stderr_log = _prepare_attempt(
                    row, job, output_root, device=device
                )
                row["status"] = "running"
                row["started_at_utc"] = _utc_now()
                _write_run_rows(csv_path, rows, plan.jobs)
                started = time.perf_counter()
                try:
                    completed = command_runner(
                        command,
                        cwd=REPO_ROOT,
                        check=False,
                        stdout_path=stdout_log,
                        stderr_path=stderr_log,
                    )
                    return_code = int(completed.returncode)
                except Exception as exc:
                    elapsed = time.perf_counter() - started
                    row.update(
                        {
                            "status": "failed",
                            "completed_at_utc": _utc_now(),
                            "elapsed_seconds": f"{elapsed:.6f}",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    _write_run_rows(csv_path, rows, plan.jobs)
                    raise FormalExecutionError(f"Formal job {job.job_id} could not be launched: {exc}") from exc
                elapsed = time.perf_counter() - started
                row.update(
                    {
                        "return_code": str(return_code),
                        "elapsed_seconds": f"{elapsed:.6f}",
                        "completed_at_utc": _utc_now(),
                    }
                )
                if return_code != 0:
                    row["status"] = "failed"
                    row["error"] = f"train_joint exited with code {return_code}"
                    _write_run_rows(csv_path, rows, plan.jobs)
                    raise FormalExecutionError(
                        f"Formal job {job.job_id} failed with return code {return_code}; resume to retry."
                    )
                try:
                    _, hashes = _validate_metrics(
                        job,
                        run_output,
                        plan=plan,
                        execution_lock_file_sha256=execution_lock_file_sha256,
                        create_attempt_manifest=True,
                    )
                except FormalExecutionError as exc:
                    row["status"] = "failed"
                    row["error"] = str(exc)
                    _write_run_rows(csv_path, rows, plan.jobs)
                    raise
                row.update(
                    {
                        "status": "completed",
                        "metrics_path": str(run_output / "metrics.json"),
                        **{key: hashes[key] for key in RUN_REGISTRY_FIELDS if key in hashes},
                        "error": "",
                    }
                )
                _write_run_rows(csv_path, rows, plan.jobs)
            _validate_seed_block(
                target,
                seed,
                block_jobs,
                rows,
                plan=plan,
                execution_lock_file_sha256=execution_lock_file_sha256,
            )
        _validate_fold_barrier(
            target,
            fold_jobs,
            rows,
            plan=plan,
            execution_lock_file_sha256=execution_lock_file_sha256,
        )

    results = build_formal_results(
        plan,
        rows,
        csv_path,
        execution_lock_file_sha256=execution_lock_file_sha256,
    )
    results_path = output_root / "formal_results.json"
    _write_or_validate_json(results_path, results)
    return {
        "mode": "execute",
        "stage": plan.stage,
        "job_count": len(plan.jobs),
        "completed_job_count": len(plan.jobs),
        "run_registry": str(csv_path),
        "formal_results": str(results_path),
        "formal_results_file_sha256": _file_sha256(results_path),
    }


def execute_formal(
    registry_path: Path,
    output_dir: Path,
    *,
    device: str,
    dry_run: bool,
    command_runner: CommandRunner = run_logged_command,
) -> Dict[str, Any]:
    """Acquire one OS-owned output lock, then plan or execute the matrix."""

    output_root = output_dir.resolve()
    bundle_dir = registry_path.resolve().parent
    if output_root in {REPO_ROOT.resolve(), bundle_dir} or _inside(output_root, bundle_dir):
        raise FormalExecutionError(f"Unsafe formal output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    with exclusive_runner_lock(output_root / ".runner.lock"):
        return _execute_formal_locked(
            registry_path,
            output_root,
            device=device,
            dry_run=dry_run,
            command_runner=command_runner,
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    summary = execute_formal(
        args.formal_registry,
        args.output_dir,
        device=args.device,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
