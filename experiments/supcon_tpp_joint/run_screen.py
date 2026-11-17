"""Run, resume, and collect the locked 18-job Stage-1 lambda screen.

Execution is deliberately opt-in: the CLI requires either ``--dry-run`` or
``--execute``.  Jobs run serially, one complete outer fold at a time.  A
failed/interrupted attempt is never deleted; resumption writes a new attempt
directory and skips only outputs that still pass all provenance checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence

try:  # Support module and direct-script execution.
    from . import check_protocol, protocol, screen_health, select_lambda
except ImportError:  # pragma: no cover
    import check_protocol  # type: ignore
    import protocol  # type: ignore
    import screen_health  # type: ignore
    import select_lambda  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = Path(__file__).resolve().parent / "generated_protocol" / "screen_job_registry.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs_supcon_tpp_joint" / "stage1_screen"
TRAIN_MODULE = "experiments.supcon_tpp_joint.train_joint"

EXPECTED_JOB_KEYS = {
    "job_id",
    "stage",
    "target_env",
    "source_envs",
    "screen_seed",
    "seed",
    "arm",
    "lambda_supcon",
    "config_path",
    "config_content_sha256",
    "config_file_sha256",
}

RUN_REGISTRY_FIELDS = [
    "job_id",
    "target_env",
    "source_envs",
    "arm",
    "lambda_supcon",
    "screen_seed",
    "seed",
    "config_path",
    "config_content_sha256",
    "config_file_sha256",
    "screen_registry_file_sha256",
    "status",
    "attempt",
    "output_dir",
    "command",
    "started_at_utc",
    "completed_at_utc",
    "elapsed_seconds",
    "return_code",
    "metrics_path",
    "metrics_file_sha256",
    "error",
]

IMMUTABLE_RUN_FIELDS = RUN_REGISTRY_FIELDS[:11]
VALID_RUN_STATUSES = {"planned", "running", "completed", "failed"}


class ScreenExecutionError(RuntimeError):
    """Raised when a screen artifact or resumable run fails closed."""


@dataclass(frozen=True)
class ScreenJob:
    job_id: str
    target_env: str
    source_envs: tuple[str, ...]
    screen_seed: int
    arm: str
    lambda_supcon: float
    config_path: Path
    config_content_sha256: str
    config_file_sha256: str
    calibration_artifact_id: str
    config: Mapping[str, Any]


@dataclass(frozen=True)
class ScreenPlan:
    registry_path: Path
    registry_file_sha256: str
    registry: Mapping[str, Any]
    jobs: tuple[ScreenJob, ...]


CommandRunner = Callable[..., Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json_object(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ScreenExecutionError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ScreenExecutionError(f"Expected a JSON object: {path}")
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


def _validate_job_file(
    raw_job: Mapping[str, Any],
    *,
    bundle_dir: Path,
) -> tuple[ScreenJob, Dict[str, Any]]:
    job_id = str(raw_job.get("job_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
        raise ScreenExecutionError(f"Unsafe or empty screen job_id: {job_id!r}")
    if set(raw_job) != EXPECTED_JOB_KEYS:
        raise ScreenExecutionError(
            f"Registry row {job_id} schema drifted; expected {sorted(EXPECTED_JOB_KEYS)}, "
            f"received {sorted(raw_job)}."
        )

    relative_config = Path(str(raw_job["config_path"]))
    if relative_config.is_absolute():
        raise ScreenExecutionError(f"Registry config_path must be relative for {job_id}.")
    config_path = (bundle_dir / relative_config).resolve()
    if not _inside(config_path, bundle_dir) or not config_path.is_file():
        raise ScreenExecutionError(f"Missing or out-of-bundle config for {job_id}: {config_path}")

    expected_file_hash = str(raw_job["config_file_sha256"])
    observed_file_hash = _file_sha256(config_path)
    if observed_file_hash != expected_file_hash:
        raise ScreenExecutionError(
            f"Config file hash mismatch for {job_id}: {observed_file_hash} != {expected_file_hash}."
        )
    config = _load_json_object(config_path)
    expected_content_hash = str(raw_job["config_content_sha256"])
    observed_content_hash = protocol.canonical_sha256(config)
    if observed_content_hash != expected_content_hash:
        raise ScreenExecutionError(
            f"Config content hash mismatch for {job_id}: {observed_content_hash} != {expected_content_hash}."
        )

    target_env = str(raw_job["target_env"])
    calibration = config.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ScreenExecutionError(f"Config {job_id} has no calibration object.")
    artifact_id = str(calibration.get("artifact_id", ""))
    if not artifact_id or bool(calibration.get("uses_target_data", True)):
        raise ScreenExecutionError(f"Config {job_id} does not use a source-only named calibration artifact.")
    evaluation = config.get("evaluation")
    training = config.get("training")
    if not isinstance(evaluation, Mapping) or evaluation.get("evaluate_target") is not False:
        raise ScreenExecutionError(f"Screen config {job_id} must keep target evaluation disabled.")
    if not isinstance(training, Mapping) or int(training.get("epochs", -1)) != protocol.EPOCHS:
        raise ScreenExecutionError(f"Screen config {job_id} must contain exactly {protocol.EPOCHS} epochs.")

    job = ScreenJob(
        job_id=job_id,
        target_env=target_env,
        source_envs=tuple(map(str, raw_job["source_envs"])),
        screen_seed=int(raw_job["screen_seed"]),
        arm=str(raw_job["arm"]),
        lambda_supcon=float(raw_job["lambda_supcon"]),
        config_path=config_path,
        config_content_sha256=expected_content_hash,
        config_file_sha256=expected_file_hash,
        calibration_artifact_id=artifact_id,
        config=config,
    )
    return job, dict(raw_job) | {"config": config}


def load_screen_plan(registry_path: Path) -> ScreenPlan:
    """Load and hash-check the v2 object registry and all 18 configs."""

    resolved_registry = registry_path.resolve()
    registry = _load_json_object(resolved_registry)
    try:
        raw_jobs = select_lambda.validate_screen_registry_header(registry)
    except (ValueError, KeyError, TypeError) as exc:
        raise ScreenExecutionError(f"Invalid screen registry header: {exc}") from exc

    jobs: List[ScreenJob] = []
    hydrated_rows: List[dict] = []
    bundle_dir = resolved_registry.parent.resolve()
    for raw_job in raw_jobs:
        job, hydrated = _validate_job_file(raw_job, bundle_dir=bundle_dir)
        jobs.append(job)
        hydrated_rows.append(hydrated)
    try:
        check_protocol.check_screen_job_registry(dict(registry) | {"jobs": hydrated_rows})
    except (ValueError, KeyError, TypeError) as exc:
        raise ScreenExecutionError(f"Screen registry/config protocol validation failed: {exc}") from exc

    ordered: List[ScreenJob] = []
    calibration_ids: Dict[str, str] = {}
    arm_order = {arm: index for index, arm in enumerate(protocol.ARMS)}
    for target_env in protocol.ENVIRONMENTS:
        fold_jobs = [job for job in jobs if job.target_env == target_env]
        if len(fold_jobs) != 6:
            raise ScreenExecutionError(f"Expected six screen jobs for {target_env}; found {len(fold_jobs)}.")
        fold_artifacts = {job.calibration_artifact_id for job in fold_jobs}
        if len(fold_artifacts) != 1:
            raise ScreenExecutionError(
                f"Calibration artifact IDs differ inside outer fold {target_env}: {sorted(fold_artifacts)}"
            )
        calibration_ids[target_env] = next(iter(fold_artifacts))
        ordered.extend(
            sorted(
                fold_jobs,
                key=lambda item: (arm_order[item.arm], item.lambda_supcon, item.job_id),
            )
        )
    if len(set(calibration_ids.values())) != len(protocol.ENVIRONMENTS):
        raise ScreenExecutionError(
            "Calibration artifact IDs must be fold-specific; refusing cross-fold calibration reuse."
        )

    return ScreenPlan(
        registry_path=resolved_registry,
        registry_file_sha256=_file_sha256(resolved_registry),
        registry=registry,
        jobs=tuple(ordered),
    )


def _base_run_row(job: ScreenJob, registry_hash: str) -> Dict[str, str]:
    return {
        "job_id": job.job_id,
        "target_env": job.target_env,
        "source_envs": json.dumps(job.source_envs, separators=(",", ":")),
        "arm": job.arm,
        "lambda_supcon": format(job.lambda_supcon, ".17g"),
        "screen_seed": str(job.screen_seed),
        "seed": str(job.screen_seed),
        "config_path": str(job.config_path),
        "config_content_sha256": job.config_content_sha256,
        "config_file_sha256": job.config_file_sha256,
        "screen_registry_file_sha256": registry_hash,
        "status": "planned",
        "attempt": "0",
        "output_dir": "",
        "command": "",
        "started_at_utc": "",
        "completed_at_utc": "",
        "elapsed_seconds": "",
        "return_code": "",
        "metrics_path": "",
        "metrics_file_sha256": "",
        "error": "",
    }


def _load_run_rows(path: Path, plan: ScreenPlan) -> Dict[str, Dict[str, str]]:
    expected = {job.job_id: _base_run_row(job, plan.registry_file_sha256) for job in plan.jobs}
    if not path.is_file():
        return expected
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RUN_REGISTRY_FIELDS:
            raise ScreenExecutionError(
                f"run_registry.csv schema drifted; expected {RUN_REGISTRY_FIELDS}, received {reader.fieldnames}."
            )
        observed: Dict[str, Dict[str, str]] = {}
        for raw_row in reader:
            row = {key: str(raw_row.get(key, "")) for key in RUN_REGISTRY_FIELDS}
            job_id = row["job_id"]
            if job_id in observed:
                raise ScreenExecutionError(f"Duplicate run_registry.csv row: {job_id}")
            if job_id not in expected:
                raise ScreenExecutionError(f"run_registry.csv references unknown job: {job_id}")
            for key in IMMUTABLE_RUN_FIELDS:
                if row[key] != expected[job_id][key]:
                    raise ScreenExecutionError(f"run_registry.csv binding mismatch for {job_id}.{key}.")
            if row["status"] not in VALID_RUN_STATUSES:
                raise ScreenExecutionError(f"Invalid run status for {job_id}: {row['status']!r}")
            try:
                attempt = int(row["attempt"])
            except ValueError as exc:
                raise ScreenExecutionError(f"Invalid attempt counter for {job_id}.") from exc
            if attempt < 0:
                raise ScreenExecutionError(f"Negative attempt counter for {job_id}.")
            observed[job_id] = row
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        raise ScreenExecutionError(f"run_registry.csv does not cover all 18 jobs; missing={missing}.")
    return observed


def _write_run_rows(path: Path, rows: Mapping[str, Mapping[str, str]], jobs: Sequence[ScreenJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_REGISTRY_FIELDS)
        writer.writeheader()
        for job in jobs:
            writer.writerow({key: rows[job.job_id].get(key, "") for key in RUN_REGISTRY_FIELDS})
    os.replace(temporary, path)


def _attempt_output_dir(output_dir: Path, job: ScreenJob, attempt: int) -> Path:
    return output_dir / "runs" / job.target_env / job.job_id / f"attempt_{attempt:03d}"


def build_train_command(job: ScreenJob, output_dir: Path, *, device: str) -> List[str]:
    """Build a formal screen command; epochs come from the hash-locked config."""

    if int(job.config["training"]["epochs"]) != protocol.EPOCHS:  # type: ignore[index]
        raise ScreenExecutionError(f"Job {job.job_id} is not a {protocol.EPOCHS}-epoch screen config.")
    return [
        sys.executable,
        "-m",
        TRAIN_MODULE,
        "--config",
        str(job.config_path),
        "--seed",
        str(job.screen_seed),
        "--evaluate-target",
        "false",
        "--device",
        device,
        "--output-dir",
        str(output_dir),
    ]


def _prepare_attempt(
    row: MutableMapping[str, str],
    job: ScreenJob,
    output_root: Path,
    *,
    device: str,
) -> tuple[Path, List[str]]:
    reuse_planned = row["status"] == "planned" and row["output_dir"] and int(row["attempt"]) > 0
    if reuse_planned:
        attempt = int(row["attempt"])
        run_output = Path(row["output_dir"]).resolve()
        if run_output.exists() and any(run_output.iterdir()):
            reuse_planned = False
    if not reuse_planned:
        attempt = int(row["attempt"] or 0) + 1
        run_output = _attempt_output_dir(output_root, job, attempt).resolve()
        while run_output.exists() and any(run_output.iterdir()):
            attempt += 1
            run_output = _attempt_output_dir(output_root, job, attempt).resolve()
    command = build_train_command(job, run_output, device=device)
    row.update(
        {
            "status": "planned",
            "attempt": str(attempt),
            "output_dir": str(run_output),
            "command": subprocess.list2cmdline(command),
            "started_at_utc": "",
            "completed_at_utc": "",
            "elapsed_seconds": "",
            "return_code": "",
            "metrics_path": "",
            "metrics_file_sha256": "",
            "error": "",
        }
    )
    return run_output, command


def _validate_metrics(job: ScreenJob, output_dir: Path) -> tuple[Dict[str, Any], Path, str]:
    metrics_path = output_dir / "metrics.json"
    metrics = _load_json_object(metrics_path)
    expected = {
        "target_env": job.target_env,
        "arm": job.arm,
        "seed": job.screen_seed,
        "evaluate_target": False,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ScreenExecutionError(
                f"Metrics provenance mismatch for {job.job_id}.{key}: {metrics.get(key)!r} != {value!r}."
            )
    resources = metrics.get("resources")
    if not isinstance(resources, Mapping):
        raise ScreenExecutionError(f"Metrics for {job.job_id} have no resources object.")
    if int(resources.get("epochs", -1)) != protocol.EPOCHS:
        raise ScreenExecutionError(f"Metrics for {job.job_id} are not from {protocol.EPOCHS} epochs.")
    if int(resources.get("optimizer_steps", -1)) != protocol.TOTAL_OPTIMIZER_STEPS:
        raise ScreenExecutionError(f"Metrics for {job.job_id} are not from 60,000 optimizer steps.")
    screen_metrics = metrics.get("screen_metrics")
    if not isinstance(screen_metrics, Mapping):
        raise ScreenExecutionError(f"Metrics for {job.job_id} have no screen_metrics object.")
    history = resources.get("history")
    if not isinstance(history, list) or not all(isinstance(row, Mapping) for row in history):
        raise ScreenExecutionError(f"Metrics for {job.job_id} have no complete source history.")
    try:
        screen_health.validate_screen_metrics(
            screen_metrics,
            expected_epoch_count=protocol.EPOCHS,
            history=history,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ScreenExecutionError(
            f"Metrics for {job.job_id} have invalid training-health evidence: {exc}"
        ) from exc

    invocation = _load_json_object(output_dir / "run_invocation.json")
    if invocation.get("raw_config_sha256") != job.config_content_sha256:
        raise ScreenExecutionError(f"Run/config content hash mismatch for {job.job_id}.")
    invoked_path = Path(str(invocation.get("config_path", ""))).resolve()
    if invoked_path != job.config_path:
        raise ScreenExecutionError(f"Run/config path mismatch for {job.job_id}.")

    calibration_hash = metrics.get("calibration_sha256")
    calibration_path = Path(str(metrics.get("calibration_artifact", "")))
    if not isinstance(calibration_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", calibration_hash):
        raise ScreenExecutionError(f"Metrics for {job.job_id} have an invalid calibration SHA256.")
    if not calibration_path.is_file() or _file_sha256(calibration_path) != calibration_hash:
        raise ScreenExecutionError(f"Calibration artifact/hash mismatch for {job.job_id}.")
    forbidden = [path.name for path in output_dir.iterdir() if path.name.startswith("target_")]
    if forbidden:
        raise ScreenExecutionError(f"Source-only screen job {job.job_id} emitted target artifacts: {forbidden}")
    return metrics, metrics_path, _file_sha256(metrics_path)


def _completed_metrics(
    row: Mapping[str, str],
    job: ScreenJob,
    *,
    require_recorded_hash: bool,
) -> tuple[Dict[str, Any], Path, str]:
    if not row.get("output_dir"):
        raise ScreenExecutionError(f"Completed row {job.job_id} has no output directory.")
    metrics, metrics_path, metrics_hash = _validate_metrics(job, Path(row["output_dir"]))
    if require_recorded_hash and row.get("metrics_file_sha256") != metrics_hash:
        raise ScreenExecutionError(f"Recorded metrics hash changed for completed job {job.job_id}.")
    return metrics, metrics_path, metrics_hash


def _recover_or_validate_rows(
    rows: MutableMapping[str, MutableMapping[str, str]],
    plan: ScreenPlan,
) -> None:
    for job in plan.jobs:
        row = rows[job.job_id]
        if row["status"] == "completed":
            _completed_metrics(row, job, require_recorded_hash=True)
            continue
        # Recover the narrow crash window after metrics.json was committed but
        # before the CSV status was updated.  Explicitly failed jobs are retried.
        if row["status"] == "running" and row.get("output_dir"):
            try:
                _, metrics_path, metrics_hash = _completed_metrics(
                    row, job, require_recorded_hash=False
                )
            except ScreenExecutionError:
                continue
            row.update(
                {
                    "status": "completed",
                    "return_code": "0",
                    "completed_at_utc": _utc_now(),
                    "metrics_path": str(metrics_path),
                    "metrics_file_sha256": metrics_hash,
                    "error": "",
                }
            )


def _validate_fold_calibration(
    target_env: str,
    jobs: Sequence[ScreenJob],
    rows: Mapping[str, Mapping[str, str]],
) -> None:
    hashes = set()
    for job in jobs:
        if rows[job.job_id]["status"] != "completed":
            raise ScreenExecutionError(f"Outer fold {target_env} is incomplete at {job.job_id}.")
        metrics, _, _ = _completed_metrics(rows[job.job_id], job, require_recorded_hash=True)
        hashes.add(str(metrics["calibration_sha256"]))
    if len(hashes) != 1:
        raise ScreenExecutionError(
            f"The six jobs in outer fold {target_env} did not use one identical calibration artifact."
        )


def build_screen_results(
    plan: ScreenPlan,
    rows: Mapping[str, Mapping[str, str]],
) -> Dict[str, Any]:
    """Collect exactly the schema consumed by ``select_lambda.py``."""

    result_rows: List[dict] = []
    for job in plan.jobs:
        row = rows[job.job_id]
        if row["status"] != "completed":
            raise ScreenExecutionError(f"Cannot collect incomplete screen job {job.job_id}.")
        metrics, _, _ = _completed_metrics(row, job, require_recorded_hash=True)
        result_rows.append(
            {
                "job_id": job.job_id,
                "target_env": job.target_env,
                "arm": job.arm,
                "screen_seed": job.screen_seed,
                "lambda_supcon": job.lambda_supcon,
                "config_content_sha256": job.config_content_sha256,
                "metrics": dict(metrics["screen_metrics"]),
            }
        )
    payload = {
        "schema_version": protocol.SCREEN_RESULTS_SCHEMA,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "screen_registry_file_sha256": plan.registry_file_sha256,
        "results": result_rows,
    }
    try:
        select_lambda.validate_screen_results(
            payload,
            plan.registry,
            registry_file_sha256=plan.registry_file_sha256,
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise ScreenExecutionError(f"Collected screen_results.json is invalid: {exc}") from exc
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_or_validate_selections(
    selection_dir: Path,
    records: Mapping[str, Mapping[str, object]],
) -> None:
    failed = [target for target, record in records.items() if record.get("status") != "selected"]
    if failed:
        raise ScreenExecutionError(
            "Lambda selection failed closed after all screen jobs: " + ", ".join(failed)
        )
    if not selection_dir.exists() or not any(selection_dir.iterdir()):
        select_lambda.write_selection_records(selection_dir, records, clean=False)
        return
    expected_names = {f"selection_{target}.json" for target in protocol.ENVIRONMENTS} | {
        "selection_summary.json"
    }
    observed_names = {path.name for path in selection_dir.iterdir() if path.is_file()}
    if observed_names != expected_names:
        raise ScreenExecutionError(
            f"Selection directory is nonempty and does not contain the exact expected files: {selection_dir}"
        )
    for target_env in protocol.ENVIRONMENTS:
        observed = _load_json_object(selection_dir / f"selection_{target_env}.json")
        if observed != records[target_env]:
            raise ScreenExecutionError(f"Existing selection record drifted for {target_env}.")
    expected_summary = {
        "schema_version": protocol.SELECTION_RECORD_SCHEMA,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "statuses": {
            target: records[target]["status"] for target in protocol.ENVIRONMENTS
        },
        "selected_lambdas": {
            target: records[target]["selected_lambda"] for target in protocol.ENVIRONMENTS
        },
    }
    if _load_json_object(selection_dir / "selection_summary.json") != expected_summary:
        raise ScreenExecutionError("Existing selection_summary.json drifted from the three records.")


def execute_screen(
    registry_path: Path,
    output_dir: Path,
    *,
    device: str,
    dry_run: bool,
    select_lambdas: bool = False,
    selection_dir: Optional[Path] = None,
    command_runner: CommandRunner = subprocess.run,
) -> Dict[str, Any]:
    """Plan or execute the 18 jobs, then strictly collect their metrics."""

    if dry_run and select_lambdas:
        raise ScreenExecutionError("--select-lambda requires --execute, not --dry-run.")
    plan = load_screen_plan(registry_path)
    output_root = output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "run_registry.csv"
    rows = _load_run_rows(csv_path, plan)
    _recover_or_validate_rows(rows, plan)

    if dry_run:
        for job in plan.jobs:
            row = rows[job.job_id]
            if row["status"] != "completed":
                _prepare_attempt(row, job, output_root, device=device)
        _write_run_rows(csv_path, rows, plan.jobs)
        return {
            "mode": "dry_run",
            "job_count": len(plan.jobs),
            "completed_job_count": sum(row["status"] == "completed" for row in rows.values()),
            "run_registry": str(csv_path),
            "screen_results": None,
        }

    for target_env in protocol.ENVIRONMENTS:
        fold_jobs = [job for job in plan.jobs if job.target_env == target_env]
        for job in fold_jobs:
            row = rows[job.job_id]
            if row["status"] == "completed":
                continue
            run_output, command = _prepare_attempt(row, job, output_root, device=device)
            row["status"] = "running"
            row["started_at_utc"] = _utc_now()
            _write_run_rows(csv_path, rows, plan.jobs)
            started = time.perf_counter()
            try:
                completed = command_runner(command, cwd=REPO_ROOT, check=False)
                return_code = int(completed.returncode)
            except Exception as exc:  # pragma: no cover - exercised through behavior tests
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
                raise ScreenExecutionError(f"Screen job {job.job_id} could not be launched: {exc}") from exc
            elapsed = time.perf_counter() - started
            row["return_code"] = str(return_code)
            row["elapsed_seconds"] = f"{elapsed:.6f}"
            row["completed_at_utc"] = _utc_now()
            if return_code != 0:
                row["status"] = "failed"
                row["error"] = f"train_joint exited with code {return_code}"
                _write_run_rows(csv_path, rows, plan.jobs)
                raise ScreenExecutionError(
                    f"Screen job {job.job_id} failed with return code {return_code}; resume to retry."
                )
            try:
                _, metrics_path, metrics_hash = _validate_metrics(job, run_output)
            except ScreenExecutionError as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                _write_run_rows(csv_path, rows, plan.jobs)
                raise
            row.update(
                {
                    "status": "completed",
                    "metrics_path": str(metrics_path),
                    "metrics_file_sha256": metrics_hash,
                    "error": "",
                }
            )
            # Required durability point: persist immediately after every job.
            _write_run_rows(csv_path, rows, plan.jobs)
        # Fold barrier: all six jobs and their one shared calibration must pass
        # before the executor is allowed to enter the next outer fold.
        _validate_fold_calibration(target_env, fold_jobs, rows)

    results = build_screen_results(plan, rows)
    results_path = output_root / "screen_results.json"
    _write_json_atomic(results_path, results)

    selected: Optional[Dict[str, object]] = None
    if select_lambdas:
        records = select_lambda.build_selection_records(plan.registry_path, results_path)
        destination = (selection_dir or output_root / "selection_records").resolve()
        _write_or_validate_selections(destination, records)
        selected = {target: records[target]["selected_lambda"] for target in protocol.ENVIRONMENTS}

    return {
        "mode": "execute",
        "job_count": len(plan.jobs),
        "completed_job_count": len(plan.jobs),
        "run_registry": str(csv_path),
        "screen_results": str(results_path),
        "selected_lambdas": selected,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Write the plan/CSV without launching training.")
    mode.add_argument("--execute", action="store_true", help="Explicitly launch the serial 18-job screen.")
    parser.add_argument(
        "--select-lambda",
        action="store_true",
        help="After all 18 jobs pass, generate the three hash-bound selection records.",
    )
    parser.add_argument("--selection-output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.dry_run and (args.select_lambda or args.selection_output_dir is not None):
        parser.error("--select-lambda/--selection-output-dir require --execute.")
    if args.selection_output_dir is not None and not args.select_lambda:
        parser.error("--selection-output-dir requires --select-lambda.")
    return args


def main() -> None:
    args = parse_args()
    summary = execute_screen(
        args.screen_registry,
        args.output_dir,
        device=args.device,
        dry_run=bool(args.dry_run),
        select_lambdas=bool(args.select_lambda),
        selection_dir=args.selection_output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
