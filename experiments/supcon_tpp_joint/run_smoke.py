"""Run and fail-closed verify the paired Stage-0 smoke gate."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

try:
    from . import check_protocol, protocol
    from .runner_utils import write_json
except ImportError:  # pragma: no cover
    import check_protocol  # type: ignore
    import protocol  # type: ignore
    from runner_utils import write_json  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = Path(__file__).resolve().parent / "generated_protocol"
SMOKE_EPOCHS = 10
# The plan asks for one same-arm/same-seed repeat and defines determinism only as
# identical batch IDs and initialization hashes.  Repeating the control arm is
# the smallest implementation of that gate; metrics and trained weights are not
# compared because the preregistration does not ask for either.
DETERMINISM_REPEAT_ARM = "Y00"


class SmokeGateError(RuntimeError):
    """Raised when any Stage-0 smoke invariant is absent or inconsistent."""


CommandRunner = Callable[..., Any]


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeGateError(f"Cannot read valid JSON artifact {path}: {exc}") from exc


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SmokeGateError(f"{context} must be a JSON object.")
    return value


def _list(value: Any, *, context: str) -> List[Any]:
    if not isinstance(value, list):
        raise SmokeGateError(f"{context} must be a JSON list.")
    return value


def _sha256(value: Any, *, context: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise SmokeGateError(f"{context} must be a lowercase SHA256 digest.")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--target-env", choices=protocol.ENVIRONMENTS, default="E1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _registry_jobs(payload: Any) -> List[Mapping[str, Any]]:
    registry = _mapping(payload, context="screen_job_registry.json")
    jobs = _list(registry.get("jobs"), context="screen_job_registry.json.jobs")
    if not all(isinstance(job, Mapping) for job in jobs):
        raise SmokeGateError("Every screen registry job must be a JSON object.")
    return list(jobs)


def _config_path(bundle_dir: Path, job: Mapping[str, Any]) -> Path:
    raw_path = job.get("config_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise SmokeGateError(f"Registry job {job.get('job_id')} has no config_path.")
    bundle_root = bundle_dir.resolve()
    path = (bundle_root / raw_path).resolve()
    if path != bundle_root and bundle_root not in path.parents:
        raise SmokeGateError(f"Registry config path escapes the bundle: {raw_path}")
    return path


def validate_registry_job_config_hash(bundle_dir: Path, job: Mapping[str, Any]) -> Path:
    """Revalidate one registry-to-config binding immediately before launch."""

    path = _config_path(bundle_dir, job)
    expected_file_hash = _sha256(
        job.get("config_file_sha256"), context=f"{job.get('job_id')} config_file_sha256"
    )
    expected_content_hash = _sha256(
        job.get("config_content_sha256"), context=f"{job.get('job_id')} config_content_sha256"
    )
    if not path.is_file():
        raise SmokeGateError(f"Registry config is missing: {path}")
    actual_file_hash = protocol.file_sha256(path)
    if actual_file_hash != expected_file_hash:
        raise SmokeGateError(f"Config file hash mismatch for {job.get('job_id')}.")
    config = _mapping(load_json(path), context=str(path))
    actual_content_hash = protocol.canonical_sha256(config)
    if actual_content_hash != expected_content_hash:
        raise SmokeGateError(f"Config content hash mismatch for {job.get('job_id')}.")
    if config.get("experiment_name") != job.get("job_id"):
        raise SmokeGateError(f"Config job binding mismatch for {job.get('job_id')}.")
    return path


def validate_registry_config_hashes(
    bundle_dir: Path, registry: Sequence[Mapping[str, Any]]
) -> Dict[str, str]:
    """Validate every registry config, not only the four selected smoke cells."""

    validated: Dict[str, str] = {}
    for job in registry:
        job_id = str(job.get("job_id"))
        if not job_id or job_id == "None" or job_id in validated:
            raise SmokeGateError(f"Invalid or duplicate registry job_id: {job_id}")
        validated[job_id] = str(validate_registry_job_config_hash(bundle_dir, job))
    return validated


def preflight_bundle(bundle_dir: Path) -> Tuple[Dict[str, Any], List[Mapping[str, Any]]]:
    """Run the complete protocol checker before creating outputs or launching work."""

    bundle_dir = bundle_dir.resolve()
    check_summary = check_protocol.check_bundle(bundle_dir)
    registry = _registry_jobs(load_json(bundle_dir / "screen_job_registry.json"))
    validate_registry_config_hashes(bundle_dir, registry)
    return check_summary, registry


def choose_jobs(
    registry: Sequence[Mapping[str, Any]], target_env: str
) -> List[Mapping[str, Any]]:
    selected: List[Mapping[str, Any]] = []
    for arm in protocol.ARMS:
        candidates = [
            job
            for job in registry
            if job.get("target_env") == target_env and job.get("arm") == arm
        ]
        if not candidates:
            raise SmokeGateError(f"No screen job exists for {target_env}/{arm}.")
        try:
            selected.append(min(candidates, key=lambda job: float(job["lambda_supcon"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise SmokeGateError(f"Invalid lambda metadata for {target_env}/{arm}.") from exc
    seeds = {int(job.get("screen_seed", -1)) for job in selected}
    if len(seeds) != 1:
        raise SmokeGateError(f"Paired smoke arms do not share one screen seed: {sorted(seeds)}")
    return selected


def build_train_command(
    job: Mapping[str, Any], config_path: Path, run_output: Path, *, device: str
) -> List[str]:
    return [
        sys.executable,
        "-m",
        "experiments.supcon_tpp_joint.train_joint",
        "--config",
        str(config_path),
        "--seed",
        str(job["screen_seed"]),
        "--epochs",
        str(SMOKE_EPOCHS),
        "--evaluate-target",
        "false",
        "--device",
        device,
        "--output-dir",
        str(run_output),
    ]


def _load_run_artifacts(row: Mapping[str, Any]) -> Dict[str, Any]:
    output_dir = Path(str(row.get("output_dir", ""))).resolve()
    if not output_dir.is_dir():
        raise SmokeGateError(f"Smoke output directory is missing: {output_dir}")

    metrics = _mapping(load_json(output_dir / "metrics.json"), context=f"{output_dir}/metrics.json")
    split_manifest = _mapping(
        load_json(output_dir / "split_manifest.json"), context=f"{output_dir}/split_manifest.json"
    )
    phase_audit = _mapping(
        load_json(output_dir / "phase_input_audit.json"),
        context=f"{output_dir}/phase_input_audit.json",
    )
    batch_audit = _mapping(
        load_json(output_dir / "batch_schedule_audit.json"),
        context=f"{output_dir}/batch_schedule_audit.json",
    )
    parameter_audit = _mapping(
        load_json(output_dir / "parameter_audit.json"), context=f"{output_dir}/parameter_audit.json"
    )
    stage0_gate = _mapping(
        load_json(output_dir / "stage0_gate.json"), context=f"{output_dir}/stage0_gate.json"
    )
    invocation = _mapping(
        load_json(output_dir / "run_invocation.json"), context=f"{output_dir}/run_invocation.json"
    )
    history = _list(load_json(output_dir / "history.json"), context=f"{output_dir}/history.json")

    arm = str(metrics.get("arm"))
    target_env = str(metrics.get("target_env"))
    seed = int(metrics.get("seed", -1))
    if arm != row.get("arm") or target_env != row.get("target_env") or seed != int(row.get("seed", -1)):
        raise SmokeGateError(f"Output identity does not match registry row for {output_dir}.")
    if invocation.get("raw_config_sha256") != row.get("config_content_sha256"):
        raise SmokeGateError(f"Run invocation is not bound to the registered config for {arm}.")

    resources = _mapping(metrics.get("resources"), context=f"{arm} metrics.resources")
    embedded_batch = _mapping(
        resources.get("batch_schedule_audit"), context=f"{arm} resources.batch_schedule_audit"
    )
    embedded_history = _list(resources.get("history"), context=f"{arm} resources.history")
    embedded_parameters = _mapping(
        metrics.get("parameter_report"), context=f"{arm} metrics.parameter_report"
    )
    if dict(embedded_batch) != dict(batch_audit):
        raise SmokeGateError(f"{arm} batch_schedule_audit.json disagrees with metrics.json.")
    if embedded_history != history:
        raise SmokeGateError(f"{arm} history.json disagrees with metrics.json.")
    if dict(embedded_parameters) != dict(parameter_audit):
        raise SmokeGateError(f"{arm} parameter_audit.json disagrees with metrics.json.")
    if stage0_gate.get("parameter_audit") != parameter_audit:
        raise SmokeGateError(f"{arm} stage0 parameter audit is inconsistent.")

    split_hash = protocol.canonical_sha256(split_manifest)
    phase_hash = protocol.canonical_sha256(phase_audit)
    if stage0_gate.get("split_manifest_sha256") != split_hash:
        raise SmokeGateError(f"{arm} split manifest hash is inconsistent.")
    if stage0_gate.get("phase_audit_sha256") != phase_hash:
        raise SmokeGateError(f"{arm} phase input audit hash is inconsistent.")

    calibration_hash = _sha256(metrics.get("calibration_sha256"), context=f"{arm} calibration hash")
    calibration_raw_path = metrics.get("calibration_artifact")
    if not isinstance(calibration_raw_path, str) or not calibration_raw_path:
        raise SmokeGateError(f"{arm} has no calibration artifact path.")
    calibration_path = Path(calibration_raw_path)
    if not calibration_path.is_absolute():
        calibration_path = REPO_ROOT / calibration_path
    calibration_path = calibration_path.resolve()
    if not calibration_path.is_file():
        raise SmokeGateError(f"{arm} calibration artifact is missing: {calibration_path}")
    if protocol.file_sha256(calibration_path) != calibration_hash:
        raise SmokeGateError(f"{arm} calibration artifact does not match its reported hash.")
    if (
        parameter_audit.get("calibration_artifact") != calibration_raw_path
        or parameter_audit.get("calibration_sha256") != calibration_hash
        or stage0_gate.get("calibration_sha256") != calibration_hash
    ):
        raise SmokeGateError(f"{arm} calibration artifact/hash bindings are inconsistent.")

    initialization_hashes = _mapping(
        parameter_audit.get("initialization_hashes"), context=f"{arm} initialization_hashes"
    )
    if set(initialization_hashes) != set(protocol.ARMS):
        raise SmokeGateError(f"{arm} initialization audit must contain all four arms.")
    validated_initialization = {
        code: _sha256(value, context=f"{arm}/{code} initialization hash")
        for code, value in initialization_hashes.items()
    }
    if len(set(validated_initialization.values())) != 1:
        raise SmokeGateError(f"{arm} initialization hashes differ within its four-arm audit.")

    if int(resources.get("epochs", -1)) != SMOKE_EPOCHS:
        raise SmokeGateError(f"{arm} did not execute exactly {SMOKE_EPOCHS} smoke epochs.")
    expected_steps = SMOKE_EPOCHS * protocol.OPTIMIZER_STEPS_PER_EPOCH
    if int(resources.get("optimizer_steps", -1)) != expected_steps:
        raise SmokeGateError(f"{arm} did not execute {expected_steps} smoke optimizer steps.")
    if len(history) != SMOKE_EPOCHS:
        raise SmokeGateError(f"{arm} history does not contain exactly {SMOKE_EPOCHS} epochs.")

    epoch_schedule_hashes: List[str] = []
    for expected_epoch, raw_record in enumerate(history, start=1):
        record = _mapping(raw_record, context=f"{arm} history epoch {expected_epoch}")
        if int(record.get("epoch", -1)) != expected_epoch:
            raise SmokeGateError(f"{arm} history epoch numbering is not contiguous.")
        train = _mapping(record.get("train"), context=f"{arm} epoch {expected_epoch} train")
        epoch_schedule_hashes.append(
            _sha256(
                train.get("schedule_sha256"),
                context=f"{arm} epoch {expected_epoch} batch schedule",
            )
        )
    initial_schedule_hash = _sha256(
        batch_audit.get("schedule_sha256"), context=f"{arm} initial batch schedule"
    )
    if initial_schedule_hash != epoch_schedule_hashes[0]:
        raise SmokeGateError(f"{arm} epoch-1 schedule disagrees with its batch audit.")
    first_batch_ids = _list(
        batch_audit.get("first_batch_sample_ids"), context=f"{arm} first batch sample IDs"
    )
    if len(first_batch_ids) != protocol.BATCH_SIZE:
        raise SmokeGateError(f"{arm} first batch audit must contain {protocol.BATCH_SIZE} sample IDs.")

    forbidden = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.name.startswith("target_")
    )
    screen_metrics = _mapping(metrics.get("screen_metrics"), context=f"{arm} screen_metrics")
    if metrics.get("evaluate_target") is not False or stage0_gate.get("target_opened") is not False or forbidden:
        raise SmokeGateError(f"{arm} emitted target-disabled artifacts: {forbidden}")
    if screen_metrics.get("finite_training") is not True or screen_metrics.get("gradient_exploded") is not False:
        raise SmokeGateError(f"{arm} did not pass the finite-training smoke gate.")

    return {
        "arm": arm,
        "target_env": target_env,
        "seed": seed,
        "metrics": metrics,
        "split_manifest": dict(split_manifest),
        "split_manifest_sha256": split_hash,
        "phase_input_audit": dict(phase_audit),
        "phase_input_audit_sha256": phase_hash,
        "calibration_artifact": str(calibration_path),
        "calibration_sha256": calibration_hash,
        "initialization_hashes": validated_initialization,
        "common_initialization_sha256": next(iter(validated_initialization.values())),
        "trainable_parameter_count": int(parameter_audit.get("trainable_parameter_count", -1)),
        "first_batch_sample_ids": list(first_batch_ids),
        "epoch_batch_schedule_sha256": epoch_schedule_hashes,
        "resources": resources,
    }


def _all_equal(values: Sequence[Any]) -> bool:
    return bool(values) and all(value == values[0] for value in values[1:])


def verify_smoke_outputs(
    run_rows: Sequence[Mapping[str, Any]], deterministic_repeat_row: Mapping[str, Any]
) -> Dict[str, Any]:
    if len(run_rows) != len(protocol.ARMS):
        raise SmokeGateError(f"Smoke verification requires exactly {len(protocol.ARMS)} primary arms.")
    artifacts = [_load_run_artifacts(row) for row in run_rows]
    by_arm = {item["arm"]: item for item in artifacts}
    if set(by_arm) != set(protocol.ARMS) or len(by_arm) != len(artifacts):
        raise SmokeGateError("Primary smoke outputs must be exactly Y00/Y01/Y10/Y11.")

    failures: List[str] = []
    split_manifests = [by_arm[arm]["split_manifest"] for arm in protocol.ARMS]
    phase_audits = [by_arm[arm]["phase_input_audit"] for arm in protocol.ARMS]
    calibration_paths = [by_arm[arm]["calibration_artifact"] for arm in protocol.ARMS]
    calibration_hashes = [by_arm[arm]["calibration_sha256"] for arm in protocol.ARMS]
    initialization_maps = [by_arm[arm]["initialization_hashes"] for arm in protocol.ARMS]
    parameter_counts = [by_arm[arm]["trainable_parameter_count"] for arm in protocol.ARMS]
    first_batch_ids = [by_arm[arm]["first_batch_sample_ids"] for arm in protocol.ARMS]
    epoch_schedules = [by_arm[arm]["epoch_batch_schedule_sha256"] for arm in protocol.ARMS]

    if not _all_equal(split_manifests):
        failures.append("split manifests differ across arms")
    if not _all_equal(phase_audits):
        failures.append("phase input audits differ across arms")
    if not _all_equal(calibration_paths):
        failures.append("calibration artifact paths differ across arms")
    if not _all_equal(calibration_hashes):
        failures.append("calibration artifact hashes differ across arms")
    if not _all_equal(initialization_maps):
        failures.append("trainable initialization hashes differ across arms")
    if not _all_equal(parameter_counts):
        failures.append("trainable parameter counts differ across arms")
    if not _all_equal(first_batch_ids):
        failures.append("first-batch sample IDs differ across arms")
    if not _all_equal(epoch_schedules):
        failures.append("epoch batch schedules differ across arms")

    repeat = _load_run_artifacts(deterministic_repeat_row)
    reference = by_arm[DETERMINISM_REPEAT_ARM]
    if (
        repeat["arm"] != reference["arm"]
        or repeat["seed"] != reference["seed"]
        or repeat["target_env"] != reference["target_env"]
    ):
        failures.append("deterministic repeat is not the same arm, seed, and target")
    if repeat["first_batch_sample_ids"] != reference["first_batch_sample_ids"]:
        failures.append("deterministic repeat batch IDs differ")
    if repeat["epoch_batch_schedule_sha256"] != reference["epoch_batch_schedule_sha256"]:
        failures.append("deterministic repeat epoch batch schedules differ")
    if repeat["initialization_hashes"] != reference["initialization_hashes"]:
        failures.append("deterministic repeat initialization hashes differ")

    report = {
        "passed": not failures,
        "failures": failures,
        "primary_arms": list(protocol.ARMS),
        "split_manifest_sha256": {
            arm: by_arm[arm]["split_manifest_sha256"] for arm in protocol.ARMS
        },
        "phase_input_audit_sha256": {
            arm: by_arm[arm]["phase_input_audit_sha256"] for arm in protocol.ARMS
        },
        "calibration_artifacts": {
            arm: by_arm[arm]["calibration_artifact"] for arm in protocol.ARMS
        },
        "calibration_hashes": {
            arm: by_arm[arm]["calibration_sha256"] for arm in protocol.ARMS
        },
        "initialization_hashes": {
            arm: by_arm[arm]["common_initialization_sha256"] for arm in protocol.ARMS
        },
        "parameter_counts": {
            arm: by_arm[arm]["trainable_parameter_count"] for arm in protocol.ARMS
        },
        "epoch_batch_schedule_sha256": {
            arm: by_arm[arm]["epoch_batch_schedule_sha256"] for arm in protocol.ARMS
        },
        "deterministic_repeat": {
            "arm": repeat["arm"],
            "seed": repeat["seed"],
            "target_env": repeat["target_env"],
            "comparison_scope": [
                "first_batch_sample_ids",
                "all_epoch_batch_schedule_sha256",
                "initialization_hashes",
            ],
            "passed": not any(failure.startswith("deterministic repeat") for failure in failures),
        },
        "resources": {
            arm: {
                "mean_seconds_per_epoch": by_arm[arm]["resources"]["mean_seconds_per_epoch"],
                "peak_cuda_allocated_bytes": by_arm[arm]["resources"]["peak_cuda_allocated_bytes"],
                "peak_cuda_reserved_bytes": by_arm[arm]["resources"]["peak_cuda_reserved_bytes"],
            }
            for arm in protocol.ARMS
        },
    }
    if failures:
        raise SmokeGateError("Smoke gate failed: " + "; ".join(failures))
    return report


def execute_smoke(
    bundle_dir: Path,
    target_env: str,
    *,
    device: str,
    output_dir: Path,
    dry_run: bool,
    command_runner: CommandRunner = subprocess.run,
) -> Dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    # This is deliberately the first operation with external effects: a broken
    # bundle must fail before the output directory is created or a process runs.
    bundle_summary, registry = preflight_bundle(bundle_dir)
    jobs = choose_jobs(registry, target_env)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_job = next(job for job in jobs if job["arm"] == DETERMINISM_REPEAT_ARM)
    run_specs: List[Tuple[Mapping[str, Any], str]] = [
        (job, "primary") for job in jobs
    ] + [(reference_job, "determinism_repeat")]
    rows: List[Dict[str, Any]] = []
    for job, role in run_specs:
        # Repeat the registry binding check immediately before every subprocess
        # to close the gap between the complete bundle preflight and launch.
        config_path = validate_registry_job_config_hash(bundle_dir, job)
        arm = str(job["arm"])
        run_name = arm if role == "primary" else f"{arm}__determinism_repeat"
        run_output = output_dir / run_name
        command = build_train_command(job, config_path, run_output, device=device)
        row: Dict[str, Any] = {
            "run_id": str(job["job_id"]) if role == "primary" else f"{job['job_id']}__repeat",
            "run_role": role,
            "job_id": job["job_id"],
            "target_env": target_env,
            "arm": arm,
            "lambda_supcon": job["lambda_supcon"],
            "seed": job["screen_seed"],
            "config_path": str(config_path),
            "config_file_sha256": job["config_file_sha256"],
            "config_content_sha256": job["config_content_sha256"],
            "output_dir": str(run_output),
            "command": subprocess.list2cmdline(command),
            "status": "planned" if dry_run else "running",
        }
        rows.append(row)
        if dry_run:
            continue
        started = time.perf_counter()
        completed = command_runner(command, cwd=REPO_ROOT, check=False)
        row["elapsed_seconds"] = time.perf_counter() - started
        row["return_code"] = int(completed.returncode)
        row["status"] = "completed" if completed.returncode == 0 else "failed"
        write_registry(output_dir / "smoke_registry.csv", rows)
        if completed.returncode != 0:
            raise SmokeGateError(
                f"Smoke run {row['run_id']} failed with code {completed.returncode}."
            )

    write_registry(output_dir / "smoke_registry.csv", rows)
    if dry_run:
        payload = {
            "dry_run": True,
            "primary_jobs": len(protocol.ARMS),
            "determinism_repeat_jobs": 1,
            "jobs": rows,
            "bundle_check": bundle_summary,
        }
        write_json(output_dir / "dry_run.json", payload)
        return payload

    primary_rows = [row for row in rows if row["run_role"] == "primary"]
    repeat_rows = [row for row in rows if row["run_role"] == "determinism_repeat"]
    if len(repeat_rows) != 1:
        raise SmokeGateError("Exactly one deterministic repeat is required.")
    report = verify_smoke_outputs(primary_rows, repeat_rows[0])
    report["bundle_check"] = bundle_summary
    write_json(output_dir / "smoke_gate_report.json", report)
    return report


def main() -> None:
    args = parse_args()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else REPO_ROOT / "outputs_supcon_tpp_joint" / "stage0_smoke" / f"target_{args.target_env}"
    )
    report = execute_smoke(
        args.bundle_dir,
        args.target_env,
        device=args.device,
        output_dir=output_dir,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def write_registry(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
