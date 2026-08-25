from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BASE_RUNNER_PATH = (
    REPO_ROOT / "experiments" / "supcon_tpp_legacy_reproduction" / "run_factorial.py"
)
DEFAULT_BUNDLE = HERE / "generated_l1248_v1"
OUTPUT_ROOT = REPO_ROOT / "outputs_tpp_level8_ablation" / "l1248_v1"
EXPECTED_PARAMETERS = 4_786_294
LEVELS = [1, 2, 4, 8]
REFERENCE_LOOKUP: dict[tuple[str, int], dict[str, Any]] = {}


def load_base():
    spec = importlib.util.spec_from_file_location("level8_base_run_factorial", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SCHEMA_VERSION = "tpp-level8-runner/v1"
    module.COMPLETION_VERSION = "tpp-level8-attempt-completion/v1"
    module.SEAL_VERSION = "tpp-level8-attempt-seal/v1"
    module.EXPERIMENT_ROOT = HERE
    module.REPO_ROOT = REPO_ROOT
    module.DEFAULT_BUNDLE = DEFAULT_BUNDLE
    module.OUTPUT_ROOT = OUTPUT_ROOT
    module.ARM_ORDER = ("H11",)
    return module


base = load_base()


def changed_paths(reference: Any, candidate: Any, prefix: str = "") -> set[str]:
    if isinstance(reference, dict) and isinstance(candidate, dict):
        result: set[str] = set()
        for key in sorted(set(reference) | set(candidate)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in reference or key not in candidate:
                result.add(path)
            else:
                result |= changed_paths(reference[key], candidate[key], path)
        return result
    return {prefix} if reference != candidate else set()


def verify_bundle(bundle_root: Path):
    global REFERENCE_LOOKUP
    bundle_root = bundle_root.resolve()
    paths = {
        name: bundle_root / name
        for name in (
            "bundle_manifest.json",
            "protocol_lock.json",
            "level8_job_registry.json",
            "reference_registry.json",
            "execution_lock.json",
        )
    }
    for path in paths.values():
        if not path.is_file():
            raise base.RunnerError(f"Bundle member missing: {path}")
    manifest = base.read_json(paths["bundle_manifest.json"])
    protocol = base.read_json(paths["protocol_lock.json"])
    registry = base.read_json(paths["level8_job_registry.json"])
    references = base.read_json(paths["reference_registry.json"])
    execution = base.read_json(paths["execution_lock.json"])
    base.assert_equal(base.sha256_file(paths["protocol_lock.json"]), manifest["protocol_lock_sha256"], "protocol lock")
    base.assert_equal(base.sha256_file(paths["level8_job_registry.json"]), manifest["registry_sha256"], "registry")
    base.assert_equal(base.sha256_file(paths["reference_registry.json"]), manifest["reference_registry_sha256"], "reference registry")
    base.assert_equal(base.sha256_file(paths["execution_lock.json"]), manifest["execution_lock_sha256"], "execution lock")
    base.assert_equal(manifest["job_count"], 9, "manifest job count")
    base.assert_equal(registry["job_count"], 9, "registry job count")
    base.assert_equal(references["row_count"], 9, "reference row count")
    base.assert_equal(registry["protocol_lock_sha256"], manifest["protocol_lock_sha256"], "registry protocol binding")
    base.assert_equal(registry["reference_registry_sha256"], manifest["reference_registry_sha256"], "registry reference binding")
    base.assert_equal(execution["protocol_lock_sha256"], manifest["protocol_lock_sha256"], "execution protocol binding")
    base.assert_equal(execution["registry_sha256"], manifest["registry_sha256"], "execution registry binding")
    base.assert_equal(execution["reference_registry_sha256"], manifest["reference_registry_sha256"], "execution reference binding")
    for name, record in execution["scripts"].items():
        path = Path(record["path"]).resolve()
        base.assert_equal(base.sha256_file(path), record["sha256"], f"execution script {name}")
    for name, record in execution["required_gate_artifacts"].items():
        path = Path(record["path"]).resolve()
        base.assert_equal(base.sha256_file(path), record["sha256"], f"gate hash {name}")
        base.assert_equal(base.read_json(path)["status"], record["required_status"], f"gate status {name}")

    provenance = protocol["provenance"]
    clean_root = Path(provenance["clean_worktree"]).resolve()
    base.assert_equal(base.git_output(["rev-parse", "HEAD"], clean_root), provenance["legacy_commit"], "legacy commit")
    base.assert_equal(base.git_output(["status", "--porcelain"], clean_root), "", "legacy worktree status")
    for name, record in provenance["legacy_modules"].items():
        base.assert_equal(base.sha256_file(Path(record["path"])), record["sha256"], f"legacy module {name}")
    tpp_path = Path(provenance["tpp_snapshot_root"]) / "experiments" / "tpp_isolated" / "tpp_models.py"
    base.assert_equal(base.sha256_file(tpp_path), provenance["tpp_models_sha256"], "TPP snapshot")

    REFERENCE_LOOKUP = {
        (row["target_env"], int(row["seed"])): row for row in references["rows"]
    }
    jobs = []
    observed_order = []
    allowed = set(protocol["allowed_config_changes"])
    for item in registry["jobs"]:
        config_path = bundle_root / item["config_path"]
        config = base.read_json(config_path)
        reference = REFERENCE_LOOKUP[(item["target_env"], int(item["seed"]))]
        reference_config_path = Path(reference["source_config_path"])
        reference_config = base.read_json(reference_config_path)
        base.assert_equal(base.sha256_file(reference_config_path), reference["source_config_file_sha256"], "reference config hash")
        base.assert_equal(base.payload_sha256(reference_config), reference["source_config_payload_sha256"], "reference config payload")
        base.assert_equal(changed_paths(reference_config, config), allowed, f"allowed config delta {item['job_id']}")
        base.assert_equal(base.sha256_file(config_path), item["config_file_sha256"], "candidate config hash")
        base.assert_equal(base.payload_sha256(config), item["config_payload_sha256"], "candidate config payload")
        base.assert_equal(config["job_id"], item["job_id"], "job id")
        base.assert_equal(config["target_env"], item["target_env"], "target")
        base.assert_equal(config["seed_list"], [item["seed"]], "seed")
        base.assert_equal(config["factorial"]["arm"], "H11", "arm")
        base.assert_equal(config["model"]["tpp_levels"], LEVELS, "TPP levels")
        base.assert_equal(config["model"].get("channel_tpp_levels", LEVELS), LEVELS, "channel TPP levels")
        base.assert_equal(float(config["model"]["tpp_pyramid_scale"]), 0.25, "residual scale")
        base.assert_equal(config["training"]["batch_size"], 32, "batch size")
        base.assert_equal(config["training"]["sampler"], "domain_class_balanced", "sampler")
        base.assert_equal(config["training"]["epochs"], 200, "epochs")
        base.assert_equal(config["losses"]["lambda_supcon"], 0.05, "SupCon lambda")
        base.assert_equal(config["losses"]["temperature"], 0.2, "temperature")
        base.assert_equal(config["protocol"]["total_optimizer_steps"], 30000, "optimizer steps")
        base.assert_equal(config["protocol"]["expected_parameter_count"], EXPECTED_PARAMETERS, "parameter count")
        if item["target_env"] in config["data"]["env_files"]:
            raise base.RunnerError(f"Target leaked into source data: {item['job_id']}")
        base.assert_equal(list(config["target_data"]["data"]["env_files"]), [item["target_env"]], "target-only data")
        jobs.append(base.Job(
            job_id=str(item["job_id"]),
            target=str(item["target_env"]),
            seed=int(item["seed"]),
            arm="H11",
            config_path=config_path,
            config_file_sha256=str(item["config_file_sha256"]),
            config_payload_sha256=str(item["config_payload_sha256"]),
        ))
        observed_order.append((str(item["target_env"]), int(item["seed"]), "H11"))
    expected_order = [(target, seed, "H11") for target in protocol["targets"] for seed in protocol["seeds"]]
    base.assert_equal(observed_order, expected_order, "nine-job registry order")
    return manifest, protocol, execution, jobs


def paired_barrier(jobs: Sequence[Any], python: Path, target: str, seed: int) -> dict[str, Any]:
    cells = [job for job in jobs if job.target == target and job.seed == seed]
    base.assert_equal(len(cells), 1, "candidate cell count")
    job = cells[0]
    state = base.scan(job, python)
    if state.successful_attempt is None:
        raise base.RunnerError(f"Paired barrier missing candidate: {job.job_id}")
    health = next(item["health"] for item in state.prior if item["status"] == "success")
    reference = REFERENCE_LOOKUP[(target, seed)]
    base.assert_equal(health["sampler_schedule_sha256"], reference["sampler_schedule_sha256"], "paired sampler schedule")
    base.assert_equal(health["split_hashes"], reference["split_hashes"], "paired split hashes")
    base.assert_equal(health["parameter_count"], EXPECTED_PARAMETERS, "candidate parameter count")
    return {
        "schema_version": "tpp-level8-paired-barrier/v1",
        "status": "passed",
        "target": target,
        "seed": seed,
        "candidate_job_id": job.job_id,
        "candidate_attempt": str(state.successful_attempt),
        "reference_id": reference["reference_id"],
        "reference_attempt": reference["attempt_dir"],
        "sampler_schedule_sha256": health["sampler_schedule_sha256"],
        "split_hashes": health["split_hashes"],
        "candidate_parameter_count": health["parameter_count"],
        "reference_parameter_count": 3_721_334,
    }


base.quartet_barrier = paired_barrier


def finalize_results(jobs: Sequence[Any], python: Path, bundle_root: Path) -> Path:
    rows = []
    barriers = []
    for target in sorted({job.target for job in jobs}):
        for seed in sorted(job.seed for job in jobs if job.target == target):
            barriers.append(paired_barrier(jobs, python, target, seed))
    for job in jobs:
        state = base.scan(job, python)
        if state.successful_attempt is None:
            raise base.RunnerError(f"Cannot finalize missing job: {job.job_id}")
        attempt = state.successful_attempt
        metrics_path = attempt / "payload" / "run" / "metrics.json"
        metrics = base.read_json(metrics_path)
        rows.append({
            "job_id": job.job_id,
            "target_env": job.target,
            "seed": job.seed,
            "arm": job.arm,
            "reference_id": REFERENCE_LOOKUP[(job.target, job.seed)]["reference_id"],
            "attempt_dir": str(attempt),
            "completion_manifest_sha256": base.sha256_file(attempt / "attempt_completed.json"),
            "metrics_file_sha256": base.sha256_file(metrics_path),
            "checkpoint_sha256": metrics["checkpoint_sha256"],
            "target_predictions_file_sha256": metrics["target_predictions_file_sha256"],
            "target_prediction_count": metrics["target_prediction_count"],
            "test_metrics": metrics["test_metrics"],
            "per_class_metrics": metrics["per_class_metrics"],
        })
    result = {
        "schema_version": "tpp-level8-results/v1",
        "status": "complete",
        "created_at_utc": base.utc_now(),
        "job_count": len(rows),
        "bundle_manifest_sha256": base.sha256_file(bundle_root / "bundle_manifest.json"),
        "paired_barriers": barriers,
        "rows": rows,
        "analysis_status": "not_yet_run",
        "no_target_metric_tuning": True,
    }
    output = OUTPUT_ROOT / "level8_results.json"
    base.atomic_write_json(output, result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Append-only H11 L1248 runner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    python = args.python.resolve()
    bundle_root = args.bundle.resolve()
    _manifest, _protocol, _execution, jobs = verify_bundle(bundle_root)
    plans = []
    for job in jobs:
        state = base.scan(job, python)
        plans.append({
            "job_id": job.job_id,
            "target_env": job.target,
            "seed": job.seed,
            "arm": job.arm,
            "action": state.action,
            "next_attempt": state.next_attempt,
            "successful_attempt": str(state.successful_attempt) if state.successful_attempt else None,
            "prior": list(state.prior),
        })
    summary = {
        "schema_version": "tpp-level8-runner/v1",
        "mode": "dry_run" if args.dry_run else "execute",
        "bundle_manifest_sha256": base.sha256_file(bundle_root / "bundle_manifest.json"),
        "planned_jobs": len(plans),
        "jobs_to_run": sum(item["action"] == "run" for item in plans),
        "jobs_to_skip": sum(item["action"] == "skip" for item in plans),
        "jobs": plans,
    }
    if args.dry_run:
        if args.summary_only:
            summary["jobs"] = [
                {key: item[key] for key in ("job_id", "target_env", "seed", "action", "next_attempt")}
                for item in plans
            ]
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return

    invocation_id = uuid.uuid4().hex
    with base.RunnerLock(invocation_id):
        invocation = OUTPUT_ROOT / "runner_invocations" / invocation_id
        invocation.mkdir(parents=True, exist_ok=False)
        base.write_json_exclusive(invocation / "plan.json", summary)
        results = []
        current_block = None
        for job, plan in zip(jobs, plans):
            block = (job.target, job.seed)
            if current_block is not None and block != current_block:
                barrier = paired_barrier(jobs, python, current_block[0], current_block[1])
                base.write_json_exclusive(invocation / f"barrier_{current_block[0]}_seed_{current_block[1]}.json", barrier)
            current_block = block
            if plan["action"] == "skip":
                results.append({"job_id": job.job_id, "status": "skipped_healthy_success", "attempt": plan["successful_attempt"]})
                continue
            attempt, health = base.execute_job(job, int(plan["next_attempt"]), python)
            results.append({"job_id": job.job_id, "status": "success", "attempt": str(attempt), "health_sha256": base.payload_sha256(health)})
        if current_block is not None:
            barrier = paired_barrier(jobs, python, current_block[0], current_block[1])
            base.write_json_exclusive(invocation / f"barrier_{current_block[0]}_seed_{current_block[1]}.json", barrier)
        base.write_json_exclusive(invocation / "results.json", {"schema_version": "tpp-level8-runner/v1", "status": "complete", "results": results})
        final = finalize_results(jobs, python, bundle_root)
        base.write_json_exclusive(invocation / "completed.json", {
            "schema_version": "tpp-level8-runner/v1",
            "status": "complete",
            "level8_results": str(final),
            "level8_results_sha256": base.sha256_file(final),
        })
    print(json.dumps({"status": "complete", "jobs": len(jobs), "level8_results": str(final)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
