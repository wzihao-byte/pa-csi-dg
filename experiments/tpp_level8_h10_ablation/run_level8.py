from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SOURCE = REPO_ROOT / "experiments" / "tpp_level8_ablation" / "run_level8.py"
DEFAULT_BUNDLE = HERE / "generated_l1248_h10_v1"
OUTPUT_ROOT = REPO_ROOT / "outputs_tpp_level8_ablation" / "l1248_h10_v1"
LEVELS = [1, 2, 4, 8]
EXPECTED_PARAMETERS = 4_786_294


def load_runner():
    spec = importlib.util.spec_from_file_location("level8_h10_runner_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()
base = runner.base
runner.DEFAULT_BUNDLE = DEFAULT_BUNDLE
runner.OUTPUT_ROOT = OUTPUT_ROOT
base.EXPERIMENT_ROOT = HERE
base.DEFAULT_BUNDLE = DEFAULT_BUNDLE
base.OUTPUT_ROOT = OUTPUT_ROOT
base.ARM_ORDER = ("H10",)


def clean_git_output(arguments: list[str], cwd: Path) -> str:
    safe = cwd.resolve().as_posix()
    return subprocess.run(
        ["git", "-c", f"safe.directory={safe}", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def verify_bundle(bundle_root: Path):
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
    base.assert_equal(clean_git_output(["rev-parse", "HEAD"], clean_root), provenance["legacy_commit"], "legacy commit")
    base.assert_equal(clean_git_output(["status", "--porcelain"], clean_root), "", "legacy worktree status")
    for name, record in provenance["legacy_modules"].items():
        base.assert_equal(base.sha256_file(Path(record["path"])), record["sha256"], f"legacy module {name}")
    tpp_path = Path(provenance["tpp_snapshot_root"]) / "experiments" / "tpp_isolated" / "tpp_models.py"
    base.assert_equal(base.sha256_file(tpp_path), provenance["tpp_models_sha256"], "TPP snapshot")

    lookup: dict[tuple[str, int], dict[str, Any]] = {
        (row["target_env"], int(row["seed"])): row for row in references["rows"]
    }
    runner.REFERENCE_LOOKUP = lookup
    jobs = []
    observed_order = []
    allowed = set(protocol["allowed_config_changes"])
    for item in registry["jobs"]:
        config_path = bundle_root / item["config_path"]
        config = base.read_json(config_path)
        reference = lookup[(item["target_env"], int(item["seed"]))]
        reference_config_path = Path(reference["source_config_path"])
        reference_config = base.read_json(reference_config_path)
        base.assert_equal(base.sha256_file(reference_config_path), reference["source_config_file_sha256"], "reference config hash")
        base.assert_equal(base.payload_sha256(reference_config), reference["source_config_payload_sha256"], "reference config payload")
        base.assert_equal(runner.changed_paths(reference_config, config), allowed, f"allowed config delta {item['job_id']}")
        base.assert_equal(base.sha256_file(config_path), item["config_file_sha256"], "candidate config hash")
        base.assert_equal(base.payload_sha256(config), item["config_payload_sha256"], "candidate config payload")
        base.assert_equal(config["job_id"], item["job_id"], "job id")
        base.assert_equal(config["target_env"], item["target_env"], "target")
        base.assert_equal(config["seed_list"], [item["seed"]], "seed")
        base.assert_equal(config["factorial"]["arm"], "H10", "arm")
        base.assert_equal(config["factorial"]["supcon_on"], False, "SupCon off")
        base.assert_equal(config["model"]["tpp_levels"], LEVELS, "TPP levels")
        base.assert_equal(config["model"].get("channel_tpp_levels", LEVELS), LEVELS, "channel TPP levels")
        base.assert_equal(float(config["model"]["tpp_pyramid_scale"]), 0.25, "residual scale")
        base.assert_equal(config["training"]["batch_size"], 32, "batch size")
        base.assert_equal(config["training"]["sampler"], "domain_class_balanced", "sampler")
        base.assert_equal(config["training"]["epochs"], 200, "epochs")
        base.assert_equal(float(config["losses"]["lambda_supcon"]), 0.0, "SupCon lambda")
        base.assert_equal(config["protocol"]["total_optimizer_steps"], 30000, "optimizer steps")
        base.assert_equal(config["protocol"]["expected_parameter_count"], EXPECTED_PARAMETERS, "parameter count")
        if item["target_env"] in config["data"]["env_files"]:
            raise base.RunnerError(f"Target leaked into source data: {item['job_id']}")
        base.assert_equal(list(config["target_data"]["data"]["env_files"]), [item["target_env"]], "target-only data")
        jobs.append(base.Job(
            job_id=str(item["job_id"]),
            target=str(item["target_env"]),
            seed=int(item["seed"]),
            arm="H10",
            config_path=config_path,
            config_file_sha256=str(item["config_file_sha256"]),
            config_payload_sha256=str(item["config_payload_sha256"]),
        ))
        observed_order.append((str(item["target_env"]), int(item["seed"]), "H10"))
    expected_order = [(target, seed, "H10") for target in protocol["targets"] for seed in protocol["seeds"]]
    base.assert_equal(observed_order, expected_order, "nine-job registry order")
    return manifest, protocol, execution, jobs


runner.verify_bundle = verify_bundle
base.quartet_barrier = runner.paired_barrier


if __name__ == "__main__":
    runner.main()
