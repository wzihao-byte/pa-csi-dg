"""Freeze 18 original-H L1248 runs with verified grouped source partitions."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import psutil
import torch

from common import (ARMS, BUNDLE, HERE, OUTPUT, PARAMETERS, ROOT, SEEDS, TARGETS,
                    grouped_split, index_hash, legacy, metadata_dataset, read,
                    replay_sampler, require, sha, state_hash, verify_config, write_new)


def main():
    require(not BUNDLE.exists(), "Preserve existing frozen bundle; use a new version for any change")
    require(all((HERE / name).is_file() for name in ("qa.py", "run.py", "analyze.py", "queue_after_inputs.py", "EXPERIMENT_PLAN.md")), "Supplement code/protocol incomplete")
    upstream = psutil.Process(18384)
    require(abs(upstream.create_time() - 1789026459.2377293) < .01, "Existing MultiEnv input coordinator identity changed")
    require("analysis/continue_inputs_after_controls.py" in upstream.cmdline(), "Unexpected upstream process")
    inputs = Path("D:/pa-csi-dg-revision-runtime/studies/r_inputs_v1")
    prerequisite = {"pid": upstream.pid, "process_create_time": upstream.create_time(), "command": upstream.cmdline(),
                    "coordinator_output": "D:/pa-csi-dg-revision-runtime/coordinators/controls_to_inputs_v2",
                    "input_bundle": str(inputs), "input_bundle_sha256": sha(inputs / "bundle_manifest.json"),
                    "widar_cancelled_by_user": True}
    BUNDLE.mkdir(parents=True)
    base = legacy()
    torch.set_num_threads(2)
    source_root = Path("D:/pa-csi-dg-revision-runtime/studies/r_core_v1")
    source_bundle = read(source_root / "bundle_manifest.json")
    historical = {}
    for arm, folder in (("H10", "l1248_h10_v1"), ("H11", "l1248_v1")):
        index = read(ROOT / "outputs_tpp_level8_ablation" / folder / "level8_results.json")
        require(index["status"] == "complete" and len(index["rows"]) == 9, "Historical reference incomplete")
        for row in index["rows"]:
            attempt = Path(row["attempt_dir"])
            require(sha(attempt / "attempt_completed.json") == row["completion_manifest_sha256"], "Historical completion changed")
            completion = read(attempt / "attempt_completed.json")
            require(completion["status"] == "success", "Historical run failed")
            stem = f"tpp_l1248__target_{row['target_env']}__seed_{row['seed']}__arm_{arm}.json"
            generated = "tpp_level8_h10_ablation/generated_l1248_h10_v1" if arm == "H10" else "tpp_level8_ablation/generated_l1248_v1"
            reference_path = ROOT / "experiments" / generated / "configs" / stem
            require(sha(reference_path) == completion["source_config_file_sha256"], "Historical generated configuration changed")
            reference = read(reference_path)
            executed = read(attempt / "job_config.json")
            executed.pop("output_dir", None)
            comparable = copy.deepcopy(reference)
            comparable.pop("output_dir", None)
            require(comparable == executed, "Reference was not the configuration actually trained")
            historical[(row["target_env"], row["seed"], arm)] = (reference, reference_path, row)

    init_records = {}
    for seed in SEEDS:
        hashes = []
        for arm in ARMS:
            cfg = historical[("E1", seed, arm)][0]
            modules = base.configure_imports(cfg)
            modules["train_dg"].set_seed(seed)
            cls = base.model_class(cfg, modules)
            model = cls(input_dim=90, num_classes=6, model_config=cfg["model"])
            require(sum(p.numel() for p in model.parameters()) == PARAMETERS, "Historical H parameter mismatch")
            hashes.append(state_hash(model))
            del model
        require(hashes[0] == hashes[1], "H10/H11 initial models differ")
        init_records[str(seed)] = hashes[0]
    registry, audits, references = [], {}, []
    external_code = {}
    for target in TARGETS:
        source_path = source_root / "manifests" / f"source_{target}.json"
        expected_source_sha = source_bundle["files_sha256"][f"manifests/source_{target}.json"]
        require(sha(source_path) == expected_source_sha, "Frozen source membership changed")
        source_copy = BUNDLE / "manifests" / source_path.name
        source_copy.parent.mkdir(exist_ok=True)
        shutil.copy2(source_path, source_copy)
        source = read(source_copy)
        for env, record in source["files"].items():
            require(sha(record["sidecar_path"]) == record["sidecar_sha256"], "Subject identity sidecar changed")
            sidecar_copy = BUNDLE / "identity" / f"{env}.json"
            sidecar_copy.parent.mkdir(exist_ok=True)
            if not sidecar_copy.exists():
                shutil.copy2(record["sidecar_path"], sidecar_copy)
        dataset = metadata_dataset(source)
        split, identity = grouped_split(dataset, source, target)
        grouped_hashes = {part: index_hash(dataset.sample_ids[split[f"{part}_indices"]]) for part in ("train", "val")}
        audits[target] = {"source_manifest_sha256": sha(source_copy), "split_hashes": grouped_hashes,
                          "counts": {part: {key: len(values) for key, values in group.items()} for part, group in identity.items()},
                          "validation_subjects": identity["val"]["subject_id"], "identity_overlap": False}
        for seed in SEEDS:
            ref = historical[(target, seed, "H11")][0]
            modules = base.configure_imports(ref)
            sampler_hash, schedule = replay_sampler(dataset, split, seed, modules["dg_dataset"].DomainClassBalancedBatchSampler)
            schedule_path = BUNDLE / "schedules" / f"{target}_{seed}.npy"
            schedule_path.parent.mkdir(exist_ok=True)
            np.save(schedule_path, schedule, allow_pickle=False)
            for arm in ARMS:
                reference, original_path, row = historical[(target, seed, arm)]
                for env, records in reference["source_file_records"].items():
                    for kind, record in records.items():
                        require(record["sha256"] == source["files"][env]["actual_file_sha256"][kind], "Grouped identity belongs to different original arrays")
                reference_copy = BUNDLE / "references" / original_path.name
                reference_copy.parent.mkdir(exist_ok=True)
                shutil.copy2(original_path, reference_copy)
                job_id = f"{target}_HG_{arm}_{seed}"
                config = copy.deepcopy(reference)
                config.update(experiment_name=f"h_grouped_l1248__{job_id}", job_id=job_id)
                config["protocol"]["version"] = "h-l1248-grouped-source/v1"
                config["protocol"]["expected_split_hashes"].update(grouped_hashes)
                config["h_grouped"] = {
                    "reference_config": str(reference_copy), "reference_config_sha256": sha(reference_copy),
                    "source_manifest": str(source_copy), "source_manifest_sha256": sha(source_copy),
                    "initial_state_sha256": init_records[str(seed)],
                    "expected_sampler_sha256": sampler_hash,
                    "sampler_global_ids_file": str(schedule_path), "sampler_global_ids_file_sha256": sha(schedule_path),
                    "only_scientific_change": "source training/validation membership; sample sizes unchanged",
                    "fixed_partition_across_seeds": True, "preprocessing_and_model_unchanged": True,
                    "target_used_previously_for_development": True, "target_access_rule_unchanged": True,
                    "new_hyperparameter_search_trials": 0}
                verify_config(config)
                config_path = BUNDLE / "configs" / f"{job_id}.json"
                write_new(config_path, config)
                registry.append({"job_id": job_id, "target": target, "seed": seed, "arm": arm,
                                 "config": f"configs/{job_id}.json"})
                references.append({"job_id": job_id, "original_config": str(original_path),
                                   "original_config_sha256": sha(original_path),
                                   "historical_attempt": row["attempt_dir"],
                                   "historical_completion_sha256": row["completion_manifest_sha256"],
                                   "historical_metrics_sha256": row["metrics_file_sha256"],
                                   "historical_predictions_sha256": row["target_predictions_file_sha256"]})
                clean_root = Path(reference["provenance"]["clean_worktree"])
                for path in clean_root.glob("*.py"):
                    external_code[str(path)] = sha(path)
                tpp = Path(reference["provenance"]["tpp_snapshot_root"]) / "experiments/tpp_isolated/tpp_models.py"
                external_code[str(tpp)] = sha(tpp)
    code = BUNDLE / "code"
    code.mkdir()
    for path in HERE.glob("*.py"):
        shutil.copy2(path, code / path.name)
    for kind, filename in (("trainer", "train_factorial.py"), ("runner", "run_factorial.py")):
        shutil.copy2(ROOT / "experiments/supcon_tpp_legacy_reproduction" / filename, code / f"base_{kind}.py")
    shutil.copy2(HERE / "EXPERIMENT_PLAN.md", BUNDLE / "EXPERIMENT_PLAN.md")
    write_new(BUNDLE / "prerequisite.json", prerequisite)
    write_new(BUNDLE / "registry.json", {"schema": "h-l1248-grouped-registry/v1", "jobs": registry})
    write_new(BUNDLE / "alignment_audit.json", {"status": "passed", "jobs": 18, "parameters_each_arm": PARAMETERS,
              "initial_state_sha256_by_seed": init_records, "source_partitions": audits,
              "legacy_loader_and_training_reused": True, "paired_full_sampler_schedules_saved": True,
              "checkpoint_rule": "source_val_accuracy_last_tied_best", "source_manifest_used_for_identity_only": True})
    write_new(BUNDLE / "historical_references.json", {"references": references})
    files = {p.relative_to(BUNDLE).as_posix(): sha(p) for p in BUNDLE.rglob("*") if p.is_file()}
    write_new(BUNDLE / "bundle_manifest.json", {"schema": "h-l1248-grouped-bundle/v1", "created_at": base.utc_now(),
              "job_count": 18, "files_sha256": files, "external_code_sha256": external_code,
              "scientific_change": "verified grouped source train/validation partition only",
              "runtime": {"python": __import__('sys').version, "torch": torch.__version__, "numpy": np.__version__},
              "formal_training_performed": False, "target_evaluation_performed": False})
    print(json.dumps({"status": "frozen", "jobs": 18, "bundle": str(BUNDLE),
                      "bundle_manifest_sha256": sha(BUNDLE / "bundle_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
