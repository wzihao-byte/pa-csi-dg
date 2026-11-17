"""Generate candidate WiMANS jobs and freeze only after real source QA.

The four-arm 36 formal candidates and nine WC source-search candidates are
generated first. WC formal candidates require a source-only gamma selection
receipt. None of these candidates can be launched by train-source until frozen.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .adapter import Manifest, ROOMS, file_sha256, read_json, require, require_formal_input, sampling_schedule, schedule_hash
from .models import ARMS, EXPECTED_PARAMETERS, MODEL_CONFIG
from .training import SCHEMA, atomic_json, canonical_hash, code_receipt, inputs_for, validate_config


def base_config(manifest, processed_dir, arm, target, seed, gamma=0.0):
    ids = manifest.folds[target]["sample_ids"]["train"]
    proxy = SimpleNamespace(labels=np.asarray([int(manifest.by_id[x]["class_id"]) for x in ids]),
                            domains=np.asarray([ROOMS.index(manifest.by_id[x]["environment"]) for x in ids]),
                            numeric_ids=np.asarray([manifest.ids.index(x) for x in ids]))
    # SimpleNamespace has no __len__; use a minimal metadata-only sampler proxy.
    class SamplerMetadata:
        def __init__(self, values):
            self.__dict__.update(vars(values))
        def __len__(self):
            return len(self.labels)
    proxy = SamplerMetadata(proxy)
    processed_dir = Path(processed_dir).resolve()
    config = {"schema": SCHEMA, "arm": arm, "target_environment": target, "seed": seed,
              "gamma": gamma, "model": copy.deepcopy(MODEL_CONFIG), "expected_parameters": EXPECTED_PARAMETERS,
              "training": {"epochs": 200, "updates_per_epoch": 150, "batch_size": 32,
                           "optimizer": "adam", "lr": 0.001, "weight_decay": 0.0,
                           "sampler": "domain_class_balanced", "lr_decay_rate": 0.9,
                           "lr_decay_steps": 10000, "grad_clip_norm": 0.0,
                           "checkpoint_rule": "source_val_accuracy_last_tied_best"},
              "losses": {**ARMS[arm], "temperature": 0.2, "positive_mode": "global_class_instance_views",
                         "include_same_domain_same_class": True, "warmup_updates": 0},
              "masked_forward_exposure": False, "subset_manifest": str(manifest.subset_path.resolve()),
              "fold_manifest": str(manifest.folds_path.resolve()), "manifest_receipt": manifest.receipt(),
              "processed_dir": str(processed_dir),
              "input_audit_sha256": file_sha256(processed_dir / "input_audit.json"),
              "sampling_schedule_sha256": schedule_hash(proxy, sampling_schedule(proxy, seed)),
              "execution_lock": {"status": "candidate_pre_freeze"}}
    validate_config(config)
    return config


def generate(subset_manifest, fold_manifest, processed_dir, output_dir):
    manifest, output_dir = Manifest(subset_manifest, fold_manifest), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    require(not any(output_dir.iterdir()), "Candidate output directory must be empty")
    jobs = []
    for target in ROOMS:
        for seed in (42, 52, 62):
            for arm in ("W00", "W01", "W10", "W11"):
                name = f"{arm}_{target}_{seed}.json"
                atomic_json(output_dir / name, base_config(manifest, processed_dir, arm, target, seed))
                jobs.append({"config": name, "arm": arm, "target_environment": target, "seed": seed,
                             "kind": "formal_candidate", "executable": False})
        for gamma in (0.1, 1.0, 10.0):
            name = f"WC_search_{target}_gamma{gamma}_20261014.json"
            atomic_json(output_dir / name, base_config(manifest, processed_dir, "WC", target, 20261014, gamma))
            jobs.append({"config": name, "arm": "WC", "target_environment": target, "seed": 20261014,
                         "gamma": gamma, "kind": "source_search_candidate", "executable": False})
    atomic_json(output_dir / "registry.json", {"schema": "wimans-candidate-jobs/v1", "status": "not_frozen",
                "code_files_sha256": code_receipt(), "jobs": jobs, "counts": {"formal_four_arm": 36, "source_coral_search": 9},
                "pending": ["full_input_qa", "real_source_qa", "source_coral_selection_before_WC_formal", "execution_freeze"]})


def freeze_config(candidate_path, source_qa_path, output_path, coral_selection_path=None):
    config, qa = read_json(candidate_path), read_json(source_qa_path)
    validate_config(config)
    inputs = inputs_for(config)
    require_formal_input(inputs.audit)
    require(qa.get("status") == "passed" and qa.get("synthetic") is False and qa.get("target_signal_rows_read") == 0,
            "Freeze requires passed real source QA without target signal reads")
    require(qa.get("code_files_sha256") == code_receipt() and qa.get("input_audit_sha256") == config["input_audit_sha256"] and
            qa.get("manifest_receipt") == config["manifest_receipt"] and qa.get("target_environment") == config["target_environment"],
            "Source QA does not bind this data/fold/code")
    require(qa.get("source_short_train_updates", 0) >= 2 and qa.get("validation_rows") == 396 and
            qa.get("source_train_batch_size") == 32 and qa.get("input_shape") == [850, 90],
            "Source QA did not verify the registered full input/batch protocol")
    require(qa.get("checked_model_arm") == "W11" and qa.get("expected_parameters") == EXPECTED_PARAMETERS,
            "Source QA must exercise the hybrid model and auxiliary forwards")
    require(qa.get("checkpoint_strict_reload_logits_exact") is True, "Source QA checkpoint reload was not exact")
    lock = {"status": "frozen", "code_files_sha256": code_receipt(), "source_qa_receipt": str(Path(source_qa_path).resolve()),
            "source_qa_sha256": file_sha256(source_qa_path)}
    if config["arm"] == "WC" and config["seed"] != 20261014:
        require(coral_selection_path is not None, "WC formal freeze requires source-only gamma selection")
        lock.update({"coral_selection_receipt": str(Path(coral_selection_path).resolve()),
                     "coral_selection_sha256": file_sha256(coral_selection_path)})
    config.pop("execution_lock", None)
    lock["config_sha256"] = canonical_hash(config)
    config["execution_lock"] = lock
    validate_config(config, formal=True)
    require(not Path(output_path).exists(), "Frozen config output already exists")
    atomic_json(output_path, config)


def select_coral(target, run_dirs, output_path):
    require(len(run_dirs) == 3 and not Path(output_path).exists(), "Provide three source-search runs and a new output path")
    candidates = []
    for directory in map(Path, run_dirs):
        config, frozen = read_json(directory / "resolved_config.json"), read_json(directory / "checkpoint_frozen.json")
        validate_config(config, formal=True)
        require(config["arm"] == "WC" and config["seed"] == 20261014 and config["target_environment"] == target,
                "Wrong CORAL source search identity")
        require(frozen["config_sha256"] == canonical_hash(config) and frozen["code_files_sha256"] == code_receipt() and
                frozen["input_audit_sha256"] == config["input_audit_sha256"] and
                frozen["manifest_receipt"] == config["manifest_receipt"], "CORAL search receipt/config binding drift")
        require(not (directory / "target_access.json").exists() and frozen["target_data_opened"] is False,
                "CORAL search must not access target model scores")
        require(frozen["optimizer_updates"] == 30000 and frozen.get("strict_reloaded") is True and
                frozen["target_environment"] == target and frozen["checkpoint_sha256"] == file_sha256(directory / "best_model.pt"),
                "CORAL search is incomplete or checkpoint changed")
        candidates.append({"gamma": config["gamma"], "source_validation_accuracy": frozen["source_validation_score"],
                           "run_dir": str(directory.resolve()), "checkpoint_sha256": frozen["checkpoint_sha256"],
                           "frozen_receipt_sha256": file_sha256(directory / "checkpoint_frozen.json"),
                           "config_sha256": canonical_hash(config), "input_audit_sha256": config["input_audit_sha256"],
                           "manifest_receipt": config["manifest_receipt"], "code_files_sha256": code_receipt()})
    require({x["gamma"] for x in candidates} == {0.1, 1.0, 10.0}, "Gamma grid incomplete")
    require(all(row["input_audit_sha256"] == candidates[0]["input_audit_sha256"] and
                row["manifest_receipt"] == candidates[0]["manifest_receipt"] for row in candidates),
            "CORAL gamma searches use different input/fold bindings")
    selected = sorted(candidates, key=lambda row: (-row["source_validation_accuracy"], row["gamma"]))[0]
    atomic_json(output_path, {"schema": "wimans-coral-source-selection/v1", "target_environment": target,
                "selected_gamma": selected["gamma"], "selection_metric": "source_validation_accuracy",
                "tie_rule": "smaller_gamma", "target_evaluated": False, "candidates": candidates,
                "input_audit_sha256": selected["input_audit_sha256"], "manifest_receipt": selected["manifest_receipt"],
                "code_files_sha256": code_receipt()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--subset-manifest", type=Path, required=True)
    gen.add_argument("--fold-manifest", type=Path, required=True)
    gen.add_argument("--processed-dir", type=Path, required=True)
    gen.add_argument("--output-dir", type=Path, required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--candidate", type=Path, required=True)
    freeze.add_argument("--source-qa", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--coral-selection", type=Path)
    selection = commands.add_parser("select-coral")
    selection.add_argument("--target", choices=ROOMS, required=True)
    selection.add_argument("--run-dirs", type=Path, nargs=3, required=True)
    selection.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.subset_manifest, args.fold_manifest, args.processed_dir, args.output_dir)
    elif args.command == "freeze":
        freeze_config(args.candidate, args.source_qa, args.output, args.coral_selection)
    else:
        select_coral(args.target, args.run_dirs, args.output)


if __name__ == "__main__":
    main()
