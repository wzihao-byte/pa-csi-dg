"""Use the original H trainer with verified grouped source membership only."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from common import PARAMETERS, grouped_split, legacy, read, require, sha, state_hash, verify_bundle, verify_config


def install(base, config):
    verify_config(config)
    source = read(config["h_grouped"]["source_manifest"])
    base.EXPECTED_PARAMETERS["isolated_pa_csi_dg_lite_tpp"] = PARAMETERS
    base.SCHEMA_VERSION = "h-l1248-grouped-train/v1"
    original_manifest = base.source_manifest
    original_imports = base.configure_imports
    identity_holder = {}

    def source_split(dataset, seed, target, _dg_dataset):
        require(seed == config["seed_list"][0], "Seed drift")
        split, identity = grouped_split(dataset, source, target)
        identity_holder.update(identity)
        return split

    def source_manifest(dataset, split, target):
        value = original_manifest(dataset, split, target)
        value["source_partition_rule"] = "verified_subject_acquisition_grouped_holdout"
        value["group_identity"] = copy.deepcopy(identity_holder)
        value["grouped_source_manifest_sha256"] = config["h_grouped"]["source_manifest_sha256"]
        return value

    def configure_imports(cfg):
        modules = original_imports(cfg)
        trainer = modules["train_dg"]
        original_build = trainer.build_model_and_optimizer
        original_epoch = trainer.run_epoch
        original_load = modules["dg_dataset"].load_dataset
        completed_epochs = 0
        run_dir = Path(cfg["output_dir"])

        def build(*args, **kwargs):
            result = original_build(*args, **kwargs)
            model = result[0]
            count = sum(p.numel() for p in model.parameters())
            actual = state_hash(model)
            require(count == PARAMETERS, "Actual model parameter count changed")
            require(actual == cfg["h_grouped"]["initial_state_sha256"], "Initial model differs from historical H configuration")
            base.atomic_write_json(run_dir / "initialization_audit.json", {
                "parameter_count": count, "initial_state_sha256": actual,
                "historical_configuration_matched": True, "seed": cfg["seed_list"][0]})
            return result

        def epoch(*args, **kwargs):
            nonlocal completed_epochs
            result = original_epoch(*args, **kwargs)
            if kwargs.get("train") is False and completed_epochs < 200:
                completed_epochs += 1
                with (run_dir / "progress.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"event": "source_validation_completed", "at": base.utc_now(),
                                        "epoch": completed_epochs, "accuracy": float(result["accuracy"])}) + "\n")
            return result

        def load(data_config, base_dir):
            environments = list(data_config["env_files"])
            if cfg["target_env"] in environments:
                require(environments == [cfg["target_env"]], "Mixed source/target loader")
                freeze = read(run_dir / "checkpoint_frozen.json")
                require(freeze["strict_load"] and freeze["target_data_opened"] is False, "Premature target access")
                require(sha(run_dir / "best_model.pt") == freeze["checkpoint_sha256"], "Unfrozen target model")
            else:
                require(environments == source["source_environments"], "Unexpected source files")
            return original_load(data_config, base_dir)

        trainer.build_model_and_optimizer = build
        trainer.run_epoch = epoch
        modules["dg_dataset"].load_dataset = load
        return modules

    base.source_split = source_split
    base.source_manifest = source_manifest
    base.configure_imports = configure_imports
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-checkpoint", type=Path)
    args = parser.parse_args()
    config = read(args.config)
    verify_config(config)
    base = legacy()
    base.EXPECTED_PARAMETERS["isolated_pa_csi_dg_lite_tpp"] = PARAMETERS
    if args.validate_checkpoint:
        base.validate_checkpoint(args.config.resolve(), args.validate_checkpoint.resolve())
    else:
        verify_bundle()
        install(base, config).train(args.config.resolve())


if __name__ == "__main__":
    main()
