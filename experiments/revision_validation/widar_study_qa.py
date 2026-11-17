"""Source-only Widar data, objective, initialization, and resume QA."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from dg_losses import DomainAwareSupConLoss
from experiments.revision_validation.experiment_data import VerifiedInputs
from experiments.revision_validation.experiment_design import build_experiment_model, objective, predict
from experiments.revision_validation.extended_engine import train_experiment
from experiments.revision_validation.freeze_widar import ROOT, WIDAR_CODE, check_widar_dataset
from experiments.revision_validation.source_data import epoch_schedule
from experiments.revision_validation.study_engine import equal_state, runtime_settings
from experiments.revision_validation.study_io import now, read, sha256, write
from experiments.revision_validation.widar_study_design import WIDAR_ARMS, widar_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    manifest_hash = check_widar_dataset(data, verify_files=False)
    if output.exists():
        raise FileExistsError("Preserve previous Widar QA attempts")
    output.mkdir(parents=True)
    runtime_settings()
    source_path = data / "native_amplitude/manifests/source_W1.json"
    results, opened, identities, states = {}, set(), None, {}
    resume_config = None
    for arm in WIDAR_ARMS:
        config = widar_config("W1", arm, 701, source_path, epochs=2, updates=2, device="cpu")
        dataset = VerifiedInputs(source_path, config["input_modalities"])
        opened.update(dataset.opened_paths)
        batch = epoch_schedule(dataset.manifest["train"], 701, 0, steps=1)[0]
        inputs, labels, rows = dataset.batch("train", batch, "cpu")
        row_identity = [{key: row[key] for key in ("sample_id", "label", "subject_id", "acquisition_id")} for row in rows]
        if identities is None:
            identities = row_identity
        elif identities != row_identity:
            raise RuntimeError("Widar arms changed the source batch identities")
        model = build_experiment_model(config)
        states[arm] = {key: value.clone() for key, value in model.state_dict().items()}
        torch.manual_seed(9711)
        before = torch.get_rng_state()
        loss, primary, regularizer, weight, _ = objective(model, inputs, labels, rows, 6000, config)
        torch.set_rng_state(before)
        reference = build_experiment_model(config)
        values = predict(reference, inputs, config, project=arm.endswith("1"))
        expected_primary = F.cross_entropy(values["logits"], labels)
        if arm.endswith("1"):
            expected_regularizer = DomainAwareSupConLoss(temperature=config["temperature"], positive_mode="all_views")(
                values["projection"], labels)
            expected_weight = config["lambda_supcon"]
        else:
            expected_regularizer = expected_primary.new_zeros(())
            expected_weight = 0.0
        expected = expected_primary + expected_weight * expected_regularizer
        if not torch.equal(loss, expected) or not torch.equal(primary, expected_primary) or not torch.equal(regularizer, expected_regularizer) or weight != expected_weight:
            raise RuntimeError("Widar objective differs from its direct CE/SupCon definition")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not torch.isfinite(loss) or not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("Widar source objective/gradient is not finite")
        results[arm] = {"loss": float(loss.detach()), "primary": float(primary.detach()),
                        "regularizer": float(regularizer.detach()), "weight": weight,
                        "parameters": config["total_parameters"],
                        "active_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.grad is not None),
                        "optimizer_updates": 0, "direct_objective_exact": True}
        if arm == "R11":
            resume_config = config
        print(f"Widar source input/model/objective checked: {arm}", flush=True)
    if not equal_state(states["R00"], states["R01"]) or not equal_state(states["R10"], states["R11"]):
        raise RuntimeError("Paired Widar CE/SupCon initializations differ")
    continuous, resumed = output / "continuous", output / "resumed"
    train_experiment(resume_config, continuous)
    train_experiment(resume_config, resumed, pause_after_epoch=1)
    train_experiment(resume_config, resumed)
    checkpoints = [torch.load(path / read(path / "progress.json")["checkpoint"], map_location="cpu", weights_only=True)
                   for path in (continuous, resumed)]
    checks = {"model_exact": equal_state(checkpoints[0]["model"], checkpoints[1]["model"]),
              "optimizer_exact": equal_state(checkpoints[0]["optimizer"], checkpoints[1]["optimizer"]),
              "rng_exact": equal_state(checkpoints[0]["rng"], checkpoints[1]["rng"]),
              "history_exact": checkpoints[0]["history"] == checkpoints[1]["history"],
              "source_predictions_exact": read(continuous / "selected_source_predictions.json") == read(resumed / "selected_source_predictions.json")}
    report = {"passed": all(checks.values()), "checked_at": now(), "control_checks": results,
              "CPU_resume": checks, "resume_trajectories": 2, "epochs_per_trajectory": 2, "updates_per_epoch": 2,
              "paired_objective_initializations_exact": True, "target_data_loaded": False,
              "target_performance_or_model_selection_used": False, "source_identities": identities,
              "opened_source_paths": sorted(opened), "compiled_dataset_manifest_sha256": manifest_hash,
              "full_training_completed": False, "code_sha256": {name: sha256(ROOT / name) for name in WIDAR_CODE}}
    write(output / "widar_study_qa.json", report)
    if not report["passed"]:
        raise RuntimeError("Widar source resume QA failed")
    print({"passed": True, "CPU_resume": checks, "target_data_loaded": False})


if __name__ == "__main__":
    main()
