"""Check core loss equivalence, every control objective and actual CPU resume."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from dg_losses import DomainAwareSupConLoss
from experiments.revision_validation.freeze_controls import EXTENSION_CODE, ROOT, SOURCE_PROTOCOL, experiment_config
from experiments.revision_validation.experiment_data import VerifiedInputs
from experiments.revision_validation.experiment_design import CONTROL_SPECS, build_experiment_model, objective
from experiments.revision_validation.extended_engine import train_experiment
from experiments.revision_validation.models import build_model
from experiments.revision_validation.source_data import epoch_schedule
from experiments.revision_validation.study_engine import equal_state, runtime_settings
from experiments.revision_validation.study_io import now, read, sha256, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Keep previous QA evidence; use a new directory")
    output.mkdir(parents=True)
    runtime_settings()
    source_path = SOURCE_PROTOCOL / "source_E1.json"
    dataset = VerifiedInputs(source_path)
    batch = epoch_schedule(dataset.manifest["train"], 401, 0, steps=1)[0]
    inputs, labels, rows = dataset.batch("train", batch, "cpu")
    identity_config = experiment_config("CORAL", "E1", 8102, source_path, epochs=2, updates=2, device="cpu")
    identity_config.update(experiment={"base_arm": "R11", "pool": "unchanged", "objective": "supcon"}, lambda_supcon=0.025)
    reference, candidate = build_model("R11", 8102), build_experiment_model(identity_config)
    torch.manual_seed(991)
    before = torch.get_rng_state()
    values = reference(inputs["amp"])
    expected = F.cross_entropy(values["logits"], labels) + 0.025 * DomainAwareSupConLoss(temperature=0.2, positive_mode="all_views")(values["projection"], labels)
    expected.backward()
    torch.optim.Adam(reference.parameters(), lr=0.001).step()
    torch.set_rng_state(before)
    actual, _, _, _, _ = objective(candidate, inputs, labels, rows, 6000, identity_config)
    actual.backward()
    torch.optim.Adam(candidate.parameters(), lr=0.001).step()
    identity_passed = bool(torch.equal(actual, expected) and equal_state(reference.state_dict(), candidate.state_dict()))
    if not identity_passed:
        raise RuntimeError("Extension adapter changes the core objective/update")
    controls = {}
    for name in CONTROL_SPECS:
        config = experiment_config(name, "E1", 8101, source_path, epochs=2, updates=2, device="cpu")
        model = build_experiment_model(config)
        loss, primary, regularizer, weight, diagnostics = objective(model, inputs, labels, rows, 6000, config)
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError(f"Nonfinite source objective/gradient: {name}")
        controls[name] = {"objective": float(loss.detach()), "primary": float(primary.detach()),
                          "regularizer": float(regularizer.detach()), "regularizer_weight": weight,
                          "diagnostics": diagnostics, "parameters": sum(p.numel() for p in model.parameters()),
                          "parameters_with_gradient": sum(p.numel() for p in model.parameters() if p.grad is not None),
                          "optimizer_updates": 0, "source_only": True}
        print(f"Source objective/backward checked: {name}", flush=True)
    config = experiment_config("CORAL", "E1", 8101, source_path, epochs=2, updates=2, device="cpu")
    continuous, resumed = output / "continuous", output / "resumed"
    train_experiment(config, continuous)
    train_experiment(config, resumed, pause_after_epoch=1)
    train_experiment(config, resumed)
    states = [torch.load(path / read(path / "progress.json")["checkpoint"], map_location="cpu", weights_only=True) for path in (continuous, resumed)]
    resume_checks = {"model_exact": equal_state(states[0]["model"], states[1]["model"]),
                     "optimizer_exact": equal_state(states[0]["optimizer"], states[1]["optimizer"]),
                     "rng_exact": equal_state(states[0]["rng"], states[1]["rng"]),
                     "history_exact": states[0]["history"] == states[1]["history"],
                     "source_predictions_exact": read(continuous / "selected_source_predictions.json") == read(resumed / "selected_source_predictions.json")}
    report = {"passed": identity_passed and all(resume_checks.values()), "checked_at": now(),
              "core_supcon_objective_and_Adam_update_exact": identity_passed, "control_objective_checks": controls,
              "CPU_resume": resume_checks, "resume_trajectories": 2, "epochs_per_trajectory": 2,
              "updates_per_epoch": 2, "source_sample_ids": [r["sample_id"] for r in rows],
              "target_data_loaded": False, "full_training_completed": False, "opened_source_paths": dataset.opened_paths,
              "code_sha256": {name: sha256(ROOT / name) for name in EXTENSION_CODE}}
    write(output / "extension_qa.json", report)
    print({k: v for k, v in report.items() if k not in ("code_sha256", "source_sample_ids", "control_objective_checks")})
    if not report["passed"]:
        raise RuntimeError("Extension QA failed")


if __name__ == "__main__":
    main()
