"""Actual source-data objective and two-stream interruption/resume checks."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.revision_validation.experiment_data import VerifiedInputs
from experiments.revision_validation.experiment_design import build_experiment_model, objective, predict
from experiments.revision_validation.extended_engine import train_experiment
from experiments.revision_validation.freeze_inputs import INPUT_CODE, ROOT, check_dataset
from experiments.revision_validation.input_study_design import INPUT_SPECS, input_config
from experiments.revision_validation.models import build_model
from experiments.revision_validation.source_data import epoch_schedule
from experiments.revision_validation.study_engine import equal_state, runtime_settings
from experiments.revision_validation.study_io import now, read, sha256, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-data", type=Path, required=True)
    parser.add_argument("--modality-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = {"native": args.native_data.resolve(), "modality": args.modality_data.resolve()}
    # Source loaders verify the actual source signal files. No target signal is
    # opened by this QA; the full compiler's immutable manifest identifies them.
    manifests = {name: check_dataset(directory, verify_files=False) for name,directory in data.items()}
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve previous input QA attempts")
    output.mkdir(parents=True)
    runtime_settings()
    results, opened, identities = {}, set(), None
    amplitude_reference, twin_state = None, None
    resume_config = None
    for name, spec in INPUT_SPECS.items():
        path = data[spec["dataset_family"]] / spec["variant"] / "manifests/source_E1.json"
        config = input_config(name, "E1", 9511, path, epochs=2, updates=2, device="cpu")
        dataset = VerifiedInputs(path, config["input_modalities"])
        opened.update(dataset.opened_paths)
        batch = epoch_schedule(dataset.manifest["train"], 401, 0, steps=1)[0]
        inputs, labels, rows = dataset.batch("train", batch, "cpu")
        row_identity = [{k:r[k] for k in ("sample_id", "label", "subject_id", "acquisition_id")} for r in rows]
        if identities is None:
            identities = row_identity
        elif row_identity != identities:
            raise RuntimeError("Input controls changed source sample/label/group identities")
        if name == "NATIVE_LINEAR":
            amplitude_reference = inputs["amp"].clone()
        if name in ("TWIN_AA", "TWIN_AP") and not torch.equal(inputs["amp"], amplitude_reference):
            raise RuntimeError("Twin amplitude input differs from the native single-stream reference")
        model = build_experiment_model(config)
        if spec.get("secondary"):
            if twin_state is None:
                twin_state = {k:v.clone() for k,v in model.state_dict().items()}
            elif not equal_state(twin_state, model.state_dict()):
                raise RuntimeError("Twin capacity/modality initializations differ")
        elif not equal_state(model.state_dict(), build_model("R10", 9511).state_dict()):
            raise RuntimeError("Single-stream modality controls changed model initialization")
        torch.manual_seed(9500)
        before = torch.get_rng_state()
        loss, primary, regularizer, weight, _ = objective(model, inputs, labels, rows, 1, config)
        torch.set_rng_state(before)
        # Same train-mode call sequence updates BN identically in both paths;
        # compare an independent fresh model rather than reusing mutable buffers.
        reference = build_experiment_model(config)
        expected = F.cross_entropy(predict(reference, inputs, config, project=False)["logits"], labels)
        if not torch.equal(loss, expected) or weight != 0 or float(regularizer) != 0:
            raise RuntimeError("Input objective differs from the direct CE computation")
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError("Input-control loss/gradient is not finite")
        results[name] = {"source_manifest_sha256": sha256(path), "loss": float(loss.detach()), "parameters": config["total_parameters"],
                         "active_CE_parameters": sum(p.numel() for p in model.parameters() if p.grad is not None),
                         "direct_CE_exact": True, "optimizer_updates": 0, "input_modalities": config["input_modalities"]}
        if name == "TWIN_AP":
            resume_config = config
        print(f"Source input/model/CE checked: {name}", flush=True)
    continuous, resumed = output / "continuous", output / "resumed"
    train_experiment(resume_config, continuous)
    train_experiment(resume_config, resumed, pause_after_epoch=1)
    train_experiment(resume_config, resumed)
    states = [torch.load(path / read(path / "progress.json")["checkpoint"], map_location="cpu", weights_only=True) for path in (continuous,resumed)]
    checks = {"model_exact": equal_state(states[0]["model"],states[1]["model"]),
              "optimizer_exact": equal_state(states[0]["optimizer"],states[1]["optimizer"]),
              "rng_exact": equal_state(states[0]["rng"],states[1]["rng"]), "history_exact": states[0]["history"]==states[1]["history"],
              "source_predictions_exact": read(continuous/"selected_source_predictions.json")==read(resumed/"selected_source_predictions.json")}
    report = {"passed": all(checks.values()), "checked_at": now(), "control_checks": results, "CPU_twin_resume": checks,
              "resume_trajectories": 2, "epochs_per_trajectory": 2, "updates_per_epoch": 2, "target_data_loaded": False,
              "source_identities": identities, "opened_source_paths": sorted(opened), "full_training_completed": False,
              "twin_same_amplitude_and_model_initialization": True, "single_same_R10_initialization": True,
              "compiled_dataset_manifest_sha256": manifests, "code_sha256": {name: sha256(ROOT/name) for name in INPUT_CODE}}
    write(output / "input_study_qa.json", report)
    if not report["passed"]:
        raise RuntimeError("Input-study source resume QA failed")
    print({"passed": True, "CPU_twin_resume": checks, "full_training_completed": False})


if __name__ == "__main__":
    main()
