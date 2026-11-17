"""Exercise actual source training uninterrupted and across an epoch boundary."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.revision_validation.freeze_study import CODE_FILES, ROOT, SOURCE_PROTOCOL, job_config
from experiments.revision_validation.study_engine import train_job
from experiments.revision_validation.study_io import read, sha256, verify_completed, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda:0", "cpu"], default="cpu")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Keep previous QA evidence; use a fresh directory")
    config = job_config("E1", "R11", 7401, SOURCE_PROTOCOL / "source_E1.json", epochs=2, updates=4)
    config["device"] = args.device
    uninterrupted, resumed = output / "uninterrupted", output / "resumed"
    train_job(config, uninterrupted)
    train_job(config, resumed, pause_after_epoch=1)
    train_job(config, resumed)
    statuses = [verify_completed(path, config) for path in (uninterrupted, resumed)]
    states = [torch.load(path / read(path / "progress.json")["checkpoint"], map_location="cpu", weights_only=True)
              for path in (uninterrupted, resumed)]
    differences = {name: float((states[0]["model"][name] - states[1]["model"][name]).abs().max())
                   for name in states[0]["model"]}
    same_schedule = all(a["schedule_sha256"] == b["schedule_sha256"]
                        for a, b in zip(states[0]["history"], states[1]["history"]))
    same_predictions = read(uninterrupted / "selected_source_predictions.json") == read(resumed / "selected_source_predictions.json")
    max_difference = max(differences.values())
    # Tolerance is fixed here before observing QA. The test verifies the state
    # plumbing; it is not a guarantee of identical long CUDA trajectories.
    passed = same_schedule and same_predictions and max_difference <= 1e-5
    report = {"passed": passed, "device": args.device, "epochs": 2, "updates_per_epoch": 4, "target_data_loaded": False,
              "same_batch_schedule": same_schedule, "same_selected_source_predictions": same_predictions,
              "max_absolute_final_parameter_difference": max_difference,
              "parameter_tolerance_predeclared": 1e-5,
              "selected_epoch": [s["best_epoch"] for s in statuses],
              "source_macro_f1": [s["best_source_macro_f1"] for s in statuses],
              "scope": "One actual E1-source/R11 short trajectory; not a full-training deterministic guarantee",
              "run_directories": [str(uninterrupted), str(resumed)],
              "status_sha256": {str(p): sha256(p / "status.json") for p in (uninterrupted, resumed)},
              "code_sha256": {name: sha256(ROOT / name) for name in CODE_FILES}}
    write(output / "resume_qa.json", report)
    print(report)
    if not passed:
        raise RuntimeError("Resume QA failed; inspect differences without relaxing the recorded tolerance")


if __name__ == "__main__":
    main()
