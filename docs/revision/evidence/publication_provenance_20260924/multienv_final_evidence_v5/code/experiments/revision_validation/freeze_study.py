"""Freeze a zero-search R factorial before full training or target evaluation."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from experiments.revision_validation.models import MODEL_CONFIG, shared_initialization_audit
from experiments.revision_validation.study_engine import runtime_settings
from experiments.revision_validation.study_io import now, read, sha256, verify_bundle, write

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PROTOCOL = ROOT / "experiments/revision_validation/verified_protocol"
CODE_FILES = [
    "dg_dataset.py", "dg_models.py", "dg_losses.py",
    "experiments/tpp_isolated/tpp_models.py",
    "experiments/supcon_tpp_joint/protocol.py", "experiments/supcon_tpp_joint/runner_utils.py",
    *["experiments/revision_validation/" + name for name in (
        "models.py", "source_data.py", "prepare_verified_protocol.py", "train_source.py",
        "study_io.py", "study_engine.py", "freeze_study.py", "evaluate_study.py", "resume_qa.py")],
]


def job_config(target, arm, seed, manifest_path, *, epochs=200, updates=300):
    return {"job_id": f"{target}_{arm}_{seed}", "stage": "full_source_development" if epochs == 200 else "resume_QA",
            "target_name_only": target, "arm": arm, "seed": seed,
            "source_manifest_path": str(manifest_path), "source_manifest_sha256": sha256(manifest_path),
            "model": MODEL_CONFIG, "modality": "amplitude_only", "phase_loaded": False,
            "input_processing": "Verified historical 850x90 amplitude; float64 to float32, no fitted normalization",
            "epochs": epochs, "updates_per_epoch": updates, "batch_size": 16, "device": "cuda:0",
            "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0,
            "scheduler": "initial_lr * 0.9 ** (completed_updates / 10000)",
            "lambda_supcon": 0.025 if arm.endswith("1") else 0.0, "temperature": 0.2, "warmup_steps": 6000,
            "checkpoint_metric": "source_validation_macro_f1", "tie_rule": "earliest",
            "expected_runtime": runtime_settings(), "target_data_loaded": False,
            "hyperparameter_search_trials": 0, "development_environments": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--qa-report", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if bundle.exists():
        raise FileExistsError("Use a new study directory; never rewrite a frozen bundle")
    qa = read(args.qa_report)
    if not qa["passed"]:
        raise RuntimeError("Resume QA must pass before freezing")
    for name, expected in qa["code_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"Code changed after QA: {name}")
    for name in CODE_FILES:
        path = bundle / "code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    for path in SOURCE_PROTOCOL.glob("*.json"):
        destination = bundle / "manifests" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    shutil.copy2(args.qa_report, bundle / "resume_qa.json")
    write(bundle / "shared_initialization_audit.json", shared_initialization_audit())
    jobs = [job_config(target, arm, seed, bundle / "manifests" / f"source_{target}.json")
            for target, base in (("E1", 400), ("E2", 500), ("E3", 600))
            for seed in range(base + 1, base + 4) for arm in ("R00", "R01", "R10", "R11")]
    protocol = {"schema": "revision-core-study/v1", "frozen_at": now(), "study_id": bundle.name,
                "jobs": jobs, "job_count": 36, "model": MODEL_CONFIG,
                "research_status": "Development comparison on previously used E1-E3; not independent confirmation",
                "hyperparameters": {"selection": "No new search for any arm", "trials_per_arm": 0,
                    "lambda": 0.025, "rationale": "Reuse the previously source-selected Y value as a fixed strength; no R target tuning"},
                "stopping": {"performance_based_early_stopping": False, "epochs_per_job": 200,
                    "numerical_or_hash_failure": "Stop and diagnose; retain failure and any deviation, no silent exclusion",
                    "external_interruption": "Restore last committed epoch, optimizer and RNG; retain event trail",
                    "historical_Y_stop": "Remains in effect; these are new R runs, not added Y seeds"},
                "target_gate": "No new R target evaluation until every frozen source run is complete and verified",
                "analysis": {"primary": "Macro-F1 on six classes", "secondary": ["accuracy", "class F1", "confusion matrix"],
                    "summary": "Seed mean and sample SD within each environment; equal-weight mean across three environments",
                    "paired_effects": ["R10-R00", "R11-R01", "R01-R00", "R11-R10", "R11-R10-R01+R00"],
                    "no_population_inference": "No seed-as-environment p values or population confidence intervals",
                    "negative_results": "Report every completed run and every environment, including harm"},
                "checkpoint_retention": "epoch 1, each tenth epoch, two most recent epochs and selected source best",
                "known_limitations": ["CUDA adaptive-max-pool backward is not strictly deterministic",
                    "Historical timestamp and phase provenance limitations remain; R uses amplitude only",
                    "This fixed-strength study does not establish the optimum of any method"]}
    write(bundle / "protocol.json", protocol)
    paths = [p for p in bundle.rglob("*") if p.is_file()]
    write(bundle / "bundle_manifest.json", {"schema": "revision-frozen-bundle/v1", "frozen_at": now(),
          "files_sha256": {p.relative_to(bundle).as_posix(): sha256(p) for p in paths}})
    verify_bundle(bundle)
    print(f"Frozen {len(jobs)} jobs at {bundle}; bundle SHA256 {sha256(bundle / 'bundle_manifest.json')}")


if __name__ == "__main__":
    main()
