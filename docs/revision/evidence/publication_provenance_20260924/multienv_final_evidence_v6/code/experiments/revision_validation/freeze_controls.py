"""Freeze the full single-factor/DG extension before any new target results."""
from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

import torch

from experiments.revision_validation.freeze_study import CODE_FILES, ROOT, SOURCE_PROTOCOL, job_config
from experiments.revision_validation.experiment_design import CONTROL_SPECS, build_experiment_model
from experiments.revision_validation.models import count_matmul_conv_macs
from experiments.revision_validation.extended_engine import verify_extension
from experiments.revision_validation.study_io import now, read, sha256, verify_bundle, write

EXTENSION_CODE = CODE_FILES + [
    "experiments/revision_validation/" + name for name in (
        "control_components.py", "experiment_data.py", "experiment_design.py", "extended_engine.py",
        "extended_evaluate.py", "freeze_controls.py", "extension_qa.py")]


def experiment_config(control_id, target, seed, source_path, *, epochs=200, updates=300, device="cuda:0"):
    specification = copy.deepcopy(CONTROL_SPECS[control_id])
    config = job_config(target, specification["base_arm"], seed, source_path, epochs=epochs, updates=updates)
    config.update(job_id=f"{target}_C_{control_id}_{seed}", control_id=control_id, experiment=specification,
                  input_modalities=["amp"], primary_input="amp", validation_sample_count=len(read(source_path)["val"]),
                  lambda_supcon=0.025 if specification["objective"] in ("mask", "supcon") else 0.0,
                  resource_profile=epochs == 200, device=device)
    config["total_parameters"] = sum(p.numel() for p in build_experiment_model(config).parameters())
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--core-bundle", type=Path, required=True)
    parser.add_argument("--qa-report", type=Path, required=True)
    args = parser.parse_args()
    bundle, dependency = args.bundle.resolve(), args.core_bundle.resolve()
    if bundle.exists():
        raise FileExistsError("Use a new extension bundle; never rewrite an existing freeze")
    core_protocol, dependency_hash = verify_bundle(dependency)
    if (dependency / "target_gate.json").exists():
        raise RuntimeError("This planned extension must freeze before the first new R target evaluation")
    qa = read(args.qa_report)
    if not qa["passed"]:
        raise RuntimeError("Extension training/resume QA must pass")
    for relative, expected in qa["code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Code changed after extension QA: {relative}")
    for relative in EXTENSION_CODE:
        destination = bundle / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    for path in SOURCE_PROTOCOL.glob("*.json"):
        destination = bundle / "manifests" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    shutil.copy2(args.qa_report, bundle / "extension_qa.json")
    jobs = [experiment_config(control, target, seed, bundle / "manifests" / f"source_{target}.json")
            for control in CONTROL_SPECS for target, base in (("E1", 400), ("E2", 500), ("E3", 600))
            for seed in range(base + 1, base + 4)]
    references = {"P1": "R10", "NO_P1": "R10", "ATTENTION": "R10", "CORAL": "R00", "MIXUP": "R00", "TEMPORAL_AUG": "R10"}
    comparisons = [{"treatment": name, "reference": reference, "reference_study": "core",
                    "interpretation": "Pool-size contrasts also change projection capacity; report recorded parameters" if name in ("P1", "NO_P1") else "Fixed-protocol descriptive comparison"}
                   for name, reference in references.items()]
    comparisons += [{"treatment": name, "reference": "MASK_HYBRID", "reference_study": "extension",
                     "interpretation": "Same four views and forward calls; only the mask changes"}
                    for name in ("MASK_ALL", "MASK_REMOVE")]
    comparisons += [{"treatment": "MASK_HYBRID", "reference": "R11", "reference_study": "core",
                     "interpretation": "Four-view versus V1 also changes input exposure and BatchNorm calls; not a pure mask contrast"}]
    costs = {}
    for name in CONTROL_SPECS:
        config = next(c for c in jobs if c["control_id"] == name)
        model = build_experiment_model(config)
        costs[name] = {"total_parameters": sum(p.numel() for p in model.parameters()),
                       "inference_counted_macs": count_matmul_conv_macs(model, torch.zeros(1, 850, 90)),
                       "counting_limit": "Conv/linear/attention/position matrix MACs only; excludes pooling arithmetic, normalization and other elementwise operations"}
    write(bundle / "model_costs.json", costs)
    protocol = {"schema": "revision-extension-study/v1", "study_id": bundle.name, "frozen_at": now(),
                "source_training_dependency": str(dependency), "dependency_bundle_sha256": dependency_hash,
                "dependency_sequence": "Finish all core source training; run its frozen deferred evaluation; then begin these controls",
                "jobs": jobs, "job_count": len(jobs), "controls": CONTROL_SPECS, "comparisons": comparisons,
                "core_reference_jobs": [c["job_id"] for c in core_protocol["jobs"]],
                "inference_scope": "E1-E3 development environments; seed means/SD and descriptive paired differences only, no population p/CI",
                "hyperparameters": "Zero new search for all controls; fixed SC lambda .025, tau .2, warmup 6000; CORAL gamma 1, Mixup alpha .2",
                "training_budget": "200 epochs x 300 optimizer updates; source-balanced class schedule and source macro-F1 earliest-tie selection match core",
                "target_gate": "Evaluate extension targets only after every frozen extension source job is complete and verified",
                "stopping": "No performance early stop or seed addition; preserve/diagnose failures, replay only externally interrupted epochs",
                "method_boundary": "New amplitude-only controlled experiments; no historical H/Y checkpoint or pseudo-Tx slicing changes",
                "remaining_other_studies": "Native-time/phase and modality controls, and cross-dataset replication remain separately versioned work"}
    write(bundle / "protocol.json", protocol)
    files = [p for p in bundle.rglob("*") if p.is_file()]
    write(bundle / "bundle_manifest.json", {"schema": "revision-frozen-bundle/v1", "frozen_at": now(),
          "files_sha256": {p.relative_to(bundle).as_posix(): sha256(p) for p in files}})
    verify_extension(bundle)
    print(f"Frozen {len(jobs)} extension jobs at {bundle}; SHA256 {sha256(bundle / 'bundle_manifest.json')}")


if __name__ == "__main__":
    main()
