"""Freeze the seven input/modality contrasts after complete data and source QA."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from torch import nn

from experiments.revision_validation.experiment_design import build_experiment_model
from experiments.revision_validation.extended_engine import verify_extension
from experiments.revision_validation.freeze_controls import EXTENSION_CODE, ROOT
from experiments.revision_validation.input_study_design import INPUT_SPECS, COMPARISONS, input_config
from experiments.revision_validation.models import count_matmul_conv_macs
from experiments.revision_validation.study_io import bound_path, now, read, sha256, verify_bundle, write

INPUT_CODE = EXTENSION_CODE + ["experiments/revision_validation/"+name for name in (
    "modality_models.py", "input_study_design.py", "input_study_engine.py", "freeze_inputs.py", "input_study_qa.py")]


class TwinCounter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, value, *, project=False):
        return self.model(value, secondary=value, project=project)


def check_dataset(directory, verify_files=True):
    manifest = read(directory / "dataset_manifest.json")
    if not manifest.get("complete") or manifest["rows"] != 9000:
        raise RuntimeError("Input studies require full, completed 9000-row compilation")
    if verify_files:
        for name, digest in manifest["artifacts_sha256"].items():
            if sha256(bound_path(directory, name)) != digest:
                raise RuntimeError("Compiled input artifact hash mismatch")
    return sha256(directory / "dataset_manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--core-bundle", type=Path, required=True)
    parser.add_argument("--controls-bundle", type=Path, required=True)
    parser.add_argument("--native-data", type=Path, required=True)
    parser.add_argument("--modality-data", type=Path, required=True)
    parser.add_argument("--qa-report", type=Path, required=True)
    args = parser.parse_args()
    bundle, core, controls = args.bundle.resolve(), args.core_bundle.resolve(), args.controls_bundle.resolve()
    if bundle.exists():
        raise FileExistsError("Use a new input-study version; preserve every existing freeze")
    _, core_hash = verify_bundle(core)
    _, controls_hash = verify_extension(controls)
    if (core / "target_gate.json").exists():
        raise RuntimeError("This planned matrix must freeze before new core target evaluation")
    qa = read(args.qa_report)
    if not qa.get("passed") or qa.get("target_data_loaded"):
        raise RuntimeError("Complete source-only input/training QA must pass")
    for name, digest in qa["code_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"Input-study code changed after QA: {name}")
    data = {"native": args.native_data.resolve(), "modality": args.modality_data.resolve()}
    manifests = {name: check_dataset(directory) for name, directory in data.items()}
    if manifests != qa["compiled_dataset_manifest_sha256"]:
        raise RuntimeError("Input data differ from the source training QA")
    for name in INPUT_CODE:
        destination = bundle / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    for family, variant in sorted({(s["dataset_family"], s["variant"]) for s in INPUT_SPECS.values()}):
        source_directory = data[family] / variant / "manifests"
        expected_names = {f"{prefix}_{target}.json" for prefix in ("source", "deferred_target")
                          for target in ("E1", "E2", "E3")}
        manifest_paths = list(source_directory.glob("*.json"))
        if {path.name for path in manifest_paths} != expected_names:
            raise RuntimeError(f"Unexpected manifest set for {variant}")
        for path in manifest_paths:
            destination = bundle / "manifests" / variant / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    shutil.copy2(args.qa_report, bundle / "input_study_qa.json")
    jobs = [input_config(control, target, seed,
              bundle / "manifests" / spec["variant"] / f"source_{target}.json",
              bundle / "manifests" / spec["variant"] / f"deferred_target_{target}.json")
            for control, spec in INPUT_SPECS.items() for target, base in (("E1",400), ("E2",500), ("E3",600))
            for seed in range(base+1,base+4)]
    costs = {}
    for control in INPUT_SPECS:
        config = next(c for c in jobs if c["control_id"] == control)
        model = build_experiment_model(config)
        wrapper = TwinCounter(model) if config["experiment"]["model_kind"] == "twin" else model
        costs[control] = {"parameters": config["total_parameters"], "counted_inference_macs": count_matmul_conv_macs(wrapper, torch.zeros(1,850,90))}
    write(bundle / "model_costs.json", costs)
    protocol = {"schema": "revision-extension-study/v1", "study_kind": "input_modality_controls", "study_id": bundle.name,
        "frozen_at": now(), "entry_module": "experiments.revision_validation.input_study_engine", "jobs": jobs, "job_count": len(jobs),
        "prerequisites": [{"bundle": str(core), "bundle_sha256": core_hash, "kind": "core"},
                          {"bundle": str(controls), "bundle_sha256": controls_hash, "kind": "extension"}],
        "core_reference_bundle": str(core), "core_reference_bundle_sha256": core_hash,
        "compiled_inputs": {name: {"directory": str(data[name]), "dataset_manifest_sha256": digest} for name,digest in manifests.items()},
        "controls": INPUT_SPECS, "comparisons": COMPARISONS,
        "training_budget": "200 epochs x 300 updates, batch 16, same Adam/LR schedule and source-balanced sampling as R; CE only",
        "selection": "Earliest maximum source validation macro-F1; original verified subject/acquisition splits and sample order",
        "hyperparameter_search_trials": 0, "complete_hardware_calibration": False,
        "inference_scope": "E1-E3 development evidence; per-environment seed means/SD and descriptive paired differences, no population p/CI",
        "target_gate": "All 63 source jobs complete and verified before any input-study target signal evaluation",
        "stopping": "No performance early stop/seed additions; retain and diagnose failures; resume only verified committed states"}
    write(bundle / "protocol.json", protocol)
    files = [p for p in bundle.rglob("*") if p.is_file()]
    write(bundle / "bundle_manifest.json", {"frozen_at": now(), "files_sha256": {p.relative_to(bundle).as_posix(): sha256(p) for p in files}})
    verified, digest = verify_extension(bundle)
    print(f"Frozen {len(verified['jobs'])} input/modality jobs; bundle SHA256 {digest}")


if __name__ == "__main__":
    main()
