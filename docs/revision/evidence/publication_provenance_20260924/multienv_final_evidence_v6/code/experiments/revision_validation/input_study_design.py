"""Fixed input/modality contrasts; each result belongs to its own data version."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from experiments.revision_validation.experiment_design import build_experiment_model
from experiments.revision_validation.freeze_study import job_config
from experiments.revision_validation.study_io import read, sha256

INPUT_SPECS = {
    "NATIVE_MERGE": {"dataset_family": "native", "variant": "native_time_legacy_merge", "modalities": ["amp"], "primary": "amp"},
    "NATIVE_LINEAR": {"dataset_family": "native", "variant": "native_time_linear", "modalities": ["amp"], "primary": "amp"},
    "HIST_PHASE": {"dataset_family": "modality", "variant": "historical_feature_axis_unwrap", "modalities": ["phase"], "primary": "phase"},
    "NATIVE_PHASE": {"dataset_family": "modality", "variant": "native_time_phase", "modalities": ["phase"], "primary": "phase"},
    "DIFFERENTIAL_PHASE": {"dataset_family": "modality", "variant": "native_time_differential_phase", "modalities": ["phase"], "primary": "phase"},
    "TWIN_AA": {"dataset_family": "native", "variant": "native_time_linear", "modalities": ["amp"], "primary": "amp", "secondary": "amp"},
    "TWIN_AP": {"dataset_family": "modality", "variant": "native_time_differential_phase", "modalities": ["amp", "phase"], "primary": "amp", "secondary": "phase"},
}

COMPARISONS = [
    {"treatment": "NATIVE_MERGE", "reference": "R10", "reference_study": "core", "interpretation": "Native relative timestamps in the historical merge; representation/origin differences are both included"},
    {"treatment": "NATIVE_LINEAR", "reference": "NATIVE_MERGE", "reference_study": "extension", "interpretation": "Same native clock, change the historical merge to linear interpolation"},
    {"treatment": "HIST_PHASE", "reference": "R10", "reference_study": "core", "interpretation": "Single-stream modality comparison on historical processing; not a calibration-only contrast"},
    {"treatment": "NATIVE_PHASE", "reference": "HIST_PHASE", "reference_study": "extension", "interpretation": "Whole phase-processing pipeline comparison; timestamp, averaging, unwrap axis and centering differ"},
    {"treatment": "DIFFERENTIAL_PHASE", "reference": "NATIVE_PHASE", "reference_study": "extension", "interpretation": "Same temporal processing, per-Rx phase becomes paired-Rx phase; common-error rejection and represented geometry both differ"},
    {"treatment": "TWIN_AA", "reference": "NATIVE_LINEAR", "reference_study": "extension", "interpretation": "Additional amplitude encoder/fusion capacity without a new modality"},
    {"treatment": "TWIN_AP", "reference": "TWIN_AA", "reference_study": "extension", "interpretation": "Identical two-stream model and initialization; second input changes from amplitude to differential phase"},
    {"treatment": "TWIN_AP", "reference": "NATIVE_LINEAR", "reference_study": "extension", "interpretation": "Combined modality and capacity change; interpret alongside the matched TWIN_AA contrast"},
]


def input_config(control, target, seed, source_path, target_path=None, *, epochs=200, updates=300, device="cuda:0"):
    spec = deepcopy(INPUT_SPECS[control])
    source_path = Path(source_path)
    source = read(source_path)
    if source["data_version"] != spec["variant"]:
        raise RuntimeError("Input-control manifest is not the declared preprocessing variant")
    if source["outer_target"] != target or target in source["source_environments"]:
        raise RuntimeError("Input-control source manifest is not the declared outer fold")
    config = job_config(target, "R10", seed, source_path, epochs=epochs, updates=updates)
    experiment = {"base_arm": "R10", "pool": "unchanged", "objective": "ce", "model_kind": "single"}
    if "secondary" in spec:
        experiment.update(model_kind="twin", secondary_input=spec["secondary"])
    config.update(job_id=f"{target}_I_{control}_{seed}", control_id=control, experiment=experiment,
        input_modalities=spec["modalities"], primary_input=spec["primary"], input_variant=spec["variant"],
        modality="twin_"+spec["secondary"] if "secondary" in spec else spec["primary"]+"_only",
        phase_loaded="phase" in spec["modalities"], input_processing="Frozen separately compiled "+spec["variant"],
        validation_sample_count=len(source["val"]), lambda_supcon=0.0, resource_profile=epochs == 200,
        device=device, complete_hardware_calibration=False)
    if target_path is not None:
        target_metadata = read(target_path)
        if target_metadata["data_version"] != spec["variant"] or target_metadata["outer_target"] != target:
            raise RuntimeError("Input-control target manifest is not the declared preprocessing variant/fold")
        if set(target_metadata["files"]) != {target}:
            raise RuntimeError("Deferred target manifest must contain exactly the declared target environment")
        config.update(target_manifest_path=str(target_path), target_manifest_sha256=sha256(target_path))
    config["total_parameters"] = sum(p.numel() for p in build_experiment_model(config).parameters())
    return config
