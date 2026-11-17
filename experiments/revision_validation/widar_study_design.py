"""Fixed R factorial on the separately predeclared Widar room cohort."""
from __future__ import annotations

from pathlib import Path

from experiments.revision_validation.experiment_design import build_experiment_model
from experiments.revision_validation.freeze_study import job_config
from experiments.revision_validation.study_io import read, sha256


WIDAR_ARMS = ("R00", "R01", "R10", "R11")
WIDAR_SEEDS = {"W1": (701, 702, 703), "W2": (801, 802, 803), "W3": (901, 902, 903)}


def widar_config(target, arm, seed, source_path, target_path=None, *, epochs=200, updates=300, device="cuda:0"):
    if arm not in WIDAR_ARMS or target not in WIDAR_SEEDS or seed not in WIDAR_SEEDS[target]:
        raise ValueError("Widar job is outside the preregistered room/arm/seed matrix")
    source_path = Path(source_path)
    source = read(source_path)
    if source["data_version"] != "widar_native_amplitude_v2" or source["outer_target"] != target:
        raise RuntimeError("Widar source manifest is not the declared data version/fold")
    if target in source["source_environments"] or set(source["source_environments"]) != ({"W1", "W2", "W3"} - {target}):
        raise RuntimeError("Widar source rooms do not match the outer fold")
    config = job_config(target, arm, seed, source_path, epochs=epochs, updates=updates)
    config.update(
        job_id=f"{target}_W_{arm}_{seed}", control_id=arm,
        experiment={"base_arm": arm, "pool": "unchanged", "objective": "supcon" if arm.endswith("1") else "ce", "model_kind": "single"},
        input_modalities=["amp"], primary_input="amp", input_variant="widar_native_amplitude_v2",
        modality="amplitude_only", phase_loaded=False,
        input_processing="Frozen Widar public DAT native-time linear amplitude, 850x90 float32; no fitted normalization",
        validation_sample_count=len(source["val"]), resource_profile=epochs == 200, device=device,
        development_environments=False, external_replication=True, complete_hardware_calibration=False,
    )
    if target_path is not None:
        target_path = Path(target_path)
        target_metadata = read(target_path)
        if target_metadata["data_version"] != "widar_native_amplitude_v2" or target_metadata["outer_target"] != target:
            raise RuntimeError("Widar target manifest is not the declared data version/fold")
        if set(target_metadata["files"]) != {target}:
            raise RuntimeError("Widar deferred target manifest contains the wrong room set")
        config.update(target_manifest_path=str(target_path), target_manifest_sha256=sha256(target_path))
    config["total_parameters"] = sum(parameter.numel() for parameter in build_experiment_model(config).parameters())
    return config
