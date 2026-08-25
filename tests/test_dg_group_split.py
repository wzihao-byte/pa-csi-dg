from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dg_dataset import (
    PA_CSI_F0_SPLIT_SEED,
    PA_CSI_GROUP_LABEL_COUNTS,
    PaCsiArrayDataset,
    build_source_only_group_split,
    build_split_indices,
    build_split_manifest,
    load_env_npy_triplets,
    sha256_file,
    validate_split_integrity,
)


ENV_NAMES = ("E1", "E2", "E3")


def valid_block_labels() -> np.ndarray:
    parts = [
        np.full(count, label, dtype=np.int64)
        for label, count in PA_CSI_GROUP_LABEL_COUNTS.items()
    ]
    return np.concatenate(parts)


def write_triplets(
    root: Path,
    *,
    with_sidecars: bool = False,
    sample_count: int = 3000,
    corrupt_first_block: bool = False,
    bad_sidecar_hash: bool = False,
    strict_group_protocol: bool = True,
) -> dict[str, Any]:
    env_files: dict[str, dict[str, str]] = {}
    full_labels = np.tile(valid_block_labels(), 10)[:sample_count]
    if corrupt_first_block:
        full_labels = full_labels.copy()
        full_labels[0] = 1

    for env_offset, env_name in enumerate(ENV_NAMES):
        amp_path = root / f"{env_name}_amp.npy"
        phase_path = root / f"{env_name}_phase.npy"
        label_path = root / f"{env_name}_label.npy"
        np.save(amp_path, np.full((sample_count, 2, 3), env_offset, dtype=np.float32))
        np.save(phase_path, np.zeros((sample_count, 2, 3), dtype=np.float32))
        np.save(label_path, full_labels)

        file_group = {
            "amp": str(amp_path),
            "phase": str(phase_path),
            "label": str(label_path),
        }
        if with_sidecars:
            hashes = {
                "amp": sha256_file(amp_path),
                "phase": sha256_file(phase_path),
                "label": sha256_file(label_path),
            }
            if bad_sidecar_hash and env_name == "E1":
                hashes["amp"] = "0" * 64
            sidecar_path = root / f"{env_name}_subjects.json"
            sidecar_payload = {
                "env_name": env_name,
                "block_subject_ids": [f"S{env_offset * 10 + index + 1:02d}" for index in range(10)],
                "data_file_sha256": hashes,
            }
            sidecar_path.write_text(json.dumps(sidecar_payload), encoding="utf-8")
            file_group["subject_sidecar"] = str(sidecar_path)
        env_files[env_name] = file_group

    return {
        "format": "env_npy_triplets",
        "unwrap_phase": False,
        "strict_group_protocol": strict_group_protocol,
        "env_files": env_files,
    }


@pytest.fixture()
def inferred_dataset(tmp_path: Path) -> PaCsiArrayDataset:
    return load_env_npy_triplets(write_triplets(tmp_path), tmp_path)


@pytest.fixture()
def subject_dataset(tmp_path: Path) -> PaCsiArrayDataset:
    return load_env_npy_triplets(write_triplets(tmp_path, with_sidecars=True), tmp_path)


def build_group_split(
    dataset: PaCsiArrayDataset,
    target_env: str,
    inner_validation: str = "inferred_group_block_holdout",
    **kwargs: Any,
) -> dict[str, Any]:
    return build_split_indices(
        dataset,
        mode="dg_loeo",
        seed=int(kwargs.pop("seed", 42)),
        val_ratio=0.99,
        target_env=target_env,
        inner_validation=inner_validation,
        **kwargs,
    )


def test_loader_infers_strict_namespaced_groups_without_claiming_subjects(
    inferred_dataset: PaCsiArrayDataset,
) -> None:
    dataset = inferred_dataset

    assert len(dataset) == 9000
    assert set(dataset.group_ids[:300]) == {"E1:B00"}
    assert set(dataset.group_ids[2700:3000]) == {"E1:B09"}
    assert dataset.has_complete_group_ids()
    assert not dataset.has_complete_subject_ids()
    assert all(subject_id is None for subject_id in dataset.subject_ids)
    assert len(dataset[0]) == 5  # The training-facing tuple stays backward compatible.
    assert dataset.provenance["sidecar_status"]["E1"]["status"] == "absent_inferred_groups"
    assert len(dataset.provenance["data_files"]["E1"]["amp"]["sha256"]) == 64


@pytest.mark.parametrize(
    ("target_env", "expected_validation_groups"),
    [
        ("E1", {"E2:B00", "E2:B06", "E3:B00", "E3:B04"}),
        ("E2", {"E1:B01", "E1:B07", "E3:B00", "E3:B04"}),
        ("E3", {"E1:B01", "E1:B07", "E2:B00", "E2:B06"}),
    ],
)
def test_inferred_group_holdout_uses_authoritative_f0_and_expected_counts(
    inferred_dataset: PaCsiArrayDataset,
    target_env: str,
    expected_validation_groups: set[str],
) -> None:
    split = build_group_split(inferred_dataset, target_env)

    assert split["group_fold"] == "F0"
    assert split["split_seed"] == PA_CSI_F0_SPLIT_SEED
    assert len(split["train_indices"]) == 4800
    assert len(split["val_indices"]) == 1200
    assert len(split["test_indices"]) == 3000
    assert set(inferred_dataset.group_counts(split["val_indices"])) == expected_validation_groups
    assert inferred_dataset.label_counts(split["train_indices"]) == {
        0: 1920,
        1: 640,
        2: 640,
        3: 640,
        4: 640,
        5: 320,
    }
    assert inferred_dataset.label_counts(split["val_indices"]) == {
        0: 480,
        1: 160,
        2: 160,
        3: 160,
        4: 160,
        5: 80,
    }


def test_model_seed_and_split_seed_never_reshuffle_explicit_f0(
    inferred_dataset: PaCsiArrayDataset,
) -> None:
    first = build_group_split(inferred_dataset, "E1", seed=1)
    second = build_group_split(
        inferred_dataset,
        "E1",
        seed=999,
        split_seed=PA_CSI_F0_SPLIT_SEED,
    )

    assert first["train_indices"] == second["train_indices"]
    assert first["val_indices"] == second["val_indices"]
    assert first["test_indices"] == second["test_indices"]
    assert first["split_seed"] == PA_CSI_F0_SPLIT_SEED
    assert second["split_seed"] == PA_CSI_F0_SPLIT_SEED

    with pytest.raises(ValueError, match="protocol-locked"):
        build_group_split(inferred_dataset, "E1", split_seed=7)


def test_explicit_validation_table_is_used_verbatim(
    inferred_dataset: PaCsiArrayDataset,
) -> None:
    table = {"E2": ["B02", "E2:B03"], "E3": [8, 9]}
    split = build_group_split(
        inferred_dataset,
        "E1",
        validation_group_ids=table,
    )

    assert split["group_fold"] == "explicit"
    assert set(inferred_dataset.group_counts(split["val_indices"])) == {
        "E2:B02",
        "E2:B03",
        "E3:B08",
        "E3:B09",
    }


def test_subject_grouped_holdout_requires_sidecars(
    inferred_dataset: PaCsiArrayDataset,
) -> None:
    with pytest.raises(ValueError, match="validated subject sidecars"):
        build_group_split(inferred_dataset, "E1", "subject_grouped_holdout")


def test_validated_sidecars_enable_subject_split_and_rich_manifest(
    subject_dataset: PaCsiArrayDataset,
) -> None:
    split = build_group_split(subject_dataset, "E1", "subject_grouped_holdout")
    manifest = build_split_manifest(subject_dataset, split)

    assert subject_dataset.has_complete_subject_ids()
    assert subject_dataset.provenance["sidecar_status"]["E1"]["hash_binding"] == "bound"
    assert manifest["subject_sidecars_complete"] is True
    assert manifest["train"]["group_count"] == 16
    assert manifest["val"]["group_count"] == 4
    assert manifest["test"]["group_count"] == 10
    assert manifest["train"]["subject_count"] == 16
    assert manifest["val"]["subject_count"] == 4
    assert manifest["test"]["subject_count"] == 10
    assert manifest["overlap_audit"]["all_clear"] is True
    assert manifest["overlap_audit"]["subject_ids_checked"] is True
    assert len(manifest["index_sha256"]) == 64
    assert len(manifest["train"]["indices_sha256"]) == 64
    assert len(manifest["file_sha256"]["E1"]["amp"]) == 64
    assert manifest["validation_group_ids"] == {
        "E2": ["E2:B00", "E2:B06"],
        "E3": ["E3:B00", "E3:B04"],
    }


def test_integrity_rejects_partial_group_leakage(
    inferred_dataset: PaCsiArrayDataset,
) -> None:
    split = build_group_split(inferred_dataset, "E1")
    leaked_index = next(
        index
        for index in split["train_indices"]
        if inferred_dataset.group_ids[index] == "E2:B01"
    )
    split["train_indices"].remove(leaked_index)
    split["val_indices"].append(leaked_index)

    with pytest.raises(ValueError, match="group ids overlap|partially assigned"):
        validate_split_integrity(inferred_dataset, split)


def test_source_only_public_api_never_requires_loaded_target(tmp_path: Path) -> None:
    config = write_triplets(tmp_path)
    config["env_files"] = {
        env_name: config["env_files"][env_name]
        for env_name in ("E2", "E3")
    }
    dataset = load_env_npy_triplets(config, tmp_path)
    assert dataset.env_names == ["E2", "E3"]
    assert "E1" not in dataset.provenance["data_files"]

    split = build_source_only_group_split(dataset, target_env="E1")
    manifest = build_split_manifest(dataset, split)

    assert split["mode"] == "dg_loeo_source_only"
    assert split["target_env"] == "E1"
    assert split["target_not_loaded"] is True
    assert split["source_envs"] == ["E2", "E3"]
    assert len(split["train_indices"]) == 4800
    assert len(split["val_indices"]) == 1200
    assert split["test_indices"] == []
    assert manifest["target_not_loaded"] is True
    assert manifest["test"]["count"] == 0
    assert set(manifest["data_files"]) == {"E2", "E3"}
    assert manifest["overlap_audit"]["all_clear"] is True


def test_source_only_api_fails_if_target_is_loaded(
    inferred_dataset: PaCsiArrayDataset,
) -> None:
    with pytest.raises(ValueError, match="must load exactly"):
        build_source_only_group_split(inferred_dataset, target_env="E1")


def test_source_only_integrity_rejects_nonempty_test_split(tmp_path: Path) -> None:
    config = write_triplets(tmp_path)
    config["env_files"] = {
        env_name: config["env_files"][env_name]
        for env_name in ("E2", "E3")
    }
    dataset = load_env_npy_triplets(config, tmp_path)
    split = build_source_only_group_split(dataset, target_env="E1")
    leaked_index = split["train_indices"].pop()
    split["test_indices"].append(leaked_index)

    with pytest.raises(ValueError, match="empty test split"):
        validate_split_integrity(dataset, split)


def test_integrity_rejects_duplicate_sample_ids(
    inferred_dataset: PaCsiArrayDataset,
) -> None:
    split = build_group_split(inferred_dataset, "E1")
    inferred_dataset.sample_ids[1] = inferred_dataset.sample_ids[0]

    with pytest.raises(ValueError, match="globally unique"):
        validate_split_integrity(inferred_dataset, split)


def test_non_strict_loader_preserves_legacy_arbitrary_dataset_shape(tmp_path: Path) -> None:
    config = write_triplets(
        tmp_path,
        sample_count=17,
        strict_group_protocol=False,
    )
    dataset = load_env_npy_triplets(config, tmp_path)

    assert len(dataset) == 51
    assert not dataset.has_complete_group_ids()
    assert all(group_id is None for group_id in dataset.group_ids)
    assert dataset.provenance["group_protocol"]["enabled"] is False
    assert dataset.provenance["data_files"]["E1"]["amp"]["sha256"] is None
    with pytest.raises(ValueError, match="group_id"):
        build_group_split(dataset, "E1")


@pytest.mark.parametrize(
    ("invalid_value", "message"),
    [
        (np.nan, "NaN or infinite"),
        (np.inf, "NaN or infinite"),
        (1.5, "non-integer"),
    ],
)
def test_loader_validates_labels_before_integer_cast(
    tmp_path: Path,
    invalid_value: float,
    message: str,
) -> None:
    config = write_triplets(tmp_path)
    label_path = Path(config["env_files"]["E1"]["label"])
    labels = np.load(label_path).astype(np.float64)
    labels[0] = invalid_value
    np.save(label_path, labels)

    with pytest.raises(ValueError, match=message):
        load_env_npy_triplets(config, tmp_path)


@pytest.mark.parametrize(
    ("sample_count", "corrupt_first_block", "message"),
    [
        (2999, False, "exactly 3000 samples"),
        (3000, True, "label counts"),
    ],
)
def test_loader_fails_closed_on_invalid_contiguous_group_protocol(
    tmp_path: Path,
    sample_count: int,
    corrupt_first_block: bool,
    message: str,
) -> None:
    config = write_triplets(
        tmp_path,
        sample_count=sample_count,
        corrupt_first_block=corrupt_first_block,
    )

    with pytest.raises(ValueError, match=message):
        load_env_npy_triplets(config, tmp_path)


def test_loader_rejects_sidecar_bound_to_different_arrays(tmp_path: Path) -> None:
    config = write_triplets(tmp_path, with_sidecars=True, bad_sidecar_hash=True)

    with pytest.raises(ValueError, match="hash mismatch"):
        load_env_npy_triplets(config, tmp_path)


def test_unbound_sidecars_cannot_enable_subject_claims(tmp_path: Path) -> None:
    config = write_triplets(tmp_path, with_sidecars=True)
    for file_group in config["env_files"].values():
        sidecar_path = Path(file_group["subject_sidecar"])
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        payload.pop("data_file_sha256")
        sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = load_env_npy_triplets(config, tmp_path)
    assert not dataset.has_complete_subject_ids()
    assert all(subject_id is None for subject_id in dataset.subject_ids)
    assert dataset.provenance["subject_sidecars_complete"] is False
    assert dataset.provenance["sidecar_status"]["E1"]["status"] == "validated_unbound"
    assert dataset.provenance["sidecar_status"]["E1"]["subject_ids"] == []
    assert len(
        dataset.provenance["sidecar_status"]["E1"]["candidate_subject_ids_unverified"]
    ) == 10
    with pytest.raises(ValueError, match="validated subject sidecars"):
        build_group_split(dataset, "E1", "subject_grouped_holdout")


def test_partially_bound_sidecars_keep_public_subject_ids_null(tmp_path: Path) -> None:
    config = write_triplets(
        tmp_path,
        with_sidecars=True,
        strict_group_protocol=False,
    )
    for file_group in config["env_files"].values():
        sidecar_path = Path(file_group["subject_sidecar"])
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        payload["data_file_sha256"].pop("label")
        sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = load_env_npy_triplets(config, tmp_path)
    assert dataset.provenance["group_protocol"]["enabled"] is True
    assert dataset.provenance["sidecar_status"]["E1"]["status"] == "validated_partially_bound"
    assert all(subject_id is None for subject_id in dataset.subject_ids)
    assert not dataset.has_complete_subject_ids()
