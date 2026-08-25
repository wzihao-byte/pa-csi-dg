from __future__ import annotations

import numpy as np
import pytest

from dg_dataset import PaCsiArrayDataset, build_split_indices, validate_split_integrity


def make_dataset() -> PaCsiArrayDataset:
    samples_per_env = 6
    env_names = ["E1", "E2", "E3"]
    num_samples = samples_per_env * len(env_names)
    amplitude = np.zeros((num_samples, 2, 3), dtype=np.float32)
    phase = np.zeros((num_samples, 2, 3), dtype=np.float32)
    labels = np.tile(np.arange(samples_per_env) % 3, len(env_names)).astype(np.int64)
    env_ids = np.repeat(np.arange(len(env_names)), samples_per_env).astype(np.int64)
    sample_ids = np.arange(num_samples, dtype=np.int64)
    return PaCsiArrayDataset(amplitude, phase, labels, env_ids, env_names, sample_ids)


def env_names_for(dataset: PaCsiArrayDataset, indices: list[int]) -> set[str]:
    return {dataset.env_name(int(dataset.env_ids[index])) for index in indices}


def test_loso_inner_validation_holds_out_one_full_source_domain() -> None:
    dataset = make_dataset()

    split = build_split_indices(
        dataset,
        mode="dg_loeo",
        seed=42,
        val_ratio=0.2,
        target_env="E1",
        inner_validation="loso",
    )

    assert split["source_envs"] == ["E2", "E3"]
    assert split["train_envs"] == ["E2"]
    assert split["val_envs"] == ["E3"]
    assert split["loso_val_env"] == "E3"
    assert env_names_for(dataset, split["train_indices"]) == {"E2"}
    assert env_names_for(dataset, split["val_indices"]) == {"E3"}
    assert env_names_for(dataset, split["test_indices"]) == {"E1"}
    assert len(split["val_indices"]) == 6


def test_loso_integrity_rejects_validation_source_leaking_into_train() -> None:
    dataset = make_dataset()
    split = build_split_indices(
        dataset,
        mode="dg_loeo",
        seed=42,
        val_ratio=0.2,
        target_env="E1",
        inner_validation="loso",
    )
    leaked_index = int(split["val_indices"][0])
    split["train_indices"] = [*split["train_indices"], leaked_index]

    with pytest.raises(ValueError, match="overlap|leaked"):
        validate_split_integrity(dataset, split)
