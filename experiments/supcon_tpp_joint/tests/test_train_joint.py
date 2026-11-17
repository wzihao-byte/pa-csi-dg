from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dg_losses import DomainAwareSupConLoss
from experiments.supcon_tpp_joint.runner_utils import DeterministicDomainClassBatchSampler
from experiments.supcon_tpp_joint.train_joint import (
    combine_deferred_target_manifest,
    effective_lambda_at_step,
    evaluate_model,
    phase_input_audit,
    resolve_runtime_config,
    resolve_single_seed,
    resolve_source_data_config,
    resolve_target_data_config,
    select_calibration_indices,
    train_one_epoch,
)


def _deferred_config() -> dict:
    return {
        "target_env": "E1",
        "source_envs": ["E2", "E3"],
        "data": {
            "format": "env_npy_triplets",
            "unwrap_phase": False,
            "env_files": {
                "E1": {"amp": "a1", "phase": "p1", "label": "l1"},
                "E2": {"amp": "a2", "phase": "p2", "label": "l2"},
                "E3": {"amp": "a3", "phase": "p3", "label": "l3"},
            },
        },
    }


def test_source_and_target_configs_are_disjoint() -> None:
    config = _deferred_config()
    target_group = config["data"]["env_files"].pop("E1")
    config["target_data"] = {
        "target_env": "E1",
        "load_for_evaluation": True,
        "env_files": target_group,
    }
    source = resolve_source_data_config(config)
    target = resolve_target_data_config(config)
    assert list(source["env_files"]) == ["E2", "E3"]
    assert list(target["env_files"]) == ["E1"]
    assert source["unwrap_phase"] is target["unwrap_phase"] is False
    assert source["strict_group_protocol"] is target["strict_group_protocol"] is True


def test_top_level_deferred_target_schema_is_supported() -> None:
    config = _deferred_config()
    target_group = config["data"]["env_files"].pop("E1")
    config["target_data"] = {
        "target_env": "E1",
        "load_for_evaluation": True,
        "env_files": target_group,
    }
    target = resolve_target_data_config(config)
    assert target["env_files"] == {"E1": target_group}


def test_target_resolver_rejects_target_inside_source_mapping() -> None:
    config = _deferred_config()
    config["target_data"] = {
        "target_env": "E1",
        "load_for_evaluation": True,
        "env_files": config["data"]["env_files"]["E1"],
    }
    with pytest.raises(ValueError, match="absent from data.env_files"):
        resolve_target_data_config(config)


def test_resolved_seed_and_formal_epochs_cannot_be_overridden() -> None:
    config = {"seed_list": [101], "protocol": {"stage": "formal_interim"}, "training": {"epochs": 200}}
    assert resolve_single_seed(config, 101) == 101
    with pytest.raises(ValueError, match="cannot change"):
        resolve_single_seed(config, 999)
    args = SimpleNamespace(epochs=10, device=None, evaluate_target=None)
    with pytest.raises(ValueError, match="only"):
        resolve_runtime_config(config, args)


def test_screen_smoke_override_is_locked_to_ten_epochs() -> None:
    config = {"protocol": {"stage": "screen"}, "training": {"epochs": 200}}
    resolved = resolve_runtime_config(
        config,
        SimpleNamespace(epochs=10, device=None, evaluate_target=None),
    )
    assert resolved["training"]["epochs"] == 10
    assert resolved["protocol"]["runtime_kind"] == "smoke_override"
    with pytest.raises(ValueError, match="locked to 10"):
        resolve_runtime_config(
            config,
            SimpleNamespace(epochs=9, device=None, evaluate_target=None),
        )


def test_linear_warmup_step_boundaries() -> None:
    target = 0.025
    assert effective_lambda_at_step(target, 0, 6000) == pytest.approx(target / 6000)
    assert effective_lambda_at_step(target, 5998, 6000) == pytest.approx(target * 5999 / 6000)
    assert effective_lambda_at_step(target, 5999, 6000) == pytest.approx(target)
    assert effective_lambda_at_step(target, 6000, 6000) == pytest.approx(target)
    assert effective_lambda_at_step(0.0, 0, 6000) == 0.0


def test_phase_audit_matches_raw_without_unwrap(tmp_path) -> None:
    raw = np.linspace(-4.0, 4.0, 24, dtype=np.float64).reshape(2, 3, 4)
    phase_path = tmp_path / "phase.npy"
    np.save(phase_path, raw)
    dataset = SimpleNamespace(
        env_names=["E2"],
        env_ids=np.array([0, 0]),
        phase=raw.astype(np.float32),
    )
    data_config = {
        "unwrap_phase": False,
        "env_files": {"E2": {"phase": str(phase_path)}},
    }
    audit = phase_input_audit(dataset, data_config, base_dir=tmp_path)
    assert audit["records"]["E2"]["exact_float32_match"] is True
    assert audit["records"]["E2"]["unwrap_applied"] is False


def test_calibration_selection_is_lowest_source_train_and_disjoint() -> None:
    dataset = SimpleNamespace(
        env_names=["E2", "E3"],
        env_ids=np.repeat(np.arange(2), 3000),
        sample_ids=np.arange(6000),
    )
    # Mimic F0 with the first 300-sample block held out in each source.
    train_indices = list(range(300, 3000)) + list(range(3300, 6000))
    calibration, smoke, calibration_ids, smoke_ids = select_calibration_indices(
        dataset,
        train_indices,
    )
    assert len(calibration) == 512
    assert len(smoke) == 16
    assert calibration[:3] == [300, 301, 302]
    assert calibration[256:259] == [3300, 3301, 3302]
    assert calibration_ids[0] == "E2:0300"
    assert calibration_ids[256] == "E3:0300"
    assert set(calibration).isdisjoint(smoke)
    assert set(calibration_ids).isdisjoint(smoke_ids)


def test_combined_deferred_manifest_recomputes_namespaced_identity_audit() -> None:
    source_dataset = SimpleNamespace(
        env_names=["E2", "E3"],
        env_ids=np.array([0, 0, 1, 1]),
        sample_ids=np.array([0, 1, 2, 3]),
        group_ids=np.array(["E2:B01", "E2:B00", "E3:B01", "E3:B00"], dtype=object),
        subject_ids=np.full(4, None, dtype=object),
    )
    target_dataset = SimpleNamespace(
        env_names=["E1"],
        env_ids=np.array([0, 0]),
        sample_ids=np.array([0, 1]),
        group_ids=np.array(["E1:B00", "E1:B01"], dtype=object),
        subject_ids=np.full(2, None, dtype=object),
    )
    source_split = {"train_indices": [0, 2], "val_indices": [1, 3]}
    source_manifest = {
        "dataset_size": 4,
        "train": {"group_ids_complete": True},
        "val": {"group_ids_complete": True},
        "test": {},
        "data_files": {"E2": {}, "E3": {}},
        "file_sha256": {"E2": {}, "E3": {}},
        "sidecar_status": {"E2": {}, "E3": {}},
        "subject_sidecars_complete": False,
    }
    target_manifest_payload = {
        "dataset_size": 2,
        "test": {"count": 2, "group_ids_complete": True, "subject_ids_complete": False},
        "data_files": {"E1": {}},
        "file_sha256": {"E1": {}},
        "sidecar_status": {"E1": {}},
        "subject_sidecars_complete": False,
    }
    combined = combine_deferred_target_manifest(
        source_manifest,
        source_dataset,
        source_split,
        target_manifest_payload,
        target_dataset,
        "E1",
    )
    assert combined["dataset_size"] == 6
    assert combined["target_not_loaded"] is False
    assert combined["target_absent_during_training"] is True
    assert combined["target_loaded_after_checkpoint"] is True
    assert combined["overlap_audit"]["all_clear"] is True
    assert combined["overlap_audit"]["sample_ids"]["train_test"] == []
    assert set(combined["file_sha256"]) == {"E1", "E2", "E3"}


class _TinyDataset(Dataset):
    def __init__(self) -> None:
        self.labels = np.tile(np.arange(6), 8).astype(np.int64)
        self.env_ids = np.repeat(np.arange(2), 24).astype(np.int64)
        self.sample_ids = np.arange(48, dtype=np.int64)
        self.env_names = ["E2", "E3"]
        self.amplitude = np.arange(48 * 4, dtype=np.float32).reshape(48, 2, 2) / 100
        self.phase = self.amplitude * 0.2
        self.group_ids = np.asarray(
            [f"{self.env_names[int(env)]}:B00" for env in self.env_ids],
            dtype=object,
        )
        self.subject_ids = np.full(48, None, dtype=object)
        self.provenance = {
            "sidecar_status": {
                "E2": {"verified": False},
                "E3": {"verified": False},
            }
        }

    def __len__(self) -> int:
        return 48

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.amplitude[index]),
            torch.from_numpy(self.phase[index]),
            torch.tensor(self.labels[index]),
            torch.tensor(self.env_ids[index]),
            torch.tensor(self.sample_ids[index]),
        )

    def env_name(self, env_id: int) -> str:
        return self.env_names[env_id]


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(4, 8)
        self.classifier = torch.nn.Linear(8, 6)
        self.projection_head = torch.nn.Linear(8, 5)

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor):
        fused = torch.tanh(self.encoder(torch.cat([amplitude.mean(1), phase.mean(1)], dim=1)))
        projection = F.normalize(self.projection_head(fused), dim=1)
        return {
            "fused_feature": fused,
            "logits": self.classifier(fused),
            "projection": projection,
            "supcon_projection_views": projection.unsqueeze(1),
        }


def test_training_loop_executes_exact_steps_and_prediction_subjects_are_null() -> None:
    torch.manual_seed(7)
    dataset = _TinyDataset()
    sampler = DeterministicDomainClassBatchSampler(
        {0: list(range(24)), 1: list(range(24, 48))},
        dataset.labels,
        batch_size=16,
        num_batches=300,
        seed=81,
    )
    loader = DataLoader(dataset, batch_sampler=sampler)
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = DomainAwareSupConLoss(temperature=0.2)
    metrics, global_step = train_one_epoch(
        model,  # type: ignore[arg-type]
        loader,
        sampler,
        optimizer,
        loss,
        epoch=1,
        device=torch.device("cpu"),
        target_lambda=0.01,
        warmup_steps=6000,
        global_step=0,
        base_lr=0.001,
        decay_rate=0.9,
        decay_steps=10_000,
    )
    assert global_step == 300
    assert metrics["global_step"] == 300
    assert metrics["positive_anchor_fraction"] == pytest.approx(1.0)
    assert metrics["gradient_norm_l2"]["count"] == 300
    assert np.isfinite(metrics["gradient_norm_l2"]["maximum"])

    eval_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    _, records = evaluate_model(model, eval_loader, dataset, device=torch.device("cpu"))  # type: ignore[arg-type]
    assert len(records) == 48
    assert all(record["subject_id"] is None for record in records)
    assert all(len(record["probability"]) == 6 for record in records)
