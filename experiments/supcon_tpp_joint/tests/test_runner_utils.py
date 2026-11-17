from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.supcon_tpp_joint.runner_utils import (
    DeterministicDomainClassBatchSampler,
    assert_phase_unwrap_disabled,
    ce_supcon_gradient_diagnostics,
    filter_data_config_to_environments,
    fixed_label_metrics,
    trainable_parameter_hash,
)


def _sampler(seed: int = 17) -> DeterministicDomainClassBatchSampler:
    labels = np.tile(np.arange(6), 8)
    return DeterministicDomainClassBatchSampler(
        {0: list(range(24)), 1: list(range(24, 48))},
        labels,
        batch_size=16,
        num_batches=5,
        seed=seed,
    )


def test_deterministic_sampler_is_exact_and_covers_every_domain_class() -> None:
    first = _sampler()
    second = _sampler()
    assert list(first) == list(second)
    assert first.schedule_hash() == second.schedule_hash()
    assert len(first) == 5

    labels = first.labels
    for batch in first:
        assert len(batch) == 16
        for domain, domain_range in ((0, range(0, 24)), (1, range(24, 48))):
            domain_set = set(domain_range)
            domain_indices = [index for index in batch if index in domain_set]
            assert len(domain_indices) == 8
            assert set(int(labels[index]) for index in domain_indices) == set(range(6))


def test_sampler_epoch_changes_schedule_but_remains_repeatable() -> None:
    sampler = _sampler()
    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)
    assert epoch_zero != epoch_one
    repeat = _sampler()
    repeat.set_epoch(1)
    assert epoch_one == list(repeat)


def test_source_filter_never_retains_target_files() -> None:
    data = {
        "unwrap_phase": False,
        "env_files": {
            "E1": {"phase": "target.npy"},
            "E2": {"phase": "source2.npy"},
            "E3": {"phase": "source3.npy"},
        },
    }
    filtered = filter_data_config_to_environments(data, ["E2", "E3"])
    assert list(filtered["env_files"]) == ["E2", "E3"]
    assert "E1" not in filtered["env_files"]
    assert list(data["env_files"]) == ["E1", "E2", "E3"]


def test_phase_guard_requires_explicit_false_in_both_configs() -> None:
    config = {"data": {"unwrap_phase": False}, "protocol": {"phase_unwrap": False}}
    assert_phase_unwrap_disabled(config, copy.deepcopy(config))
    bad = copy.deepcopy(config)
    bad["data"]["unwrap_phase"] = True
    with pytest.raises(ValueError, match="unwrap_phase=false"):
        assert_phase_unwrap_disabled(config, bad)


def test_fixed_metrics_keep_six_classes_and_pair_confusion() -> None:
    labels = np.arange(6)
    predictions = np.array([0, 4, 2, 3, 1, 5])
    metrics = fixed_label_metrics(labels, predictions)
    assert set(metrics["per_class"]) == {str(index) for index in range(6)}
    assert metrics["c1_c4_pair_confusion"]["bidirectional_count"] == 2
    assert np.asarray(metrics["confusion_matrix"]).shape == (6, 6)


class _TinyJoint(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.amp_encoder = torch.nn.Linear(2, 3)
        self.phase_encoder = torch.nn.Linear(2, 3)
        self.fusion = torch.nn.Linear(6, 4)
        self.classifier = torch.nn.Linear(4, 2)
        self.projection_head = torch.nn.Linear(4, 3)

    def forward(self, amp: torch.Tensor, phase: torch.Tensor):
        fused = self.fusion(torch.cat([self.amp_encoder(amp), self.phase_encoder(phase)], dim=1))
        return self.classifier(fused), F.normalize(self.projection_head(fused), dim=1)


def test_parameter_hash_and_gradient_diagnostics_are_stable() -> None:
    torch.manual_seed(5)
    model_a = _TinyJoint()
    torch.manual_seed(5)
    model_b = _TinyJoint()
    assert trainable_parameter_hash(model_a) == trainable_parameter_hash(model_b)

    amp = torch.randn(6, 2)
    phase = torch.randn(6, 2)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    logits, projection = model_a(amp, phase)
    ce = F.cross_entropy(logits, labels)
    # A differentiable supervised compactness objective is sufficient to test
    # the separate-gradient accounting without coupling this unit test to the
    # production SupCon implementation.
    supcon = sum(projection[labels == class_id].var(dim=0).mean() for class_id in (0, 1))
    report = ce_supcon_gradient_diagnostics(model_a, ce, supcon, lambda_supcon=0.025)
    assert report["amplitude_encoder"]["ce_grad_norm"] > 0
    assert report["amplitude_encoder"]["supcon_grad_norm"] > 0
    assert "weighted_supcon_to_ce_ratio" not in report["classifier"]
    assert report["projection_head"]["ce_grad_norm"] == pytest.approx(0.0)

