"""Meaningful component checks before freezing later control experiments."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dg_losses import DomainAwareSupConLoss
from experiments.revision_validation.models import build_model
from experiments.revision_validation.source_data import VerifiedSources, epoch_schedule
from experiments.revision_validation.control_components import (
    build_pool_control, cross_domain_mixup, mask_control_loss, mixed_cross_entropy,
    pseudo_tx_views, source_coral, temporal_intervention,
)
from experiments.revision_validation.study_io import now, sha256, write


def main():
    torch.set_num_threads(4)
    torch.manual_seed(8101)
    dataset = VerifiedSources(ROOT / "experiments/revision_validation/verified_protocol/source_E1.json")
    schedule = epoch_schedule(dataset.manifest["train"], 401, 0, steps=1)
    amplitude, labels, rows = dataset.batch("train", schedule[0], "cpu")
    reference = build_model("R10", 401)
    reference_state = reference.state_dict()
    pool_checks = {}
    for kind in ("p1_only", "without_p1", "attention"):
        model = build_pool_control(kind, 401)
        for name, value in reference_state.items():
            if not name.startswith(("encoder.temporal_pool.projection.", "encoder.channel_pool.projection.")):
                assert torch.equal(value, model.state_dict()[name]), name
        output = model(amplitude, project=False)
        loss = F.cross_entropy(output["logits"], labels)
        loss.backward()
        assert output["logits"].shape == (16, 6) and torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        pool_checks[kind] = {"parameters": sum(p.numel() for p in model.parameters()),
                             "source_forward_backward_finite": True, "common_initial_tensors_equal": True,
                             "source_CE_smoke_only": float(loss)}
    assert pool_checks["attention"]["parameters"] == sum(p.numel() for p in reference.parameters())
    # Check actual pseudo-Tx slicing, not a renamed physical-antenna assumption.
    views = pseudo_tx_views(amplitude)
    assert views.shape == (16, 4, 850, 90) and torch.equal(views[:, 0], amplitude)
    for group in range(3):
        assert torch.equal(views[:, group + 1, :, group * 30:(group + 1) * 30], amplitude[:, :, group * 30:(group + 1) * 30])
        remaining = [i for i in range(90) if i // 30 != group]
        assert not torch.count_nonzero(views[:, group + 1, :, remaining])
    features = torch.randn(4, 4, 8, dtype=torch.double, requires_grad=True)
    toy_labels = torch.tensor([0, 0, 1, 1])
    mask_checks = {}
    for mode, expected in (("hybrid", (52, 240)), ("all_view", (112, 240)),
                           ("hybrid_remove_same_class_nonpositives", (52, 180))):
        loss, diagnostics, positive, allowed = mask_control_loss(features, toy_labels, mode)
        assert (diagnostics["positive_directed_pairs"], diagnostics["denominator_directed_pairs"]) == expected
        assert not positive.diagonal().any() and not allowed.diagonal().any()
        assert torch.isfinite(torch.autograd.grad(loss, features, retain_graph=True)[0]).all()
        if mode != "hybrid_remove_same_class_nonpositives":
            native_mode = "all_views" if mode == "all_view" else "global_class_instance_views"
            assert torch.equal(loss, DomainAwareSupConLoss(positive_mode=native_mode)(features, toy_labels))
        else:
            assert not positive[1, 5] and not allowed[1, 5]
        mask_checks[mode] = diagnostics
    source_features = torch.randn(4, 3, dtype=torch.double)
    domains = torch.tensor([0] * 4 + [1] * 4)
    identical = torch.cat([source_features, source_features])
    shifted = torch.cat([source_features, source_features + 2])
    assert source_coral(identical, domains) == 0
    assert torch.allclose(source_coral(shifted, domains), torch.tensor(4., dtype=torch.double))
    gradient_input = torch.randn(8, 3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: source_coral(x, domains), (gradient_input,))
    signal = torch.cat([torch.ones(4, 3, 2), 3 * torch.ones(4, 3, 2)])
    toy_labels = domains.clone()
    random_before = torch.get_rng_state()
    mixed, a, b, weights = cross_domain_mixup(signal, toy_labels, domains, seed=1, step=7)
    assert torch.equal(random_before, torch.get_rng_state())
    repeated = cross_domain_mixup(signal, toy_labels, domains, seed=1, step=7)
    assert all(torch.equal(x, y) for x, y in zip((mixed, a, b, weights), repeated))
    assert torch.all(a != b) and torch.all((weights >= 0) & (weights <= 1))
    expected = weights * (1 + 2 * a) + (1 - weights) * (1 + 2 * b)
    assert torch.allclose(mixed[:, 0, 0], expected)
    logits = torch.randn(8, 2, requires_grad=True)
    mixed_cross_entropy(logits, a, b, weights).backward()
    assert torch.isfinite(logits.grad).all()
    ordered = torch.arange(850).float().reshape(1, 850, 1).repeat(2, 1, 1)
    perturbation_checks = {}
    for kind in ("shift_85", "shift_170", "within_85_shuffle", "ten_block_shuffle"):
        value = temporal_intervention(ordered, ["a", "b"], kind)
        assert torch.equal(value.sort(dim=1).values, ordered)
        assert torch.equal(value, temporal_intervention(ordered, ["a", "b"], kind))
        assert not torch.equal(value, ordered)
        perturbation_checks[kind] = "deterministic per sample; preserves every input frame exactly once"
    report = {"passed": True, "checked_at": now(), "target_data_loaded": False, "training_completed": False,
              "pool_controls": pool_checks, "pseudo_tx_slicing_exact": True, "mask_controls": mask_checks,
              "coral_identity_shift_and_gradient_checks": True, "mixup_cross_domain_and_rng_checks": True,
              "temporal_interventions": perturbation_checks, "source_sample_ids": [r["sample_id"] for r in rows],
              "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
                  Path(__file__), ROOT / "experiments/revision_validation/control_components.py")}}
    write(ROOT / "docs/revision/evidence/control_component_qa.json", report)
    print(report)


if __name__ == "__main__":
    main()
