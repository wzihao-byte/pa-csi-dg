from __future__ import annotations

import sys
from pathlib import Path
from math import isclose

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dg_losses import supcon_positive_pair_diagnostics  # noqa: E402


def test_all_views_positive_pair_diagnostics() -> None:
    features = torch.randn(3, 2, 4)
    labels = torch.tensor([0, 0, 1])
    domains = torch.tensor([0, 1, 0])
    sample_ids = torch.tensor([10, 11, 12])

    diagnostics = supcon_positive_pair_diagnostics(
        features,
        labels=labels,
        domains=domains,
        sample_ids=sample_ids,
        include_same_domain_same_class=True,
        positive_mode="all_views",
    )

    assert diagnostics["supcon_num_anchors"] == 6.0
    assert diagnostics["supcon_num_valid_positive_pairs"] == 14.0
    assert isclose(diagnostics["supcon_mean_positives_per_anchor"], 14.0 / 6.0, rel_tol=1e-6)
    assert diagnostics["supcon_min_positives_per_anchor"] == 1.0
    assert diagnostics["supcon_frac_anchors_with_positive"] == 1.0


def test_global_class_instance_views_positive_pair_diagnostics() -> None:
    features = torch.randn(3, 2, 4)
    labels = torch.tensor([0, 0, 1])
    domains = torch.tensor([0, 1, 0])
    sample_ids = torch.tensor([10, 11, 12])

    diagnostics = supcon_positive_pair_diagnostics(
        features,
        labels=labels,
        domains=domains,
        sample_ids=sample_ids,
        include_same_domain_same_class=True,
        positive_mode="global_class_instance_views",
    )

    assert diagnostics["supcon_num_anchors"] == 6.0
    assert diagnostics["supcon_num_valid_positive_pairs"] == 8.0
    assert isclose(diagnostics["supcon_mean_positives_per_anchor"], 8.0 / 6.0, rel_tol=1e-6)
    assert diagnostics["supcon_min_positives_per_anchor"] == 1.0
    assert diagnostics["supcon_frac_anchors_with_positive"] == 1.0


def test_cross_domain_only_positive_pair_diagnostics() -> None:
    features = torch.randn(3, 2, 4)
    labels = torch.tensor([0, 0, 0])
    domains = torch.tensor([0, 0, 1])
    sample_ids = torch.tensor([10, 11, 12])

    diagnostics = supcon_positive_pair_diagnostics(
        features,
        labels=labels,
        domains=domains,
        sample_ids=sample_ids,
        include_same_domain_same_class=False,
        positive_mode="all_views",
    )

    assert diagnostics["supcon_num_anchors"] == 6.0
    assert diagnostics["supcon_num_valid_positive_pairs"] == 22.0
    assert isclose(diagnostics["supcon_mean_positives_per_anchor"], 22.0 / 6.0, rel_tol=1e-6)
    assert diagnostics["supcon_min_positives_per_anchor"] == 3.0
    assert diagnostics["supcon_frac_anchors_with_positive"] == 1.0


if __name__ == "__main__":
    test_all_views_positive_pair_diagnostics()
    test_global_class_instance_views_positive_pair_diagnostics()
    test_cross_domain_only_positive_pair_diagnostics()
    print("SupCon positive-pair diagnostic smoke check passed")
