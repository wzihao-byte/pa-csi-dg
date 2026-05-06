from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_views(
    features: torch.Tensor,
    labels: Optional[torch.Tensor],
    domains: Optional[torch.Tensor],
    sample_ids: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if features.dim() == 2:
        features = features.unsqueeze(1)
    if features.dim() != 3:
        raise ValueError(f"Expected features with shape [batch, views, dim] or [batch, dim], received {features.shape}.")

    batch_size, num_views, feature_dim = features.shape
    flat_features = features.reshape(batch_size * num_views, feature_dim)

    def repeat_target(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        return value.repeat_interleave(num_views)

    return (
        flat_features,
        repeat_target(labels),
        repeat_target(domains),
        repeat_target(sample_ids),
    )


def masked_contrastive_loss(features: torch.Tensor, positive_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    if features.size(0) < 2:
        return features.new_zeros(())

    normalized = F.normalize(features, dim=-1)
    similarity = torch.matmul(normalized, normalized.T) / temperature
    logits = similarity - similarity.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(features.size(0), device=features.device, dtype=torch.bool)
    positive_mask = positive_mask & (~self_mask)
    negative_mask = ~self_mask

    exp_logits = torch.exp(logits) * negative_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positive_count = positive_mask.sum(dim=1)
    valid_rows = positive_count > 0
    if not valid_rows.any():
        return features.new_zeros(())

    mean_log_prob_pos = (positive_mask[valid_rows] * log_prob[valid_rows]).sum(dim=1) / positive_count[valid_rows]
    return -mean_log_prob_pos.mean()


class DomainAwareSupConLoss(nn.Module):
    VALID_POSITIVE_MODES = {"all_views", "global_class_instance_views"}

    def __init__(
        self,
        temperature: float = 0.2,
        include_same_domain_same_class: bool = True,
        positive_mode: str = "all_views",
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.include_same_domain_same_class = include_same_domain_same_class
        self.positive_mode = positive_mode
        if self.positive_mode not in self.VALID_POSITIVE_MODES:
            raise ValueError(
                f"Unsupported SupCon positive mode '{self.positive_mode}'. "
                f"Choose from {sorted(self.VALID_POSITIVE_MODES)}."
            )

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        domains: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.dim() == 2:
            batch_size, num_views = features.size(0), 1
        elif features.dim() == 3:
            batch_size, num_views = features.size(0), features.size(1)
        else:
            raise ValueError(
                f"Expected features with shape [batch, views, dim] or [batch, dim], received {features.shape}."
            )

        flat_features, flat_labels, flat_domains, flat_sample_ids = _flatten_views(features, labels, domains, sample_ids)
        if flat_labels is None:
            raise ValueError("labels are required for supervised contrastive loss.")

        same_class = flat_labels[:, None].eq(flat_labels[None, :])
        class_positive_mask = same_class
        if flat_domains is not None and not self.include_same_domain_same_class:
            same_domain = flat_domains[:, None].eq(flat_domains[None, :])
            class_positive_mask = class_positive_mask & (~same_domain)

        if self.positive_mode == "all_views":
            positive_mask = class_positive_mask.clone()
        elif self.positive_mode == "global_class_instance_views":
            view_ids = torch.arange(num_views, device=flat_features.device).repeat(batch_size)
            is_global = view_ids == 0
            global_global = is_global[:, None] & is_global[None, :]
            positive_mask = class_positive_mask & global_global
        else:
            raise AssertionError(f"Unhandled SupCon positive mode: {self.positive_mode}")

        if flat_sample_ids is not None:
            same_sample = flat_sample_ids[:, None].eq(flat_sample_ids[None, :])
        else:
            implicit_sample_ids = torch.arange(batch_size, device=flat_features.device).repeat_interleave(num_views)
            same_sample = implicit_sample_ids[:, None].eq(implicit_sample_ids[None, :])
        positive_mask = positive_mask | same_sample

        return masked_contrastive_loss(flat_features, positive_mask, self.temperature)


class SubCenterPrototypeContrastiveLoss(nn.Module):
    """Sub-center prototype contrastive loss.

    Replaces the pairwise SupCon "pull all same-class samples together" geometry with a
    soft assignment to per-class learnable sub-centers. Each anchor is pulled toward the
    closest sub-center of its true class while being pushed away from every other
    prototype across all classes.
    """

    def __init__(
        self,
        temperature: float = 0.2,
        lambda_pair_margin: float = 0.0,
        prototype_pair_margins: Optional[Sequence[Mapping[str, float]]] = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.lambda_pair_margin = float(lambda_pair_margin)
        self.prototype_pair_margins: List[Mapping[str, float]] = list(prototype_pair_margins or [])

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        if features.dim() == 2:
            features = features.unsqueeze(1)
        if features.dim() != 3:
            raise ValueError(
                f"Expected features with shape [batch, views, dim] or [batch, dim], received {features.shape}."
            )
        if prototypes.dim() != 3:
            raise ValueError(
                f"Expected prototypes with shape [num_classes, num_subcenters, dim], received {prototypes.shape}."
            )

        batch_size, num_views, feature_dim = features.shape
        if prototypes.size(-1) != feature_dim:
            raise ValueError(
                f"Prototype dim {prototypes.size(-1)} does not match feature dim {feature_dim}."
            )

        anchors = features.reshape(batch_size * num_views, feature_dim)
        anchor_labels = labels.repeat_interleave(num_views)
        num_anchors = anchors.size(0)

        anchors = F.normalize(anchors, dim=-1)
        prototypes_norm = F.normalize(prototypes, dim=-1)
        num_classes, num_subcenters, _ = prototypes_norm.shape
        proto_flat = prototypes_norm.reshape(num_classes * num_subcenters, feature_dim)

        sim = anchors @ proto_flat.T
        sim = sim.reshape(num_anchors, num_classes, num_subcenters)

        scaled = sim / self.temperature
        anchor_index = torch.arange(num_anchors, device=anchors.device)
        positive_logit = scaled[anchor_index, anchor_labels].max(dim=-1).values
        denom = torch.logsumexp(scaled.reshape(num_anchors, num_classes * num_subcenters), dim=-1)
        ce_loss = (-positive_logit + denom).mean()

        margin_term = anchors.new_zeros(())
        if self.lambda_pair_margin > 0.0 and self.prototype_pair_margins:
            per_class_max = sim.max(dim=-1).values  # [N, C]
            collected: List[torch.Tensor] = []
            for spec in self.prototype_pair_margins:
                true_cls = int(spec["true"])
                neg_cls = int(spec["negative"])
                margin_value = float(spec["margin"])
                if not (0 <= true_cls < num_classes and 0 <= neg_cls < num_classes):
                    raise ValueError(
                        f"prototype_pair_margins references out-of-range class indices: "
                        f"true={true_cls}, negative={neg_cls}, num_classes={num_classes}."
                    )
                mask = anchor_labels == true_cls
                if not mask.any():
                    continue
                true_score = per_class_max[mask, true_cls]
                neg_score = per_class_max[mask, neg_cls]
                hinge = F.relu(margin_value + neg_score - true_score)
                collected.append(hinge)
            if collected:
                margin_term = torch.cat(collected).mean()

        return ce_loss + self.lambda_pair_margin * margin_term


class InstanceContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
        flat_features, _, _, flat_sample_ids = _flatten_views(features, None, None, sample_ids)
        if flat_sample_ids is None:
            raise ValueError("sample_ids are required for instance contrastive loss.")
        positive_mask = flat_sample_ids[:, None].eq(flat_sample_ids[None, :])
        return masked_contrastive_loss(flat_features, positive_mask, self.temperature)


def choose_style_donors(domains: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if domains is None:
        return None

    donors = []
    device = domains.device
    for index in range(domains.size(0)):
        candidate_mask = domains != domains[index]
        candidates = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if candidates.numel() == 0:
            return None
        donors.append(candidates[torch.randint(candidates.numel(), (1,), device=device)].item())
    return torch.tensor(donors, device=device, dtype=torch.long)


def adain_mix(features: torch.Tensor, domains: Optional[torch.Tensor] = None, eps: float = 1e-5) -> torch.Tensor:
    if features.dim() != 2:
        raise ValueError(f"AdaIN expects [batch, dim] fused features, received {features.shape}.")
    if features.size(0) < 2:
        return features

    donor_indices = choose_style_donors(domains)
    if donor_indices is None:
        permutation = torch.randperm(features.size(0), device=features.device)
        if torch.equal(permutation, torch.arange(features.size(0), device=features.device)):
            permutation = torch.roll(permutation, shifts=1)
        donor_indices = permutation

    donor_features = features[donor_indices]
    content_mean = features.mean(dim=1, keepdim=True)
    content_std = features.std(dim=1, keepdim=True, unbiased=False) + eps
    style_mean = donor_features.mean(dim=1, keepdim=True)
    style_std = donor_features.std(dim=1, keepdim=True, unbiased=False) + eps

    normalized = (features - content_mean) / content_std
    return normalized * style_std + style_mean
