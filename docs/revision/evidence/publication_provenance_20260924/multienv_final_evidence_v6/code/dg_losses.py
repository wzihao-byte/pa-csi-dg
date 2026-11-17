from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DOMAIN_AWARE_SUPCON_POSITIVE_MODES = {"all_views", "global_class_instance_views"}


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


def domain_aware_supcon_positive_mask(
    features: torch.Tensor,
    labels: torch.Tensor,
    domains: Optional[torch.Tensor] = None,
    sample_ids: Optional[torch.Tensor] = None,
    include_same_domain_same_class: bool = True,
    positive_mode: str = "all_views",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the positive mask used by DomainAwareSupConLoss.

    The returned mask still includes self-pairs when they are implied by
    same-sample positives; masked_contrastive_loss removes self-pairs before
    computing the loss. Diagnostics should do the same.
    """
    if features.dim() == 2:
        batch_size, num_views = features.size(0), 1
    elif features.dim() == 3:
        batch_size, num_views = features.size(0), features.size(1)
    else:
        raise ValueError(
            f"Expected features with shape [batch, views, dim] or [batch, dim], received {features.shape}."
        )
    if positive_mode not in DOMAIN_AWARE_SUPCON_POSITIVE_MODES:
        raise ValueError(
            f"Unsupported SupCon positive mode '{positive_mode}'. "
            f"Choose from {sorted(DOMAIN_AWARE_SUPCON_POSITIVE_MODES)}."
        )

    flat_features, flat_labels, flat_domains, flat_sample_ids = _flatten_views(features, labels, domains, sample_ids)
    if flat_labels is None:
        raise ValueError("labels are required for supervised contrastive loss.")

    same_class = flat_labels[:, None].eq(flat_labels[None, :])
    class_positive_mask = same_class
    if flat_domains is not None and not include_same_domain_same_class:
        same_domain = flat_domains[:, None].eq(flat_domains[None, :])
        class_positive_mask = class_positive_mask & (~same_domain)

    if positive_mode == "all_views":
        positive_mask = class_positive_mask.clone()
    elif positive_mode == "global_class_instance_views":
        view_ids = torch.arange(num_views, device=flat_features.device).repeat(batch_size)
        is_global = view_ids == 0
        global_global = is_global[:, None] & is_global[None, :]
        positive_mask = class_positive_mask & global_global
    else:
        raise AssertionError(f"Unhandled SupCon positive mode: {positive_mode}")

    if flat_sample_ids is not None:
        same_sample = flat_sample_ids[:, None].eq(flat_sample_ids[None, :])
    else:
        implicit_sample_ids = torch.arange(batch_size, device=flat_features.device).repeat_interleave(num_views)
        same_sample = implicit_sample_ids[:, None].eq(implicit_sample_ids[None, :])
    positive_mask = positive_mask | same_sample
    return flat_features, positive_mask


def supcon_positive_pair_diagnostics(
    features: torch.Tensor,
    labels: torch.Tensor,
    domains: Optional[torch.Tensor] = None,
    sample_ids: Optional[torch.Tensor] = None,
    include_same_domain_same_class: bool = True,
    positive_mode: str = "all_views",
) -> Dict[str, float]:
    """Return logging-only positive-pair availability metrics for pairwise SupCon."""
    flat_features, positive_mask = domain_aware_supcon_positive_mask(
        features,
        labels=labels,
        domains=domains,
        sample_ids=sample_ids,
        include_same_domain_same_class=include_same_domain_same_class,
        positive_mode=positive_mode,
    )
    num_anchors = int(flat_features.size(0))
    if num_anchors == 0:
        return {
            "supcon_num_anchors": 0.0,
            "supcon_mean_positives_per_anchor": 0.0,
            "supcon_min_positives_per_anchor": 0.0,
            "supcon_frac_anchors_with_positive": 0.0,
            "supcon_num_valid_positive_pairs": 0.0,
        }

    self_mask = torch.eye(num_anchors, device=flat_features.device, dtype=torch.bool)
    positive_mask = positive_mask & (~self_mask)
    positive_counts = positive_mask.sum(dim=1)
    anchors_with_positive = positive_counts > 0
    return {
        "supcon_num_anchors": float(num_anchors),
        "supcon_mean_positives_per_anchor": float(positive_counts.float().mean().item()),
        "supcon_min_positives_per_anchor": float(positive_counts.min().item()),
        "supcon_frac_anchors_with_positive": float(anchors_with_positive.float().mean().item()),
        "supcon_num_valid_positive_pairs": float(positive_counts.sum().item()),
    }


class DomainAwareSupConLoss(nn.Module):
    VALID_POSITIVE_MODES = DOMAIN_AWARE_SUPCON_POSITIVE_MODES

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
        flat_features, positive_mask = domain_aware_supcon_positive_mask(
            features,
            labels=labels,
            domains=domains,
            sample_ids=sample_ids,
            include_same_domain_same_class=self.include_same_domain_same_class,
            positive_mode=self.positive_mode,
        )
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
        self.prototype_pair_margins: List[Dict[str, float]] = [
            dict(spec) for spec in (prototype_pair_margins or [])
        ]
        self.last_pair_margin_loss = 0.0
        self.last_pair_margin_active_anchors = 0

    def set_pair_margins(self, prototype_pair_margins: Sequence[Mapping[str, float]]) -> None:
        self.prototype_pair_margins = [dict(spec) for spec in prototype_pair_margins]

    def get_pair_margins(self) -> List[Dict[str, float]]:
        return [dict(spec) for spec in self.prototype_pair_margins]

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

        self.last_pair_margin_loss = 0.0
        self.last_pair_margin_active_anchors = 0

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
                hinge_values = torch.cat(collected)
                margin_term = hinge_values.mean()
                self.last_pair_margin_loss = float(margin_term.detach().cpu().item())
                self.last_pair_margin_active_anchors = int(hinge_values.numel())

        return ce_loss + self.lambda_pair_margin * margin_term


class AdaptiveKMarginLoss(nn.Module):
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        cached_prototypes: Optional[torch.Tensor],
        selected_pairs: Optional[torch.Tensor],
        pair_weights: Optional[torch.Tensor],
        pair_margins: Optional[torch.Tensor],
        prototype_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cached_prototypes is None or selected_pairs is None or pair_weights is None or pair_margins is None:
            return embeddings.new_zeros(())
        if embeddings.numel() == 0 or labels.numel() == 0:
            return embeddings.new_zeros(())

        prototypes = F.normalize(cached_prototypes.detach().to(embeddings.device), dim=-1)
        selected = selected_pairs.detach().to(embeddings.device)
        weights = pair_weights.detach().to(embeddings.device)
        margins = pair_margins.detach().to(embeddings.device)
        if prototype_valid_mask is None:
            prototype_valid = torch.ones(prototypes.size(0), device=embeddings.device, dtype=torch.bool)
        else:
            prototype_valid = prototype_valid_mask.detach().to(embeddings.device)

        if prototypes.dim() != 2 or prototypes.size(1) != embeddings.size(1):
            return embeddings.new_zeros(())
        if selected.shape != weights.shape or selected.shape != margins.shape:
            return embeddings.new_zeros(())
        if selected.dim() != 2 or selected.size(0) != prototypes.size(0) or selected.size(1) != prototypes.size(0):
            return embeddings.new_zeros(())

        embeddings = F.normalize(embeddings, dim=-1)
        sims = embeddings @ prototypes.T
        batch_losses: List[torch.Tensor] = []
        num_classes = prototypes.size(0)

        for class_id in labels.unique().tolist():
            true_class = int(class_id)
            if not (0 <= true_class < num_classes) or not bool(prototype_valid[true_class].item()):
                continue

            pair_mask = (selected[true_class] & prototype_valid).clone()
            pair_mask[true_class] = False
            if not pair_mask.any():
                continue

            sample_mask = labels == true_class
            true_scores = sims[sample_mask, true_class].unsqueeze(1)
            neg_scores = sims[sample_mask][:, pair_mask]
            hinge = F.relu(margins[true_class, pair_mask].unsqueeze(0) - (true_scores - neg_scores))
            weighted = hinge * weights[true_class, pair_mask].unsqueeze(0)
            batch_losses.append(weighted.sum(dim=1))

        if not batch_losses:
            return embeddings.new_zeros(())
        return torch.cat(batch_losses, dim=0).mean()
