"""Explicit components for later single-factor and DG controls, not core R v1.

The running core bundle imports its own immutable snapshot. These controls need
their own frozen manifests and complete training before any efficacy claim.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dg_models import AntennaViewBuilder
from dg_losses import domain_aware_supcon_positive_mask, masked_contrastive_loss
from experiments.revision_validation.models import build_model


PSEUDO_TX_LAYOUT = {"mode": "tx", "num_rx": 1, "num_tx": 3, "num_subcarriers": 30}


class AttentionResidualPool(nn.Module):
    """Per-convolution attention followed by a parameter-matched conventional MLP.

    Each convolution has its own temporal length. Its bias-free scalar scorer
    uses c parameters; summed over branches these use d parameters. The two
    bias-free MLP matrices use 7d^2, matching the reference TPP's 7d^2+d exactly.
    Additional time-dependent scorer MACs are real and must be reported.
    """
    def __init__(self, reference):
        super().__init__()
        self.convs, self.activation, self.dropout = reference.convs, reference.activation, reference.dropout
        self.output_dim, self.pyramid_scale = reference.output_dim, reference.pyramid_scale
        d = self.output_dim
        self.scorers = nn.ModuleList([nn.Linear(conv.out_channels, 1, bias=False) for conv in self.convs])
        self.projection = nn.Sequential(nn.Linear(d, 7 * d // 2, bias=False), nn.ReLU(),
                                        nn.Linear(7 * d // 2, d, bias=False))
        assert sum(p.numel() for p in self.scorers.parameters()) + sum(p.numel() for p in self.projection.parameters()) == sum(p.numel() for p in reference.projection.parameters())

    def forward(self, x):
        pooled, global_features = [], []
        for conv, scorer in zip(self.convs, self.scorers):
            response = self.activation(conv(x.transpose(1, 2)))
            weight = scorer(response.transpose(1, 2)).softmax(dim=1).transpose(1, 2)
            pooled.append((response * weight).sum(dim=-1))
            global_features.append(response.amax(dim=-1))
        return self.dropout(torch.cat(global_features, dim=1) +
                            self.pyramid_scale * self.projection(torch.cat(pooled, dim=1)))


def build_pool_control(kind, seed):
    if kind not in ("p1_only", "without_p1", "attention"):
        raise ValueError(kind)
    model = build_model("R10", seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 60000)
        for name in ("temporal_pool", "channel_pool"):
            pool = getattr(model.encoder, name)
            if kind == "attention":
                setattr(model.encoder, name, AttentionResidualPool(pool))
            else:
                pool.levels = [1] if kind == "p1_only" else [2, 4]
                pool.segment_count = sum(pool.levels)
                pool.pyramid_dim = pool.output_dim * pool.segment_count
                pool.projection = nn.Linear(pool.pyramid_dim, pool.output_dim)
    return model


def pseudo_tx_views(amplitude):
    """Keep the historical three contiguous 30-column masks exactly."""
    builder = AntennaViewBuilder(90, PSEUDO_TX_LAYOUT)
    auxiliary = builder(amplitude)
    return torch.cat((amplitude.unsqueeze(1), auxiliary), dim=1)


def four_view_projection(model, amplitude):
    """Use the historical global/auxiliary forward-call separation.

    All three mask controls share these calls and their BatchNorm exposure. A V1
    comparison changes that exposure as well as views and is reported separately.
    """
    global_output = model(amplitude)
    auxiliary = pseudo_tx_views(amplitude)[:, 1:]
    b, v, t, c = auxiliary.shape
    auxiliary_output = model(auxiliary.reshape(b * v, t, c))
    projections = torch.cat((global_output["projection"].unsqueeze(1),
                             auxiliary_output["projection"].reshape(b, v, -1)), dim=1)
    return global_output, projections


def mask_control_loss(features, labels, mode, temperature=0.2):
    if mode not in ("hybrid", "all_view", "hybrid_remove_same_class_nonpositives"):
        raise ValueError(mode)
    positive_mode = "all_views" if mode == "all_view" else "global_class_instance_views"
    flat, positive = domain_aware_supcon_positive_mask(features, labels, positive_mode=positive_mode)
    self_pairs = torch.eye(len(flat), device=flat.device, dtype=torch.bool)
    positive = positive & ~self_pairs
    allowed = ~self_pairs
    if mode == "hybrid_remove_same_class_nonpositives":
        repeated = labels.repeat_interleave(features.size(1))
        same_class = repeated[:, None] == repeated[None, :]
        allowed = allowed & ~(same_class & ~positive)
        normalized = F.normalize(flat, dim=-1)
        similarity = normalized @ normalized.T / temperature
        logits = similarity - similarity.max(dim=1, keepdim=True).values.detach()
        log_prob = logits - ((logits.exp() * allowed).sum(dim=1, keepdim=True) + 1e-12).log()
        counts = positive.sum(dim=1)
        valid = counts > 0
        loss = -((positive[valid] * log_prob[valid]).sum(dim=1) / counts[valid]).mean()
    else:
        # Reuse the exact existing loss; no numerical rewrite of H's mask loss.
        loss = masked_contrastive_loss(flat, positive, temperature)
    diagnostics = {"positive_directed_pairs": int(positive.sum()), "denominator_directed_pairs": int(allowed.sum()),
                   "anchors": len(flat), "anchors_without_positive": int((positive.sum(dim=1) == 0).sum())}
    return loss, diagnostics, positive, allowed


def source_coral(features, environments):
    """DomainBed mean/covariance penalty, measured on joint-batch features.

    The joint source forward pass deliberately retains the core CE/BatchNorm
    convention. The formula is DomainBed's CORAL penalty, but the entire
    DomainBed training program is not claimed to be reproduced verbatim.
    """
    unique = environments.unique(sorted=True)
    if len(unique) != 2:
        raise ValueError("The control expects two source environments")
    x, y = [features[environments == env] for env in unique]
    if min(len(x), len(y)) < 2:
        raise ValueError("Sample covariance requires at least two examples per source")
    mx, my = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
    cx, cy = x - mx, y - my
    covariance_x, covariance_y = cx.T @ cx / (len(x) - 1), cy.T @ cy / (len(y) - 1)
    return (mx - my).square().mean() + (covariance_x - covariance_y).square().mean()


def cross_domain_mixup(amplitude, labels, environments, seed, step, alpha=0.2):
    """Two directed source-domain mixtures, with an independent step-keyed RNG."""
    unique = environments.unique(sorted=True)
    if len(unique) != 2:
        raise ValueError("Need exactly two source domains")
    indices = [torch.where(environments == env)[0] for env in unique]
    if len(indices[0]) != len(indices[1]):
        raise ValueError("Use the frozen source-balanced batches")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(step), 20260904]))
    left, right = indices
    right = right[torch.as_tensor(rng.permutation(len(right)), device=right.device)]
    weights = rng.beta(alpha, alpha, size=2)
    inputs, labels_a, labels_b, coefficients = [], [], [], []
    for a, b, weight in ((left, right, weights[0]), (right, left, weights[1])):
        inputs.append(float(weight) * amplitude[a] + (1 - float(weight)) * amplitude[b])
        labels_a.append(labels[a]); labels_b.append(labels[b])
        coefficients.append(amplitude.new_full((len(a),), float(weight)))
    return torch.cat(inputs), torch.cat(labels_a), torch.cat(labels_b), torch.cat(coefficients)


def mixed_cross_entropy(logits, labels_a, labels_b, coefficients):
    return (coefficients * F.cross_entropy(logits, labels_a, reduction="none") +
            (1 - coefficients) * F.cross_entropy(logits, labels_b, reduction="none")).mean()


def temporal_intervention(amplitude, sample_ids, kind, seed=0):
    """Predeclared signal perturbations; never choose strength using outcomes."""
    if amplitude.shape[1] != 850 or len(sample_ids) != len(amplitude):
        raise ValueError("Expected aligned 850-frame inputs and identities")
    if kind in ("shift_85", "shift_170"):
        return amplitude.roll(int(kind.split("_")[1]), dims=1)
    if kind not in ("within_85_shuffle", "ten_block_shuffle"):
        raise ValueError(kind)
    output = []
    for item, sample_id in zip(amplitude, sample_ids):
        digest = hashlib.sha256(f"{seed}:{sample_id}:{kind}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        if kind == "within_85_shuffle":
            indices = np.concatenate([85 * block + rng.permutation(85) for block in range(10)])
        else:
            indices = np.concatenate([np.arange(85 * block, 85 * (block + 1)) for block in rng.permutation(10)])
        output.append(item[torch.as_tensor(indices, device=item.device)])
    return torch.stack(output)
