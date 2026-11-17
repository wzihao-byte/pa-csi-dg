"""Audited extensions of the frozen dual-stream H L1248 model.

The caller supplies the bound H model class. No mutable workspace model or
dataset is imported here. Construct all common layers before replacing the
four pooling projections, preserving paired initialization and CPU RNG state.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Optional, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F


POOL_PATHS = (
    "amp_encoder.temporal_pool",
    "amp_encoder.channel_pool",
    "phase_encoder.temporal_pool",
    "phase_encoder.channel_pool",
)
POOL_PROJECTION_PREFIXES = tuple(path + ".projection." for path in POOL_PATHS)
POOL_MODES = frozenset({"tpp", "global_mlp"})
POSITIVE_MODE_ALIASES = {
    "hybrid": "hybrid",
    "global_class_instance_views": "hybrid",
    "all_views": "all_views",
    "all_view_sameclass": "all_views",
    "hybrid_remove_other_sameclass_denominator": "hybrid_remove_other_sameclass_denominator",
}
DOMAIN_AWARE_SUPCON_POSITIVE_MODES = frozenset(POSITIVE_MODE_ALIASES)


def canonical_positive_mode(mode: str) -> str:
    try:
        return POSITIVE_MODE_ALIASES[mode]
    except KeyError as error:
        raise ValueError(
            f"Unsupported H positive mode {mode!r}; choose from {sorted(POSITIVE_MODE_ALIASES)}"
        ) from error


class CapacityMatchedGlobalMlpPool(nn.Module):
    """Global max plus d -> 15d/2 -> d correction on one frozen H path.

    Linear(15d,d,bias=True) and the MLP each have 15d^2+d parameters.
    Convolutions, ReLU and terminal dropout are retained by reference.
    The MLP contains no hidden dropout or normalization.
    """

    def __init__(self, reference: nn.Module) -> None:
        super().__init__()
        if list(reference.levels) != [1, 2, 4, 8] or reference.segment_count != 15:
            raise ValueError("Capacity control requires the frozen H L1248 / 15-bin pool")
        if reference.pool_type != "max" or not reference.preserve_global:
            raise ValueError("Capacity control requires residual global-max TPP")
        if reference.pyramid_scale != 0.25 or reference.dropout.p != 0.1:
            raise ValueError("Capacity control requires H residual scale 0.25 and dropout 0.1")
        d = int(reference.output_dim)
        if d not in (256, 32):
            raise ValueError(f"Expected H temporal/context widths 256/32, received {d}")
        if reference.projection.in_features != 15 * d or reference.projection.out_features != d:
            raise ValueError("Reference TPP projection dimensions disagree with 15 bins")
        self.convs = reference.convs
        self.activation = reference.activation
        self.dropout = reference.dropout
        self.output_dim = d
        self.pyramid_scale = reference.pyramid_scale
        self.hidden_dim = 15 * d // 2
        self.projection = nn.Sequential(
            nn.Linear(d, self.hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, d, bias=True),
        )
        expected = sum(parameter.numel() for parameter in reference.projection.parameters())
        actual = sum(parameter.numel() for parameter in self.projection.parameters())
        if actual != expected:
            raise ValueError(f"Projection capacity differs: global MLP {actual}, frozen TPP {expected}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        # Retain the frozen adaptive-max kernel, including tie/backward semantics.
        global_feature = torch.cat([
            F.adaptive_max_pool1d(self.activation(conv(x_t)), output_size=1).flatten(start_dim=1)
            for conv in self.convs
        ], dim=1)
        return self.dropout(global_feature + self.pyramid_scale * self.projection(global_feature))


def install_capacity_matched_global(model: nn.Module) -> nn.Module:
    """Replace exactly four corrections after full H construction on CPU.

    Restore CPU RNG after drawing MLP weights so later paired stochastic
    operations start from the same state as TPP. Shared layers are never
    reinitialized. The caller then moves the complete model to its device.
    """
    references = []
    for path in POOL_PATHS:
        parent_path, attribute = path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        reference = getattr(parent, attribute)
        if any(parameter.device.type != "cpu" for parameter in reference.parameters()):
            raise ValueError("Install the capacity control on CPU before moving the H model")
        references.append((parent, attribute, reference))
    # Construct every replacement before mutation; failed validation leaves H intact.
    with torch.random.fork_rng(devices=[]):
        replacements = [CapacityMatchedGlobalMlpPool(reference) for _, _, reference in references]
    for (parent, attribute, _), replacement in zip(references, replacements):
        setattr(parent, attribute, replacement)
    return model


def make_model_class(frozen_tpp_class: Type[nn.Module], pool_mode: str = "tpp") -> Type[nn.Module]:
    """Factory compatible with the historical trainer's model_class hook."""
    if pool_mode not in POOL_MODES:
        raise ValueError(f"Unsupported H pool mode {pool_mode!r}; choose from {sorted(POOL_MODES)}")
    if pool_mode == "tpp":
        return frozen_tpp_class

    class CapacityMatchedH(frozen_tpp_class):
        def __init__(self, input_dim, num_classes, model_config) -> None:
            super().__init__(input_dim, num_classes, model_config)
            install_capacity_matched_global(self)

    CapacityMatchedH.__name__ = "CapacityMatchedH"
    return CapacityMatchedH


def _state_hash(state, names) -> str:
    # Same representation as the bound H-S state_hash: name/dtype/shape/bytes.
    digest = hashlib.sha256()
    for name in names:
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def shared_state_hash(model: nn.Module) -> str:
    state = model.state_dict()
    return _state_hash(state, [name for name in state if not name.startswith(POOL_PROJECTION_PREFIXES)])


def initialization_audit(model: nn.Module, tpp_reference: nn.Module) -> Dict[str, object]:
    """JSON-ready exact shared-tensor and full/common state fingerprints."""
    candidate = model.state_dict()
    reference = tpp_reference.state_dict()
    names = [name for name in reference if not name.startswith(POOL_PROJECTION_PREFIXES)]
    candidate_names = [name for name in candidate if not name.startswith(POOL_PROJECTION_PREFIXES)]
    missing = sorted(set(names) - set(candidate_names))
    extra = sorted(set(candidate_names) - set(names))
    unequal = [name for name in names if name not in candidate or not torch.equal(candidate[name], reference[name])]
    return {
        "shared_state_tensors_exactly_equal": not (missing or extra or unequal),
        "shared_state_tensor_count": len(names),
        "missing_shared_names": missing,
        "extra_shared_names": extra,
        "unequal_shared_names": unequal,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "reference_parameter_count": sum(parameter.numel() for parameter in tpp_reference.parameters()),
        "model_full_state_sha256": _state_hash(candidate, candidate),
        "reference_full_state_sha256": _state_hash(reference, reference),
        "model_common_state_sha256": _state_hash(candidate, candidate_names),
        "reference_common_state_sha256": _state_hash(reference, names),
        "excluded_prefixes": list(POOL_PROJECTION_PREFIXES),
    }


def control_pair_masks(
    features: torch.Tensor,
    labels: torch.Tensor,
    domains: Optional[torch.Tensor] = None,
    sample_ids: Optional[torch.Tensor] = None,
    include_same_domain_same_class: bool = True,
    positive_mode: str = "global_class_instance_views",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sample-major features, nonself positives and denominator pairs.

    View zero is complete input. Other views are masked pseudo-Tx feature
    groups. Original record identity, not draw position, defines instance
    positives, including repeated records at the same view. The denominator
    control removes only same-class pairs not selected by the hybrid rule;
    every selected positive remains in its anchor's denominator.
    """
    if features.dim() == 2:
        features = features.unsqueeze(1)
    if features.dim() != 3:
        raise ValueError(f"Expected [batch, views, dim] or [batch, dim], received {features.shape}")
    batch_size, num_views, feature_dim = features.shape
    if num_views < 1:
        raise ValueError("At least one feature view is required")
    for name, target in (("labels", labels), ("domains", domains), ("sample_ids", sample_ids)):
        if target is not None and (target.dim() != 1 or target.size(0) != batch_size):
            raise ValueError(f"{name} must have shape [{batch_size}]")
        if target is not None and target.device != features.device:
            raise ValueError(f"{name} and features must be on the same device")
    if labels is None:
        raise ValueError("labels are required for supervised contrastive loss")
    mode = canonical_positive_mode(positive_mode)
    flat_features = features.reshape(batch_size * num_views, feature_dim)
    flat_labels = labels.repeat_interleave(num_views)
    flat_ids = (
        sample_ids if sample_ids is not None else torch.arange(batch_size, device=features.device)
    ).repeat_interleave(num_views)
    same_class = flat_labels[:, None].eq(flat_labels[None, :])
    class_positives = same_class
    if domains is not None and not include_same_domain_same_class:
        flat_domains = domains.repeat_interleave(num_views)
        class_positives = class_positives & ~flat_domains[:, None].eq(flat_domains[None, :])
    same_sample = flat_ids[:, None].eq(flat_ids[None, :])
    if mode == "all_views":
        positives = same_sample | class_positives
    else:
        is_full = torch.arange(num_views, device=features.device).repeat(batch_size).eq(0)
        positives = same_sample | (class_positives & is_full[:, None] & is_full[None, :])
    nonself = ~torch.eye(flat_features.size(0), device=features.device, dtype=torch.bool)
    positives = positives & nonself
    denominator = nonself
    if mode == "hybrid_remove_other_sameclass_denominator":
        denominator = nonself & (~same_class | positives)
    return flat_features, positives, denominator


def domain_aware_supcon_positive_mask(*args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Diagnostic-compatible wrapper; self pairs are already removed."""
    flat_features, positives, _ = control_pair_masks(*args, **kwargs)
    return flat_features, positives


def masked_contrastive_loss(
    features: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float,
    denominator_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Historical H numerical reduction with an explicit denominator mask."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    n = features.size(0)
    if positive_mask.shape != (n, n) or positive_mask.dtype != torch.bool:
        raise ValueError("positive_mask must be a square boolean mask matching features")
    nonself = ~torch.eye(n, device=features.device, dtype=torch.bool)
    positives = positive_mask & nonself
    denominator = nonself if denominator_mask is None else denominator_mask & nonself
    if denominator.shape != (n, n) or denominator.dtype != torch.bool:
        raise ValueError("denominator_mask must be a square boolean mask matching features")
    if (positives & ~denominator).any():
        raise ValueError("Every positive must remain in its anchor's denominator")
    if n < 2:
        return features.new_zeros(())
    normalized = F.normalize(features, dim=-1)
    similarity = torch.matmul(normalized, normalized.T) / temperature
    logits = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * denominator
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    positive_count = positives.sum(dim=1)
    valid_rows = positive_count > 0
    if not valid_rows.any():
        return features.new_zeros(())
    mean_log_prob_pos = (positives[valid_rows] * log_prob[valid_rows]).sum(dim=1) / positive_count[valid_rows]
    return -mean_log_prob_pos.mean()


class ControlDomainAwareSupConLoss(nn.Module):
    """Drop-in replacement for the bound H training module's loss symbol."""
    VALID_POSITIVE_MODES = DOMAIN_AWARE_SUPCON_POSITIVE_MODES

    def __init__(self, temperature: float = 0.2, include_same_domain_same_class: bool = True,
                 positive_mode: str = "global_class_instance_views") -> None:
        super().__init__()
        canonical_positive_mode(positive_mode)
        self.temperature = temperature
        self.include_same_domain_same_class = include_same_domain_same_class
        self.positive_mode = positive_mode

    def forward(self, features: torch.Tensor, labels: torch.Tensor,
                domains: Optional[torch.Tensor] = None,
                sample_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        flat, positives, denominator = control_pair_masks(
            features, labels, domains, sample_ids,
            self.include_same_domain_same_class, self.positive_mode,
        )
        return masked_contrastive_loss(flat, positives, self.temperature, denominator)


DomainAwareSupConLoss = ControlDomainAwareSupConLoss


def supcon_positive_pair_diagnostics(
    features: torch.Tensor, labels: torch.Tensor, domains: Optional[torch.Tensor] = None,
    sample_ids: Optional[torch.Tensor] = None, include_same_domain_same_class: bool = True,
    positive_mode: str = "global_class_instance_views",
) -> Dict[str, float]:
    flat, positives, denominator = control_pair_masks(
        features, labels, domains, sample_ids, include_same_domain_same_class, positive_mode,
    )
    count = positives.sum(dim=1)
    return {
        "supcon_num_anchors": float(flat.size(0)),
        "supcon_mean_positives_per_anchor": float(count.float().mean().item()) if count.numel() else 0.0,
        "supcon_min_positives_per_anchor": float(count.min().item()) if count.numel() else 0.0,
        "supcon_frac_anchors_with_positive": float((count > 0).float().mean().item()) if count.numel() else 0.0,
        "supcon_num_valid_positive_pairs": float(count.sum().item()),
        "supcon_num_denominator_pairs": float(denominator.sum().item()),
        "supcon_mean_denominator_per_anchor": float(denominator.sum(dim=1).float().mean().item()) if count.numel() else 0.0,
    }
