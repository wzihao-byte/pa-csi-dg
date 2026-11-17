"""Nine-class models bound to the audited H/H-S implementation snapshot."""
from __future__ import annotations

import copy

import torch
from torch.nn import functional as F

from .frozen_models.tpp_models import IsolatedPaCsiDGLiteTPP
from .frozen_models.control_components import (
    ControlDomainAwareSupConLoss, initialization_audit, install_capacity_matched_global,
)

MODEL_CONFIG = {
    "time_downsample": 2, "max_positions": 500, "position_kernels": 10,
    "mcat_layers": 5, "mcat_heads": 9, "temporal_kernel_num": 128,
    "temporal_kernel_sizes": [20, 40], "channel_group_width": 30,
    "channel_encoder_layers": 1, "channel_encoder_heads": 6,
    "channel_kernel_num": 16, "channel_kernel_sizes": [2, 4],
    "fusion_dim": 256, "projection_hidden_dim": 256, "projection_dim": 128,
    "dropout": 0.1, "antenna_layout": {"mode": "tx", "num_rx": 1, "num_tx": 3, "num_subcarriers": 30},
    "tpp_levels": [1, 2, 4, 8], "tpp_pool_type": "max",
    "tpp_preserve_global": True, "tpp_pyramid_scale": 0.25,
}
ARMS = {
    "W00": {"pool": "global_mlp", "lambda_supcon": 0.0, "masked_forward": False, "coral": False},
    "W01": {"pool": "global_mlp", "lambda_supcon": 0.05, "masked_forward": True, "coral": False},
    "W10": {"pool": "tpp", "lambda_supcon": 0.0, "masked_forward": False, "coral": False},
    "W11": {"pool": "tpp", "lambda_supcon": 0.05, "masked_forward": True, "coral": False},
    "WC": {"pool": "global_mlp", "lambda_supcon": 0.0, "masked_forward": False, "coral": True},
}
EXPECTED_PARAMETERS = 4_787_065


def make_model(pool_mode, seed):
    if pool_mode not in {"tpp", "global_mlp"}:
        raise ValueError("Unsupported pool mode")
    torch.manual_seed(seed)
    model = IsolatedPaCsiDGLiteTPP(90, 9, copy.deepcopy(MODEL_CONFIG))
    if pool_mode == "global_mlp":
        install_capacity_matched_global(model)
    count = sum(p.numel() for p in model.parameters())
    if count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Nine-class model capacity drift: {count}")
    return model


def source_coral(features, domains):
    """Mean plus unbiased-covariance difference; joint source forward features."""
    unique = domains.unique(sorted=True)
    if len(unique) != 2:
        raise ValueError("CORAL requires two source domains")
    x, y = [features[domains == domain] for domain in unique]
    if min(len(x), len(y)) < 2:
        raise ValueError("Unbiased covariance requires >=2 observations per source")
    mx, my = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
    cx, cy = x - mx, y - my
    covx, covy = cx.T @ cx / (len(x) - 1), cy.T @ cy / (len(y) - 1)
    return (mx - my).square().mean() + (covx - covy).square().mean()


def objective(model, batch, arm, gamma=0.0, *, masked_forward_exposure=False):
    """Full batch first, then one sample-major auxiliary batch of 3B.

    An explicit exposure flag executes auxiliaries even at lambda_supcon=0.
    This API supports the exposure mechanism; MultiEnv ME still needs its own
    identity-bound adapter and is not registered as a WiMANS formal arm.
    """
    settings = ARMS[arm]
    amp, phase, labels, domains, sample_ids = batch
    full = model(amp, phase)
    ce = F.cross_entropy(full["logits"], labels)
    sc = ce.new_zeros(())
    auxiliary = settings["masked_forward"] or masked_forward_exposure
    if auxiliary:
        views = model.encode_antenna_views(amp, phase)
        projections = torch.cat((full["projection"].unsqueeze(1), views["projection_views"]), dim=1)
        if settings["lambda_supcon"] > 0:
            sc = ControlDomainAwareSupConLoss(temperature=0.2, include_same_domain_same_class=True,
                                             positive_mode="global_class_instance_views")(
                                                 projections, labels, domains, sample_ids)
    coral = source_coral(full["fused_feature"], domains) if settings["coral"] else ce.new_zeros(())
    loss = ce + settings["lambda_supcon"] * sc + gamma * coral
    return loss, {"ce": float(ce.detach()), "supcon": float(sc.detach()),
                  "coral": float(coral.detach()), "masked_forward_executed": bool(auxiliary)}


def paired_initialization_audit(seed):
    tpp, global_model = make_model("tpp", seed), make_model("global_mlp", seed)
    return initialization_audit(global_model, tpp)
