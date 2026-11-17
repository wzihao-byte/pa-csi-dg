"""Generic domain-generalization baselines on the frozen dual-stream H model.

Two objectives are added to complete-input cross-entropy training, without
masked views or contrastive terms, on two backbones of the H-S four-arm study:
the capacity-matched global-pooling ERM arm (G10; arms GC10 / GX10) and the
TPP + CE arm (historical HS10; arms C10 / X10):

* ``coral``: DomainBed's CORAL penalty (squared mean and covariance differences)
  between the two source environments of each joint batch, measured on the fused
  256-d feature that feeds the classifier, with the DomainBed default weight 1.0.
* ``mixup``: DomainBed-style inter-domain Mixup with Beta(0.2, 0.2) coefficients.
  Two directed source-to-source mixtures (16 + 16) replace the batch; amplitude
  and phase are mixed with the same coefficients and the same pairing, and the
  cross-entropy is the coefficient-weighted mixture of the two label terms.

``baseline_epoch`` replays the frozen trainer's cross-entropy training loop
(forward, loss, backward, optimizer and per-step scheduler) so that the ``ce``
mode is bitwise identical to the historical ``run_epoch``; QA checks this on
real source observations. Evaluation passes are never replaced.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

METHODS = ("ce", "coral", "mixup")
FORMAL_METHODS = ("coral", "mixup")
RNG_NAMESPACE = 20261105
CORAL_GAMMA = 1.0
MIXUP_ALPHA = 0.2
CORAL_FEATURE = "fused_feature"
ARM_METHOD = {"GC10": "coral", "GX10": "mixup", "C10": "coral", "X10": "mixup"}
ARM_POOL = {"GC10": "global_mlp", "GX10": "global_mlp", "C10": "tpp", "X10": "tpp"}


def arm_specification(arm: str) -> Dict[str, Any]:
    method = ARM_METHOD[arm]
    if method == "coral":
        return {"method": "coral", "coral_gamma": CORAL_GAMMA, "feature": CORAL_FEATURE,
                "reference": "DomainBed CORAL (mmd_gamma default 1.0), joint-batch source statistics"}
    return {"method": "mixup", "mixup_alpha": MIXUP_ALPHA, "rng_namespace": RNG_NAMESPACE,
            "mix_inputs": ["amplitude", "phase"], "directed_mixtures": 2, "forward_batch": "concatenated 32",
            "reference": "DomainBed Mixup (mixup_alpha default 0.2), step-keyed independent RNG"}


def validate_specification(spec: Mapping[str, Any], arm: str) -> None:
    expected = arm_specification(arm)
    if dict(spec) != expected:
        raise ValueError(f"dg_baseline specification drift for {arm}: {dict(spec)} != {expected}")


def source_coral(features: torch.Tensor, environments: torch.Tensor) -> torch.Tensor:
    """DomainBed CORAL penalty between the two source environments of one batch."""
    if features.dim() != 2:
        raise ValueError("CORAL expects [batch, dim] features")
    unique = environments.unique(sorted=True)
    if len(unique) != 2:
        raise ValueError("The control expects exactly two source environments per batch")
    x, y = [features[environments == env] for env in unique]
    if min(len(x), len(y)) < 2:
        raise ValueError("Sample covariance requires at least two examples per source")
    mean_x, mean_y = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
    centered_x, centered_y = x - mean_x, y - mean_y
    covariance_x = centered_x.t() @ centered_x / (len(x) - 1)
    covariance_y = centered_y.t() @ centered_y / (len(y) - 1)
    return (mean_x - mean_y).pow(2).mean() + (covariance_x - covariance_y).pow(2).mean()


def mixup_plan(environments: torch.Tensor, seed: int, step: int, alpha: float = MIXUP_ALPHA,
               namespace: int = RNG_NAMESPACE) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (first indices, second indices, coefficients) for two directed cross-domain mixtures."""
    unique = environments.unique(sorted=True)
    if len(unique) != 2:
        raise ValueError("Need exactly two source domains")
    left, right = [torch.where(environments == env)[0] for env in unique]
    if len(left) != len(right):
        raise ValueError("Use the frozen source-balanced batches")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(step), int(namespace)]))
    right = right[torch.as_tensor(rng.permutation(len(right)), device=right.device)]
    weights = rng.beta(alpha, alpha, size=2)
    first = torch.cat([left, right])
    second = torch.cat([right, left])
    coefficients = torch.cat([
        torch.full((len(left),), float(weights[0]), dtype=torch.float32, device=environments.device),
        torch.full((len(right),), float(weights[1]), dtype=torch.float32, device=environments.device),
    ])
    return first, second, coefficients


def mix_inputs(tensor: torch.Tensor, first: torch.Tensor, second: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    shape = (-1,) + (1,) * (tensor.dim() - 1)
    weight = coefficients.to(tensor.dtype).reshape(shape)
    return weight * tensor[first] + (1 - weight) * tensor[second]


def mixed_cross_entropy(logits: torch.Tensor, labels_a: torch.Tensor, labels_b: torch.Tensor,
                        coefficients: torch.Tensor) -> torch.Tensor:
    coefficients = coefficients.to(logits.dtype)
    return (coefficients * F.cross_entropy(logits, labels_a, reduction="none")
            + (1 - coefficients) * F.cross_entropy(logits, labels_b, reduction="none")).mean()


def baseline_epoch(trainer, model, loader, optimizer, scheduler, scheduler_step_mode, device,
                   loss_config: Mapping[str, Any], *, spec: Mapping[str, Any], seed: int,
                   step_counter: Mapping[str, int]) -> Dict[str, Any]:
    """Cross-entropy training epoch of the frozen trainer with an optional DG objective.

    ``trainer`` is the bound legacy ``train_dg`` module (metric helpers only).
    ``step_counter['steps']`` is the number of optimizer updates completed so far;
    it indexes the Mixup RNG before each update.
    """
    method = str(spec["method"])
    if method not in METHODS:
        raise ValueError(f"Unsupported baseline method {method!r}")
    model.train()
    lambda_supcon = float(loss_config.get("lambda_supcon", 0.0))
    if lambda_supcon != 0.0:
        raise ValueError("DG baselines are cross-entropy arms; the contrastive coefficient must be zero")
    lambda_supcon_target = float(loss_config.get("lambda_supcon_target", lambda_supcon))
    lambda_supcon_warmup_epochs = int(loss_config.get("lambda_supcon_warmup_epochs", 0))
    grad_clip_norm = float(loss_config.get("grad_clip_norm", 0.0))
    total_loss = total_ce = total_regularizer = 0.0
    weight = 0.0
    coefficient_sum = 0.0
    coefficient_count = 0
    all_predictions, all_labels = [], []
    shape_debug: Optional[Dict[str, Any]] = None
    for amplitude, phase, labels, domains, _sample_ids in loader:
        amplitude = amplitude.to(device)
        phase = phase.to(device)
        labels = labels.to(device)
        domains = domains.to(device)
        with torch.set_grad_enabled(True):
            if method == "mixup":
                first, second, coefficients = mixup_plan(domains, seed, int(step_counter["steps"]),
                                                        float(spec["mixup_alpha"]), int(spec["rng_namespace"]))
                outputs = model(mix_inputs(amplitude, first, second, coefficients),
                                mix_inputs(phase, first, second, coefficients))
                ce_loss = mixed_cross_entropy(outputs["logits"], labels[first], labels[second], coefficients)
                regularizer = ce_loss.new_zeros(())
                total_batch_loss = ce_loss
                report_labels = labels[first]
                coefficient_sum += float(coefficients.sum().item())
                coefficient_count += int(coefficients.numel())
            else:
                outputs = model(amplitude, phase)
                ce_loss = F.cross_entropy(outputs["logits"], labels)
                report_labels = labels
                if method == "coral":
                    weight = float(spec["coral_gamma"])
                    regularizer = source_coral(outputs[str(spec["feature"])], domains)
                    total_batch_loss = ce_loss + weight * regularizer
                else:
                    regularizer = ce_loss.new_zeros(())
                    total_batch_loss = ce_loss
            if not torch.isfinite(total_batch_loss):
                raise FloatingPointError("Encountered a non-finite loss value.")
            optimizer.zero_grad(set_to_none=True)
            total_batch_loss.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            if scheduler is not None and scheduler_step_mode == "step":
                scheduler.step()
        predictions = outputs["logits"].argmax(dim=1)
        all_predictions.append(predictions.detach().cpu().numpy())
        all_labels.append(report_labels.detach().cpu().numpy())
        total_loss += float(total_batch_loss.item())
        total_ce += float(ce_loss.item())
        total_regularizer += float(regularizer.item())
        if shape_debug is None:
            shape_debug = {
                "amplitude": list(amplitude.shape),
                "phase": list(phase.shape),
                "fused_feature": list(outputs["fused_feature"].shape),
                "projection": list(outputs["projection"].shape),
            }
    if not all_labels:
        raise RuntimeError("Empty training loader")
    mean_ce = total_ce / max(1, len(loader))
    labels_np = np.concatenate(all_labels, axis=0)
    predictions_np = np.concatenate(all_predictions, axis=0)
    metrics = trainer.metric_dict(labels_np, predictions_np)
    metrics.update({
        "loss_total": total_loss / max(1, len(loader)),
        "loss_ce": mean_ce,
        "loss_supcon": 0.0,
        "loss_supcon_weighted": 0.0,
        "loss_supcon_to_ce": 0.0,
        "lambda_supcon_target": lambda_supcon_target,
        "lambda_supcon_effective": lambda_supcon,
        "lambda_supcon_warmup_epochs": lambda_supcon_warmup_epochs,
        "effective_pair_margin_weight": 0.0,
        **trainer.zero_supcon_diagnostics(),
        "loss_pair_margin": 0.0,
        "akm_loss": 0.0,
        "labels": labels_np.tolist(),
        "predictions": predictions_np.tolist(),
        "shape_debug": shape_debug or {},
    })
    if method != "ce":
        metrics.update({
            "loss_dg_regularizer": total_regularizer / max(1, len(loader)),
            "dg_regularizer_weight": weight,
            "dg_method_is_coral": 1.0 if method == "coral" else 0.0,
            "dg_method_is_mixup": 1.0 if method == "mixup" else 0.0,
            "mixup_mean_coefficient": (coefficient_sum / coefficient_count) if coefficient_count else 0.0,
        })
    return metrics
