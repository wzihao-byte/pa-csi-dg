"""Model/loss definitions for an explicitly frozen extension study."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from dg_losses import DomainAwareSupConLoss
from experiments.revision_validation.models import MODEL_CONFIG, build_model
from experiments.revision_validation.control_components import (
    build_pool_control, cross_domain_mixup, four_view_projection,
    mask_control_loss, mixed_cross_entropy, source_coral,
)

CONTROL_SPECS = {
    "P1": {"base_arm": "R10", "pool": "p1_only", "objective": "ce"},
    "NO_P1": {"base_arm": "R10", "pool": "without_p1", "objective": "ce"},
    "ATTENTION": {"base_arm": "R10", "pool": "attention", "objective": "ce"},
    "MASK_HYBRID": {"base_arm": "R10", "pool": "unchanged", "objective": "mask", "mask": "hybrid"},
    "MASK_ALL": {"base_arm": "R10", "pool": "unchanged", "objective": "mask", "mask": "all_view"},
    "MASK_REMOVE": {"base_arm": "R10", "pool": "unchanged", "objective": "mask", "mask": "hybrid_remove_same_class_nonpositives"},
    "CORAL": {"base_arm": "R00", "pool": "unchanged", "objective": "coral", "coral_gamma": 1.0},
    "MIXUP": {"base_arm": "R00", "pool": "unchanged", "objective": "mixup", "mixup_alpha": 0.2},
    "TEMPORAL_AUG": {"base_arm": "R10", "pool": "unchanged", "objective": "ce", "max_circular_shift": 85},
}


def build_experiment_model(config):
    if config["model"] != MODEL_CONFIG:
        raise ValueError("Base architecture configuration differs from this implementation")
    spec = config["experiment"]
    if spec.get("model_kind", "single") == "twin":
        if spec["base_arm"] != "R10" or spec["pool"] != "unchanged" or spec["objective"] != "ce" or spec.get("max_circular_shift"):
            raise ValueError("Twin modality controls require the explicit R10 CE specification")
        if spec.get("secondary_input") not in ("amp", "phase"):
            raise ValueError("Specify the second stream's input modality")
        from experiments.revision_validation.modality_models import build_twin_model
        return build_twin_model(config["seed"])
    if spec.get("model_kind", "single") != "single":
        raise ValueError("Unknown revision model kind")
    if spec["pool"] == "unchanged":
        return build_model(spec["base_arm"], config["seed"])
    return build_pool_control(spec["pool"], config["seed"])


def predict(model, inputs, config, *, project=False):
    key = config.get("primary_input", "amp")
    if config["experiment"].get("model_kind") == "twin":
        return model(inputs[key], secondary=inputs[config["experiment"]["secondary_input"]], project=project)
    return model(inputs[key], project=project)


def temporal_augmentation(amplitude, seed, step, maximum_shift):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(step), 901]))
    shifts = torch.as_tensor(rng.integers(-maximum_shift, maximum_shift + 1, size=len(amplitude)), device=amplitude.device)
    positions = (torch.arange(amplitude.shape[1], device=amplitude.device)[None, :] - shifts[:, None]) % amplitude.shape[1]
    return amplitude[torch.arange(len(amplitude), device=amplitude.device)[:, None], positions]


def objective(model, inputs, labels, rows, step, config):
    spec = config["experiment"]
    kind = spec["objective"]
    primary = inputs[config.get("primary_input", "amp")]
    if spec.get("max_circular_shift"):
        primary = temporal_augmentation(primary, config["seed"], step, spec["max_circular_shift"])
    environment_names = sorted({r["environment"] for r in rows})
    environments = torch.tensor([environment_names.index(r["environment"]) for r in rows], device=labels.device)
    diagnostics = {}
    weight = 0.0
    if kind == "mixup":
        mixed, labels_a, labels_b, coefficients = cross_domain_mixup(primary, labels, environments,
            config["seed"], step, alpha=spec["mixup_alpha"])
        values = model(mixed, project=False)
        ce = mixed_cross_entropy(values["logits"], labels_a, labels_b, coefficients)
        regularizer = ce.new_zeros(())
        diagnostics = {"mixup_weight_first_direction": float(coefficients[0]),
                       "mixup_weight_second_direction": float(coefficients[len(coefficients) // 2])}
    elif kind == "mask":
        values, views = four_view_projection(model, primary)
        ce = F.cross_entropy(values["logits"], labels)
        regularizer, diagnostics, _, _ = mask_control_loss(views, labels, spec["mask"], config["temperature"])
        weight = config["lambda_supcon"] * min(1.0, step / config["warmup_steps"])
    elif kind in ("ce", "supcon", "coral"):
        if spec.get("model_kind") == "twin":
            if kind != "ce" or spec.get("max_circular_shift"):
                raise ValueError("Unsupported twin-stream objective/augmentation")
            values = model(primary, secondary=inputs[spec["secondary_input"]], project=False)
        else:
            values = model(primary, project=kind == "supcon")
        ce = F.cross_entropy(values["logits"], labels)
        if kind == "supcon":
            regularizer = DomainAwareSupConLoss(temperature=config["temperature"], positive_mode="all_views")(values["projection"], labels)
            weight = config["lambda_supcon"] * min(1.0, step / config["warmup_steps"])
        elif kind == "coral":
            regularizer = source_coral(values["feature"], environments)
            weight = spec["coral_gamma"]
        else:
            regularizer = ce.new_zeros(())
    else:
        raise ValueError(f"Unknown objective {kind}")
    return ce + weight * regularizer, ce, regularizer, weight, diagnostics


def score_partition(model, dataset, config, part="val"):
    model.eval()
    predictions = []
    confusion = np.zeros((6, 6), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(dataset.manifest[part]), 32):
            inputs, labels, rows = dataset.batch(part, range(start, min(start + 32, len(dataset.manifest[part]))), config["device"])
            logits = predict(model, inputs, config)["logits"]
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite evaluation logits")
            predicted = logits.argmax(dim=1).cpu().tolist()
            for row, actual, value in zip(rows, labels.cpu().tolist(), predicted):
                confusion[actual, value] += 1
                predictions.append({"sample_id": row["sample_id"], "subject_id": row["subject_id"],
                                    "acquisition_id": row["acquisition_id"], "true": actual, "predicted": value})
    diagonal = confusion.diagonal()
    denominator = confusion.sum(0) + confusion.sum(1)
    class_f1 = np.divide(2 * diagonal, denominator, out=np.zeros(6), where=denominator > 0)
    return {"macro_f1": float(class_f1.mean()), "class_f1": class_f1.tolist(),
            "accuracy": float(diagonal.sum() / confusion.sum()), "confusion_matrix": confusion.tolist()}, predictions
