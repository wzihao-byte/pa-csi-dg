"""Pre-registered, source-only health gates for the 18-run lambda screen.

The compact evidence object intentionally carries the complete epoch series
used by the decision.  ``select_lambda`` can therefore reproduce every health
flag without opening a training dataset, checkpoint, or target artifact.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # Support module and direct-script execution.
    from . import protocol
except ImportError:  # pragma: no cover
    import protocol  # type: ignore


SCREEN_HEALTH_SCHEMA = protocol.SCREEN_HEALTH_SCHEMA

_SERIES_KEYS: Tuple[str, ...] = (
    "train_ce_by_epoch",
    "source_val_macro_f1_by_epoch",
    "gradient_l2_median_by_epoch",
    "gradient_l2_p95_by_epoch",
    "gradient_l2_max_by_epoch",
)


def _thresholds() -> Dict[str, float]:
    names = (
        "minimum_source_val_macro_f1",
        "maximum_late_to_early_train_ce_ratio",
        "train_ce_window_epochs",
        "gradient_reference_epochs",
        "gradient_relative_multiplier",
        "gradient_absolute_floor_l2",
        "gradient_catastrophic_l2",
        "gradient_persistent_epochs",
    )
    return {name: float(protocol.SCREEN_THRESHOLDS[name]) for name in names}


def _numeric_audit(value: object, path: str, failures: List[str]) -> int:
    """Count numeric leaves and record non-finite paths; bools are not numbers."""

    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not math.isfinite(float(value)):
            failures.append(path)
        return 1
    if isinstance(value, Mapping):
        return sum(
            _numeric_audit(child, f"{path}.{key}", failures)
            for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sum(
            _numeric_audit(child, f"{path}[{index}]", failures)
            for index, child in enumerate(value)
        )
    return 0


def _longest_consecutive(values: Sequence[int]) -> int:
    longest = 0
    current = 0
    previous: Optional[int] = None
    for value in values:
        if previous is not None and value == previous + 1:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
        previous = value
    return longest


def _derive(evidence: Mapping[str, object]) -> Dict[str, object]:
    thresholds = _thresholds()
    epoch_count = int(evidence["epoch_count"])
    train_ce = [float(value) for value in evidence["train_ce_by_epoch"]]  # type: ignore[index]
    val_f1 = [float(value) for value in evidence["source_val_macro_f1_by_epoch"]]  # type: ignore[index]
    gradient_median = [float(value) for value in evidence["gradient_l2_median_by_epoch"]]  # type: ignore[index]
    gradient_p95 = [float(value) for value in evidence["gradient_l2_p95_by_epoch"]]  # type: ignore[index]
    gradient_max = [float(value) for value in evidence["gradient_l2_max_by_epoch"]]  # type: ignore[index]

    registered_window = int(thresholds["train_ce_window_epochs"])
    effective_window = min(registered_window, max(1, epoch_count // 2))
    early_ce = float(np.median(train_ce[:effective_window]))
    late_ce = float(np.median(train_ce[-effective_window:]))
    if early_ce > 0.0:
        ce_ratio = late_ce / early_ce
    elif late_ce <= 0.0:
        ce_ratio = 0.0
    else:
        # Keep the evidence strict-JSON-compatible while ensuring the gate
        # fails for an impossible increase from exactly zero CE.
        ce_ratio = float(np.finfo(np.float64).max)
    best_val = max(val_f1)

    reference_epochs = min(int(thresholds["gradient_reference_epochs"]), epoch_count)
    reference_l2 = float(np.median(gradient_median[:reference_epochs]))
    explosion_threshold = max(
        thresholds["gradient_absolute_floor_l2"],
        thresholds["gradient_relative_multiplier"] * reference_l2,
    )
    explosive_epochs = [
        epoch
        for epoch, value in enumerate(gradient_p95, start=1)
        if value > explosion_threshold
    ]
    longest_explosive_run = _longest_consecutive(explosive_epochs)
    catastrophic_epochs = [
        epoch
        for epoch, value in enumerate(gradient_max, start=1)
        if value >= thresholds["gradient_catastrophic_l2"]
    ]

    finite_training = bool(evidence["all_required_history_values_finite"])
    validation_above_chance = best_val >= thresholds["minimum_source_val_macro_f1"]
    training_ce_decreased = ce_ratio <= thresholds["maximum_late_to_early_train_ce_ratio"]
    converged = finite_training and validation_above_chance and training_ce_decreased
    gradient_exploded = bool(catastrophic_epochs) or (
        longest_explosive_run >= int(thresholds["gradient_persistent_epochs"])
    )
    return {
        "finite_training": finite_training,
        "converged": converged,
        "gradient_exploded": gradient_exploded,
        "observations": {
            "effective_train_ce_window_epochs": effective_window,
            "early_train_ce_median": early_ce,
            "late_train_ce_median": late_ce,
            "late_to_early_train_ce_ratio": ce_ratio,
            "best_source_val_macro_f1": best_val,
            "gradient_reference_median_l2": reference_l2,
            "gradient_explosion_threshold_l2": explosion_threshold,
            "gradient_max_l2": max(gradient_max),
            "explosive_epoch_ids": explosive_epochs,
            "longest_consecutive_explosive_epochs": longest_explosive_run,
            "catastrophic_epoch_ids": catastrophic_epochs,
        },
        "criteria": {
            "source_val_at_or_above_uniform_six_class_chance": validation_above_chance,
            "late_train_ce_at_most_95_percent_of_early_train_ce": training_ce_decreased,
            "no_persistent_or_catastrophic_gradient_explosion": not gradient_exploded,
        },
    }


def build_screen_health(history: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Build decision evidence only from source train/validation history."""

    if not history:
        raise ValueError("Screen health requires at least one completed epoch.")
    failures: List[str] = []
    checked_values = 0
    train_ce: List[float] = []
    val_f1: List[float] = []
    gradient_median: List[float] = []
    gradient_p95: List[float] = []
    gradient_max: List[float] = []
    for index, row in enumerate(history, start=1):
        if int(row.get("epoch", -1)) != index:
            raise ValueError("Screen history epochs must be contiguous and one-indexed.")
        train = row.get("train")
        val = row.get("val")
        if not isinstance(train, Mapping) or not isinstance(val, Mapping):
            raise ValueError(f"Screen history epoch {index} lacks train/val metrics.")
        gradient = train.get("gradient_norm_l2")
        if not isinstance(gradient, Mapping):
            raise ValueError(f"Screen history epoch {index} lacks gradient_norm_l2.")
        train_ce.append(float(train["loss_ce"]))
        val_f1.append(float(val[protocol.CHECKPOINT_METRIC]))
        gradient_median.append(float(gradient["median"]))
        gradient_p95.append(float(gradient["p95"]))
        gradient_max.append(float(gradient["maximum"]))
        for section_name, section in (
            ("train", train),
            ("val", val),
            ("gradient_diagnostic", row.get("gradient_diagnostic")),
        ):
            checked_values += _numeric_audit(section, f"epoch[{index}].{section_name}", failures)

    evidence: Dict[str, object] = {
        "schema_version": SCREEN_HEALTH_SCHEMA,
        "thresholds": _thresholds(),
        "epoch_count": len(history),
        "train_ce_by_epoch": train_ce,
        "source_val_macro_f1_by_epoch": val_f1,
        "gradient_l2_median_by_epoch": gradient_median,
        "gradient_l2_p95_by_epoch": gradient_p95,
        "gradient_l2_max_by_epoch": gradient_max,
        "all_required_history_values_finite": not failures,
        "finite_numeric_audit": {
            "checked_value_count": checked_values,
            "nonfinite_paths": failures,
        },
    }
    evidence["derived"] = _derive(evidence)
    return evidence


def validate_screen_health(
    evidence: Mapping[str, object],
    *,
    expected_epoch_count: Optional[int] = None,
) -> Dict[str, object]:
    """Validate evidence and independently reproduce all decision flags."""

    if evidence.get("schema_version") != SCREEN_HEALTH_SCHEMA:
        raise ValueError("Unknown or missing screen-health schema.")
    if evidence.get("thresholds") != _thresholds():
        raise ValueError("Screen-health thresholds differ from the protocol lock.")
    epoch_count = evidence.get("epoch_count")
    if not isinstance(epoch_count, int) or isinstance(epoch_count, bool) or epoch_count <= 0:
        raise ValueError("screen_health.epoch_count must be a positive integer.")
    if expected_epoch_count is not None and epoch_count != expected_epoch_count:
        raise ValueError(
            f"Screen health must cover {expected_epoch_count} epochs; received {epoch_count}."
        )
    for key in _SERIES_KEYS:
        values = evidence.get(key)
        if not isinstance(values, list) or len(values) != epoch_count:
            raise ValueError(f"screen_health.{key} must contain one value per epoch.")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError(f"screen_health.{key} must contain only finite numbers.")
        numeric = [float(value) for value in values]
        if key == "source_val_macro_f1_by_epoch":
            if any(value < 0.0 or value > 1.0 for value in numeric):
                raise ValueError("Source-val macro-F1 health series must stay in [0,1].")
        elif any(value < 0.0 for value in numeric):
            raise ValueError(f"screen_health.{key} cannot contain negative values.")
    finite_audit = evidence.get("finite_numeric_audit")
    if not isinstance(finite_audit, Mapping):
        raise ValueError("screen_health.finite_numeric_audit must be an object.")
    failures = finite_audit.get("nonfinite_paths")
    finite_flag = evidence.get("all_required_history_values_finite")
    if not isinstance(failures, list) or not isinstance(finite_flag, bool):
        raise ValueError("Screen finite-audit fields have invalid types.")
    if finite_flag != (len(failures) == 0):
        raise ValueError("Screen finite-audit flag and paths disagree.")
    checked = finite_audit.get("checked_value_count")
    if not isinstance(checked, int) or isinstance(checked, bool) or checked <= 0:
        raise ValueError("Screen finite audit must report a positive checked-value count.")

    reproduced = _derive(evidence)
    if evidence.get("derived") != reproduced:
        raise ValueError("Screen-health derived evidence does not reproduce from its epoch series.")
    return reproduced


def build_screen_metrics(
    history: Sequence[Mapping[str, object]],
    *,
    best_source_val_sit_stand_f1: float,
    positive_anchor_fraction: float,
) -> Dict[str, object]:
    evidence = build_screen_health(history)
    derived = evidence["derived"]
    assert isinstance(derived, Mapping)
    observations = derived["observations"]
    assert isinstance(observations, Mapping)
    return {
        "finite_training": bool(derived["finite_training"]),
        "converged": bool(derived["converged"]),
        "gradient_exploded": bool(derived["gradient_exploded"]),
        "best_source_val_macro_f1": float(observations["best_source_val_macro_f1"]),
        "best_source_val_sit_stand_f1": float(best_source_val_sit_stand_f1),
        "positive_anchor_fraction": float(positive_anchor_fraction),
        "health_evidence": evidence,
    }


def validate_screen_metrics(
    metrics: Mapping[str, object],
    *,
    expected_epoch_count: Optional[int] = None,
    history: Optional[Sequence[Mapping[str, object]]] = None,
) -> None:
    evidence = metrics.get("health_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("screen_metrics.health_evidence must be an object.")
    reproduced = validate_screen_health(evidence, expected_epoch_count=expected_epoch_count)
    for key in ("finite_training", "converged", "gradient_exploded"):
        if metrics.get(key) is not reproduced[key]:
            raise ValueError(f"screen_metrics.{key} does not reproduce from health evidence.")
    observations = reproduced["observations"]
    assert isinstance(observations, Mapping)
    if float(metrics["best_source_val_macro_f1"]) != float(
        observations["best_source_val_macro_f1"]
    ):
        raise ValueError("Best source-val macro-F1 disagrees with the health epoch series.")
    if history is not None:
        rebuilt = build_screen_health(history)
        if evidence != rebuilt:
            raise ValueError("Screen-health evidence disagrees with the hash-bound training history.")
