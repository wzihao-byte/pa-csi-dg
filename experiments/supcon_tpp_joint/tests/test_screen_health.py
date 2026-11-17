from __future__ import annotations

import copy

import pytest

from experiments.supcon_tpp_joint import protocol, screen_health


def _history(
    *,
    epochs: int = 200,
    early_ce: float = 1.5,
    late_ce: float = 1.0,
    val_f1: float = 0.40,
    gradient_median: float = 1.0,
    gradient_p95_by_epoch: dict[int, float] | None = None,
    gradient_max_by_epoch: dict[int, float] | None = None,
) -> list[dict]:
    gradient_p95_by_epoch = gradient_p95_by_epoch or {}
    gradient_max_by_epoch = gradient_max_by_epoch or {}
    return [
        {
            "epoch": epoch,
            "train": {
                "loss_ce": early_ce + (late_ce - early_ce) * (epoch - 1) / (epochs - 1),
                "loss_total": early_ce + (late_ce - early_ce) * (epoch - 1) / (epochs - 1),
                "gradient_norm_l2": {
                    "count": 300,
                    "minimum": 0.5,
                    "median": gradient_median,
                    "p95": gradient_p95_by_epoch.get(epoch, 2.0),
                    "maximum": gradient_max_by_epoch.get(epoch, 3.0),
                    "mean": 1.2,
                },
            },
            "val": {"f1_macro": val_f1, "loss_ce": 1.0},
        }
        for epoch in range(1, epochs + 1)
    ]


def _metrics(history: list[dict]) -> dict:
    return screen_health.build_screen_metrics(
        history,
        best_source_val_sit_stand_f1=0.3,
        positive_anchor_fraction=1.0,
    )


def test_healthy_screen_is_reproducible_from_epoch_series() -> None:
    history = _history()
    metrics = _metrics(history)
    screen_health.validate_screen_metrics(
        metrics,
        expected_epoch_count=protocol.EPOCHS,
        history=history,
    )
    assert metrics["finite_training"] is True
    assert metrics["converged"] is True
    assert metrics["gradient_exploded"] is False
    derived = metrics["health_evidence"]["derived"]
    assert derived["observations"]["effective_train_ce_window_epochs"] == 20


def test_chance_level_and_ce_descent_are_independent_convergence_gates() -> None:
    below_chance = _metrics(_history(val_f1=1.0 / 6.0 - 1e-6))
    no_descent = _metrics(_history(early_ce=1.0, late_ce=1.0))
    assert below_chance["converged"] is False
    assert no_descent["converged"] is False
    assert below_chance["gradient_exploded"] is False
    assert no_descent["gradient_exploded"] is False


def test_gradient_gate_requires_persistence_but_catches_catastrophic_spike() -> None:
    # Baseline median=1 => registered relative threshold=max(100, 100*1)=100.
    two_epoch_warning = _metrics(
        _history(gradient_p95_by_epoch={50: 101.0, 51: 101.0})
    )
    three_epoch_explosion = _metrics(
        _history(gradient_p95_by_epoch={50: 101.0, 51: 101.0, 52: 101.0})
    )
    catastrophic = _metrics(
        _history(gradient_max_by_epoch={80: protocol.SCREEN_THRESHOLDS["gradient_catastrophic_l2"]})
    )
    assert two_epoch_warning["gradient_exploded"] is False
    assert three_epoch_explosion["gradient_exploded"] is True
    assert catastrophic["gradient_exploded"] is True


def test_tampered_health_flag_or_epoch_series_fails_closed() -> None:
    metrics = _metrics(_history())
    metrics["converged"] = False
    with pytest.raises(ValueError, match="does not reproduce"):
        screen_health.validate_screen_metrics(metrics, expected_epoch_count=protocol.EPOCHS)

    metrics = _metrics(_history())
    tampered = copy.deepcopy(metrics)
    tampered["health_evidence"]["train_ce_by_epoch"][-1] = 10.0
    with pytest.raises(ValueError, match="derived evidence"):
        screen_health.validate_screen_metrics(tampered, expected_epoch_count=protocol.EPOCHS)
