"""Native-time phase controls with explicit shared-error invariance.

The differential representation cancels a common complex phase factor at each
packet/subcarrier. It is not a complete hardware calibration. No unwrap crosses
a subcarrier or Rx boundary; all temporal processing stays within a recording.
"""
from __future__ import annotations

import numpy as np

VERSION = "native-time-phase-controls-v2-explicit-pi-ties"
RX_PAIRS = ((0, 1), (1, 2), (2, 0))
VARIANTS = ("native_time_phase", "native_time_differential_phase")


def phase_channels(csi, differential):
    values = np.asarray(csi, dtype=np.complex128)
    if values.ndim != 2 or values.shape[1] != 90 or not np.isfinite(values).all():
        raise ValueError("Expected finite, permutation-corrected [packet, 3 Rx x 30] CSI")
    if differential:
        links = values.reshape(-1, 3, 30)
        return np.concatenate([links[:, left] * links[:, right].conj() for left, right in RX_PAIRS], axis=1)
    return values


def reconstruct_phase(csi, elapsed, *, differential):
    channels = phase_channels(csi, differential)
    elapsed = np.asarray(elapsed, dtype=np.float64)
    if elapsed.shape != (len(channels),) or elapsed[0] != 0 or not np.isfinite(elapsed).all():
        raise ValueError("Phase reconstruction requires record-relative finite native times")
    if np.any(np.diff(elapsed) < 0) or elapsed[-1] <= 0:
        raise ValueError("Phase reconstruction requires monotonic positive-duration times")
    unique, inverse = np.unique(elapsed, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("At least two distinct native times are required")
    valid = np.abs(channels) > 0
    unit = np.zeros_like(channels)
    np.divide(channels, np.abs(channels), out=unit, where=valid)
    circular_sum = np.zeros((len(unique), 90), dtype=np.complex128)
    counts = np.zeros((len(unique), 90), dtype=np.int64)
    np.add.at(circular_sum, inverse, unit)
    np.add.at(counts, inverse, valid)
    # A zero-amplitude observation has no phase. Opposite phase observations at
    # an identical timestamp can also cancel, so do not assign them angle(0).
    magnitude = np.abs(circular_sum)
    usable = (counts > 0) & (magnitude > 1e-12 * counts)
    grid = np.linspace(0, elapsed[-1], 850)
    output = np.empty((850, 90), dtype=np.float64)
    gaps, edge_extensions, near_pi = [], [], 0
    for column in range(90):
        good = usable[:, column]
        if good.sum() < 2:
            raise ValueError(f"Phase column {column} has fewer than two usable native times")
        times = unique[good]
        observations = circular_sum[good, column]
        observations = observations / np.abs(observations)
        increments = np.angle(observations[1:] * observations[:-1].conj())
        near_pi += int(np.count_nonzero(np.abs(increments) >= np.pi - 1e-6))
        # Exactly pi is physically ambiguous. Fix the numerical tie instead of
        # letting a tiny common-error roundoff change an entire later trajectory.
        increments[np.abs(np.abs(increments) - np.pi) <= 1e-10] = np.pi
        unwrapped = np.r_[0., np.cumsum(increments)]
        output[:, column] = np.interp(grid, times, unwrapped)
        # This removes recording-specific constants; it is not train-fitted
        # normalization and does not estimate physical AoA/ToF or absolute phase.
        output[:, column] -= output[:, column].mean()
        gaps.append(float(np.diff(times).max()))
        edge_extensions.append(float(times[0] + elapsed[-1] - times[-1]))
    if not np.isfinite(output).all():
        raise FloatingPointError("Phase processing produced nonfinite values")
    diagnostics = {
        "raw_channel_zero_magnitude_count": int((~valid).sum()),
        "raw_channel_element_count": int(valid.size),
        "duplicate_timestamp_packets": int(len(elapsed) - len(unique)),
        "circular_cancellation_bins": int(((counts > 0) & ~usable).sum()),
        "max_unusable_time_fraction_per_channel": float((~usable).mean(axis=0).max()),
        "max_usable_time_gap_seconds": max(gaps) / 1e6,
        "max_endpoint_constant_extension_seconds": max(edge_extensions) / 1e6,
        "wrapped_adjacent_steps_within_1e_minus6_of_pi": near_pi,
        "max_absolute_centered_phase": float(np.abs(output).max()),
        "max_abs_channel_mean_before_float32": float(np.abs(output.mean(axis=0)).max()),
        "unwrap_axis": "native time, independently for each of the 90 channels",
        "pi_tie_rule": "principal phase increment; within 1e-10 of +/-pi choose +pi; true direction unresolved",
        "complete_hardware_calibration": False,
    }
    return output.astype(np.float32), diagnostics


def build_phase_variants(csi, elapsed):
    values, diagnostics = {}, {}
    for variant, differential in zip(VARIANTS, (False, True)):
        values[variant], diagnostics[variant] = reconstruct_phase(csi, elapsed, differential=differential)
    return values, diagnostics
