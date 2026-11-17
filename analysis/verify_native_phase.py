"""Check temporal phase geometry, invalid observations and common-error rejection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.audit_multienv_timestamps import decode_dat_csi
from analysis.native_phase_processing import RX_PAIRS, VARIANTS, build_phase_variants, phase_channels, reconstruct_phase
from analysis.prepare_native_multienv import native_elapsed
from analysis.verify_multienv_raw_mapping import read_dat_timestamps
from experiments.revision_validation.study_io import now, read, sha256, write

CODE_FILES = ("analysis/native_phase_processing.py", "analysis/verify_native_phase.py", "analysis/audit_multienv_timestamps.py",
              "analysis/prepare_native_multienv.py", "analysis/verify_multienv_raw_mapping.py")


def common_error_check(csi, elapsed, seed):
    rng = np.random.default_rng(seed)
    # Random packet offsets and frequency slopes common to all three Rx. The
    # arbitrary subcarrier coordinates test cancellation, not carrier mapping.
    theta = rng.uniform(-np.pi, np.pi, (len(csi), 1))
    slope = rng.uniform(-0.4, 0.4, (len(csi), 1))
    phase = theta + slope * np.linspace(-1, 1, 30)[None, :]
    errors = np.exp(1j * phase)[:, None, :]
    changed = (csi.reshape(-1, 3, 30) * errors).reshape(-1, 90)
    reference, _ = build_phase_variants(csi, elapsed)
    perturbed, _ = build_phase_variants(changed, elapsed)
    differential_error = float(np.max(np.abs(reference[VARIANTS[1]] - perturbed[VARIANTS[1]])))
    direct_change = float(np.max(np.abs(reference[VARIANTS[0]] - perturbed[VARIANTS[0]])))
    if differential_error > 1e-5 or direct_change <= 0.1:
        raise RuntimeError("Common-error injection did not distinguish raw and differential phase as expected")
    return {"differential_max_abs_error": differential_error, "direct_phase_max_change": direct_change,
            "absolute_tolerance": 1e-5, "seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new phase QA output to preserve previous attempts")
    times = np.linspace(0, 4e6, 41)
    time_fraction = times / times[-1]
    slopes = np.linspace(-1.1, 1.3, 90).reshape(3, 30)
    offsets = np.linspace(-4.0, 4.0, 90).reshape(3, 30)
    csi = (2 * np.exp(1j * (offsets[None] + time_fraction[:, None, None] * slopes[None]))).reshape(-1, 90)
    values, _ = build_phase_variants(csi, times)
    expected_time = np.linspace(-0.5, 0.5, 850)[:, None]
    expected_direct = expected_time * slopes.reshape(1, 90)
    expected_differential = expected_time * np.concatenate([slopes[a] - slopes[b] for a, b in RX_PAIRS])[None]
    errors = {VARIANTS[0]: float(np.max(np.abs(values[VARIANTS[0]] - expected_direct))),
              VARIANTS[1]: float(np.max(np.abs(values[VARIANTS[1]] - expected_differential)))}
    assert max(errors.values()) < 1e-6
    synthetic = common_error_check(csi, times, 9130)
    pi_csi = np.tile(np.repeat(np.array([1+2j, 3+4j, 5+6j]), 30)[None], (41, 1))
    pi_csi[1::2, :30] *= -1
    pi_boundary = common_error_check(pi_csi, np.arange(41)*100000., 9129)
    missing = csi.copy()
    missing[10] = 0
    missing_values, missing_stats = build_phase_variants(missing, times)
    assert all(np.max(np.abs(missing_values[name] - values[name])) < 1e-6 for name in VARIANTS)
    # Duplicating identical timestamps and values must leave the result intact.
    duplicated_times = np.repeat(times, 2)
    duplicated_csi = np.repeat(csi, 2, axis=0)
    duplicate_values, _ = build_phase_variants(duplicated_csi, duplicated_times)
    assert all(np.array_equal(values[name], duplicate_values[name]) for name in VARIANTS)
    # A pair of opposing direct-phase observations at one timestamp has no
    # usable circular mean. Removing that interior time still recovers a ramp.
    cancelled_csi = duplicated_csi.copy()
    cancelled_csi[21] *= -1
    cancellation_values, cancellation_stats = reconstruct_phase(cancelled_csi, duplicated_times, differential=False)
    assert cancellation_stats["circular_cancellation_bins"] == 90
    assert np.max(np.abs(cancellation_values - values[VARIANTS[0]])) < 1e-6
    all_zero_rejected = False
    try:
        reconstruct_phase(np.zeros_like(csi), times, differential=True)
    except ValueError as error:
        all_zero_rejected = "fewer than two usable" in str(error)
    assert all_zero_rejected
    # Altering one direct phase channel must not unwrap or center any neighbour.
    one_channel = csi.copy()
    one_channel[:, 30] *= np.exp(1j * np.sin(time_fraction * 3))
    local, _ = reconstruct_phase(one_channel, times, differential=False)
    assert np.array_equal(np.delete(local, 30, axis=1), np.delete(values[VARIANTS[0]], 30, axis=1))
    # Pair orientation and column order are checked against scalar products.
    pairs = phase_channels(csi, True)
    for group, (a, b) in enumerate(RX_PAIRS):
        assert np.array_equal(pairs[:, group*30:(group+1)*30], csi[:, a*30:(a+1)*30] * csi[:, b*30:(b+1)*30].conj())
    summary = read(args.native_smoke / "summary.json")
    actual = []
    for index, (name, expected_hash) in enumerate(sorted(summary["record_sha256"].items())):
        record_path = args.native_smoke / "records" / (Path(name).stem + ".json")
        if sha256(record_path) != expected_hash:
            raise RuntimeError("Native smoke evidence changed")
        row = read(record_path)
        raw_path = args.native_smoke / "dat_cache" / (Path(name).stem + ".dat")
        if sha256(raw_path) != row["dat_sha256"]:
            raise RuntimeError("Native raw bytes changed")
        data = raw_path.read_bytes()
        native = decode_dat_csi(data)
        elapsed, _ = native_elapsed(read_dat_timestamps(data)[0])
        _, stats = build_phase_variants(native, elapsed)
        actual.append({"raw_path": str(raw_path), "raw_sha256": row["dat_sha256"],
                       "shared_error_invariance": common_error_check(native, elapsed, 9200 + index), "diagnostics": stats})
    report = {"passed": True, "checked_at": now(), "source": "synthetic controls and metadata-selected native smoke rows",
              "analytic_temporal_ramp_max_errors": errors, "synthetic_shared_error_invariance": synthetic,
              "pi_boundary_shared_error_invariance": pi_boundary, "pi_tie_direction_is_a_convention": True,
              "zero_interior_packet_interpolation_exact_within_1e_minus6": True,
              "identical_duplicate_time_invariance_exact": True, "all_zero_channel_rejected": all_zero_rejected,
              "opposite_duplicate_time_phase_marked_unusable_and_interpolated": True,
              "no_cross_channel_unwrap_or_centering": True, "pair_direction_and_order_exact": True,
              "actual_rows": actual, "model_training_or_selection": False,
              "full_preparation_completed": False, "complete_hardware_calibration": False,
              "code_sha256": {name: sha256(ROOT / name) for name in CODE_FILES}}
    write(args.output, report)
    print({"passed": True, "actual_rows": len(actual), "largest_real_invariance_error": max(r["shared_error_invariance"]["differential_max_abs_error"] for r in actual),
           "output": str(args.output)})


if __name__ == "__main__":
    main()
