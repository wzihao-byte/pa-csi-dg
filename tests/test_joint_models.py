from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from dg_models import PaCsiStreamEncoder
from experiments.supcon_tpp_joint import protocol
from experiments.supcon_tpp_joint.joint_models import (
    BASIS_NAMES,
    CALIBRATION_INITIALIZATION_SEED,
    CalibrationError,
    CalibrationProvenance,
    JointModelProtocolError,
    MatchedResidualPyramidPool,
    TPP_LEVELS,
    TPP_RESIDUAL_SCALE,
    audit_joint_model_parameters,
    build_joint_arm_models,
    build_joint_model,
    calibrate_global_basis_control,
    derive_basis_statistics,
    exact_global_basis,
    validate_calibration_sample_membership,
)


def _joint_temporal_config(arm: str) -> dict:
    calibration = dict(protocol.GLOBAL_BASIS_CALIBRATION)
    return {
        "branch": "amplitude_only",
        "factor": protocol.ARMS[arm].temporal_factor,
        "global_basis": {
            "basis": [dict(item) for item in protocol.GLOBAL_BASIS],
            "calibration": calibration,
            "input_normalization": {
                "affine": False,
                "clip_max": calibration["clip_max"],
                "clip_min": calibration["clip_min"],
                "epsilon": calibration["layer_norm_epsilon"],
                "name": "layer_norm",
            },
        },
        "levels": list(protocol.TPP_LEVELS),
        "pool_type": protocol.TPP_POOL_TYPE,
        "preserve_global": protocol.TPP_PRESERVE_GLOBAL,
        "residual_scale": protocol.TPP_RESIDUAL_SCALE,
    }


def _small_model_config(arm: str = "Y00") -> dict:
    return {
        "time_downsample": 1,
        "max_positions": 64,
        "position_kernels": 2,
        "mcat_layers": 1,
        "mcat_heads": 2,
        "dropout": 0.0,
        "temporal_kernel_num": 4,
        "temporal_kernel_sizes": [3],
        "channel_group_width": 3,
        "channel_encoder_layers": 1,
        "channel_encoder_heads": 1,
        "channel_kernel_num": 4,
        "channel_kernel_sizes": [2, 3],
        "fusion_dim": 8,
        "projection_hidden_dim": 8,
        "projection_dim": 5,
        "joint_temporal": _joint_temporal_config(arm),
    }


def _sample_ids(sources: tuple[str, ...], count: int, kind: str) -> tuple[str, ...]:
    return tuple(
        f"{source}:{kind}-{index:03d}"
        for source in sources
        for index in range(count)
    )


def _source_hashes(sources: tuple[str, ...]) -> dict[str, str]:
    return {
        f"{source}.{kind}": f"{offset:x}" * 64
        for offset, (source, kind) in enumerate(
            ((source, kind) for source in sources for kind in ("amp", "phase", "label")),
            start=1,
        )
    }


def _valid_provenance() -> CalibrationProvenance:
    sources = ("E2", "E3")
    return CalibrationProvenance(
        outer_target="E1",
        source_environments=sources,
        calibration_sample_ids=_sample_ids(sources, 256, "cal"),
        smoke_sample_ids=_sample_ids(sources, 8, "smoke"),
        source_file_sha256=_source_hashes(sources),
        source_index_sha256="f" * 64,
    )


def test_exact_global_basis_matches_locked_formulas() -> None:
    global_feature = torch.tensor(
        [
            [0.2, 0.7, 1.4, 2.8],
            [3.0, -1.0, 0.5, 1.2],
        ],
        dtype=torch.float64,
    )

    actual = exact_global_basis(global_feature)
    normalized = F.layer_norm(global_feature, (4,), weight=None, bias=None, eps=1e-5)
    clipped = normalized.clamp(-3.0, 3.0)
    expected = torch.stack(
        (
            clipped,
            clipped.square(),
            clipped.pow(3) / 9.0,
            clipped.tanh(),
            torch.sin(math.pi * clipped / 3.0),
            torch.cos(math.pi * clipped / 3.0),
            clipped.sign() * torch.sqrt(clipped.abs() + 1e-6),
        ),
        dim=1,
    )

    assert BASIS_NAMES == (
        "identity",
        "square",
        "scaled_cube",
        "tanh",
        "sine",
        "cosine",
        "signed_sqrt",
    )
    assert BASIS_NAMES == tuple(item["name"] for item in protocol.GLOBAL_BASIS)
    assert actual.shape == (2, 7, 4)
    torch.testing.assert_close(actual, expected)


def test_tpp_pool_is_exact_l124_residual_with_scale_point_25() -> None:
    torch.manual_seed(17)
    pool = MatchedResidualPyramidPool(
        input_dim=5,
        output_channels=3,
        kernel_sizes=[3, 5],
        representation="tpp",
        dropout=0.0,
    ).eval()
    inputs = torch.randn(4, 20, 5)

    global_feature, pyramid_feature = pool.extract_residual_inputs(inputs)
    expected = global_feature + 0.25 * pool.projection(pyramid_feature)
    actual = pool(inputs)

    assert tuple(pool.levels) == TPP_LEVELS
    assert pool.pool_type == "max"
    assert pool.preserve_global is True
    assert pool.pyramid_scale == TPP_RESIDUAL_SCALE == 0.25
    torch.testing.assert_close(actual, expected)


def test_four_arms_have_equal_parameters_and_identical_global_phase_path() -> None:
    models = build_joint_arm_models(
        input_dim=6,
        num_classes=6,
        model_config=_small_model_config(),
        initialization_seed=123,
    )

    audit = audit_joint_model_parameters(models, strict=True, check_initialization=True)
    assert audit["all_equal"] is True
    assert len(set(audit["counts"].values())) == 1
    assert models["Y00"].temporal_mode == models["Y01"].temporal_mode == "global_basis"
    assert models["Y10"].temporal_mode == models["Y11"].temporal_mode == "tpp"
    assert all(isinstance(model.phase_encoder, PaCsiStreamEncoder) for model in models.values())

    reference_state = models["Y00"].phase_encoder.state_dict()
    for model in models.values():
        for name, value in model.phase_encoder.state_dict().items():
            torch.testing.assert_close(value, reference_state[name], rtol=0.0, atol=0.0)

    phase = torch.randn(3, 18, 6)
    for model in models.values():
        model.eval()
    reference_output = models["Y00"].phase_encoder(phase)
    for model in models.values():
        torch.testing.assert_close(model.phase_encoder(phase), reference_output, rtol=0.0, atol=0.0)


def test_global_only_supcon_path_has_one_view_and_never_calls_antenna_encoder() -> None:
    model = build_joint_model(
        "Y11",
        input_dim=6,
        num_classes=6,
        model_config={**_small_model_config("Y11"), "antenna_layout": {"mode": "rx"}},
        initialization_seed=321,
    ).eval()
    amplitude = torch.randn(2, 18, 6)
    phase = torch.randn(2, 18, 6)
    antenna_spy = Mock(side_effect=AssertionError("antenna views must not be called"))
    model.encode_antenna_views = antenna_spy  # type: ignore[method-assign]

    views = model.encode_supcon_views(amplitude, phase)

    assert views.shape == (2, 1, 5)
    assert model.view_builder is None
    antenna_spy.assert_not_called()
    with pytest.raises(JointModelProtocolError, match="global_only"):
        model.encode_supcon_views(amplitude, phase, view_scope="all_views")


def test_degenerate_global_basis_is_rejected_by_calibration_gate() -> None:
    degenerate_global = torch.ones(16, 8)

    with pytest.raises(CalibrationError, match="Degenerate|rank deficient"):
        derive_basis_statistics(degenerate_global)


def test_nested_joint_temporal_config_is_checked_exactly() -> None:
    wrong_levels = _small_model_config("Y10")
    wrong_levels["joint_temporal"]["levels"] = [1, 2]
    with pytest.raises(JointModelProtocolError, match="joint_temporal"):
        build_joint_model("Y10", 6, 6, wrong_levels, initialization_seed=1)

    wrong_basis = _small_model_config("Y00")
    wrong_basis["joint_temporal"]["global_basis"]["basis"][2]["name"] = "cubic_over_9"
    with pytest.raises(JointModelProtocolError, match="joint_temporal"):
        build_joint_model("Y00", 6, 6, wrong_basis, initialization_seed=1)


def test_calibration_provenance_enforces_counts_prefixes_uniqueness_and_hashes() -> None:
    provenance = _valid_provenance()
    provenance.validate()

    duplicate_ids = list(provenance.calibration_sample_ids)
    duplicate_ids[-1] = duplicate_ids[0]
    with pytest.raises(CalibrationError, match="unique"):
        replace(provenance, calibration_sample_ids=tuple(duplicate_ids)).validate()

    target_id = list(provenance.calibration_sample_ids)
    target_id[-1] = "E1:cal-target"
    with pytest.raises(CalibrationError, match="allowed source"):
        replace(provenance, calibration_sample_ids=tuple(target_id)).validate()

    with pytest.raises(CalibrationError, match="exactly 256"):
        replace(
            provenance,
            calibration_sample_ids=provenance.calibration_sample_ids[:-1],
        ).validate()

    with pytest.raises(CalibrationError, match="exactly 8"):
        replace(provenance, smoke_sample_ids=provenance.smoke_sample_ids[:-1]).validate()

    bad_hashes = dict(provenance.source_file_sha256)
    bad_hashes["E2.amp"] = "not-a-sha256"
    with pytest.raises(CalibrationError, match="64 hexadecimal"):
        replace(provenance, source_file_sha256=bad_hashes).validate()

    missing_hash = dict(provenance.source_file_sha256)
    missing_hash.pop("E3.label")
    with pytest.raises(CalibrationError, match="keys must be exactly"):
        replace(provenance, source_file_sha256=missing_hash).validate()


def test_calibration_membership_rejects_non_source_train_ids() -> None:
    provenance = _valid_provenance()
    allowed = set(provenance.calibration_sample_ids)
    allowed.remove(provenance.calibration_sample_ids[-1])

    with pytest.raises(CalibrationError, match="outside the source-train allowlist"):
        validate_calibration_sample_membership(
            provenance,
            allowed_calibration_sample_ids=allowed,
        )

    wrong_expected = list(provenance.smoke_sample_ids)
    wrong_expected[-1] = "E3:smoke-other"
    with pytest.raises(CalibrationError, match="do not match"):
        validate_calibration_sample_membership(
            provenance,
            expected_smoke_sample_ids=wrong_expected,
        )


def test_source_only_calibration_artifact_loads_control_and_passes_forward() -> None:
    torch.manual_seed(77)
    provenance = _valid_provenance()
    model = build_joint_model(
        "Y00",
        input_dim=6,
        num_classes=6,
        model_config=_small_model_config("Y00"),
        initialization_seed=CALIBRATION_INITIALIZATION_SEED,
    )
    calibration = torch.randn(len(provenance.calibration_sample_ids), 18, 6)
    smoke = torch.randn(len(provenance.smoke_sample_ids), 18, 6) * 1.05 + 0.03

    artifact = calibrate_global_basis_control(
        model,
        calibration,
        smoke,
        provenance,
        expected_calibration_sample_ids=provenance.calibration_sample_ids,
        expected_smoke_sample_ids=provenance.smoke_sample_ids,
    )

    assert model.calibration_ready is True
    assert set(artifact.paths) == {"temporal", "channel"}
    assert all(path.svd_rank == 7 for path in artifact.paths.values())
    restored = type(artifact).from_dict(artifact.to_dict())
    restored.validate(expected_paths=("temporal", "channel"))
    model.eval()
    output = model(torch.randn(2, 18, 6), torch.randn(2, 18, 6))
    assert output["logits"].shape == (2, 6)
    assert output["supcon_projection_views"].shape == (2, 1, 5)

    probe = torch.randn(8, 18, 6)
    model.zero_grad(set_to_none=True)
    path_inputs = model.amp_encoder.encode_pool_inputs(probe)
    for path_name, pool in model.amp_encoder.residual_pools().items():
        global_feature, _ = pool.extract_residual_inputs(path_inputs[path_name])
        basis_vector = pool.global_basis_vector(global_feature)
        assert basis_vector.shape[1] == 7 * pool.output_dim
        for block_index in range(7):
            block = basis_vector[:, block_index * pool.output_dim : (block_index + 1) * pool.output_dim]
            assert torch.count_nonzero(block).item() > 0

    amplitude_feature = model.amp_encoder(probe)
    amplitude_feature.square().mean().backward()
    for pool in model.amp_encoder.residual_pools().values():
        gradient = pool.projection.weight.grad
        assert gradient is not None
        for block_index in range(7):
            block_gradient = gradient[
                :,
                block_index * pool.output_dim : (block_index + 1) * pool.output_dim,
            ]
            assert torch.count_nonzero(block_gradient).item() > 0

    corrupted_payloads = []

    missing_singular_value = deepcopy(artifact.to_dict())
    missing_singular_value["paths"]["temporal"]["singular_values"] = missing_singular_value[
        "paths"
    ]["temporal"]["singular_values"][:-1]
    corrupted_payloads.append(missing_singular_value)

    unsorted_singular_values = deepcopy(artifact.to_dict())
    unsorted_singular_values["paths"]["temporal"]["singular_values"] = list(
        reversed(unsorted_singular_values["paths"]["temporal"]["singular_values"])
    )
    corrupted_payloads.append(unsorted_singular_values)

    wrong_condition = deepcopy(artifact.to_dict())
    wrong_condition["paths"]["temporal"]["condition_number"] *= 1.1
    corrupted_payloads.append(wrong_condition)

    inconsistent_kappa = deepcopy(artifact.to_dict())
    inconsistent_kappa["paths"]["temporal"]["kappa"] *= 1.1
    corrupted_payloads.append(inconsistent_kappa)

    negative_rms = deepcopy(artifact.to_dict())
    negative_rms["paths"]["temporal"]["smoke_control_scaled_rms"] = -0.1
    corrupted_payloads.append(negative_rms)

    inconsistent_smoke_ratio = deepcopy(artifact.to_dict())
    inconsistent_smoke_ratio["paths"]["temporal"]["smoke_rms_ratio"] *= 1.1
    corrupted_payloads.append(inconsistent_smoke_ratio)

    for payload in corrupted_payloads:
        with pytest.raises(CalibrationError):
            type(artifact).from_dict(payload)
