from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dg_models import (
    ChannelContextEncoder,
    GaussianRelativeEncoding,
    GatedResidualFusion,
    MCATEncoder,
    PaCsiStreamEncoder,
    ProjectionHead,
    compatible_heads,
)
from experiments.tpp_isolated.tpp_models import IsolatedTemporalPyramidPool
from experiments.supcon_tpp_joint import protocol


ARM_NAMES = ("Y00", "Y01", "Y10", "Y11")
TPP_LEVELS = tuple(protocol.TPP_LEVELS)
TPP_SEGMENT_COUNT = sum(TPP_LEVELS)
TPP_RESIDUAL_SCALE = float(protocol.TPP_RESIDUAL_SCALE)
SUPCON_VIEW_SCOPE = str(protocol.SUPCON_VIEW_SCOPE)
BASIS_NAMES = tuple(str(item["name"]) for item in protocol.GLOBAL_BASIS)
BASIS_COUNT = len(BASIS_NAMES)
_CALIBRATION_PROTOCOL = protocol.GLOBAL_BASIS_CALIBRATION
BASIS_LAYER_NORM_EPS = float(_CALIBRATION_PROTOCOL["layer_norm_epsilon"])
BASIS_VALUE_EPS = float(_CALIBRATION_PROTOCOL["standardization_epsilon"])
BASIS_CLIP_MIN = float(_CALIBRATION_PROTOCOL["clip_min"])
BASIS_CLIP_MAX = float(_CALIBRATION_PROTOCOL["clip_max"])
CALIBRATION_SAMPLES_PER_SOURCE = int(_CALIBRATION_PROTOCOL["samples_per_source_environment"])
SMOKE_SAMPLES_PER_SOURCE = 8
SVD_RELATIVE_TOLERANCE = float(_CALIBRATION_PROTOCOL["svd_relative_tolerance"])
SVD_CONDITION_MAX = float(_CALIBRATION_PROTOCOL["maximum_condition_number"])
KAPPA_BOUNDS = (
    float(_CALIBRATION_PROTOCOL["kappa_min"]),
    float(_CALIBRATION_PROTOCOL["kappa_max"]),
)
SMOKE_RMS_RATIO_BOUNDS = (
    float(_CALIBRATION_PROTOCOL["residual_rms_ratio_min"]),
    float(_CALIBRATION_PROTOCOL["residual_rms_ratio_max"]),
)
CALIBRATION_INITIALIZATION_SEED = int(_CALIBRATION_PROTOCOL["initialization_seed"])
CALIBRATION_PROTOCOL_VERSION = (
    f"{protocol.PROTOCOL_NAME}:{protocol.PROTOCOL_VERSION}:global_basis"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_FLOAT_REL_TOL = 1e-5
_FLOAT_ABS_TOL = 1e-8

if BASIS_COUNT != int(_CALIBRATION_PROTOCOL["required_rank"]):
    raise RuntimeError("Protocol basis count and required calibration rank must match.")
if TPP_SEGMENT_COUNT != BASIS_COUNT:
    raise RuntimeError("The seven global bases must match the [1,2,4] TPP segment count.")


class JointModelProtocolError(ValueError):
    """Raised when a requested model violates the locked joint protocol."""


class CalibrationError(ValueError):
    """Raised when source-only global-basis calibration fails a protocol gate."""


def _sample_environment(sample_id: str) -> str:
    environment, separator, suffix = str(sample_id).partition(":")
    if not separator or not environment or not suffix:
        raise CalibrationError(
            f"Calibration sample ID '{sample_id}' must be namespaced as ENV:local_id."
        )
    return environment


@dataclass(frozen=True)
class JointArmSpec:
    name: str
    temporal_mode: str
    supcon_enabled: bool
    supcon_view_scope: str = SUPCON_VIEW_SCOPE


JOINT_ARM_SPECS: Dict[str, JointArmSpec] = {
    "Y00": JointArmSpec("Y00", "global_basis", False),
    "Y01": JointArmSpec("Y01", "global_basis", True),
    "Y10": JointArmSpec("Y10", "tpp", False),
    "Y11": JointArmSpec("Y11", "tpp", True),
}


@dataclass(frozen=True)
class CalibrationProvenance:
    """Source-only provenance bound to one outer fold's calibration tensors."""

    outer_target: str
    source_environments: tuple[str, ...]
    calibration_sample_ids: tuple[str, ...]
    smoke_sample_ids: tuple[str, ...]
    source_file_sha256: Mapping[str, str]
    source_index_sha256: str
    initialization_seed: int = CALIBRATION_INITIALIZATION_SEED

    def validate(self) -> None:
        target = str(self.outer_target)
        sources = tuple(str(item) for item in self.source_environments)
        if not target:
            raise CalibrationError("Calibration provenance requires a non-empty outer_target.")
        if not sources or len(set(sources)) != len(sources):
            raise CalibrationError("Calibration provenance requires distinct source environments.")
        if target in sources:
            raise CalibrationError("The outer target must not appear in source_environments.")
        if not self.calibration_sample_ids or not self.smoke_sample_ids:
            raise CalibrationError("Calibration and smoke sample IDs must both be recorded.")
        calibration_ids = tuple(map(str, self.calibration_sample_ids))
        smoke_ids = tuple(map(str, self.smoke_sample_ids))
        if len(set(calibration_ids)) != len(calibration_ids):
            raise CalibrationError("Calibration sample IDs must be unique.")
        if len(set(smoke_ids)) != len(smoke_ids):
            raise CalibrationError("Smoke sample IDs must be unique.")
        overlap = set(calibration_ids) & set(smoke_ids)
        if overlap:
            raise CalibrationError(
                "Calibration and smoke sample IDs must be disjoint; "
                f"found overlap such as {sorted(overlap)[:3]}."
            )
        calibration_counts = {source: 0 for source in sources}
        smoke_counts = {source: 0 for source in sources}
        for sample_id in calibration_ids:
            environment = _sample_environment(sample_id)
            if environment not in calibration_counts:
                raise CalibrationError(
                    f"Calibration sample ID '{sample_id}' is not prefixed by an allowed source environment."
                )
            calibration_counts[environment] += 1
        for sample_id in smoke_ids:
            environment = _sample_environment(sample_id)
            if environment not in smoke_counts:
                raise CalibrationError(
                    f"Smoke sample ID '{sample_id}' is not prefixed by an allowed source environment."
                )
            smoke_counts[environment] += 1
        invalid_calibration_counts = {
            source: count
            for source, count in calibration_counts.items()
            if count != CALIBRATION_SAMPLES_PER_SOURCE
        }
        if invalid_calibration_counts:
            raise CalibrationError(
                "Each source environment must contribute exactly "
                f"{CALIBRATION_SAMPLES_PER_SOURCE} calibration samples; "
                f"received {invalid_calibration_counts}."
            )
        invalid_smoke_counts = {
            source: count for source, count in smoke_counts.items() if count != SMOKE_SAMPLES_PER_SOURCE
        }
        if invalid_smoke_counts:
            raise CalibrationError(
                "Each source environment must contribute exactly "
                f"{SMOKE_SAMPLES_PER_SOURCE} smoke samples; received {invalid_smoke_counts}."
            )

        expected_hash_keys = {
            f"{source}.{kind}" for source in sources for kind in ("amp", "phase", "label")
        }
        actual_hashes = {str(key): str(value) for key, value in self.source_file_sha256.items()}
        if set(actual_hashes) != expected_hash_keys:
            raise CalibrationError(
                "source_file_sha256 keys must be exactly ENV.amp/ENV.phase/ENV.label for every source; "
                f"expected {sorted(expected_hash_keys)}, received {sorted(actual_hashes)}."
            )
        invalid_hashes = [key for key, value in actual_hashes.items() if not _SHA256_PATTERN.fullmatch(value)]
        if invalid_hashes:
            raise CalibrationError(
                f"Source file SHA256 values must be 64 hexadecimal characters; invalid keys={invalid_hashes}."
            )
        if not str(self.source_index_sha256):
            raise CalibrationError("The source index SHA256 must be recorded.")
        if not _SHA256_PATTERN.fullmatch(str(self.source_index_sha256)):
            raise CalibrationError("The source index SHA256 must be 64 hexadecimal characters.")
        if int(self.initialization_seed) != CALIBRATION_INITIALIZATION_SEED:
            raise CalibrationError(
                "Global-basis calibration must use initialization seed "
                f"{CALIBRATION_INITIALIZATION_SEED}."
            )


def validate_calibration_sample_membership(
    provenance: CalibrationProvenance,
    *,
    allowed_calibration_sample_ids: Iterable[str] | None = None,
    expected_calibration_sample_ids: Iterable[str] | None = None,
    allowed_smoke_sample_ids: Iterable[str] | None = None,
    expected_smoke_sample_ids: Iterable[str] | None = None,
) -> None:
    """Strictly bind recorded IDs to caller-supplied source-train membership.

    The caller still derives the allowlist/expected IDs from the split manifest
    and enforces the locked ``lowest_sample_ids`` selection rule. This helper
    prevents the calibration tensors from silently using any other IDs.
    """

    provenance.validate()

    def _check(
        recorded: Sequence[str],
        *,
        allowed: Iterable[str] | None,
        expected: Iterable[str] | None,
        label: str,
    ) -> None:
        recorded_ids = tuple(map(str, recorded))
        recorded_set = set(recorded_ids)
        if allowed is not None:
            allowed_set = set(map(str, allowed))
            unexpected = recorded_set - allowed_set
            if unexpected:
                raise CalibrationError(
                    f"Recorded {label} IDs are outside the source-train allowlist: {sorted(unexpected)[:3]}."
                )
        if expected is not None:
            expected_ids = tuple(map(str, expected))
            if len(set(expected_ids)) != len(expected_ids):
                raise CalibrationError(f"Expected {label} IDs must be unique.")
            expected_set = set(expected_ids)
            if recorded_ids != expected_ids:
                missing = sorted(expected_set - recorded_set)[:3]
                extra = sorted(recorded_set - expected_set)[:3]
                order_mismatch = not missing and not extra
                raise CalibrationError(
                    f"Recorded {label} IDs do not match the expected ordered IDs; "
                    f"missing={missing}, extra={extra}, order_mismatch={order_mismatch}."
                )

    _check(
        provenance.calibration_sample_ids,
        allowed=allowed_calibration_sample_ids,
        expected=expected_calibration_sample_ids,
        label="calibration",
    )
    _check(
        provenance.smoke_sample_ids,
        allowed=allowed_smoke_sample_ids,
        expected=expected_smoke_sample_ids,
        label="smoke",
    )


@dataclass(frozen=True)
class BasisPathCalibration:
    basis_mean: tuple[float, ...]
    basis_std: tuple[float, ...]
    singular_values: tuple[float, ...]
    svd_rank: int
    condition_number: float
    kappa: float
    calibration_tpp_rms: float
    calibration_control_unscaled_rms: float
    smoke_tpp_rms: float
    smoke_control_scaled_rms: float
    smoke_rms_ratio: float

    def validate(self) -> None:
        if len(self.basis_mean) != BASIS_COUNT or len(self.basis_std) != BASIS_COUNT:
            raise CalibrationError(f"Each path requires exactly {BASIS_COUNT} basis statistics.")
        if len(self.singular_values) != BASIS_COUNT:
            raise CalibrationError(f"Each path requires exactly {BASIS_COUNT} singular values.")
        values = (
            *self.basis_mean,
            *self.basis_std,
            *self.singular_values,
            self.condition_number,
            self.kappa,
            self.calibration_tpp_rms,
            self.calibration_control_unscaled_rms,
            self.smoke_tpp_rms,
            self.smoke_control_scaled_rms,
            self.smoke_rms_ratio,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise CalibrationError("Calibration artifacts may not contain non-finite values.")
        if any(float(value) <= 0.0 for value in self.basis_std):
            raise CalibrationError("Every basis standard deviation must be positive.")
        singular_values = tuple(float(value) for value in self.singular_values)
        if any(value <= 0.0 for value in singular_values):
            raise CalibrationError("Every stored singular value must be positive.")
        if any(left < right for left, right in zip(singular_values, singular_values[1:])):
            raise CalibrationError("Stored singular values must be in non-increasing order.")
        largest = singular_values[0]
        smallest = singular_values[-1]
        recomputed_rank = sum(value > largest * SVD_RELATIVE_TOLERANCE for value in singular_values)
        if int(self.svd_rank) != recomputed_rank or recomputed_rank != BASIS_COUNT:
            raise CalibrationError(
                "Global-basis stored rank is inconsistent with its singular values: "
                f"stored={self.svd_rank}, recomputed={recomputed_rank}, expected={BASIS_COUNT}."
            )
        recomputed_condition = largest / smallest
        condition_number = float(self.condition_number)
        if condition_number < 1.0:
            raise CalibrationError("Global-basis condition number must be at least one.")
        if not math.isclose(
            condition_number,
            recomputed_condition,
            rel_tol=_FLOAT_REL_TOL,
            abs_tol=_FLOAT_ABS_TOL,
        ):
            raise CalibrationError(
                "Stored condition number is inconsistent with its singular values: "
                f"stored={condition_number:.9g}, recomputed={recomputed_condition:.9g}."
            )
        if condition_number > SVD_CONDITION_MAX:
            raise CalibrationError(
                "Global-basis condition number exceeds the locked maximum: "
                f"{self.condition_number:.6g} > {SVD_CONDITION_MAX:.6g}."
            )
        rms_values = {
            "calibration_tpp_rms": float(self.calibration_tpp_rms),
            "calibration_control_unscaled_rms": float(self.calibration_control_unscaled_rms),
            "smoke_tpp_rms": float(self.smoke_tpp_rms),
            "smoke_control_scaled_rms": float(self.smoke_control_scaled_rms),
        }
        if any(value < 0.0 for value in rms_values.values()):
            raise CalibrationError(f"Residual RMS values must be non-negative; received {rms_values}.")
        if rms_values["calibration_control_unscaled_rms"] <= 0.0:
            raise CalibrationError("Calibration control RMS must be positive to define kappa.")
        if rms_values["smoke_tpp_rms"] <= 0.0:
            raise CalibrationError("Smoke TPP RMS must be positive to define the smoke ratio.")

        kappa = float(self.kappa)
        expected_kappa = rms_values["calibration_tpp_rms"] / (
            rms_values["calibration_control_unscaled_rms"] + 1e-8
        )
        if not math.isclose(kappa, expected_kappa, rel_tol=_FLOAT_REL_TOL, abs_tol=_FLOAT_ABS_TOL):
            raise CalibrationError(
                "Stored kappa is inconsistent with calibration RMS values: "
                f"stored={kappa:.9g}, recomputed={expected_kappa:.9g}."
            )
        if not KAPPA_BOUNDS[0] <= kappa <= KAPPA_BOUNDS[1]:
            raise CalibrationError(
                f"kappa={self.kappa:.6g} is outside the locked bounds {KAPPA_BOUNDS}."
            )
        smoke_rms_ratio = float(self.smoke_rms_ratio)
        expected_smoke_ratio = (
            rms_values["smoke_control_scaled_rms"] / rms_values["smoke_tpp_rms"]
        )
        if not math.isclose(
            smoke_rms_ratio,
            expected_smoke_ratio,
            rel_tol=_FLOAT_REL_TOL,
            abs_tol=_FLOAT_ABS_TOL,
        ):
            raise CalibrationError(
                "Stored smoke RMS ratio is inconsistent with smoke RMS values: "
                f"stored={smoke_rms_ratio:.9g}, recomputed={expected_smoke_ratio:.9g}."
            )
        if not SMOKE_RMS_RATIO_BOUNDS[0] <= smoke_rms_ratio <= SMOKE_RMS_RATIO_BOUNDS[1]:
            raise CalibrationError(
                "Smoke residual RMS ratio is outside the locked bounds: "
                f"{self.smoke_rms_ratio:.6g} not in {SMOKE_RMS_RATIO_BOUNDS}."
            )


@dataclass(frozen=True)
class GlobalBasisCalibrationArtifact:
    protocol_version: str
    basis_names: tuple[str, ...]
    provenance: CalibrationProvenance
    paths: Mapping[str, BasisPathCalibration]

    def validate(self, expected_paths: Sequence[str] | None = None) -> None:
        if self.protocol_version != CALIBRATION_PROTOCOL_VERSION:
            raise CalibrationError(
                f"Unsupported calibration protocol '{self.protocol_version}'; "
                f"expected '{CALIBRATION_PROTOCOL_VERSION}'."
            )
        if tuple(self.basis_names) != BASIS_NAMES:
            raise CalibrationError("Calibration artifact basis names do not match the locked protocol.")
        self.provenance.validate()
        if expected_paths is not None and set(self.paths) != set(expected_paths):
            raise CalibrationError(
                f"Calibration paths {sorted(self.paths)} do not match expected paths {sorted(expected_paths)}."
            )
        if not self.paths:
            raise CalibrationError("Calibration artifact contains no amplitude pooling paths.")
        for path_calibration in self.paths.values():
            path_calibration.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GlobalBasisCalibrationArtifact":
        provenance = CalibrationProvenance(
            outer_target=str(payload["provenance"]["outer_target"]),
            source_environments=tuple(payload["provenance"]["source_environments"]),
            calibration_sample_ids=tuple(map(str, payload["provenance"]["calibration_sample_ids"])),
            smoke_sample_ids=tuple(map(str, payload["provenance"]["smoke_sample_ids"])),
            source_file_sha256=dict(payload["provenance"]["source_file_sha256"]),
            source_index_sha256=str(payload["provenance"]["source_index_sha256"]),
            initialization_seed=int(payload["provenance"].get("initialization_seed", CALIBRATION_INITIALIZATION_SEED)),
        )
        paths = {
            str(name): BasisPathCalibration(
                basis_mean=tuple(map(float, values["basis_mean"])),
                basis_std=tuple(map(float, values["basis_std"])),
                singular_values=tuple(map(float, values["singular_values"])),
                svd_rank=int(values["svd_rank"]),
                condition_number=float(values["condition_number"]),
                kappa=float(values["kappa"]),
                calibration_tpp_rms=float(values["calibration_tpp_rms"]),
                calibration_control_unscaled_rms=float(values["calibration_control_unscaled_rms"]),
                smoke_tpp_rms=float(values["smoke_tpp_rms"]),
                smoke_control_scaled_rms=float(values["smoke_control_scaled_rms"]),
                smoke_rms_ratio=float(values["smoke_rms_ratio"]),
            )
            for name, values in payload["paths"].items()
        }
        artifact = cls(
            protocol_version=str(payload["protocol_version"]),
            basis_names=tuple(payload["basis_names"]),
            provenance=provenance,
            paths=paths,
        )
        artifact.validate()
        return artifact

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "GlobalBasisCalibrationArtifact":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class _BasisStatistics:
    mean: torch.Tensor
    std: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    condition_number: float


def exact_global_basis(global_feature: torch.Tensor) -> torch.Tensor:
    """Return the seven locked basis values with shape ``[batch, 7, feature]``."""

    if global_feature.ndim != 2:
        raise JointModelProtocolError(
            f"Global features must have shape [batch, feature], received {tuple(global_feature.shape)}."
        )
    if global_feature.shape[-1] < 2:
        raise JointModelProtocolError("LayerNorm-based global basis requires at least two features.")
    normalized = F.layer_norm(
        global_feature,
        (global_feature.shape[-1],),
        weight=None,
        bias=None,
        eps=BASIS_LAYER_NORM_EPS,
    )
    clipped = normalized.clamp(min=BASIS_CLIP_MIN, max=BASIS_CLIP_MAX)
    bases = (
        clipped,
        clipped.square(),
        clipped.pow(3) / 9.0,
        clipped.tanh(),
        torch.sin(math.pi * clipped / 3.0),
        torch.cos(math.pi * clipped / 3.0),
        clipped.sign() * torch.sqrt(clipped.abs() + BASIS_VALUE_EPS),
    )
    return torch.stack(bases, dim=1)


def derive_basis_statistics(
    global_feature: torch.Tensor,
    *,
    relative_tolerance: float = SVD_RELATIVE_TOLERANCE,
    condition_max: float = SVD_CONDITION_MAX,
) -> _BasisStatistics:
    """Fit source-only scalar basis statistics and enforce rank/conditioning gates."""

    raw_basis = exact_global_basis(global_feature)
    design = raw_basis.permute(0, 2, 1).reshape(-1, BASIS_COUNT).double()
    if design.shape[0] < BASIS_COUNT:
        raise CalibrationError(
            f"Basis SVD needs at least {BASIS_COUNT} rows, received {design.shape[0]}."
        )
    mean = design.mean(dim=0)
    std = design.std(dim=0, unbiased=False)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise CalibrationError("Basis calibration produced non-finite mean or standard deviation.")
    if torch.any(std <= BASIS_VALUE_EPS):
        raise CalibrationError("Degenerate global-basis column has near-zero standard deviation.")

    standardized = (design - mean) / (std + BASIS_VALUE_EPS)
    singular_values = torch.linalg.svdvals(standardized)
    if singular_values.numel() != BASIS_COUNT or not torch.isfinite(singular_values).all():
        raise CalibrationError("Global-basis SVD failed to produce seven finite singular values.")
    largest = float(singular_values[0].item())
    threshold = largest * float(relative_tolerance)
    rank = int((singular_values > threshold).sum().item()) if largest > 0.0 else 0
    smallest = float(singular_values[-1].item())
    condition_number = math.inf if smallest <= 0.0 else largest / smallest
    if rank != BASIS_COUNT:
        raise CalibrationError(
            f"Global-basis design is rank deficient: rank={rank}, expected={BASIS_COUNT}."
        )
    if not math.isfinite(condition_number) or condition_number > float(condition_max):
        raise CalibrationError(
            "Global-basis design is ill-conditioned: "
            f"condition={condition_number:.6g}, maximum={condition_max:.6g}."
        )
    return _BasisStatistics(
        mean=mean,
        std=std,
        singular_values=singular_values,
        rank=rank,
        condition_number=condition_number,
    )


def _standardized_basis_vector(
    global_feature: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    raw_basis = exact_global_basis(global_feature)
    mean = mean.to(device=global_feature.device, dtype=global_feature.dtype).view(1, BASIS_COUNT, 1)
    std = std.to(device=global_feature.device, dtype=global_feature.dtype).view(1, BASIS_COUNT, 1)
    standardized = (raw_basis - mean) / (std + BASIS_VALUE_EPS)
    return standardized.reshape(global_feature.shape[0], BASIS_COUNT * global_feature.shape[1])


def _rms(value: torch.Tensor) -> float:
    if value.numel() == 0:
        raise CalibrationError("Cannot compute RMS of an empty tensor.")
    rms = value.detach().double().square().mean().sqrt()
    result = float(rms.item())
    if not math.isfinite(result):
        raise CalibrationError("Residual RMS is non-finite.")
    return result


class MatchedResidualPyramidPool(IsolatedTemporalPyramidPool):
    """Parameter-identical TPP/global-basis pooling used by the factorial arms."""

    def __init__(
        self,
        input_dim: int,
        output_channels: int,
        kernel_sizes: Sequence[int],
        *,
        representation: str,
        dropout: float = 0.1,
    ) -> None:
        if representation not in {"tpp", "global_basis"}:
            raise JointModelProtocolError(
                f"Unsupported matched pooling representation '{representation}'."
            )
        super().__init__(
            input_dim=input_dim,
            output_channels=output_channels,
            kernel_sizes=kernel_sizes,
            levels=TPP_LEVELS,
            dropout=dropout,
            pool_type="max",
            preserve_global=True,
            pyramid_scale=TPP_RESIDUAL_SCALE,
        )
        self.representation = representation
        self.register_buffer("basis_mean", torch.zeros(BASIS_COUNT), persistent=True)
        self.register_buffer("basis_std", torch.ones(BASIS_COUNT), persistent=True)
        self.register_buffer("control_kappa", torch.tensor(1.0), persistent=True)
        self.register_buffer("calibration_loaded", torch.tensor(False, dtype=torch.bool), persistent=True)

    def extract_residual_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return original global max ``g`` and existing TPP vector ``z_T``."""

        x_t = x.transpose(1, 2)
        pyramid_parts = []
        global_parts = []
        for conv in self.convs:
            response = self.activation(conv(x_t))
            global_parts.append(F.adaptive_max_pool1d(response, output_size=1).flatten(start_dim=1))
            for level in self.levels:
                pyramid_parts.append(self.pool_response(response, level).flatten(start_dim=1))
        global_feature = torch.cat(global_parts, dim=1)
        pyramid_feature = torch.cat(pyramid_parts, dim=1)
        expected_width = TPP_SEGMENT_COUNT * global_feature.shape[1]
        if pyramid_feature.shape[1] != expected_width:
            raise JointModelProtocolError(
                f"TPP vector width {pyramid_feature.shape[1]} does not equal 7d={expected_width}."
            )
        return global_feature, pyramid_feature

    def global_basis_vector(self, global_feature: torch.Tensor) -> torch.Tensor:
        if not bool(self.calibration_loaded.item()):
            raise CalibrationError("Global-basis statistics have not been loaded for this pooling path.")
        return _standardized_basis_vector(global_feature, self.basis_mean, self.basis_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_feature, pyramid_feature = self.extract_residual_inputs(x)
        if self.representation == "tpp":
            residual = self.projection(pyramid_feature)
        else:
            basis_feature = self.global_basis_vector(global_feature)
            residual = self.control_kappa.to(dtype=global_feature.dtype) * self.projection(basis_feature)
        return self.dropout(global_feature + TPP_RESIDUAL_SCALE * residual)

    @torch.no_grad()
    def derive_path_calibration(
        self,
        calibration_global: torch.Tensor,
        calibration_tpp: torch.Tensor,
        smoke_global: torch.Tensor,
        smoke_tpp: torch.Tensor,
    ) -> BasisPathCalibration:
        stats = derive_basis_statistics(calibration_global)
        calibration_basis = _standardized_basis_vector(calibration_global, stats.mean, stats.std)
        calibration_tpp_residual = self.projection(calibration_tpp)
        calibration_control_residual = self.projection(calibration_basis)
        tpp_rms = _rms(calibration_tpp_residual)
        control_rms = _rms(calibration_control_residual)
        if control_rms <= 1e-8:
            raise CalibrationError("Unscaled global-basis residual RMS is too small to calibrate kappa.")
        kappa = tpp_rms / (control_rms + 1e-8)
        if not KAPPA_BOUNDS[0] <= kappa <= KAPPA_BOUNDS[1]:
            raise CalibrationError(f"kappa={kappa:.6g} is outside the locked bounds {KAPPA_BOUNDS}.")

        smoke_basis = _standardized_basis_vector(smoke_global, stats.mean, stats.std)
        smoke_tpp_residual = self.projection(smoke_tpp)
        smoke_control_residual = kappa * self.projection(smoke_basis)
        smoke_tpp_rms = _rms(smoke_tpp_residual)
        smoke_control_rms = _rms(smoke_control_residual)
        if smoke_tpp_rms <= 1e-8:
            raise CalibrationError("Smoke TPP residual RMS is too small for the RMS-ratio gate.")
        smoke_ratio = smoke_control_rms / smoke_tpp_rms
        artifact = BasisPathCalibration(
            basis_mean=tuple(float(value) for value in stats.mean.tolist()),
            basis_std=tuple(float(value) for value in stats.std.tolist()),
            singular_values=tuple(float(value) for value in stats.singular_values.tolist()),
            svd_rank=stats.rank,
            condition_number=stats.condition_number,
            kappa=kappa,
            calibration_tpp_rms=tpp_rms,
            calibration_control_unscaled_rms=control_rms,
            smoke_tpp_rms=smoke_tpp_rms,
            smoke_control_scaled_rms=smoke_control_rms,
            smoke_rms_ratio=smoke_ratio,
        )
        artifact.validate()
        return artifact

    @torch.no_grad()
    def load_path_calibration(self, calibration: BasisPathCalibration) -> None:
        calibration.validate()
        self.basis_mean.copy_(torch.tensor(calibration.basis_mean, device=self.basis_mean.device))
        self.basis_std.copy_(torch.tensor(calibration.basis_std, device=self.basis_std.device))
        self.control_kappa.copy_(torch.tensor(calibration.kappa, device=self.control_kappa.device))
        self.calibration_loaded.fill_(True)


class JointAmplitudeStreamEncoder(nn.Module):
    """Original stream backbone with a matched residual pool on amplitude only."""

    def __init__(self, input_dim: int, model_config: Mapping[str, Any], *, representation: str) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.representation = representation
        self.time_downsample = int(model_config.get("time_downsample", 2))
        self.position_encoding = GaussianRelativeEncoding(
            input_dim,
            max_positions=int(model_config.get("max_positions", 2048)),
            num_kernels=int(model_config.get("position_kernels", 10)),
        )
        self.temporal_encoder = MCATEncoder(
            input_dim,
            num_layers=int(model_config.get("mcat_layers", 2)),
            num_heads=compatible_heads(input_dim, int(model_config.get("mcat_heads", 6))),
            dropout=float(model_config.get("dropout", 0.1)),
        )
        self.temporal_pool = MatchedResidualPyramidPool(
            input_dim=input_dim,
            output_channels=int(model_config.get("temporal_kernel_num", 128)),
            kernel_sizes=list(model_config.get("temporal_kernel_sizes", [20, 40])),
            representation=representation,
            dropout=float(model_config.get("dropout", 0.1)),
        )

        self.channel_group_width = int(model_config.get("channel_group_width", 30))
        self.num_groups = input_dim // self.channel_group_width if input_dim % self.channel_group_width == 0 else 0
        self.use_channel_path = self.num_groups > 0
        if self.use_channel_path:
            self.channel_encoder = ChannelContextEncoder(
                feature_dim=self.channel_group_width,
                num_layers=int(model_config.get("channel_encoder_layers", 1)),
                num_heads=int(model_config.get("channel_encoder_heads", 6)),
                dropout=float(model_config.get("dropout", 0.1)),
            )
            self.channel_pool = MatchedResidualPyramidPool(
                input_dim=self.channel_group_width,
                output_channels=int(model_config.get("channel_kernel_num", 16)),
                kernel_sizes=list(model_config.get("channel_kernel_sizes", [2, 4])),
                representation=representation,
                dropout=float(model_config.get("dropout", 0.1)),
            )
            self.output_dim = self.temporal_pool.output_dim + self.channel_pool.output_dim
        else:
            self.output_dim = self.temporal_pool.output_dim

    def temporal_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.time_downsample <= 1:
            return x
        trimmed_steps = (x.size(1) // self.time_downsample) * self.time_downsample
        if trimmed_steps == 0:
            raise ValueError(
                f"Input time steps {x.size(1)} are shorter than time_downsample={self.time_downsample}."
            )
        x = x[:, :trimmed_steps, :]
        batch_size, _, feature_dim = x.shape
        return x.reshape(
            batch_size,
            trimmed_steps // self.time_downsample,
            self.time_downsample,
            feature_dim,
        ).mean(dim=2)

    def encode_pool_inputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.temporal_reduce(x)
        positioned = self.position_encoding(x)
        path_inputs = {"temporal": self.temporal_encoder(positioned)}
        if self.use_channel_path:
            channel_context = positioned.reshape(
                positioned.size(0),
                positioned.size(1),
                self.num_groups,
                self.channel_group_width,
            ).mean(dim=2)
            path_inputs["channel"] = self.channel_encoder(channel_context)
        return path_inputs

    def residual_pools(self) -> Dict[str, MatchedResidualPyramidPool]:
        pools = {"temporal": self.temporal_pool}
        if self.use_channel_path:
            pools["channel"] = self.channel_pool
        return pools

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        path_inputs = self.encode_pool_inputs(x)
        pooled = [self.temporal_pool(path_inputs["temporal"])]
        if self.use_channel_path:
            pooled.append(self.channel_pool(path_inputs["channel"]))
        return torch.cat(pooled, dim=1)


class JointPaCsiDGLite(nn.Module):
    """Four-arm joint model with amplitude-only matched pooling and V=1 SupCon."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        model_config: Mapping[str, Any],
        *,
        arm: str,
        initialization_seed: int | None = None,
    ) -> None:
        super().__init__()
        if arm not in JOINT_ARM_SPECS:
            raise JointModelProtocolError(f"Unknown joint arm '{arm}'; expected one of {ARM_NAMES}.")
        spec = JOINT_ARM_SPECS[arm]
        self.arm = arm
        self.temporal_mode = spec.temporal_mode
        self.supcon_enabled = spec.supcon_enabled
        self.supcon_view_scope = spec.supcon_view_scope
        self.initialization_seed = initialization_seed

        self.amp_encoder = JointAmplitudeStreamEncoder(
            input_dim,
            model_config,
            representation=spec.temporal_mode,
        )
        # Phase intentionally remains the unmodified global-pooling encoder in all arms.
        self.phase_encoder = PaCsiStreamEncoder(input_dim, model_config)
        if self.amp_encoder.output_dim != self.phase_encoder.output_dim:
            raise JointModelProtocolError(
                "Amplitude and phase output dimensions differ: "
                f"{self.amp_encoder.output_dim} != {self.phase_encoder.output_dim}."
            )

        fusion_dim = int(model_config.get("fusion_dim", 256))
        self.fusion = GatedResidualFusion(self.amp_encoder.output_dim, fusion_dim)
        self.classifier = nn.Linear(fusion_dim, num_classes)
        self.projection_head = ProjectionHead(
            input_dim=fusion_dim,
            hidden_dim=int(model_config.get("projection_hidden_dim", fusion_dim)),
            output_dim=int(model_config.get("projection_dim", 128)),
        )
        self.view_builder = None
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        self.projection_dim = int(model_config.get("projection_dim", 128))

        self.prototype_num_subcenters = int(model_config.get("prototype_num_subcenters", 0))
        if self.prototype_num_subcenters > 0:
            prototypes = torch.empty(num_classes, self.prototype_num_subcenters, self.projection_dim)
            nn.init.xavier_uniform_(prototypes)
            self.class_prototypes = nn.Parameter(prototypes)
        else:
            self.class_prototypes = None

    def has_class_prototypes(self) -> bool:
        return isinstance(self.class_prototypes, nn.Parameter) and self.prototype_num_subcenters > 0

    @property
    def calibration_ready(self) -> bool:
        if self.temporal_mode != "global_basis":
            return True
        return all(bool(pool.calibration_loaded.item()) for pool in self.amp_encoder.residual_pools().values())

    def load_calibration_artifact(self, artifact: GlobalBasisCalibrationArtifact) -> None:
        if self.temporal_mode != "global_basis":
            raise JointModelProtocolError("Calibration artifacts apply only to Y00/Y01 global-basis arms.")
        pools = self.amp_encoder.residual_pools()
        artifact.validate(expected_paths=tuple(pools))
        for name, pool in pools.items():
            pool.load_path_calibration(artifact.paths[name])

    def encode_features(self, amplitude: torch.Tensor, phase: torch.Tensor) -> Dict[str, torch.Tensor]:
        amp_feature = self.amp_encoder(amplitude)
        phase_feature = self.phase_encoder(phase)
        fused_feature = self.fusion(amp_feature, phase_feature)
        return {
            "amp_feature": amp_feature,
            "phase_feature": phase_feature,
            "fused_feature": fused_feature,
        }

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encode_features(amplitude, phase)
        fused_feature = features["fused_feature"]
        logits = self.classifier(fused_feature)
        projection = F.normalize(self.projection_head(fused_feature), dim=-1)
        features.update(
            {
                "logits": logits,
                "projection": projection,
                "supcon_projection_views": projection.unsqueeze(1),
            }
        )
        return features

    def encode_supcon_views(
        self,
        amplitude: torch.Tensor,
        phase: torch.Tensor,
        *,
        view_scope: str = SUPCON_VIEW_SCOPE,
    ) -> torch.Tensor:
        if view_scope != SUPCON_VIEW_SCOPE:
            raise JointModelProtocolError(
                f"Joint protocol supports only view_scope='{SUPCON_VIEW_SCOPE}', received '{view_scope}'."
            )
        return self.forward(amplitude, phase)["supcon_projection_views"]

    def encode_antenna_views(self, amplitude: torch.Tensor, phase: torch.Tensor) -> Dict[str, torch.Tensor]:
        del amplitude, phase
        raise JointModelProtocolError(
            "Antenna-view encoding is disabled: the joint protocol uses one global projection per sample."
        )


def _expected_joint_temporal_config(arm: str) -> Dict[str, Any]:
    if arm not in protocol.ARMS:
        raise JointModelProtocolError(f"Unknown joint arm '{arm}'; expected one of {ARM_NAMES}.")
    calibration = dict(protocol.GLOBAL_BASIS_CALIBRATION)
    return {
        "branch": "amplitude_only",
        "factor": protocol.ARMS[arm].temporal_factor,
        "global_basis": {
            "basis": [dict(item) for item in protocol.GLOBAL_BASIS],
            "calibration": calibration,
            "input_normalization": {
                "affine": False,
                "clip_max": float(calibration["clip_max"]),
                "clip_min": float(calibration["clip_min"]),
                "epsilon": float(calibration["layer_norm_epsilon"]),
                "name": "layer_norm",
            },
        },
        "levels": list(protocol.TPP_LEVELS),
        "pool_type": protocol.TPP_POOL_TYPE,
        "preserve_global": protocol.TPP_PRESERVE_GLOBAL,
        "residual_scale": protocol.TPP_RESIDUAL_SCALE,
    }


def _validate_locked_model_config(model_config: Mapping[str, Any], arm: str) -> None:
    joint_temporal = model_config.get("joint_temporal")
    if not isinstance(joint_temporal, Mapping):
        raise JointModelProtocolError("model.joint_temporal must be present as a mapping.")
    expected = _expected_joint_temporal_config(arm)
    actual = deepcopy(dict(joint_temporal))
    if actual != expected:
        raise JointModelProtocolError(
            "model.joint_temporal does not exactly match the locked protocol for "
            f"{arm}; expected={expected!r}, received={actual!r}."
        )


def build_joint_model(
    arm: str,
    input_dim: int,
    num_classes: int,
    model_config: Mapping[str, Any],
    *,
    initialization_seed: int | None = None,
    calibration_artifact: GlobalBasisCalibrationArtifact | None = None,
) -> JointPaCsiDGLite:
    """Construct one locked factorial arm without changing the caller's CPU RNG state."""

    _validate_locked_model_config(model_config, arm)
    if initialization_seed is None:
        model = JointPaCsiDGLite(input_dim, num_classes, model_config, arm=arm)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            model = JointPaCsiDGLite(
                input_dim,
                num_classes,
                model_config,
                arm=arm,
                initialization_seed=int(initialization_seed),
            )
    if calibration_artifact is not None:
        model.load_calibration_artifact(calibration_artifact)
    return model


def build_joint_arm_models(
    input_dim: int,
    num_classes: int,
    model_config: Mapping[str, Any],
    *,
    initialization_seed: int,
    calibration_artifact: GlobalBasisCalibrationArtifact | None = None,
) -> Dict[str, JointPaCsiDGLite]:
    """Build all four arms from identical RNG initialization for paired runs/audits."""

    models: Dict[str, JointPaCsiDGLite] = {}
    for arm in ARM_NAMES:
        arm_config = deepcopy(dict(model_config))
        if not isinstance(arm_config.get("joint_temporal"), Mapping):
            raise JointModelProtocolError("model.joint_temporal must be present for four-arm construction.")
        arm_config["joint_temporal"] = deepcopy(dict(arm_config["joint_temporal"]))
        arm_config["joint_temporal"]["factor"] = protocol.ARMS[arm].temporal_factor
        models[arm] = build_joint_model(
            arm,
            input_dim,
            num_classes,
            arm_config,
            initialization_seed=initialization_seed,
            calibration_artifact=calibration_artifact if arm in {"Y00", "Y01"} else None,
        )
    audit_joint_model_parameters(models, strict=True, check_initialization=True)
    return models


def audit_joint_model_parameters(
    models: Mapping[str, JointPaCsiDGLite],
    *,
    strict: bool = True,
    check_initialization: bool = False,
) -> Dict[str, Any]:
    """Audit exact trainable names/shapes/counts and every 7d-to-d residual projection."""

    if not models:
        raise JointModelProtocolError("Parameter audit requires at least one model.")
    signatures: Dict[str, tuple[tuple[str, tuple[int, ...], int], ...]] = {}
    counts: Dict[str, int] = {}
    projection_checks: Dict[str, Dict[str, Dict[str, int]]] = {}
    for arm, model in models.items():
        signature = tuple(
            (name, tuple(parameter.shape), parameter.numel())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        signatures[arm] = signature
        counts[arm] = sum(item[2] for item in signature)
        projection_checks[arm] = {}
        for path, pool in model.amp_encoder.residual_pools().items():
            expected_input = BASIS_COUNT * pool.output_dim
            projection_checks[arm][path] = {
                "input_features": pool.projection.in_features,
                "output_features": pool.projection.out_features,
                "expected_input_features": expected_input,
            }
            if pool.projection.in_features != expected_input or pool.projection.out_features != pool.output_dim:
                raise JointModelProtocolError(
                    f"{arm}/{path} projection is not the required 7d-to-d mapping."
                )

    reference_arm = next(iter(models))
    signature_match = all(signature == signatures[reference_arm] for signature in signatures.values())
    count_match = len(set(counts.values())) == 1
    initialization_match = True
    if check_initialization:
        reference_parameters = dict(models[reference_arm].named_parameters())
        for arm, model in models.items():
            parameters = dict(model.named_parameters())
            if parameters.keys() != reference_parameters.keys() or any(
                not torch.equal(parameters[name].detach().cpu(), reference_parameters[name].detach().cpu())
                for name in reference_parameters
            ):
                initialization_match = False
                break

    all_equal = signature_match and count_match and initialization_match
    if strict and not all_equal:
        raise JointModelProtocolError(
            "Joint arm parameter audit failed: "
            f"signature_match={signature_match}, count_match={count_match}, "
            f"initialization_match={initialization_match}."
        )
    return {
        "all_equal": all_equal,
        "signature_match": signature_match,
        "count_match": count_match,
        "initialization_match": initialization_match,
        "counts": counts,
        "projection_checks": projection_checks,
    }


def _as_amplitude_batches(batches: torch.Tensor | Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
    if isinstance(batches, torch.Tensor):
        return (batches,)
    return batches


@torch.no_grad()
def _collect_pool_components(
    model: JointPaCsiDGLite,
    amplitude_batches: torch.Tensor | Iterable[torch.Tensor],
) -> tuple[Dict[str, tuple[torch.Tensor, torch.Tensor]], int]:
    pools = model.amp_encoder.residual_pools()
    collected: Dict[str, Dict[str, list[torch.Tensor]]] = {
        name: {"global": [], "tpp": []} for name in pools
    }
    sample_count = 0
    was_training = model.training
    model.eval()
    try:
        for amplitude in _as_amplitude_batches(amplitude_batches):
            if not isinstance(amplitude, torch.Tensor) or amplitude.ndim != 3:
                raise CalibrationError("Calibration batches must be amplitude tensors shaped [batch, time, feature].")
            sample_count += int(amplitude.shape[0])
            path_inputs = model.amp_encoder.encode_pool_inputs(amplitude)
            for name, pool in pools.items():
                global_feature, tpp_feature = pool.extract_residual_inputs(path_inputs[name])
                collected[name]["global"].append(global_feature.detach())
                collected[name]["tpp"].append(tpp_feature.detach())
    finally:
        model.train(was_training)
    if sample_count == 0:
        raise CalibrationError("Calibration received no amplitude samples.")
    return {
        name: (
            torch.cat(values["global"], dim=0),
            torch.cat(values["tpp"], dim=0),
        )
        for name, values in collected.items()
    }, sample_count


@torch.no_grad()
def calibrate_global_basis_control(
    model: JointPaCsiDGLite,
    calibration_amplitude_batches: torch.Tensor | Iterable[torch.Tensor],
    smoke_amplitude_batches: torch.Tensor | Iterable[torch.Tensor],
    provenance: CalibrationProvenance,
    *,
    allowed_calibration_sample_ids: Iterable[str] | None = None,
    expected_calibration_sample_ids: Iterable[str] | None = None,
    allowed_smoke_sample_ids: Iterable[str] | None = None,
    expected_smoke_sample_ids: Iterable[str] | None = None,
) -> GlobalBasisCalibrationArtifact:
    """Derive, validate, load, and return one source-only outer-fold artifact.

    The caller is responsible for selecting the locked source-train sample IDs;
    this function verifies their recorded counts and calibration/smoke disjointness.
    """

    if model.temporal_mode != "global_basis":
        raise JointModelProtocolError("Calibration requires a Y00/Y01 global-basis model.")
    if model.initialization_seed != CALIBRATION_INITIALIZATION_SEED:
        raise CalibrationError(
            "Calibration model must be built with initialization_seed="
            f"{CALIBRATION_INITIALIZATION_SEED}."
        )
    validate_calibration_sample_membership(
        provenance,
        allowed_calibration_sample_ids=allowed_calibration_sample_ids,
        expected_calibration_sample_ids=expected_calibration_sample_ids,
        allowed_smoke_sample_ids=allowed_smoke_sample_ids,
        expected_smoke_sample_ids=expected_smoke_sample_ids,
    )
    calibration_components, calibration_count = _collect_pool_components(
        model,
        calibration_amplitude_batches,
    )
    smoke_components, smoke_count = _collect_pool_components(model, smoke_amplitude_batches)
    if calibration_count != len(provenance.calibration_sample_ids):
        raise CalibrationError(
            "Calibration tensor/sample-ID count mismatch: "
            f"{calibration_count} != {len(provenance.calibration_sample_ids)}."
        )
    if smoke_count != len(provenance.smoke_sample_ids):
        raise CalibrationError(
            f"Smoke tensor/sample-ID count mismatch: {smoke_count} != {len(provenance.smoke_sample_ids)}."
        )

    path_artifacts: Dict[str, BasisPathCalibration] = {}
    for name, pool in model.amp_encoder.residual_pools().items():
        calibration_global, calibration_tpp = calibration_components[name]
        smoke_global, smoke_tpp = smoke_components[name]
        path_artifacts[name] = pool.derive_path_calibration(
            calibration_global,
            calibration_tpp,
            smoke_global,
            smoke_tpp,
        )

    artifact = GlobalBasisCalibrationArtifact(
        protocol_version=CALIBRATION_PROTOCOL_VERSION,
        basis_names=BASIS_NAMES,
        provenance=provenance,
        paths=path_artifacts,
    )
    artifact.validate(expected_paths=tuple(model.amp_encoder.residual_pools()))
    # Commit buffers only after every path passes all gates.
    model.load_calibration_artifact(artifact)
    return artifact


__all__ = [
    "ARM_NAMES",
    "BASIS_NAMES",
    "BasisPathCalibration",
    "CALIBRATION_INITIALIZATION_SEED",
    "CALIBRATION_PROTOCOL_VERSION",
    "CalibrationError",
    "CalibrationProvenance",
    "GlobalBasisCalibrationArtifact",
    "JointModelProtocolError",
    "JointPaCsiDGLite",
    "MatchedResidualPyramidPool",
    "SUPCON_VIEW_SCOPE",
    "TPP_LEVELS",
    "TPP_RESIDUAL_SCALE",
    "audit_joint_model_parameters",
    "build_joint_arm_models",
    "build_joint_model",
    "calibrate_global_basis_control",
    "derive_basis_statistics",
    "exact_global_basis",
    "validate_calibration_sample_membership",
]
