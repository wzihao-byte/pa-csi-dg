"""Machine-checkable constants for the SupCon + TPP joint protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple


PROTOCOL_NAME = "supcon_tpp_joint"
PROTOCOL_VERSION = "1.2.0-stage0"
PROTOCOL_LOCK_SCHEMA = "supcon-tpp-protocol-lock/v2"
REGISTRY_SCHEMA = "supcon-tpp-job-registry/v2"
SCREEN_RESULTS_SCHEMA = "supcon-tpp-screen-results/v2"
SCREEN_HEALTH_SCHEMA = "supcon-tpp-screen-health/v1"
SELECTION_RECORD_SCHEMA = "supcon-tpp-lambda-selection/v1"
INTERIM_DECISION_SCHEMA = "supcon-tpp-interim-decision/v1"
BUNDLE_MANIFEST_SCHEMA = "supcon-tpp-bundle-manifest/v1"

ENVIRONMENTS: Tuple[str, ...] = ("E1", "E2", "E3")


@dataclass(frozen=True)
class ArmSpec:
    code: str
    temporal_factor: str
    supcon_enabled: bool


ARMS: Mapping[str, ArmSpec] = {
    "Y00": ArmSpec("Y00", "global_basis_control", False),
    "Y01": ArmSpec("Y01", "global_basis_control", True),
    "Y10": ArmSpec("Y10", "amplitude_residual_tpp", False),
    "Y11": ArmSpec("Y11", "amplitude_residual_tpp", True),
}


@dataclass(frozen=True)
class LabelSpec:
    label: int
    name: str
    support_per_environment: int
    support_per_group: int


LABELS: Mapping[int, LabelSpec] = {
    0: LabelSpec(0, "no_movement", 1200, 120),
    1: LabelSpec(1, "falling", 400, 40),
    2: LabelSpec(2, "walking", 400, 40),
    3: LabelSpec(3, "sitting_down_standing_up", 400, 40),
    4: LabelSpec(4, "turning", 400, 40),
    5: LabelSpec(5, "picking_up_a_pen", 200, 20),
}

# No verified subject sidecar exists on this machine.  The generated protocol
# therefore fails closed to anonymous, inferred group blocks.
SIDECAR_AVAILABLE = False
STRICT_GROUP_PROTOCOL = True
INNER_VALIDATION = "inferred_group_block_holdout"
GROUPS_PER_ENVIRONMENT = 10
GROUP_BLOCK_SIZE = 300
TRAIN_GROUPS_PER_SOURCE_ENVIRONMENT = 8
VALIDATION_GROUPS_PER_SOURCE_ENVIRONMENT = 2
SPLIT_SEED = 20260714
VALIDATION_GROUP_IDS: Mapping[str, Tuple[str, ...]] = {
    "E1": ("B01", "B07"),
    "E2": ("B00", "B06"),
    "E3": ("B00", "B04"),
}
# Backward-compatible symbol for model code; generated schema uses only
# ``validation_group_ids``.
F0_VALIDATION_BLOCKS = VALIDATION_GROUP_IDS

SCREEN_SEEDS: Mapping[str, int] = {"E1": 81, "E2": 82, "E3": 83}
FORMAL_INITIAL_SEEDS: Mapping[str, Tuple[int, ...]] = {
    "E1": (101, 102, 103),
    "E2": (201, 202, 203),
    "E3": (301, 302, 303),
}
FORMAL_EXTENSION_SEEDS: Mapping[str, Tuple[int, ...]] = {
    "E1": (104, 105),
    "E2": (204, 205),
    "E3": (304, 305),
}

EPOCHS = 200
SAMPLES_PER_SOURCE_DOMAIN_PER_BATCH = 8
SOURCE_DOMAINS_PER_OUTER_FOLD = 2
BATCH_SIZE = SAMPLES_PER_SOURCE_DOMAIN_PER_BATCH * SOURCE_DOMAINS_PER_OUTER_FOLD
OPTIMIZER_STEPS_PER_EPOCH = 300
TOTAL_OPTIMIZER_STEPS = EPOCHS * OPTIMIZER_STEPS_PER_EPOCH
SAMPLER = "domain_class_balanced"
CHECKPOINT_METRIC = "f1_macro"
PHASE_UNWRAP = False

LAMBDA_CANDIDATES: Tuple[float, ...] = (0.01, 0.025)
SUPCON_TEMPERATURE = 0.2
LINEAR_WARMUP_STEPS = 6000
SUPCON_VIEW_SCOPE = "global_only"

TPP_LEVELS: Tuple[int, ...] = (1, 2, 4)
TPP_POOL_TYPE = "max"
TPP_PRESERVE_GLOBAL = True
TPP_RESIDUAL_SCALE = 0.25

GLOBAL_BASIS: Tuple[Mapping[str, str], ...] = (
    {"name": "identity", "expression": "c"},
    {"name": "square", "expression": "c**2"},
    {"name": "scaled_cube", "expression": "c**3/9"},
    {"name": "tanh", "expression": "tanh(c)"},
    {"name": "sine", "expression": "sin(pi*c/3)"},
    {"name": "cosine", "expression": "cos(pi*c/3)"},
    {"name": "signed_sqrt", "expression": "sign(c)*sqrt(abs(c)+1e-6)"},
)
GLOBAL_BASIS_CALIBRATION: Mapping[str, object] = {
    "layer_norm_epsilon": 1e-5,
    "layer_norm_affine": False,
    "clip_min": -3.0,
    "clip_max": 3.0,
    "samples_per_source_environment": 256,
    "sample_selection": "lowest_sample_ids",
    "initialization_seed": 9001,
    "standardization_epsilon": 1e-6,
    "svd_relative_tolerance": 1e-6,
    "required_rank": 7,
    "maximum_condition_number": 1e4,
    "kappa_min": 0.25,
    "kappa_max": 4.0,
    "residual_rms_ratio_min": 0.8,
    "residual_rms_ratio_max": 1.25,
}

# F1/ratio thresholds are fractions (0.01 is one percentage point); epoch and
# gradient entries carry the units named in their keys.
SCREEN_THRESHOLDS: Mapping[str, float] = {
    "minimum_positive_anchor_fraction": 0.99,
    "maximum_combo_macro_f1_drop_vs_tpp": 0.02,
    "maximum_combo_sit_stand_f1_drop_vs_tpp": 0.03,
    "smallest_lambda_score_tolerance": 0.005,
    # Source-only, pre-registered training-health gates.  These deliberately
    # flag only clear failures: a model must beat neither a tuned baseline nor
    # another arm, only uniform six-class chance, and its late training CE must
    # show a modest improvement over its own early trajectory.
    "minimum_source_val_macro_f1": 1.0 / 6.0,
    "maximum_late_to_early_train_ce_ratio": 0.95,
    "train_ce_window_epochs": 20.0,
    # A finite gradient is called persistently explosive only when its epoch
    # p95 crosses a very conservative, early-run-relative threshold for three
    # consecutive epochs.  A separate catastrophic ceiling catches one-off
    # finite overflows before they can be hidden by epoch aggregation.
    "gradient_reference_epochs": 5.0,
    "gradient_relative_multiplier": 100.0,
    "gradient_absolute_floor_l2": 100.0,
    "gradient_catastrophic_l2": 1_000_000.0,
    "gradient_persistent_epochs": 3.0,
}
SUCCESS_THRESHOLDS: Mapping[str, float] = {
    "one_sided_confidence_level": 0.95,
    "minimum_combo_gain_vs_each_single": 0.01,
    "minimum_interaction_for_synergy": 0.01,
    "minimum_combo_gain_vs_control": 0.01,
    "sit_stand_noninferiority_margin_vs_control": -0.02,
    "minimum_sit_stand_recovery_vs_tpp": 0.05,
    "maximum_walking_drop_vs_tpp": 0.02,
    "minimum_hard_pair_absolute_reduction_vs_control": 0.025,
    "minimum_target_specific_gain_vs_each_single": -0.02,
}
FUTILITY_THRESHOLDS: Mapping[str, float] = {
    "sit_stand_harm_margin_vs_control": -0.02,
    "target_specific_mean_harm": -0.05,
    "target_specific_upper_bound": 0.0,
    "minimum_practical_effect": 0.01,
    "harm_confidence_level": 0.95,
    "futility_confidence_level": 0.90,
}

BUDGETS: Mapping[str, int] = {
    "smoke_runs": 4,
    "screen_runs": 18,
    "formal_interim_jobs": 36,
    "formal_extension_jobs": 24,
    "formal_full_jobs": 60,
    "internal_stop_path_full_runs": 54,
    "internal_complete_path_full_runs": 78,
    "interim_residual_degrees_of_freedom": 6,
    "full_residual_degrees_of_freedom": 12,
}


def source_environments(target_env: str) -> Tuple[str, ...]:
    if target_env not in ENVIRONMENTS:
        raise ValueError(f"Unknown target environment: {target_env}")
    return tuple(environment for environment in ENVIRONMENTS if environment != target_env)


def formal_seeds(target_env: str, stage: str) -> Tuple[int, ...]:
    if target_env not in ENVIRONMENTS:
        raise ValueError(f"Unknown target environment: {target_env}")
    if stage == "formal_interim":
        return FORMAL_INITIAL_SEEDS[target_env]
    if stage == "formal_extension":
        return FORMAL_EXTENSION_SEEDS[target_env]
    raise ValueError(f"Unknown formal stage: {stage}")


def linear_warmup_multiplier(step: int) -> float:
    """Return the registered step-based linear warm-up multiplier."""

    if not isinstance(step, int) or isinstance(step, bool):
        raise TypeError("Warm-up step must be an integer.")
    if step <= 0:
        return 0.0
    if step >= LINEAR_WARMUP_STEPS:
        return 1.0
    return float(step) / float(LINEAR_WARMUP_STEPS)


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_screen_run_count() -> int:
    return len(ENVIRONMENTS) * (2 + 2 * len(LAMBDA_CANDIDATES))


def expected_formal_job_count(stage: str) -> int:
    seeds_per_target = len(formal_seeds(ENVIRONMENTS[0], stage))
    return len(ENVIRONMENTS) * len(ARMS) * seeds_per_target


def protocol_lock_payload() -> Dict[str, object]:
    """Return the exact semantic content of ``protocol_lock.json``."""

    return {
        "schema_version": PROTOCOL_LOCK_SCHEMA,
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "environments": list(ENVIRONMENTS),
        "arms": {code: asdict(spec) for code, spec in ARMS.items()},
        "labels": {str(label): asdict(spec) for label, spec in LABELS.items()},
        "split": {
            "sidecar_available": SIDECAR_AVAILABLE,
            "strict_group_protocol": STRICT_GROUP_PROTOCOL,
            "inner_validation": INNER_VALIDATION,
            "subject_block_size": GROUP_BLOCK_SIZE,
            "groups_per_env": GROUPS_PER_ENVIRONMENT,
            "train_groups_per_source_env": TRAIN_GROUPS_PER_SOURCE_ENVIRONMENT,
            "validation_groups_per_source_env": VALIDATION_GROUPS_PER_SOURCE_ENVIRONMENT,
            "validation_group_ids": {
                environment: list(groups) for environment, groups in VALIDATION_GROUP_IDS.items()
            },
            "split_seed": SPLIT_SEED,
        },
        "screen_seeds": dict(SCREEN_SEEDS),
        "formal_initial_seeds": {
            environment: list(seeds) for environment, seeds in FORMAL_INITIAL_SEEDS.items()
        },
        "formal_extension_seeds": {
            environment: list(seeds) for environment, seeds in FORMAL_EXTENSION_SEEDS.items()
        },
        "training": {
            "epochs": EPOCHS,
            "samples_per_source_domain_per_batch": SAMPLES_PER_SOURCE_DOMAIN_PER_BATCH,
            "batch_size": BATCH_SIZE,
            "optimizer_steps_per_epoch": OPTIMIZER_STEPS_PER_EPOCH,
            "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
            "sampler": SAMPLER,
            "checkpoint_metric": CHECKPOINT_METRIC,
        },
        "phase_unwrap": PHASE_UNWRAP,
        "lambda_candidates": list(LAMBDA_CANDIDATES),
        "supcon": {
            "temperature": SUPCON_TEMPERATURE,
            "linear_warmup_steps": LINEAR_WARMUP_STEPS,
            "view_scope": SUPCON_VIEW_SCOPE,
        },
        "tpp": {
            "levels": list(TPP_LEVELS),
            "pool_type": TPP_POOL_TYPE,
            "preserve_global": TPP_PRESERVE_GLOBAL,
            "residual_scale": TPP_RESIDUAL_SCALE,
        },
        "global_basis": [dict(item) for item in GLOBAL_BASIS],
        "global_basis_calibration": dict(GLOBAL_BASIS_CALIBRATION),
        "screen_thresholds": dict(SCREEN_THRESHOLDS),
        "success_thresholds": dict(SUCCESS_THRESHOLDS),
        "futility_thresholds": dict(FUTILITY_THRESHOLDS),
        "budgets": dict(BUDGETS),
        "artifact_schemas": {
            "registry": REGISTRY_SCHEMA,
            "screen_results": SCREEN_RESULTS_SCHEMA,
            "screen_health": SCREEN_HEALTH_SCHEMA,
            "selection_record": SELECTION_RECORD_SCHEMA,
            "interim_decision": INTERIM_DECISION_SCHEMA,
            "bundle_manifest": BUNDLE_MANIFEST_SCHEMA,
        },
    }


def protocol_lock_sha256() -> str:
    return canonical_sha256(protocol_lock_payload())


def validate_protocol_constants() -> None:
    if tuple(ARMS) != ("Y00", "Y01", "Y10", "Y11"):
        raise ValueError("The arm set must be exactly Y00/Y01/Y10/Y11.")
    if tuple(LABELS) != tuple(range(6)):
        raise ValueError("Labels must be the contiguous IDs 0..5.")
    if LABELS[2].name != "walking" or LABELS[3].name != "sitting_down_standing_up":
        raise ValueError("Label 2/3 semantics drifted.")
    if SIDECAR_AVAILABLE or INNER_VALIDATION != "inferred_group_block_holdout":
        raise ValueError("This machine must use the strict inferred-group protocol until a sidecar is verified.")
    if any(len(groups) != 2 for groups in VALIDATION_GROUP_IDS.values()):
        raise ValueError("F0 must hold out exactly two groups per environment.")
    if BATCH_SIZE != 16 or SAMPLES_PER_SOURCE_DOMAIN_PER_BATCH != 8:
        raise ValueError("The outer-fold batch must remain 8 samples per source domain.")
    if TOTAL_OPTIMIZER_STEPS != 60_000 or LINEAR_WARMUP_STEPS != 6_000:
        raise ValueError("Training or warm-up step budget drifted.")
    if PHASE_UNWRAP is not False:
        raise ValueError("Formal phase unwrap must remain disabled.")
    if expected_screen_run_count() != BUDGETS["screen_runs"]:
        raise ValueError("Screen budget mismatch.")
    if SCREEN_THRESHOLDS["minimum_source_val_macro_f1"] != 1.0 / len(LABELS):
        raise ValueError("Screen convergence chance floor must be uniform six-class macro-F1.")
    if int(SCREEN_THRESHOLDS["train_ce_window_epochs"]) != 20:
        raise ValueError("Screen convergence must compare the registered 20-epoch CE windows.")
    if int(SCREEN_THRESHOLDS["gradient_reference_epochs"]) != 5:
        raise ValueError("Screen gradient baseline must use the first five epochs.")
    if int(SCREEN_THRESHOLDS["gradient_persistent_epochs"]) != 3:
        raise ValueError("Persistent explosion must require three consecutive epochs.")
    if expected_formal_job_count("formal_interim") != BUDGETS["formal_interim_jobs"]:
        raise ValueError("Interim budget mismatch.")
    if expected_formal_job_count("formal_extension") != BUDGETS["formal_extension_jobs"]:
        raise ValueError("Extension budget mismatch.")
    all_seeds = [
        seed
        for stage in ("formal_interim", "formal_extension")
        for environment in ENVIRONMENTS
        for seed in formal_seeds(environment, stage)
    ]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("Formal seeds must be unique across target and stage.")


validate_protocol_constants()
