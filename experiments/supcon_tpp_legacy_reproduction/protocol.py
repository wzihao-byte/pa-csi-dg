from __future__ import annotations

import os
from pathlib import Path


PROTOCOL_NAME = "supcon_tpp_legacy_reproduction"
PROTOCOL_VERSION = "1.0.0"
LEGACY_COMMIT = "7b8ec0947846daef65c6c6f37b912f9eeb5efcd6"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEAN_WORKTREE = Path(
    os.environ.get(
        "PA_CSI_LEGACY_WORKTREE",
        REPO_ROOT.parent / f"{REPO_ROOT.name}-legacy-replay-7b8ec09",
    )
)
OUTPUT_ROOT = REPO_ROOT / "outputs_supcon_tpp_legacy_reproduction"
GENERATED_ROOT = Path(__file__).resolve().parent / "generated_v1"

TARGETS = ("E1", "E2", "E3")
SEEDS = (42, 52, 62)
ENV_ORDER = ("E1", "E2", "E3")

DATA_ROOT = Path(os.environ.get("PA_CSI_DATA_ROOT", REPO_ROOT / "data" / "MultiEnv"))

DATA_FILES = {
    "E1": {
        "amp": {
            "path": DATA_ROOT / "data_6c_1.npy",
            "sha256": "26e4700ac8b9f0b1af3feca99d05d8502251c0274b3ce8718873179416aaa74a",
            "shape": (3000, 850, 90),
            "dtype": "float64",
        },
        "phase": {
            "path": DATA_ROOT / "data_angle_6c_1.npy",
            "sha256": "14f0837b28188b3cce9bea02ec1f19fddf0ee2f1abba36ade0b2531faa3ece02",
            "shape": (3000, 850, 90),
            "dtype": "float64",
        },
        "label": {
            "path": DATA_ROOT / "label_6c_1.npy",
            "sha256": "619e56cc7768625b0f2cdff8f973501b3b09ea2bf9300a045470e7a99c7a8b85",
            "shape": (3000,),
            "dtype": "int32",
        },
    },
    "E2": {
        "amp": {
            "path": DATA_ROOT / "data_6c_2.npy",
            "sha256": "83040641875495ccea66b1cc92dde3619a6dd74c0da62e9ce07a31ba19419ed5",
            "shape": (3000, 850, 90),
            "dtype": "float64",
        },
        "phase": {
            "path": DATA_ROOT / "data_angle_6c_2.npy",
            "sha256": "809c21b2de876bdede92e7e42ac1b6734863ad6f1b225c9c963440906a9961e7",
            "shape": (3000, 850, 90),
            "dtype": "float64",
        },
        "label": {
            "path": DATA_ROOT / "label_6c_2.npy",
            "sha256": "619e56cc7768625b0f2cdff8f973501b3b09ea2bf9300a045470e7a99c7a8b85",
            "shape": (3000,),
            "dtype": "int32",
        },
    },
    "E3": {
        "amp": {
            "path": DATA_ROOT / "data_6c_3.npy",
            "sha256": "daac7fd38677369c7450e099e87761c4f8d30e263f722d1fe27ab0b5577fce89",
            "shape": (3000, 850, 90),
            "dtype": "float64",
        },
        "phase": {
            "path": DATA_ROOT / "data_angle_6c_3.npy",
            "sha256": "6fd35344c48d420501c8ea53ba07cc1757c38a6537426f1d8ea81401ef56850e",
            "shape": (3000, 850, 90),
            "dtype": "float64",
        },
        "label": {
            "path": DATA_ROOT / "label_6c_3.npy",
            "sha256": "619e56cc7768625b0f2cdff8f973501b3b09ea2bf9300a045470e7a99c7a8b85",
            "shape": (3000,),
            "dtype": "int32",
        },
    },
}

LABEL_COUNTS = {0: 1200, 1: 400, 2: 400, 3: 400, 4: 400, 5: 200}
LABEL_NAMES = {
    0: "no_movement",
    1: "falling",
    2: "walking",
    3: "sit_stand",
    4: "turning",
    5: "pick_up_pen",
}

REFERENCE_PROVENANCE = {
    "anchor_supcon": {
        "artifact_complete": True,
        "matrix_complete": True,
        "training_code_commit": LEGACY_COMMIT,
        "per_run_csv": str(REPO_ROOT / "supcon_final_eval_per_run.csv"),
        "per_run_csv_sha256": "91377a944e6fdd5d459be1246089825cac74888361ef15930708de26c6ddfbb2",
        "aggregate_csv": str(REPO_ROOT / "supcon_final_eval_aggregate.csv"),
        "aggregate_csv_sha256": "331e77cf56d8ec23f973ddee45dc2f45a7d4ee1cb3da1e3eb1f59b1a1f7d2a88",
    },
    "anchor_ce": {
        "artifact_complete": True,
        "matrix_complete": True,
        "training_code_commit": LEGACY_COMMIT,
        "artifact_layout": "seed42 is in outputs_anchor_ce_baseline; seeds52/62 are in outputs_tpp_isolated/baseline_ref",
        "combined_reference_note": "No single historical CSV contains all nine CE cells.",
    },
    "anchor_tpp_l124": {
        "artifact_complete": True,
        "matrix_complete": True,
        "training_code_commit": None,
        "training_code_provenance": False,
        "warning": "The unversioned TPP source files currently on disk are newer than the l124 runs.",
    },
    "anchor_tpp_residual025_provisional": {
        "artifact_complete": True,
        "matrix_complete": False,
        "available_seeds": [52, 62],
        "missing_cells": ["E1/seed_42", "E2/seed_42", "E3/seed_42"],
        "training_code_commit": None,
        "code_unversioned": True,
    },
    "external_supcon_validation": {
        "primary_anchor": False,
        "matrix_complete": False,
        "best_complete_cell": "E2/seed_42",
        "macro_f1": 0.8565312774764505,
        "accuracy": 0.874,
        "warning": "Single-cell exploratory output without git or resolved-config provenance.",
    },
}

UNVERSIONED_TPP_SNAPSHOT = {
    "train_tpp_isolated.py": {
        "path": str(REPO_ROOT / "experiments" / "tpp_isolated" / "train_tpp_isolated.py"),
        "sha256": "e209c7e4d4bcee617040ba896cd74e9183b5aa65b4cd9c00ab7d2dac6d1f133d",
    },
    "tpp_models.py": {
        "path": str(REPO_ROOT / "experiments" / "tpp_isolated" / "tpp_models.py"),
        "sha256": "fc6aeb3858763c282b892f8c8e91ac31425d93df4343d83bb33cd1c6749ebccf",
    },
    "tpp_l124.json": {
        "path": str(REPO_ROOT / "experiments" / "tpp_isolated" / "configs" / "tpp_l124.json"),
        "sha256": "1ce64320af38f4836c14f868dec2e35cd7d4fc581440babaa24f23c391a4ede1",
    },
    "tpp_c3_residual025_l124.json": {
        "path": str(REPO_ROOT / "experiments" / "tpp_isolated" / "configs" / "tpp_c3_residual025_l124.json"),
        "sha256": "01086be6ccf198360a170a4ed0a74768bd4e1f587bd2a221204700aafd1491d9",
    },
}

# One historical resolved config per anchor family.  These are immutable
# witnesses used to prove that the newly generated anchor templates have the
# same execution-relevant fields after removing only run identity/output
# fields (experiment name, target, seed, output root and device_actual).
LEGACY_RESOLVED_CONFIG_REFERENCES = {
    "anchor_supcon": {
        "path": str(REPO_ROOT / "outputs_supcon_final_eval" / "dg_loeo" / "pa_csi_dg_supcon_final_lam0050_warm000_tau020_bs32_gci" / "target_E1" / "seed_42" / "resolved_config.json"),
        "sha256": "c4bdb683047301c06b4c12d689e606c9392f4e83a59e7acd4438c4565c20f823",
    },
    "anchor_ce": {
        "path": str(REPO_ROOT / "outputs_anchor_ce_baseline" / "dg_loeo" / "anchor_ce_baseline" / "target_E1" / "seed_42" / "resolved_config.json"),
        "sha256": "c4a725c129a2abf91477af174d0e2a8489724622394a20421c494ac7f747c377",
    },
    "anchor_tpp_l124": {
        "path": str(REPO_ROOT / "outputs_tpp_isolated" / "dg_loeo" / "tpp_l124" / "target_E1" / "seed_42" / "resolved_config.json"),
        "sha256": "cca978e6482f9dc2ffe75e0f9bae6df37046c2944a978ba9d0882aee31a25c1d",
    },
    "anchor_tpp_residual025": {
        "path": str(REPO_ROOT / "outputs_tpp_c3_recovery" / "dg_loeo" / "tpp_c3_residual025_l124" / "target_E1" / "seed_52" / "resolved_config.json"),
        "sha256": "68e6da532aea0a56b31ea8cfcaf57a7388a13158efa78209e29dce43501c112a",
    },
}

COMMON_MODEL = {
    "time_downsample": 2,
    "max_positions": 500,
    "position_kernels": 10,
    "mcat_layers": 5,
    "mcat_heads": 9,
    "temporal_kernel_num": 128,
    "temporal_kernel_sizes": [20, 40],
    "channel_group_width": 30,
    "channel_encoder_layers": 1,
    "channel_encoder_heads": 6,
    "channel_kernel_num": 16,
    "channel_kernel_sizes": [2, 4],
    "fusion_dim": 256,
    "projection_hidden_dim": 256,
    "projection_dim": 128,
    "dropout": 0.1,
    "antenna_layout": {
        "mode": "tx",
        "num_rx": 1,
        "num_tx": 3,
        "num_subcarriers": 30,
    },
}

TPP_RESIDUAL025 = {
    "tpp_levels": [1, 2, 4],
    "channel_tpp_levels": [1, 2, 4],
    "tpp_pool_type": "max",
    "channel_tpp_pool_type": "max",
    "tpp_mixed_max_weight": 0.5,
    "channel_tpp_mixed_max_weight": 0.5,
    "tpp_preserve_global": True,
    "channel_tpp_preserve_global": True,
    "tpp_pyramid_scale": 0.25,
    "channel_tpp_pyramid_scale": 0.25,
}

# Historical residual config relied on the channel path inheriting the
# temporal TPP values.  Anchor replay keeps this compact representation; the
# expanded effective values above remain locked for audit/factorial use.
TPP_RESIDUAL025_HISTORICAL_FIELDS = {
    "tpp_levels": [1, 2, 4],
    "tpp_pool_type": "max",
    "tpp_preserve_global": True,
    "tpp_pyramid_scale": 0.25,
}

COMMON_OPTIMIZER = {
    "epochs": 200,
    "lr": 0.001,
    "optimizer": "adam",
    "lr_scheduler": {
        "name": "exponential_decay",
        "update": "step",
        "decay_rate": 0.9,
        "decay_steps": 10000,
        "staircase": False,
    },
    "weight_decay": 0.0,
    "patience": 200,
    "selection_metric": "accuracy",
    "num_workers": 0,
    "domain_balanced_batch": False,
    "grad_clip_norm": 0.0,
}

COMMON_LOSSES = {
    "temperature": 0.2,
    "include_same_domain_same_class": True,
    "lambda_supcon_warmup_epochs": 0,
    "supcon_positive_mode": "global_class_instance_views",
    "contrastive_loss_type": "pairwise_supcon",
}

EXPECTED_PARAMETERS = {
    "pa_csi_dg_lite": 2_788_918,
    "isolated_pa_csi_dg_lite_tpp": 3_721_334,
}

REFERENCE_AGGREGATES = {
    "anchor_ce": {
        "E1": 0.862654267694322,
        "E2": 0.773163176037139,
        "E3": 0.776874240880971,
        "overall": 0.8042305615374775,
        "runs": 9,
    },
    "anchor_supcon": {
        "E1": 0.869359768233068,
        "E2": 0.827050473576008,
        "E3": 0.8368718713216716,
        "overall": 0.8444273710435826,
        "runs": 9,
    },
    "anchor_tpp_l124": {
        "E1": 0.819374282263638,
        "E2": 0.75795509752873,
        "E3": 0.777081030000157,
        "overall": 0.7848034699308416,
        "runs": 9,
    },
    "anchor_tpp_residual025_provisional": {
        "E1": 0.869136771728237,
        "E2": 0.802971610322479,
        "E3": 0.892563932341407,
        "overall": 0.854890771464041,
        "runs": 6,
        "available_seeds": [52, 62],
        "missing_seed": 42,
    },
}

SUPCON_REFERENCE_CELLS = {
    ("E1", 42): {"best_epoch": 72, "best_epoch_derived": True, "best_val_accuracy": 0.9983333333333333, "accuracy": 0.8966666666666666, "f1_macro": 0.8632369330841257},
    ("E1", 52): {"best_epoch": 53, "best_epoch_derived": True, "best_val_accuracy": 1.0, "accuracy": 0.903, "f1_macro": 0.8827100829531513},
    ("E1", 62): {"best_epoch": 200, "best_epoch_derived": True, "best_val_accuracy": 0.995, "accuracy": 0.885, "f1_macro": 0.8621322886619271},
    ("E2", 42): {"best_epoch": 126, "best_epoch_derived": True, "best_val_accuracy": 0.9991666666666666, "accuracy": 0.858, "f1_macro": 0.8307236167905083},
    ("E2", 52): {"best_epoch": 121, "best_epoch_derived": True, "best_val_accuracy": 0.9991666666666666, "accuracy": 0.8496666666666667, "f1_macro": 0.8132365875173516},
    ("E2", 62): {"best_epoch": 198, "best_epoch_derived": True, "best_val_accuracy": 0.9983333333333333, "accuracy": 0.8556666666666667, "f1_macro": 0.8371912164201641},
    ("E3", 42): {"best_epoch": 96, "best_epoch_derived": True, "best_val_accuracy": 0.9991666666666666, "accuracy": 0.8296666666666667, "f1_macro": 0.8237664832191314},
    ("E3", 52): {"best_epoch": 143, "best_epoch_derived": True, "best_val_accuracy": 0.9991666666666666, "accuracy": 0.841, "f1_macro": 0.8349506406969379},
    ("E3", 62): {"best_epoch": 106, "best_epoch_derived": True, "best_val_accuracy": 0.9966666666666667, "accuracy": 0.86, "f1_macro": 0.8518984900489456},
}

# Deterministically reconstructed from the 7b8ec094 split function and the
# locked labels. Historical manifests did not contain indices or these hashes.
RECONSTRUCTED_SPLIT_HASHES = {
    ("E1", 42): {"train": "0f44ac7f25e1290a06d05323aafeb573bf3ed8ee5b09659577e41bc1935e7cb8", "val": "ca8b840cbcdca4d3481eab3bc80597ca3d259e4411bfd5ead66a9298a674bb13", "test": "e8c9ceaf5aacc63c25b4cdd8542592f9d58aff50e3e8fc6c55591d3d8f596562"},
    ("E1", 52): {"train": "d2ca290230c040220cbaee77a5e1617e92f106b66f055933bb5d0f5d4d79c5fe", "val": "086d9fe9592ebb6b7dfe398ea566b7336ed128dd2ef5c5537e359ff452b15d48", "test": "e8c9ceaf5aacc63c25b4cdd8542592f9d58aff50e3e8fc6c55591d3d8f596562"},
    ("E1", 62): {"train": "0c4984898b48b59c4b3ce83e127cf610e7273a4fd44b86dae445aec66c24842c", "val": "cfcc0fff9f4ff1094754ec8f572899d8a38becd304bd9ca26a1a8822dffbe173", "test": "e8c9ceaf5aacc63c25b4cdd8542592f9d58aff50e3e8fc6c55591d3d8f596562"},
    ("E2", 42): {"train": "865d9a94823fdc43ae78674eb55a103c027764713a4f859acb0bd071dca2a48d", "val": "251e39c0fe93bbf1427f1b99f00e1427fe504941f4df323b64fa4babb777c300", "test": "df541fe3c6cba6b4a4003c7e3b6e70ea47c85e0976c3c1842a268401e2d3ddab"},
    ("E2", 52): {"train": "c11b95e183f45b43f1578f4ee98e3bd75a813bcc3826cc879b55dd7f9130e800", "val": "ac63276a5d4322e45534aff1963ab8a97c27f04056307182802f548c60be25ef", "test": "df541fe3c6cba6b4a4003c7e3b6e70ea47c85e0976c3c1842a268401e2d3ddab"},
    ("E2", 62): {"train": "4b3d0afc7bf1758949abaf5634d32f98e4fd7e1b9221b05bfea52b5bce744601", "val": "2c97cea52960971ebb1799fa649489404a1fc9e936a961fdd3758d5200ced275", "test": "df541fe3c6cba6b4a4003c7e3b6e70ea47c85e0976c3c1842a268401e2d3ddab"},
    ("E3", 42): {"train": "e58fe2824d1443011e3638d80999e7b390172c776634956870a65d4aa5102dfb", "val": "483638e65f222305406a386efc6c269bf9e0447c461c8c4668ce8d85ab7e0b42", "test": "61014996542255421ced7f4f3fbbdd8d5bfa09e023b5c88c23bdc3f95b79a136"},
    ("E3", 52): {"train": "46ceed39f5aa066908880bb33a713c7a0af74dc54411823dff82e502ed6722ec", "val": "b6caa457dfc2af94c76f9b1f30da7dcd881366e662b2c34a2456af007397948a", "test": "61014996542255421ced7f4f3fbbdd8d5bfa09e023b5c88c23bdc3f95b79a136"},
    ("E3", 62): {"train": "ccee8906acd23df8628c4092b6e7c60e1130a2852a8624899e1356b3cb7ea71e", "val": "f1b5f4e88c37648a7a8f59f97292266fd0b0866d014a5039bf5cf773460af7ae", "test": "61014996542255421ced7f4f3fbbdd8d5bfa09e023b5c88c23bdc3f95b79a136"},
}

REPRODUCTION_TOLERANCE = {
    "per_run_accuracy_abs": 0.0033333333333333335,
    "per_run_macro_f1_abs": 0.005,
    "per_target_mean_macro_f1_abs": 0.003,
    "overall_mean_macro_f1_abs": 0.002,
}

FACTORIAL_SUCCESS = {
    "overall_noninferiority_margin": -0.01,
    "per_environment_maximum_drop": -0.02,
    "pick_up_pen_overall_margin": -0.03,
    "pick_up_pen_per_environment_margin": -0.05,
    "minimum_better_gain": 0.01,
    "minimum_interaction": 0.01,
}
