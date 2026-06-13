from __future__ import annotations

import sys
from pathlib import Path
from math import isclose
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import train_dg  # noqa: E402


def parse_with(argv: List[str]):
    original_argv = sys.argv[:]
    try:
        sys.argv = ["train_dg.py", *argv]
        return train_dg.parse_args()
    finally:
        sys.argv = original_argv


def test_cli_overrides_land_in_runtime_config() -> None:
    raw_config = {
        "experiment_name": "base_experiment",
        "mode": "dg_loeo",
        "target_env": "all",
        "device": "cpu",
        "output_root": "outputs",
        "seed_list": [42],
        "training": {
            "epochs": 200,
            "batch_size": 8,
            "sampler": "none",
        },
        "losses": {
            "temperature": 0.2,
            "lambda_supcon": 0.1,
            "include_same_domain_same_class": True,
        },
    }

    args = parse_with(
        [
            "--config",
            "dummy.json",
            "--target-env",
            "E2",
            "--device",
            "cuda:0",
            "--output-root",
            "outputs_supcon_lambda_sweep",
            "--seeds",
            "7",
            "11",
            "--epochs",
            "80",
            "--lambda-supcon",
            "0.025",
            "--lambda-supcon-warmup-epochs",
            "20",
            "--temperature",
            "0.2",
            "--supcon-positive-mode",
            "all_views",
            "--include-same-domain-same-class",
            "0",
            "--batch-size",
            "16",
            "--sampler",
            "domain_class_balanced",
            "--experiment-name",
            "pa_csi_dg_supcon_lam0025_tau020",
            "--contrastive-loss-type",
            "pairwise_supcon",
            "--lambda-pair-margin",
            "0.05",
            "--prototype-num-subcenters",
            "3",
        ]
    )
    config = train_dg.resolve_runtime_config(raw_config, args)

    assert config["target_env"] == "E2"
    assert config["device"] == "cuda:0"
    assert config["output_root"] == "outputs_supcon_lambda_sweep"
    assert config["seed_list"] == [7, 11]
    assert config["experiment_name"] == "pa_csi_dg_supcon_lam0025_tau020"
    assert config["training"]["epochs"] == 80
    assert config["training"]["batch_size"] == 16
    assert config["training"]["sampler"] == "domain_class_balanced"
    assert config["losses"]["lambda_supcon"] == 0.025
    assert config["losses"]["lambda_supcon_warmup_epochs"] == 20
    assert config["losses"]["temperature"] == 0.2
    assert config["losses"]["supcon_positive_mode"] == "all_views"
    assert config["losses"]["include_same_domain_same_class"] is False
    assert config["losses"]["contrastive_loss_type"] == "pairwise_supcon"
    assert config["losses"]["lambda_pair_margin"] == 0.05
    assert config["model"]["prototype_num_subcenters"] == 3


def test_new_cli_args_are_noops_when_omitted() -> None:
    raw_config = {
        "experiment_name": "base_experiment",
        "mode": "dg_loeo",
        "seed_list": [42],
        "training": {
            "epochs": 200,
            "batch_size": 8,
            "sampler": "none",
        },
        "losses": {
            "temperature": 0.2,
            "lambda_supcon": 0.1,
            "include_same_domain_same_class": True,
        },
    }

    args = parse_with(["--config", "dummy.json"])
    config = train_dg.resolve_runtime_config(raw_config, args)

    assert config == raw_config


def test_scheduled_lambda_supcon_preserves_zero_warmup_behavior() -> None:
    assert train_dg.scheduled_lambda_supcon({"lambda_supcon": 0.1}, epoch=1) == 0.1
    assert train_dg.scheduled_lambda_supcon(
        {"lambda_supcon": 0.1, "lambda_supcon_warmup_epochs": 0},
        epoch=10,
    ) == 0.1


def test_scheduled_lambda_supcon_warms_up_by_epoch() -> None:
    loss_config = {"lambda_supcon": 0.1, "lambda_supcon_warmup_epochs": 20}
    assert isclose(train_dg.scheduled_lambda_supcon(loss_config, epoch=1), 0.005)
    assert isclose(train_dg.scheduled_lambda_supcon(loss_config, epoch=10), 0.05)
    assert isclose(train_dg.scheduled_lambda_supcon(loss_config, epoch=20), 0.1)
    assert isclose(train_dg.scheduled_lambda_supcon(loss_config, epoch=30), 0.1)


if __name__ == "__main__":
    test_cli_overrides_land_in_runtime_config()
    test_new_cli_args_are_noops_when_omitted()
    test_scheduled_lambda_supcon_preserves_zero_warmup_behavior()
    test_scheduled_lambda_supcon_warms_up_by_epoch()
    print("train_dg CLI override smoke check passed")
