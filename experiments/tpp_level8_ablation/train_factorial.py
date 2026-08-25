"""Thin locked adapter around the completed Factorial-S32 trainer.

The training implementation is reused byte-for-byte.  This adapter changes
only the expected parameter count required by the level-8 architecture.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BASE_PATH = (
    REPO_ROOT
    / "experiments"
    / "supcon_tpp_legacy_reproduction"
    / "train_factorial.py"
)
EXPECTED_PARAMETERS = 4_786_294
EXPECTED_LEVELS = [1, 2, 4, 8]


def load_base():
    spec = importlib.util.spec_from_file_location("level8_base_train_factorial", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base trainer: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_config(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = config["model"]
    protocol = config["protocol"]
    if model["name"] != "isolated_pa_csi_dg_lite_tpp":
        raise RuntimeError(f"Unexpected model: {model['name']}")
    if model.get("tpp_levels") != EXPECTED_LEVELS:
        raise RuntimeError(f"Unexpected TPP levels: {model.get('tpp_levels')}")
    effective_channel = model.get("channel_tpp_levels", model.get("tpp_levels"))
    if effective_channel != EXPECTED_LEVELS:
        raise RuntimeError(f"Unexpected channel TPP levels: {effective_channel}")
    if model.get("tpp_pool_type") != "max":
        raise RuntimeError("TPP pool type drifted")
    if model.get("tpp_preserve_global") is not True:
        raise RuntimeError("TPP global preservation drifted")
    if float(model.get("tpp_pyramid_scale")) != 0.25:
        raise RuntimeError("TPP residual scale drifted")
    if int(protocol.get("expected_parameter_count", -1)) != EXPECTED_PARAMETERS:
        raise RuntimeError("Expected parameter count is not protocol-locked")
    if config["factorial"] != {
        "name": "Factorial-S32",
        "arm": "H11",
        "tpp_on": True,
        "supcon_on": True,
        "classifier_representation": "dual_stream_residual025_tpp",
        "supcon_views": 4,
        "lambda_supcon": 0.05,
        "fully_coupled_tpp_views": True,
    }:
        raise RuntimeError("H11 factorial metadata drifted")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked H11 L1248 training adapter")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-checkpoint", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    verify_config(config_path)
    base = load_base()
    base.SCHEMA_VERSION = "tpp-level8-train/v1"
    base.EXPECTED_PARAMETERS["isolated_pa_csi_dg_lite_tpp"] = EXPECTED_PARAMETERS
    if args.validate_checkpoint is not None:
        base.validate_checkpoint(config_path, args.validate_checkpoint.resolve())
    else:
        base.train(config_path)


if __name__ == "__main__":
    main()
