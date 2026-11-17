"""Locked H11 L1248 trainer adapter for the residual-scale ablation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BASE_PATH = REPO_ROOT / "experiments" / "supcon_tpp_legacy_reproduction" / "train_factorial.py"
EXPECTED_PARAMETERS = 4_786_294
EXPECTED_LEVELS = [1, 2, 4, 8]
ALLOWED_SCALES = (0.125, 0.5, 0.75, 1.0)


def load_base():
    spec = importlib.util.spec_from_file_location("residual_scale_base_train_factorial", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base trainer: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scale_tag(scale: float) -> str:
    return f"{int(round(scale * 1000)):04d}"


def verify_config(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = config["model"]
    protocol = config["protocol"]
    if model["name"] != "isolated_pa_csi_dg_lite_tpp":
        raise RuntimeError(f"Unexpected model: {model['name']}")
    if model.get("tpp_levels") != EXPECTED_LEVELS:
        raise RuntimeError(f"Unexpected TPP levels: {model.get('tpp_levels')}")
    if model.get("channel_tpp_levels", model.get("tpp_levels")) != EXPECTED_LEVELS:
        raise RuntimeError("Channel TPP levels drifted")
    if model.get("tpp_pool_type") != "max" or model.get("tpp_preserve_global") is not True:
        raise RuntimeError("TPP pooling semantics drifted")
    scale = float(model.get("tpp_pyramid_scale"))
    if not math.isfinite(scale) or scale not in ALLOWED_SCALES:
        raise RuntimeError(f"Residual scale is not registered: {scale}")
    if float(model.get("channel_tpp_pyramid_scale", scale)) != scale:
        raise RuntimeError("Temporal and channel residual scales disagree")
    if int(protocol.get("expected_parameter_count", -1)) != EXPECTED_PARAMETERS:
        raise RuntimeError("Expected parameter count drifted")
    if protocol.get("version") != "tpp-residual-scale-ablation/v1":
        raise RuntimeError("Protocol version drifted")
    expected_factorial = {
        "name": "Factorial-S32",
        "arm": "H11",
        "tpp_on": True,
        "supcon_on": True,
        "classifier_representation": f"dual_stream_residual{scale_tag(scale)}_tpp",
        "supcon_views": 4,
        "lambda_supcon": 0.05,
        "fully_coupled_tpp_views": True,
    }
    if config["factorial"] != expected_factorial:
        raise RuntimeError("H11 factorial metadata drifted")
    losses = config["losses"]
    if float(losses["lambda_supcon"]) != 0.05 or float(losses["temperature"]) != 0.2:
        raise RuntimeError("SupCon scalar configuration drifted")
    if int(losses["lambda_supcon_warmup_epochs"]) != 0:
        raise RuntimeError("SupCon warm-up drifted")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked H11 L1248 residual-scale trainer")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-checkpoint", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    verify_config(config_path)
    base = load_base()
    base.SCHEMA_VERSION = "tpp-residual-scale-train/v1"
    base.EXPECTED_PARAMETERS["isolated_pa_csi_dg_lite_tpp"] = EXPECTED_PARAMETERS
    if args.validate_checkpoint is not None:
        base.validate_checkpoint(config_path, args.validate_checkpoint.resolve())
    else:
        base.train(config_path)


if __name__ == "__main__":
    main()
