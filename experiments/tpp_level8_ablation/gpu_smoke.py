from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BASE_TRAIN_PATH = (
    REPO_ROOT / "experiments" / "supcon_tpp_legacy_reproduction" / "train_factorial.py"
)
REFERENCE_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "supcon_tpp_legacy_reproduction"
    / "generated_factorial_s32_v1"
    / "configs"
    / "factorial_s32__target_E1__seed_42__arm_H11.json"
)
OUTPUT_PATH = REPO_ROOT / "outputs_tpp_level8_ablation" / "audit" / "gpu_smoke.json"
LEVELS = [1, 2, 4, 8]
EXPECTED_PARAMETERS = 4_786_294


class SmokeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def load_base():
    spec = importlib.util.spec_from_file_location("level8_smoke_base_train", BASE_TRAIN_PATH)
    if spec is None or spec.loader is None:
        raise SmokeError(f"Cannot load base trainer: {BASE_TRAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    started = time.perf_counter()
    require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    config = json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    config["model"]["tpp_levels"] = LEVELS
    # The channel path deliberately inherits the same levels, matching the
    # historical H11 configuration semantics.
    config["model"].pop("channel_tpp_levels", None)
    base = load_base()
    modules = base.configure_imports(config)
    model_cls = base.model_class(config, modules)
    modules["train_dg"].set_seed(42)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    model = model_cls(input_dim=90, num_classes=6, model_config=config["model"]).to(device)
    model.train()
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    require(parameter_count == EXPECTED_PARAMETERS, f"parameter count {parameter_count}")

    amplitude = torch.randn(32, 850, 90, device=device)
    phase = torch.randn_like(amplitude)
    labels = torch.arange(32, device=device, dtype=torch.long) % 6
    domains = torch.cat(
        [torch.zeros(16, device=device, dtype=torch.long), torch.ones(16, device=device, dtype=torch.long)]
    )
    sample_ids = torch.arange(32, device=device, dtype=torch.long)
    loss_config = config["losses"]
    supcon_loss_fn = modules["dg_losses"].DomainAwareSupConLoss(
        temperature=float(loss_config["temperature"]),
        include_same_domain_same_class=bool(loss_config["include_same_domain_same_class"]),
        positive_mode=str(loss_config["supcon_positive_mode"]),
    )

    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    forward_started = time.perf_counter()
    outputs = model(amplitude, phase)
    view_outputs = model.encode_antenna_views(amplitude, phase)
    views = torch.cat([outputs["projection"].unsqueeze(1), view_outputs["projection_views"]], dim=1)
    require(tuple(outputs["logits"].shape) == (32, 6), "logit shape drift")
    require(tuple(views.shape) == (32, 4, 128), "SupCon view shape drift")
    ce = F.cross_entropy(outputs["logits"], labels)
    supcon = supcon_loss_fn(views, labels=labels, domains=domains, sample_ids=sample_ids)
    total = ce + 0.05 * supcon
    require(bool(torch.isfinite(total).item()), "non-finite total loss")
    total.backward()
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - forward_started

    missing_gradients = []
    nonzero_gradients = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing_gradients.append(name)
            continue
        require(bool(torch.isfinite(parameter.grad).all().item()), f"non-finite gradient: {name}")
        nonzero_gradients += int(torch.count_nonzero(parameter.grad).item())
    require(not missing_gradients, f"missing gradients: {missing_gradients}")
    require(nonzero_gradients > 0, "all gradients are zero")

    diagnostics = modules["dg_losses"].supcon_positive_pair_diagnostics(
        views,
        labels=labels,
        domains=domains,
        sample_ids=sample_ids,
        include_same_domain_same_class=True,
        positive_mode="global_class_instance_views",
    )
    require(float(diagnostics["supcon_frac_anchors_with_positive"]) == 1.0, "positive coverage")
    properties = torch.cuda.get_device_properties(0)
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    require(peak_allocated <= int(properties.total_memory), "GPU memory overflow")
    report = {
        "schema_version": "tpp-level8-gpu-smoke/v1",
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_only": True,
        "real_data_opened": False,
        "formal_training_started": False,
        "reference_config": str(REFERENCE_CONFIG.resolve()),
        "reference_config_sha256": sha256_file(REFERENCE_CONFIG),
        "base_trainer": str(BASE_TRAIN_PATH.resolve()),
        "base_trainer_sha256": sha256_file(BASE_TRAIN_PATH),
        "tpp_models_sha256": config["provenance"]["tpp_models_sha256"],
        "effective_tpp": {
            "tpp_levels": LEVELS,
            "channel_tpp_levels": LEVELS,
            "pool_type": "max",
            "preserve_global": True,
            "pyramid_scale": 0.25,
        },
        "h11": {
            "batch_size": 32,
            "views": 4,
            "lambda_supcon": 0.05,
            "temperature": 0.2,
            "parameter_count": parameter_count,
            "input_shape": [32, 850, 90],
            "logit_shape": list(outputs["logits"].shape),
            "projection_views_shape": list(views.shape),
            "ce_loss": float(ce.detach().item()),
            "supcon_loss": float(supcon.detach().item()),
            "total_loss": float(total.detach().item()),
            "positive_anchor_fraction": float(diagnostics["supcon_frac_anchors_with_positive"]),
            "backward_complete": True,
            "missing_gradients": missing_gradients,
            "nonzero_gradient_elements": nonzero_gradients,
        },
        "device": {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "elapsed_forward_backward_seconds": elapsed,
        "elapsed_total_seconds": time.perf_counter() - started,
    }
    atomic_json(OUTPUT_PATH, report)
    print(json.dumps({
        "status": "pass",
        "output": str(OUTPUT_PATH.resolve()),
        "parameters": parameter_count,
        "peak_allocated_mib": peak_allocated / (1024 ** 2),
        "peak_reserved_mib": peak_reserved / (1024 ** 2),
    }, ensure_ascii=False))

    del total, supcon, ce, views, view_outputs, outputs
    del sample_ids, domains, labels, phase, amplitude, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
