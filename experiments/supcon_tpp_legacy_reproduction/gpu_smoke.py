from __future__ import annotations

import gc
import hashlib
import importlib
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


# Keep the detached replay worktree byte-for-byte clean during dynamic imports.
sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
GENERATED_ROOT = EXPERIMENT_ROOT / "generated_v1"
OUTPUT_PATH = (
    REPO_ROOT
    / "outputs_supcon_tpp_legacy_reproduction"
    / "audit"
    / "gpu_smoke.json"
)
TPP_ROOT = REPO_ROOT / "experiments" / "tpp_isolated"

CONFIG_RELATIVE_PATHS = {
    "anchor_ce": "anchor_configs/anchor_ce.json",
    "anchor_supcon": "anchor_configs/anchor_supcon.json",
    "anchor_tpp_residual025": "anchor_configs/anchor_tpp_residual025.json",
}

EXPECTED_INPUT_SHAPE = (32, 850, 90)
EXPECTED_NUM_CLASSES = 6
EXPECTED_MODEL_PARAMETERS = {
    "pa_csi_dg_lite": 2_788_918,
    "isolated_pa_csi_dg_lite_tpp": 3_721_334,
}
EXPECTED_TPP_RESIDUAL025 = {
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


class SmokeFailure(RuntimeError):
    """Fail-closed protocol or runtime mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def require_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise SmokeFailure(f"{context}: expected {expected!r}, got {actual!r}")


def read_json(path: Path) -> Any:
    require(path.is_file(), f"Required JSON file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_checked(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SmokeFailure(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout.strip()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def verify_bundle() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = GENERATED_ROOT / "bundle_manifest.json"
    lock_path = GENERATED_ROOT / "protocol_lock.json"
    registry_path = GENERATED_ROOT / "anchor_registry.json"
    manifest = read_json(manifest_path)
    lock = read_json(lock_path)
    registry = read_json(registry_path)

    require_equal(manifest.get("schema_version"), 1, "bundle schema_version")
    require_equal(lock.get("schema_version"), 1, "protocol lock schema_version")
    require_equal(registry.get("schema_version"), 1, "registry schema_version")
    require_equal(
        lock.get("protocol_name"),
        "supcon_tpp_legacy_reproduction",
        "protocol name",
    )
    require_equal(lock.get("protocol_version"), "1.0.0", "protocol version")

    manifest_hashes: dict[str, str] = {}
    for relative_name, expected_hash in manifest.get("files", {}).items():
        member_path = GENERATED_ROOT / relative_name
        require(member_path.is_file(), f"Bundle member is missing: {member_path}")
        actual_hash = sha256_file(member_path)
        require_equal(actual_hash, expected_hash, f"bundle hash {relative_name}")
        manifest_hashes[relative_name] = actual_hash

    require_equal(
        Path(registry["protocol_lock"]).resolve(),
        lock_path.resolve(),
        "registry protocol_lock path",
    )
    require_equal(
        registry["protocol_lock_sha256"],
        sha256_file(lock_path),
        "registry protocol_lock SHA256",
    )
    require_equal(
        registry.get("anchor_configs"),
        lock.get("anchor_configs"),
        "registry/lock anchor config records",
    )

    records = {record["config_id"]: record for record in registry["anchor_configs"]}
    for config_id, relative_name in CONFIG_RELATIVE_PATHS.items():
        require(config_id in records, f"Missing registry config record: {config_id}")
        record = records[config_id]
        config_path = (GENERATED_ROOT / relative_name).resolve()
        require_equal(Path(record["path"]).resolve(), config_path, f"{config_id} path")
        payload = read_json(config_path)
        require_equal(sha256_file(config_path), record["file_sha256"], f"{config_id} file SHA256")
        require_equal(
            hashlib.sha256(canonical_json(payload)).hexdigest(),
            record["canonical_payload_sha256"],
            f"{config_id} canonical payload SHA256",
        )

    lock_expected_parameters = {
        str(name): int(value)
        for name, value in lock.get("expected_parameters", {}).items()
    }
    require_equal(
        lock_expected_parameters,
        EXPECTED_MODEL_PARAMETERS,
        "locked expected model parameters",
    )
    return lock, registry, {
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "protocol_lock_sha256": sha256_file(lock_path),
        "anchor_registry_sha256": sha256_file(registry_path),
        "members": manifest_hashes,
    }


def verify_and_configure_imports(lock: Mapping[str, Any]) -> dict[str, Any]:
    clean_worktree = Path(lock["clean_worktree"]).resolve()
    require(clean_worktree.is_dir(), f"Locked clean worktree is missing: {clean_worktree}")
    expected_commit = str(lock["legacy_commit"])
    actual_commit = run_checked(["git", "rev-parse", "HEAD"], clean_worktree)
    require_equal(actual_commit, expected_commit, "clean worktree HEAD")
    status = run_checked(["git", "status", "--porcelain", "--untracked-files=all"], clean_worktree)
    require_equal(status, "", "clean worktree status")

    tpp_snapshot = lock.get("unversioned_tpp_snapshot", {})
    for required_name in ("tpp_models.py", "train_tpp_isolated.py"):
        require(required_name in tpp_snapshot, f"Missing locked TPP snapshot record: {required_name}")
        item = tpp_snapshot[required_name]
        snapshot_path = Path(item["path"]).resolve()
        require(snapshot_path.is_file(), f"Locked TPP snapshot file is missing: {snapshot_path}")
        require_equal(sha256_file(snapshot_path), item["sha256"], f"TPP snapshot {required_name}")

    # The clean legacy worktree must win for dg_models/dg_losses/train_dg.
    # The SHA-locked unversioned directory supplies only tpp_models.
    for path in (str(TPP_ROOT), str(clean_worktree)):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(clean_worktree))
    sys.path.insert(1, str(TPP_ROOT))

    dg_models = importlib.import_module("dg_models")
    dg_losses = importlib.import_module("dg_losses")
    train_dg = importlib.import_module("train_dg")
    tpp_models = importlib.import_module("tpp_models")

    module_paths = {
        "dg_models": Path(inspect.getfile(dg_models)).resolve(),
        "dg_losses": Path(inspect.getfile(dg_losses)).resolve(),
        "train_dg": Path(inspect.getfile(train_dg)).resolve(),
        "tpp_models": Path(inspect.getfile(tpp_models)).resolve(),
    }
    for module_name in ("dg_models", "dg_losses", "train_dg"):
        require(
            clean_worktree in module_paths[module_name].parents,
            f"{module_name} resolved outside the locked clean worktree: {module_paths[module_name]}",
        )
    require_equal(
        module_paths["tpp_models"],
        Path(tpp_snapshot["tpp_models.py"]["path"]).resolve(),
        "tpp_models import path",
    )

    return {
        "clean_worktree": str(clean_worktree),
        "legacy_commit": actual_commit,
        "module_paths": {name: str(path) for name, path in module_paths.items()},
        "module_sha256": {name: sha256_file(path) for name, path in module_paths.items()},
        "classes": {
            "PaCsiDGLite": dg_models.PaCsiDGLite,
            "IsolatedPaCsiDGLiteTPP": tpp_models.IsolatedPaCsiDGLiteTPP,
            "DomainAwareSupConLoss": dg_losses.DomainAwareSupConLoss,
        },
        "functions": {
            "set_seed": train_dg.set_seed,
            "supcon_positive_pair_diagnostics": dg_losses.supcon_positive_pair_diagnostics,
        },
    }


def load_and_verify_configs(lock: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    configs = {
        config_id: read_json(GENERATED_ROOT / relative_name)
        for config_id, relative_name in CONFIG_RELATIVE_PATHS.items()
    }
    ce = configs["anchor_ce"]
    supcon = configs["anchor_supcon"]
    residual = configs["anchor_tpp_residual025"]

    require_equal(ce["model"]["name"], "pa_csi_dg_lite", "CE model name")
    require_equal(supcon["model"]["name"], "pa_csi_dg_lite", "SupCon model name")
    require_equal(
        residual["model"]["name"],
        "isolated_pa_csi_dg_lite_tpp",
        "residual TPP model name",
    )
    require_equal(ce["model"], supcon["model"], "CE/SupCon model payload")
    require_equal(ce["data"], supcon["data"], "CE/SupCon data payload")
    require_equal(ce["data"], residual["data"], "CE/TPP data payload")
    require_equal(ce["split"], supcon["split"], "CE/SupCon split payload")
    require_equal(ce["split"], residual["split"], "CE/TPP split payload")

    require_equal(ce["losses"]["lambda_supcon"], 0.0, "CE lambda_supcon")
    require_equal(supcon["losses"]["lambda_supcon"], 0.05, "SupCon lambda_supcon")
    require_equal(supcon["losses"]["temperature"], 0.2, "SupCon temperature")
    require_equal(
        supcon["losses"]["supcon_positive_mode"],
        "global_class_instance_views",
        "SupCon positive mode",
    )
    require_equal(
        supcon["losses"]["include_same_domain_same_class"],
        True,
        "SupCon same-domain same-class setting",
    )
    require_equal(
        supcon["losses"]["lambda_supcon_warmup_epochs"],
        0,
        "SupCon warmup",
    )
    require_equal(
        supcon["losses"]["contrastive_loss_type"],
        "pairwise_supcon",
        "SupCon loss type",
    )
    require_equal(supcon["training"]["batch_size"], 32, "SupCon batch size")
    require_equal(
        supcon["training"]["sampler"],
        "domain_class_balanced",
        "SupCon sampler",
    )
    require_equal(supcon["device"], "cuda:0", "SupCon device")

    residual_model = residual["model"]
    effective_tpp = {
        "tpp_levels": residual_model.get("tpp_levels", [1, 2, 4]),
        "channel_tpp_levels": residual_model.get(
            "channel_tpp_levels",
            residual_model.get("tpp_levels", [1, 2, 4]),
        ),
        "tpp_pool_type": residual_model.get("tpp_pool_type", "max"),
        "channel_tpp_pool_type": residual_model.get(
            "channel_tpp_pool_type",
            residual_model.get("tpp_pool_type", "max"),
        ),
        "tpp_mixed_max_weight": residual_model.get("tpp_mixed_max_weight", 0.5),
        "channel_tpp_mixed_max_weight": residual_model.get(
            "channel_tpp_mixed_max_weight",
            residual_model.get("tpp_mixed_max_weight", 0.5),
        ),
        "tpp_preserve_global": residual_model.get("tpp_preserve_global", False),
        "channel_tpp_preserve_global": residual_model.get(
            "channel_tpp_preserve_global",
            residual_model.get("tpp_preserve_global", False),
        ),
        "tpp_pyramid_scale": residual_model.get("tpp_pyramid_scale", 1.0),
        "channel_tpp_pyramid_scale": residual_model.get(
            "channel_tpp_pyramid_scale",
            residual_model.get("tpp_pyramid_scale", 1.0),
        ),
    }
    for field_name, expected_value in EXPECTED_TPP_RESIDUAL025.items():
        require_equal(
            effective_tpp[field_name],
            expected_value,
            f"effective residual TPP field {field_name}",
        )

    allowed_tpp_fields = set(EXPECTED_TPP_RESIDUAL025)
    actual_tpp_fields = {
        key
        for key in residual_model
        if key.startswith("tpp_") or key.startswith("channel_tpp_")
    }
    require(
        actual_tpp_fields.issubset(allowed_tpp_fields),
        f"Residual config contains an unregistered TPP field: "
        f"{sorted(actual_tpp_fields - allowed_tpp_fields)}",
    )

    base_tpp_model = {
        key: value
        for key, value in residual_model.items()
        if key not in EXPECTED_TPP_RESIDUAL025 and key != "name"
    }
    base_legacy_model = {
        key: value
        for key, value in supcon["model"].items()
        if key != "name"
    }
    require_equal(base_tpp_model, base_legacy_model, "legacy/TPP common model fields")

    environment_order = list(supcon["data"]["env_files"].keys())
    require_equal(environment_order, ["E1", "E2", "E3"], "environment order")
    require_equal(supcon["data"]["format"], "env_npy_triplets", "data format")
    require_equal(supcon["data"]["unwrap_phase"], True, "historical phase unwrap")
    require_equal(supcon["data"]["phase_unwrap_axis"], -1, "historical phase unwrap axis")

    for env_name, records in lock["data_files"].items():
        for kind in ("amp", "phase"):
            shape = tuple(int(value) for value in records[kind]["shape"])
            require_equal(shape, (3000, 850, 90), f"locked {env_name}/{kind} shape")
        label_shape = tuple(int(value) for value in records["label"]["shape"])
        require_equal(label_shape, (3000,), f"locked {env_name}/label shape")
    require_equal(len(lock["label_names"]), EXPECTED_NUM_CLASSES, "locked class count")

    antenna_layout = supcon["model"]["antenna_layout"]
    feature_width = (
        int(antenna_layout["num_rx"])
        * int(antenna_layout["num_tx"])
        * int(antenna_layout["num_subcarriers"])
    )
    require_equal(feature_width, EXPECTED_INPUT_SHAPE[-1], "antenna layout feature width")
    require_equal(antenna_layout["mode"], "tx", "antenna view mode")
    require_equal(int(antenna_layout["num_tx"]), 3, "Tx view count")

    config_audit = {
        config_id: {
            "path": str((GENERATED_ROOT / CONFIG_RELATIVE_PATHS[config_id]).resolve()),
            "file_sha256": sha256_file(GENERATED_ROOT / CONFIG_RELATIVE_PATHS[config_id]),
            "model_name": config["model"]["name"],
            "batch_size": int(config["training"]["batch_size"]),
            "sampler": str(config["training"]["sampler"]),
            "lambda_supcon": float(config["losses"]["lambda_supcon"]),
        }
        for config_id, config in configs.items()
    }
    return configs, config_audit


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }


def verify_model_parameters(
    configs: Mapping[str, Mapping[str, Any]],
    imports: Mapping[str, Any],
) -> dict[str, Any]:
    pa_class = imports["classes"]["PaCsiDGLite"]
    tpp_class = imports["classes"]["IsolatedPaCsiDGLiteTPP"]
    records: dict[str, Any] = {}

    definitions = (
        ("anchor_ce", pa_class),
        ("anchor_supcon", pa_class),
        ("anchor_tpp_residual025", tpp_class),
    )
    for config_id, model_class in definitions:
        config = configs[config_id]
        model_name = str(config["model"]["name"])
        model = model_class(
            input_dim=EXPECTED_INPUT_SHAPE[-1],
            num_classes=EXPECTED_NUM_CLASSES,
            model_config=config["model"],
        )
        parameter_counts = count_parameters(model)
        expected_count = EXPECTED_MODEL_PARAMETERS[model_name]
        require_equal(parameter_counts["total"], expected_count, f"{config_id} total parameters")
        require_equal(
            parameter_counts["trainable"],
            expected_count,
            f"{config_id} trainable parameters",
        )
        records[config_id] = {
            "class": type(model).__name__,
            "model_name": model_name,
            **parameter_counts,
            "expected": expected_count,
        }
        del model
    gc.collect()
    return records


def tensor_shape(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.shape]


def require_tensor(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
    context: str,
) -> None:
    require_equal(tuple(tensor.shape), expected_shape, f"{context} shape")
    require(bool(torch.isfinite(tensor).all().item()), f"{context} contains NaN or infinity")


def make_balanced_metadata(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    per_domain_labels = torch.arange(16, dtype=torch.long, device=device) % EXPECTED_NUM_CLASSES
    labels = torch.cat([per_domain_labels, per_domain_labels], dim=0)
    domains = torch.cat(
        [
            torch.zeros(16, dtype=torch.long, device=device),
            torch.ones(16, dtype=torch.long, device=device),
        ],
        dim=0,
    )
    sample_ids = torch.arange(EXPECTED_INPUT_SHAPE[0], dtype=torch.long, device=device)
    return labels, domains, sample_ids


def run_h11_smoke(
    configs: Mapping[str, Mapping[str, Any]],
    imports: Mapping[str, Any],
) -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA is unavailable; refusing CPU fallback")
    require(torch.cuda.device_count() > 0, "cuda:0 does not exist")
    device_index = 0
    device = torch.device("cuda:0")
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()

    set_seed = imports["functions"]["set_seed"]
    set_seed(42)

    residual_config = configs["anchor_tpp_residual025"]
    supcon_config = configs["anchor_supcon"]
    tpp_class = imports["classes"]["IsolatedPaCsiDGLiteTPP"]
    supcon_loss_class = imports["classes"]["DomainAwareSupConLoss"]
    diagnostics_function = imports["functions"]["supcon_positive_pair_diagnostics"]

    model = tpp_class(
        input_dim=EXPECTED_INPUT_SHAPE[-1],
        num_classes=EXPECTED_NUM_CLASSES,
        model_config=residual_config["model"],
    ).to(device)
    model.train()
    parameter_counts = count_parameters(model)
    require_equal(
        parameter_counts["total"],
        EXPECTED_MODEL_PARAMETERS["isolated_pa_csi_dg_lite_tpp"],
        "H11 total parameters",
    )

    amplitude = torch.randn(*EXPECTED_INPUT_SHAPE, dtype=torch.float32, device=device)
    phase = torch.randn_like(amplitude)
    labels, domains, sample_ids = make_balanced_metadata(device)
    require_tensor(amplitude, EXPECTED_INPUT_SHAPE, "synthetic amplitude")
    require_tensor(phase, EXPECTED_INPUT_SHAPE, "synthetic phase")
    require_equal(tensor_shape(labels), [32], "synthetic labels shape")
    require_equal(tensor_shape(domains), [32], "synthetic domains shape")
    require_equal(tensor_shape(sample_ids), [32], "synthetic sample_ids shape")

    loss_config = supcon_config["losses"]
    supcon_loss_fn = supcon_loss_class(
        temperature=float(loss_config["temperature"]),
        include_same_domain_same_class=bool(loss_config["include_same_domain_same_class"]),
        positive_mode=str(loss_config["supcon_positive_mode"]),
    )

    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device_index)
    memory_before = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device_index)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device_index)),
    }
    torch.cuda.reset_peak_memory_stats(device_index)
    start = time.perf_counter()

    outputs = model(amplitude, phase)
    view_outputs = model.encode_antenna_views(amplitude, phase)
    supcon_views = torch.cat(
        [outputs["projection"].unsqueeze(1), view_outputs["projection_views"]],
        dim=1,
    )

    require_tensor(outputs["amp_feature"], (32, 288), "H11 amp_feature")
    require_tensor(outputs["phase_feature"], (32, 288), "H11 phase_feature")
    require_tensor(outputs["fused_feature"], (32, 256), "H11 fused_feature")
    require_tensor(outputs["logits"], (32, 6), "H11 logits")
    require_tensor(outputs["projection"], (32, 128), "H11 global projection")
    require_tensor(view_outputs["amp_views"], (32, 3, 850, 90), "H11 amplitude Tx views")
    require_tensor(view_outputs["phase_views"], (32, 3, 850, 90), "H11 phase Tx views")
    require_tensor(view_outputs["fused_views"], (32, 3, 256), "H11 fused Tx views")
    require_tensor(
        view_outputs["projection_views"],
        (32, 3, 128),
        "H11 projected Tx views",
    )
    require_tensor(supcon_views, (32, 4, 128), "H11 combined SupCon views")

    ce_loss = F.cross_entropy(outputs["logits"], labels)
    supcon_loss = supcon_loss_fn(
        supcon_views,
        labels=labels,
        domains=domains,
        sample_ids=sample_ids,
    )
    lambda_supcon = float(loss_config["lambda_supcon"])
    total_loss = ce_loss + lambda_supcon * supcon_loss
    for name, loss in (
        ("cross_entropy", ce_loss),
        ("supcon", supcon_loss),
        ("total", total_loss),
    ):
        require(loss.ndim == 0, f"{name} loss is not scalar: {tuple(loss.shape)}")
        require(bool(torch.isfinite(loss).item()), f"{name} loss is non-finite")

    positive_diagnostics = diagnostics_function(
        supcon_views,
        labels=labels,
        domains=domains,
        sample_ids=sample_ids,
        include_same_domain_same_class=bool(loss_config["include_same_domain_same_class"]),
        positive_mode=str(loss_config["supcon_positive_mode"]),
    )
    require_equal(
        float(positive_diagnostics["supcon_frac_anchors_with_positive"]),
        1.0,
        "H11 positive-anchor coverage",
    )
    require(
        float(positive_diagnostics["supcon_min_positives_per_anchor"]) >= 1.0,
        "H11 has an anchor without a positive",
    )

    total_loss.backward()
    torch.cuda.synchronize(device_index)
    elapsed_seconds = time.perf_counter() - start

    parameters_with_gradient = 0
    gradient_elements = 0
    nonzero_gradient_elements = 0
    maximum_absolute_gradient = 0.0
    missing_gradient_parameters: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing_gradient_parameters.append(name)
            continue
        gradient = parameter.grad
        require(bool(torch.isfinite(gradient).all().item()), f"Non-finite gradient in {name}")
        parameters_with_gradient += 1
        gradient_elements += int(gradient.numel())
        nonzero_gradient_elements += int(torch.count_nonzero(gradient).item())
        if gradient.numel():
            maximum_absolute_gradient = max(
                maximum_absolute_gradient,
                float(gradient.detach().abs().max().item()),
            )
    require_equal(missing_gradient_parameters, [], "H11 parameters missing gradients")
    require(parameters_with_gradient > 0, "H11 produced no parameter gradients")
    require(nonzero_gradient_elements > 0, "H11 produced only zero gradients")
    require(math.isfinite(maximum_absolute_gradient), "H11 maximum gradient is non-finite")

    memory_after = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device_index)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device_index)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
    }
    device_properties = torch.cuda.get_device_properties(device_index)
    require(
        memory_after["peak_allocated_bytes"] <= int(device_properties.total_memory),
        "Reported peak allocated memory exceeds physical GPU memory",
    )

    result = {
        "device": {
            "requested": "cuda:0",
            "actual": str(device),
            "index": device_index,
            "name": str(device_properties.name),
            "total_memory_bytes": int(device_properties.total_memory),
            "compute_capability": [
                int(device_properties.major),
                int(device_properties.minor),
            ],
        },
        "seed": 42,
        "dtype": str(amplitude.dtype),
        "input_shapes": {
            "amplitude": tensor_shape(amplitude),
            "phase": tensor_shape(phase),
            "labels": tensor_shape(labels),
            "domains": tensor_shape(domains),
            "sample_ids": tensor_shape(sample_ids),
        },
        "output_shapes": {
            "amp_feature": tensor_shape(outputs["amp_feature"]),
            "phase_feature": tensor_shape(outputs["phase_feature"]),
            "fused_feature": tensor_shape(outputs["fused_feature"]),
            "logits": tensor_shape(outputs["logits"]),
            "global_projection": tensor_shape(outputs["projection"]),
            "amplitude_tx_views": tensor_shape(view_outputs["amp_views"]),
            "phase_tx_views": tensor_shape(view_outputs["phase_views"]),
            "fused_tx_views": tensor_shape(view_outputs["fused_views"]),
            "projected_tx_views": tensor_shape(view_outputs["projection_views"]),
            "combined_supcon_views": tensor_shape(supcon_views),
        },
        "loss": {
            "formula": "cross_entropy + 0.05 * pairwise_supcon",
            "cross_entropy": float(ce_loss.detach().item()),
            "supcon": float(supcon_loss.detach().item()),
            "lambda_supcon": lambda_supcon,
            "total": float(total_loss.detach().item()),
            "all_finite": True,
            "temperature": float(loss_config["temperature"]),
            "positive_mode": str(loss_config["supcon_positive_mode"]),
            "include_same_domain_same_class": bool(
                loss_config["include_same_domain_same_class"]
            ),
            "positive_diagnostics": {
                key: float(value)
                for key, value in positive_diagnostics.items()
            },
        },
        "backward": {
            "completed": True,
            "parameters_with_gradient": parameters_with_gradient,
            "gradient_elements": gradient_elements,
            "nonzero_gradient_elements": nonzero_gradient_elements,
            "maximum_absolute_gradient": maximum_absolute_gradient,
            "missing_gradient_parameters": missing_gradient_parameters,
            "all_gradients_finite": True,
        },
        "parameters": parameter_counts,
        "memory": {
            "before_forward_backward": memory_before,
            "after_backward": memory_after,
            "peak_allocated_mib": memory_after["peak_allocated_bytes"] / (1024.0**2),
            "peak_reserved_mib": memory_after["peak_reserved_bytes"] / (1024.0**2),
        },
        "elapsed_forward_backward_seconds": elapsed_seconds,
        "optimizer_step_performed": False,
    }

    del total_loss, supcon_loss, ce_loss, supcon_views, view_outputs, outputs
    del sample_ids, domains, labels, phase, amplitude, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    started = time.perf_counter()
    lock, registry, bundle_audit = verify_bundle()
    imports = verify_and_configure_imports(lock)
    configs, config_audit = load_and_verify_configs(lock)
    model_audit = verify_model_parameters(configs, imports)
    h11_audit = run_h11_smoke(configs, imports)

    report = {
        "schema_version": 1,
        "protocol_name": str(lock["protocol_name"]),
        "protocol_version": str(lock["protocol_version"]),
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_type": "synthetic_full_shape_gpu_smoke",
        "synthetic_only": True,
        "real_data_opened": False,
        "formal_training_started": False,
        "output_path": str(OUTPUT_PATH.resolve()),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": int(torch.backends.cudnn.version() or 0),
            "elapsed_total_seconds": time.perf_counter() - started,
        },
        "bundle": bundle_audit,
        "code": {
            "legacy_commit": imports["legacy_commit"],
            "clean_worktree": imports["clean_worktree"],
            "module_paths": imports["module_paths"],
            "module_sha256": imports["module_sha256"],
        },
        "configs": config_audit,
        "model_parameter_audit": model_audit,
        "h11_batch32_v4": h11_audit,
        "registry_total_anchor_jobs": int(registry["total_anchor_jobs"]),
    }
    atomic_write_json(OUTPUT_PATH, report)
    print(json.dumps({
        "status": "pass",
        "output": str(OUTPUT_PATH.resolve()),
        "peak_allocated_mib": h11_audit["memory"]["peak_allocated_mib"],
        "peak_reserved_mib": h11_audit["memory"]["peak_reserved_mib"],
        "combined_supcon_views": h11_audit["output_shapes"]["combined_supcon_views"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
