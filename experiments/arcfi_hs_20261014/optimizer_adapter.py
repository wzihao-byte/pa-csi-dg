"""Correct optimizer coverage on the pinned author DG class, on CPU only.

The author class, forward, head, style randomization and loss stay unchanged.
This module is deliberately limited to a fresh-model optimizer repair. It has
no dataset loader, source training loop, target loader or evaluation command.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

PINNED_AUTHOR_COMMIT = "f31e655a8414415de58659c0af7d7dba6281a0c1"
PINNED_FILES_SHA256 = {
    "models/contrast_methods.py": "40a392f83bf4d68553b9682c64b7278d80efb3b1653f0e921d64119397d5e2a0",
    "models/sdaresnet.py": "377fbce2277182ac1820637a57a783ee3076197b47446dfce40bf0ce9c1cad2c",
    "util/optimizer.py": "b12db0f2f1f9e45eaa7ce9638677fcb7591e97b9d9c967c2dd080cb82a5baf2d",
}
DEFAULT_AUTHOR_SOURCE = (Path(__file__).resolve().parents[2] /
                        "work/wimans_20261014/author_baseline_audit/UniCrossFi")


def load_pinned_author_class(source_root=DEFAULT_AUTHOR_SOURCE):
    """Execute the exact audited DG/loss classes and their small dependencies.

    Read-and-hash bytes are the same bytes compiled. AST isolation avoids
    importing unrelated UDA methods, torchvision, W&B or author data drivers.
    Source hashes pin class semantics; no external repository code is patched.
    """
    source_root = Path(source_root)
    contents = {}
    for relative, expected in PINNED_FILES_SHA256.items():
        content = (source_root / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError("Pinned author source changed: " + relative)
        contents[relative] = content
    encoder = {"__name__": "arcfi_pinned_author_encoder"}
    exec(compile(contents["models/sdaresnet.py"], str(source_root / "models/sdaresnet.py"), "exec"), encoder)
    optimizers = {"__name__": "arcfi_pinned_author_optimizers"}
    exec(compile(contents["util/optimizer.py"], str(source_root / "util/optimizer.py"), "exec"), optimizers)
    parsed = ast.parse(contents["models/contrast_methods.py"], filename=str(source_root / "models/contrast_methods.py"))
    selected = [node for node in parsed.body if isinstance(node, ast.ClassDef) and
                node.name in {"SupConLoss", "UniCrossFi_dg"}]
    if [node.name for node in selected] != ["SupConLoss", "UniCrossFi_dg"]:
        raise RuntimeError("Audited author class roster changed")
    namespace = {"__name__": "arcfi_pinned_author_dg", "torch": torch, "nn": nn, "F": F,
                 "CosineAnnealingLR": CosineAnnealingLR, "ResNet": encoder["ResNet"],
                 "BasicBlock": encoder["BasicBlock"], "get_optimizer": optimizers["get_optimizer"]}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_root / "models/contrast_methods.py"), "exec"), namespace)
    return namespace["UniCrossFi_dg"]


def parameter_coverage(model):
    """Count each trainable model parameter across the two author optimizers."""
    named = dict(model.named_parameters())
    all_model_ids = {id(parameter) for parameter in named.values()}
    optimized = [parameter for optimizer in (model.optimizer_f, model.optimizer_c)
                 for group in optimizer.param_groups for parameter in group["params"]]
    if any(id(parameter) not in all_model_ids for parameter in optimized):
        raise RuntimeError("Optimizer contains a parameter outside the author model")
    counts = Counter(map(id, optimized))
    return {name: counts[id(parameter)] for name, parameter in named.items() if parameter.requires_grad}


def require_exact_parameter_coverage(model):
    counts = parameter_coverage(model)
    invalid = {name: count for name, count in counts.items() if count != 1}
    if invalid:
        raise RuntimeError("Trainable parameter coverage must be exactly once: " + repr(invalid))
    return counts


def correct_fresh_optimizer_coverage(model):
    """Include encoder+neck in Adam F; retain the author classifier Adam C.

    Repair before any optimizer/scheduler update, so no trained state is lost.
    Recreate F's original cosine scheduler with the same source-defined horizon.
    All original loss/feature/head functions remain the exact author methods.
    """
    if model.config.device != "cpu" or any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("This reviewable prototype only accepts a CPU author model")
    if model.config.optimizer != "Adam" or not isinstance(model.optimizer_f, torch.optim.Adam):
        raise RuntimeError("The registered prototype covers the author Adam configuration only")
    if model.optimizer_f.state or model.optimizer_c.state or model.scheduler_f.last_epoch != 0 or model.scheduler_c.last_epoch != 0:
        raise RuntimeError("Repair must precede all optimizer and scheduler updates")
    counts = parameter_coverage(model)
    neck_names = {"neck." + name for name, _parameter in model.neck.named_parameters()}
    missing = {name for name, count in counts.items() if count == 0}
    if missing != neck_names or any(count != (0 if name in neck_names else 1) for name, count in counts.items()):
        raise RuntimeError("Expected only the pinned author's omitted neck parameters")
    old_f = model.optimizer_f
    old_scheduler = model.scheduler_f
    get_optimizer = type(model).__init__.__globals__["get_optimizer"]
    new_f = get_optimizer(chain(model.feature_extractor.parameters(), model.neck.parameters()), model.config)
    if new_f.defaults != old_f.defaults or len(old_f.param_groups) != 1 or old_f.param_groups[0]["lr"] != new_f.param_groups[0]["lr"]:
        raise RuntimeError("Optimizer repair would change author Adam settings")
    model.optimizer_f = new_f
    model.scheduler_f = CosineAnnealingLR(new_f, T_max=old_scheduler.T_max, eta_min=old_scheduler.eta_min)
    coverage = require_exact_parameter_coverage(model)
    return {"schema": "arcfi-corrected-optimizer-prototype/v1", "author_commit": PINNED_AUTHOR_COMMIT,
            "author_files_sha256": dict(PINNED_FILES_SHA256), "device": "cpu", "optimizer": "Adam",
            "feature_optimizer_modules": ["feature_extractor", "neck"], "classifier_optimizer_modules": ["classifier"],
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "corrected_neck_parameter_count": sum(p.numel() for p in model.neck.parameters()),
            "trainable_parameter_tensor_count": len(coverage), "all_trainable_parameters_owned_exactly_once": True,
            "lr": model.config.lr, "weight_decay": model.config.weight_decay,
            "cosine_T_max": model.scheduler_f.T_max, "cosine_eta_min": model.scheduler_f.eta_min,
            "source_adapter_implemented": False, "formal_training_qualified": False}


def make_cpu_prototype(*, source_root=DEFAULT_AUTHOR_SOURCE, seed=42, lr=0.003, weight_decay=0.0003,
                       temperature=0.07, beta=0.5, scale_cap=6, source_update_horizon=30000):
    """Construct the two-ratio author branch, then apply only the coverage repair.

    Inputs must be [B,2 ratio views,2 amplitude/phase,time,30 subcarriers].
    The horizon is explicit and independent of any target identity/loader/size.
    Parameters are prospective values, not a selected/frozen MA configuration.
    """
    if source_update_horizon <= 0:
        raise ValueError("Source update horizon must be positive")
    author_class = load_pinned_author_class(source_root)
    config = SimpleNamespace(csidataset="Widar3.0", backbone="ResNet", device="cpu", last_dim=512,
                             classify="linear", num_classes=6, optimizer="Adam", lr=float(lr),
                             weight_decay=float(weight_decay), temperature=float(temperature),
                             beta=float(beta), scale_cap=int(scale_cap), ratio=None)
    torch.manual_seed(int(seed))
    model = author_class(config, None, int(source_update_horizon))
    receipt = correct_fresh_optimizer_coverage(model)
    return model, receipt
