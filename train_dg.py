from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from adaptive_kmargin import AdaptiveKMarginConfig, AdaptiveKMarginRecalibrator
from dg_dataset import (
    DomainBalancedBatchSampler,
    DomainClassBalancedBatchSampler,
    build_split_indices,
    build_split_manifest,
    load_dataset,
)
from dg_losses import (
    AdaptiveKMarginLoss,
    DomainAwareSupConLoss,
    SubCenterPrototypeContrastiveLoss,
    supcon_positive_pair_diagnostics,
)
from dg_models import PaCsiDGLite


SUPPORTED_CONTRASTIVE_LOSS_TYPES = {"pairwise_supcon", "subcenter_prototype"}
SUPCON_DIAGNOSTIC_KEYS = (
    "supcon_num_anchors",
    "supcon_mean_positives_per_anchor",
    "supcon_min_positives_per_anchor",
    "supcon_frac_anchors_with_positive",
    "supcon_num_valid_positive_pairs",
)


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PA-CSI domain generalization runner.")
    parser.add_argument("--config", required=True, help="Path to a JSON config preset.")
    parser.add_argument("--mode", choices=["random_split", "dg_loeo"], help="Override split mode.")
    parser.add_argument("--target-env", dest="target_env", help="Run only one held-out environment.")
    parser.add_argument("--device", help="Override the configured device.")
    parser.add_argument("--output-root", help="Override the configured output directory.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override the configured seed list.")
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count.")
    parser.add_argument("--lambda-supcon", type=float, help="Override losses.lambda_supcon.")
    parser.add_argument(
        "--lambda-supcon-warmup-epochs",
        type=int,
        help="Override losses.lambda_supcon_warmup_epochs.",
    )
    parser.add_argument("--temperature", type=float, help="Override losses.temperature.")
    parser.add_argument(
        "--supcon-positive-mode",
        choices=["all_views", "global_class_instance_views"],
        help="Override losses.supcon_positive_mode.",
    )
    parser.add_argument(
        "--include-same-domain-same-class",
        type=int,
        choices=[0, 1],
        help="Override losses.include_same_domain_same_class.",
    )
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size.")
    parser.add_argument(
        "--sampler",
        choices=["none", "domain_balanced", "domain_class_balanced"],
        help="Override training.sampler.",
    )
    parser.add_argument("--experiment-name", help="Override experiment_name.")
    parser.add_argument(
        "--contrastive-loss-type",
        choices=sorted(SUPPORTED_CONTRASTIVE_LOSS_TYPES),
        help="Override losses.contrastive_loss_type.",
    )
    parser.add_argument("--lambda-pair-margin", type=float, help="Override losses.lambda_pair_margin.")
    parser.add_argument(
        "--prototype-num-subcenters",
        type=int,
        help="Override model.prototype_num_subcenters.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_confusion_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(matrix.tolist())


def zero_supcon_diagnostics() -> Dict[str, float]:
    return {key: 0.0 for key in SUPCON_DIAGNOSTIC_KEYS}


def resolve_runtime_config(raw_config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = copy.deepcopy(raw_config)
    config["mode"] = args.mode or config.get("mode", "dg_loeo")
    if args.target_env is not None:
        config["target_env"] = args.target_env
    if args.device is not None:
        config["device"] = args.device
    if args.output_root is not None:
        config["output_root"] = args.output_root
    if args.seeds is not None:
        config["seed_list"] = args.seeds
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if getattr(args, "experiment_name", None) is not None:
        config["experiment_name"] = args.experiment_name
    if getattr(args, "batch_size", None) is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if getattr(args, "sampler", None) is not None:
        config.setdefault("training", {})["sampler"] = args.sampler
    if getattr(args, "lambda_supcon", None) is not None:
        config.setdefault("losses", {})["lambda_supcon"] = args.lambda_supcon
    if getattr(args, "lambda_supcon_warmup_epochs", None) is not None:
        config.setdefault("losses", {})["lambda_supcon_warmup_epochs"] = args.lambda_supcon_warmup_epochs
    if getattr(args, "temperature", None) is not None:
        config.setdefault("losses", {})["temperature"] = args.temperature
    if getattr(args, "supcon_positive_mode", None) is not None:
        config.setdefault("losses", {})["supcon_positive_mode"] = args.supcon_positive_mode
    if getattr(args, "include_same_domain_same_class", None) is not None:
        config.setdefault("losses", {})["include_same_domain_same_class"] = bool(
            args.include_same_domain_same_class
        )
    if getattr(args, "contrastive_loss_type", None) is not None:
        config.setdefault("losses", {})["contrastive_loss_type"] = args.contrastive_loss_type
    if getattr(args, "lambda_pair_margin", None) is not None:
        config.setdefault("losses", {})["lambda_pair_margin"] = args.lambda_pair_margin
    if getattr(args, "prototype_num_subcenters", None) is not None:
        config.setdefault("model", {})["prototype_num_subcenters"] = args.prototype_num_subcenters
    return config


def build_data_loader(
    dataset,
    indices: List[int],
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    sampler_mode: str = "none",
) -> DataLoader:
    if sampler_mode == "domain_balanced":
        grouped = dataset.indices_by_env(indices)
        batch_sampler = DomainBalancedBatchSampler(grouped, batch_size=batch_size)
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
    if sampler_mode == "domain_class_balanced":
        grouped = dataset.indices_by_env(indices)
        batch_sampler = DomainClassBalancedBatchSampler(grouped, dataset.labels, batch_size=batch_size)
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
    if sampler_mode != "none":
        raise ValueError(
            f"Unsupported sampler mode '{sampler_mode}'. "
            "Choose from ['none', 'domain_balanced', 'domain_class_balanced']."
        )

    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def metric_dict(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def scheduled_lambda_supcon(loss_config: Mapping[str, Any], epoch: int) -> float:
    target_lambda = float(loss_config.get("lambda_supcon", 0.0))
    warmup_epochs = int(loss_config.get("lambda_supcon_warmup_epochs", 0))
    if warmup_epochs <= 0:
        return target_lambda
    return target_lambda * min(1.0, float(epoch) / float(warmup_epochs))


def loss_config_for_lambda(loss_config: Mapping[str, Any], effective_lambda_supcon: float) -> Dict[str, Any]:
    epoch_loss_config = dict(loss_config)
    epoch_loss_config["lambda_supcon_target"] = float(loss_config.get("lambda_supcon", 0.0))
    epoch_loss_config["lambda_supcon"] = float(effective_lambda_supcon)
    epoch_loss_config["lambda_supcon_effective"] = float(effective_lambda_supcon)
    epoch_loss_config["lambda_supcon_warmup_epochs"] = int(
        loss_config.get("lambda_supcon_warmup_epochs", 0)
    )
    return epoch_loss_config


def run_epoch(
    model: PaCsiDGLite,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scheduler_step_mode: Optional[str],
    device: torch.device,
    loss_config: Mapping[str, Any],
    contrastive_loss_type: str,
    contrastive_loss_fn: Optional[torch.nn.Module],
    adaptive_kmargin_loss_fn: Optional[AdaptiveKMarginLoss],
    adaptive_kmargin_state: Optional[Mapping[str, torch.Tensor]],
    train: bool,
) -> Dict[str, Any]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_supcon = 0.0
    total_akm = 0.0
    total_pair_margin = 0.0
    total_supcon_diag_anchors = 0.0
    total_supcon_diag_positive_pairs = 0.0
    total_supcon_diag_anchors_with_positive = 0.0
    min_supcon_diag_positives = math.inf
    all_predictions: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    shape_debug: Optional[Dict[str, List[int]]] = None
    lambda_supcon = float(loss_config.get("lambda_supcon", 0.0))
    lambda_supcon_target = float(loss_config.get("lambda_supcon_target", lambda_supcon))
    lambda_supcon_warmup_epochs = int(loss_config.get("lambda_supcon_warmup_epochs", 0))
    lambda_pair_margin = float(loss_config.get("lambda_pair_margin", 0.0))
    effective_pair_margin_weight = lambda_supcon * lambda_pair_margin
    adaptive_config = loss_config.get("adaptive_kmargin", {}) or {}
    lambda_akm = float(adaptive_config.get("lambda_akm", 0.0)) if bool(adaptive_config.get("enabled", False)) else 0.0

    iterator = tqdm(loader, leave=False, disable=False)
    for amplitude, phase, labels, domains, sample_ids in iterator:
        amplitude = amplitude.to(device)
        phase = phase.to(device)
        labels = labels.to(device)
        domains = domains.to(device)
        sample_ids = sample_ids.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(amplitude, phase)
            ce_loss = F.cross_entropy(outputs["logits"], labels)

            supcon_loss = torch.zeros((), device=device)
            supcon_views = outputs["projection"].unsqueeze(1)
            view_outputs: Optional[Dict[str, torch.Tensor]] = None
            supcon_diag = zero_supcon_diagnostics()

            if lambda_supcon > 0.0:
                if model.view_builder is not None:
                    view_outputs = model.encode_antenna_views(amplitude, phase)
                    supcon_views = torch.cat([supcon_views, view_outputs["projection_views"]], dim=1)

            if lambda_supcon > 0.0:
                if contrastive_loss_fn is None:
                    raise RuntimeError(
                        "lambda_supcon > 0 but no contrastive loss function was constructed."
                    )
                if contrastive_loss_type == "subcenter_prototype":
                    prototypes = getattr(model, "class_prototypes", None)
                    if prototypes is None or not model.has_class_prototypes():
                        raise ValueError(
                            "contrastive_loss_type='subcenter_prototype' requires "
                            "model.class_prototypes; set model.prototype_num_subcenters > 0."
                        )
                    supcon_loss = contrastive_loss_fn(
                        supcon_views,
                        labels=labels,
                        prototypes=prototypes,
                    )
                elif contrastive_loss_type == "pairwise_supcon":
                    supcon_loss = contrastive_loss_fn(
                        supcon_views,
                        labels=labels,
                        domains=domains,
                        sample_ids=sample_ids,
                    )
                    with torch.no_grad():
                        supcon_diag = supcon_positive_pair_diagnostics(
                            supcon_views,
                            labels=labels,
                            domains=domains,
                            sample_ids=sample_ids,
                            include_same_domain_same_class=bool(
                                loss_config.get("include_same_domain_same_class", True)
                            ),
                            positive_mode=str(loss_config.get("supcon_positive_mode", "all_views")),
                        )
                else:
                    raise ValueError(
                        f"Unsupported contrastive_loss_type '{contrastive_loss_type}'. "
                        f"Choose from {sorted(SUPPORTED_CONTRASTIVE_LOSS_TYPES)}."
                    )

            akm_loss = torch.zeros((), device=device)
            if lambda_akm > 0.0 and adaptive_kmargin_loss_fn is not None and adaptive_kmargin_state is not None:
                akm_loss = adaptive_kmargin_loss_fn(
                    outputs["projection"],
                    labels,
                    cached_prototypes=adaptive_kmargin_state.get("cached_prototypes"),
                    selected_pairs=adaptive_kmargin_state.get("selected_pairs"),
                    pair_weights=adaptive_kmargin_state.get("pair_weights"),
                    pair_margins=adaptive_kmargin_state.get("pair_margins"),
                    prototype_valid_mask=adaptive_kmargin_state.get("prototype_valid_mask"),
                )

            total_batch_loss = ce_loss + lambda_supcon * supcon_loss + lambda_akm * akm_loss

            if not torch.isfinite(total_batch_loss):
                raise FloatingPointError("Encountered a non-finite loss value.")

            if train:
                optimizer.zero_grad(set_to_none=True)
                total_batch_loss.backward()
                grad_clip_norm = float(loss_config.get("grad_clip_norm", 0.0))
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                if scheduler is not None and scheduler_step_mode == "step":
                    scheduler.step()

        predictions = outputs["logits"].argmax(dim=1)
        all_predictions.append(predictions.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        total_loss += float(total_batch_loss.item())
        total_ce += float(ce_loss.item())
        total_supcon += float(supcon_loss.item())
        total_akm += float(akm_loss.item())
        supcon_diag_anchors = float(supcon_diag["supcon_num_anchors"])
        total_supcon_diag_anchors += supcon_diag_anchors
        total_supcon_diag_positive_pairs += float(supcon_diag["supcon_num_valid_positive_pairs"])
        total_supcon_diag_anchors_with_positive += (
            float(supcon_diag["supcon_frac_anchors_with_positive"]) * supcon_diag_anchors
        )
        if supcon_diag_anchors > 0:
            min_supcon_diag_positives = min(
                min_supcon_diag_positives,
                float(supcon_diag["supcon_min_positives_per_anchor"]),
            )
        if contrastive_loss_type == "subcenter_prototype" and isinstance(
            contrastive_loss_fn,
            SubCenterPrototypeContrastiveLoss,
        ):
            total_pair_margin += float(contrastive_loss_fn.last_pair_margin_loss)

        if shape_debug is None:
            shape_debug = {
                "amplitude": list(amplitude.shape),
                "phase": list(phase.shape),
                "fused_feature": list(outputs["fused_feature"].shape),
                "projection": list(outputs["projection"].shape),
            }
            if view_outputs is not None:
                shape_debug["multi_antenna_views"] = list(view_outputs["amp_views"].shape)

    if not all_labels:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "loss_total": 0.0,
            "loss_ce": 0.0,
            "loss_supcon": 0.0,
            "loss_supcon_weighted": 0.0,
            "loss_supcon_to_ce": 0.0,
            "lambda_supcon_target": lambda_supcon_target,
            "lambda_supcon_effective": lambda_supcon,
            "lambda_supcon_warmup_epochs": lambda_supcon_warmup_epochs,
            "effective_pair_margin_weight": effective_pair_margin_weight,
            **zero_supcon_diagnostics(),
            "loss_pair_margin": 0.0,
            "akm_loss": 0.0,
            "labels": [],
            "predictions": [],
            "shape_debug": shape_debug or {},
        }

    mean_ce = total_ce / max(1, len(loader))
    mean_supcon = total_supcon / max(1, len(loader))
    weighted_supcon = lambda_supcon * mean_supcon
    supcon_to_ce = weighted_supcon / (mean_ce + 1e-12)
    if total_supcon_diag_anchors > 0:
        supcon_diagnostics = {
            "supcon_num_anchors": total_supcon_diag_anchors,
            "supcon_mean_positives_per_anchor": (
                total_supcon_diag_positive_pairs / total_supcon_diag_anchors
            ),
            "supcon_min_positives_per_anchor": min_supcon_diag_positives,
            "supcon_frac_anchors_with_positive": (
                total_supcon_diag_anchors_with_positive / total_supcon_diag_anchors
            ),
            "supcon_num_valid_positive_pairs": total_supcon_diag_positive_pairs,
        }
    else:
        supcon_diagnostics = zero_supcon_diagnostics()
    labels_np = np.concatenate(all_labels, axis=0)
    predictions_np = np.concatenate(all_predictions, axis=0)
    metrics = metric_dict(labels_np, predictions_np)
    metrics.update(
        {
            "loss_total": total_loss / max(1, len(loader)),
            "loss_ce": mean_ce,
            "loss_supcon": mean_supcon,
            "loss_supcon_weighted": weighted_supcon,
            "loss_supcon_to_ce": supcon_to_ce,
            "lambda_supcon_target": lambda_supcon_target,
            "lambda_supcon_effective": lambda_supcon,
            "lambda_supcon_warmup_epochs": lambda_supcon_warmup_epochs,
            "effective_pair_margin_weight": effective_pair_margin_weight,
            **supcon_diagnostics,
            "loss_pair_margin": total_pair_margin / max(1, len(loader)),
            "akm_loss": total_akm / max(1, len(loader)),
            "labels": labels_np.tolist(),
            "predictions": predictions_np.tolist(),
            "shape_debug": shape_debug or {},
        }
    )
    return metrics


def choose_target_envs(config: Mapping[str, Any], dataset) -> List[Optional[str]]:
    if config["mode"] == "random_split":
        return [None]
    target_env = config.get("target_env")
    if target_env in (None, "all"):
        return list(dataset.env_names)
    return [target_env]


def resolve_sampler_mode(config: Mapping[str, Any], split: Mapping[str, Any]) -> str:
    sampler_mode = str(config["training"].get("sampler", "none"))
    if config["mode"] != "dg_loeo" or len(split["source_envs"]) <= 1:
        return "none" if sampler_mode != "none" else sampler_mode
    return sampler_mode


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: Mapping[str, Any],
) -> tuple[Optional[torch.optim.lr_scheduler.LRScheduler], Optional[str]]:
    scheduler_config = training_config.get("lr_scheduler")
    if not scheduler_config:
        return None, None

    scheduler_name = str(scheduler_config.get("name", "")).lower()
    update_mode = str(scheduler_config.get("update", "step")).lower()
    if update_mode not in {"step", "epoch"}:
        raise ValueError(f"Unsupported lr scheduler update mode '{update_mode}'. Choose from ['step', 'epoch'].")

    if scheduler_name == "exponential_decay":
        decay_rate = float(scheduler_config.get("decay_rate", 0.9))
        decay_steps = max(1, int(scheduler_config.get("decay_steps", 10000)))
        staircase = bool(scheduler_config.get("staircase", False))

        def lr_lambda(step_index: int) -> float:
            exponent = step_index // decay_steps if staircase else step_index / decay_steps
            return decay_rate**exponent

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler, update_mode

    if scheduler_name == "exponential":
        gamma = float(scheduler_config.get("gamma", 0.9))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        return scheduler, update_mode

    raise ValueError(
        f"Unsupported lr scheduler '{scheduler_name}'. "
        "Choose from ['exponential_decay', 'exponential']."
    )


def pair_key(true_class: int, negative_class: int) -> Tuple[int, int]:
    return int(true_class), int(negative_class)


def normalize_pair_margin_specs(specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, float]]:
    normalized: List[Dict[str, float]] = []
    for spec in specs:
        normalized.append(
            {
                "true": int(spec["true"]),
                "negative": int(spec["negative"]),
                "margin": float(spec.get("margin", 0.0)),
            }
        )
    return normalized


def pair_margin_map(specs: Sequence[Mapping[str, Any]]) -> Dict[Tuple[int, int], float]:
    return {
        pair_key(int(spec["true"]), int(spec["negative"])): float(spec.get("margin", 0.0))
        for spec in specs
    }


def parse_per_pair_caps(
    specs: Optional[Sequence[Mapping[str, Any]]],
    field: str,
) -> Dict[Tuple[int, int], float]:
    if not specs:
        return {}
    parsed: Dict[Tuple[int, int], float] = {}
    for spec in specs:
        if field not in spec:
            continue
        parsed[pair_key(int(spec["true"]), int(spec["negative"]))] = float(spec[field])
    return parsed


def dynamic_margin_initial_specs(loss_config: Mapping[str, Any]) -> List[Dict[str, float]]:
    dynamic_config = loss_config.get("dynamic_pair_margin", {}) or {}
    initial_specs = dynamic_config.get("initial_pair_margins")
    if initial_specs is None:
        initial_specs = loss_config.get("prototype_pair_margins", []) or []
    return normalize_pair_margin_specs(initial_specs)


def dynamic_margin_candidate_pairs(loss_config: Mapping[str, Any]) -> List[Tuple[int, int]]:
    dynamic_config = loss_config.get("dynamic_pair_margin", {}) or {}
    pair_specs = dynamic_config.get("candidate_pairs")
    if pair_specs is None:
        pair_specs = dynamic_config.get("initial_pair_margins")
    if pair_specs is None:
        pair_specs = loss_config.get("prototype_pair_margins", []) or []
    return [pair_key(int(spec["true"]), int(spec["negative"])) for spec in pair_specs]


def collect_subcenter_pair_gaps(
    model: PaCsiDGLite,
    loader: DataLoader,
    device: torch.device,
    candidate_pairs: Sequence[Tuple[int, int]],
) -> Dict[Tuple[int, int], torch.Tensor]:
    if not model.has_class_prototypes() or model.class_prototypes is None:
        raise ValueError("Dynamic pair margins require model.class_prototypes.")

    was_training = model.training
    model.eval()
    pair_chunks: Dict[Tuple[int, int], List[torch.Tensor]] = {pair: [] for pair in candidate_pairs}

    with torch.no_grad():
        prototypes = F.normalize(model.class_prototypes.detach(), dim=-1)
        num_classes, num_subcenters, feature_dim = prototypes.shape
        proto_flat = prototypes.reshape(num_classes * num_subcenters, feature_dim)

        for amplitude, phase, labels, domains, sample_ids in loader:
            amplitude = amplitude.to(device)
            phase = phase.to(device)
            labels = labels.to(device)

            outputs = model(amplitude, phase)
            views = outputs["projection"].unsqueeze(1)
            if model.view_builder is not None:
                view_outputs = model.encode_antenna_views(amplitude, phase)
                views = torch.cat([views, view_outputs["projection_views"]], dim=1)

            batch_size, num_views, current_dim = views.shape
            if current_dim != feature_dim:
                raise ValueError(
                    f"Projection dim {current_dim} does not match prototype dim {feature_dim}."
                )

            anchors = F.normalize(views.reshape(batch_size * num_views, current_dim), dim=-1)
            anchor_labels = labels.repeat_interleave(num_views)
            sim = anchors @ proto_flat.T
            per_class_max = sim.reshape(anchors.size(0), num_classes, num_subcenters).max(dim=-1).values

            for true_class, negative_class in candidate_pairs:
                if not (0 <= true_class < num_classes and 0 <= negative_class < num_classes):
                    raise ValueError(
                        f"Dynamic pair margin references out-of-range class indices: "
                        f"true={true_class}, negative={negative_class}, num_classes={num_classes}."
                    )
                mask = anchor_labels == true_class
                if not mask.any():
                    continue
                gaps = per_class_max[mask, true_class] - per_class_max[mask, negative_class]
                pair_chunks[(true_class, negative_class)].append(gaps.detach().cpu())

    if was_training:
        model.train()

    return {
        pair: torch.cat(chunks) if chunks else torch.empty(0)
        for pair, chunks in pair_chunks.items()
    }


def recalibrate_dynamic_pair_margins(
    model: PaCsiDGLite,
    loader: DataLoader,
    device: torch.device,
    loss_config: Mapping[str, Any],
    contrastive_loss_fn: SubCenterPrototypeContrastiveLoss,
    current_specs: Sequence[Mapping[str, Any]],
    artifact_path: Path,
    epoch: int,
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    dynamic_config = loss_config.get("dynamic_pair_margin", {}) or {}
    candidate_pairs = dynamic_margin_candidate_pairs(loss_config)
    current_map = pair_margin_map(current_specs)
    gap_by_pair = collect_subcenter_pair_gaps(model, loader, device, candidate_pairs)

    quantile = min(max(float(dynamic_config.get("quantile", 0.15)), 0.0), 1.0)
    ema_momentum = min(max(float(dynamic_config.get("ema_momentum", 0.8)), 0.0), 1.0)
    min_margin = float(dynamic_config.get("m_min", 0.03))
    max_margin = float(dynamic_config.get("m_max", 0.18))
    max_delta = float(dynamic_config.get("max_delta_per_update", 0.03))
    min_class_samples = int(dynamic_config.get("min_class_samples", 20))
    m_max_per_pair = parse_per_pair_caps(dynamic_config.get("m_max_per_pair"), "m_max")
    m_min_per_pair = parse_per_pair_caps(dynamic_config.get("m_min_per_pair"), "m_min")

    next_specs: List[Dict[str, float]] = []
    pair_records: List[Dict[str, Any]] = []

    for true_class, negative_class in candidate_pairs:
        key = pair_key(true_class, negative_class)
        pair_max = float(m_max_per_pair.get(key, max_margin))
        pair_min = float(m_min_per_pair.get(key, min_margin))
        if pair_min > pair_max:
            raise ValueError(
                f"Per-pair m_min ({pair_min}) exceeds m_max ({pair_max}) for pair "
                f"{true_class}->{negative_class}."
            )
        old_margin = float(current_map.get(key, pair_min))

        gaps = gap_by_pair.get(key, torch.empty(0))
        sample_count = int(gaps.numel())

        if sample_count >= min_class_samples:
            raw_margin = float(torch.quantile(gaps.float(), quantile).item())
            clipped_margin = min(max(raw_margin, pair_min), pair_max)
            ema_margin = ema_momentum * old_margin + (1.0 - ema_momentum) * clipped_margin
            if max_delta > 0.0:
                lower = old_margin - max_delta
                upper = old_margin + max_delta
                ema_margin = min(max(ema_margin, lower), upper)
            new_margin = min(max(ema_margin, pair_min), pair_max)
            updated = True
        else:
            raw_margin = None
            clipped_margin = None
            new_margin = old_margin
            updated = False

        next_specs.append({"true": true_class, "negative": negative_class, "margin": float(new_margin)})

        if sample_count > 0:
            gaps_float = gaps.float()
            gap_summary = {
                "mean": float(gaps_float.mean().item()),
                "p10": float(torch.quantile(gaps_float, 0.10).item()),
                "p50": float(torch.quantile(gaps_float, 0.50).item()),
                "p90": float(torch.quantile(gaps_float, 0.90).item()),
                "min": float(gaps_float.min().item()),
                "max": float(gaps_float.max().item()),
            }
        else:
            gap_summary = {}

        pair_records.append(
            {
                "true": true_class,
                "negative": negative_class,
                "old_margin": old_margin,
                "raw_quantile_margin": raw_margin,
                "clipped_margin": clipped_margin,
                "new_margin": float(new_margin),
                "updated": updated,
                "sample_count": sample_count,
                "gap_summary": gap_summary,
                "m_min": pair_min,
                "m_max": pair_max,
            }
        )

    contrastive_loss_fn.set_pair_margins(next_specs)
    margins = [spec["margin"] for spec in next_specs]
    debug = {
        "epoch": int(epoch),
        "source_split": str(dynamic_config.get("source_split", "val")),
        "quantile": quantile,
        "ema_momentum": ema_momentum,
        "m_min": min_margin,
        "m_max": max_margin,
        "m_min_per_pair": {f"{t}->{n}": v for (t, n), v in m_min_per_pair.items()},
        "m_max_per_pair": {f"{t}->{n}": v for (t, n), v in m_max_per_pair.items()},
        "max_delta_per_update": max_delta,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "min_margin": float(np.min(margins)) if margins else 0.0,
        "max_margin": float(np.max(margins)) if margins else 0.0,
        "updated_pair_count": int(sum(1 for record in pair_records if record["updated"])),
        "pair_margins": pair_records,
    }
    save_json(artifact_path, debug)
    return next_specs, debug


def build_model_and_optimizer(
    dataset,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[
    PaCsiDGLite,
    torch.optim.Optimizer,
    Optional[torch.optim.lr_scheduler.LRScheduler],
    Optional[str],
]:
    model = PaCsiDGLite(
        input_dim=dataset.input_dim,
        num_classes=dataset.num_classes,
        model_config=config["model"],
    ).to(device)
    training_config = config["training"]
    optimizer_name = str(training_config.get("optimizer", "adamw")).lower()
    learning_rate = float(training_config.get("lr", 1e-4))
    weight_decay = float(training_config.get("weight_decay", 1e-4))

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Choose from ['adamw', 'adam'].")

    scheduler, scheduler_step_mode = build_scheduler(optimizer, training_config)
    return model, optimizer, scheduler, scheduler_step_mode


def run_single_experiment(
    dataset,
    config: Mapping[str, Any],
    seed: int,
    target_env: Optional[str],
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    set_seed(seed)
    split = build_split_indices(
        dataset,
        mode=config["mode"],
        seed=seed,
        val_ratio=float(config["split"].get("val_ratio", 0.2)),
        test_ratio=float(config["split"].get("test_ratio", 0.2)),
        target_env=target_env,
    )
    split_manifest = build_split_manifest(dataset, split)
    save_json(output_dir / "split_manifest.json", split_manifest)

    batch_size = int(config["training"].get("batch_size", 32))
    num_workers = int(config["training"].get("num_workers", 0))
    train_loader = build_data_loader(
        dataset,
        split["train_indices"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=resolve_sampler_mode(config, split) == "none",
        sampler_mode=resolve_sampler_mode(config, split),
    )
    val_loader = build_data_loader(
        dataset,
        split["val_indices"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler_mode="none",
    )
    test_loader = build_data_loader(
        dataset,
        split["test_indices"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler_mode="none",
    )

    model, optimizer, scheduler, scheduler_step_mode = build_model_and_optimizer(dataset, config, device)
    loss_config = dict(config["losses"])
    loss_config["grad_clip_norm"] = config["training"].get("grad_clip_norm", 0.0)

    contrastive_loss_type = str(config["losses"].get("contrastive_loss_type", "pairwise_supcon"))
    if contrastive_loss_type not in SUPPORTED_CONTRASTIVE_LOSS_TYPES:
        raise ValueError(
            f"Unsupported contrastive_loss_type '{contrastive_loss_type}'. "
            f"Choose from {sorted(SUPPORTED_CONTRASTIVE_LOSS_TYPES)}."
        )
    dynamic_pair_margin_config = config["losses"].get("dynamic_pair_margin", {}) or {}
    dynamic_pair_margin_enabled = (
        contrastive_loss_type == "subcenter_prototype"
        and bool(dynamic_pair_margin_config.get("enabled", False))
    )
    current_pair_margin_specs = (
        dynamic_margin_initial_specs(config["losses"])
        if dynamic_pair_margin_enabled
        else normalize_pair_margin_specs(config["losses"].get("prototype_pair_margins", []) or [])
    )

    contrastive_loss_fn: Optional[torch.nn.Module]
    if contrastive_loss_type == "subcenter_prototype":
        if not model.has_class_prototypes():
            raise ValueError(
                "contrastive_loss_type='subcenter_prototype' requires the model to expose "
                "class_prototypes; set model.prototype_num_subcenters > 0 in the config."
            )
        contrastive_loss_fn = SubCenterPrototypeContrastiveLoss(
            temperature=float(config["losses"].get("temperature", 0.2)),
            lambda_pair_margin=float(config["losses"].get("lambda_pair_margin", 0.0)),
            prototype_pair_margins=current_pair_margin_specs,
        )
    else:
        contrastive_loss_fn = DomainAwareSupConLoss(
            temperature=float(config["losses"].get("temperature", 0.2)),
            include_same_domain_same_class=bool(config["losses"].get("include_same_domain_same_class", True)),
            positive_mode=str(config["losses"].get("supcon_positive_mode", "all_views")),
        )

    adaptive_kmargin_config = AdaptiveKMarginConfig.from_mapping(config["losses"].get("adaptive_kmargin"))
    adaptive_kmargin_loss_fn: Optional[AdaptiveKMarginLoss] = None
    adaptive_kmargin_recalibrator: Optional[AdaptiveKMarginRecalibrator] = None
    adaptive_kmargin_state: Optional[Mapping[str, torch.Tensor]] = None
    adaptive_kmargin_debug: Dict[str, Any] = {}
    dynamic_pair_margin_debug: Dict[str, Any] = {}
    if adaptive_kmargin_config.enabled and adaptive_kmargin_config.lambda_akm > 0.0:
        adaptive_kmargin_loss_fn = AdaptiveKMarginLoss()
        adaptive_kmargin_recalibrator = AdaptiveKMarginRecalibrator(
            adaptive_kmargin_config,
            num_classes=dataset.num_classes,
            device=device,
        )

    selection_metric = str(config["training"].get("selection_metric", "f1_macro"))
    best_val_score = -math.inf
    best_state: Optional[Dict[str, Any]] = None
    patience = int(config["training"].get("patience", 10))
    epochs = int(config["training"].get("epochs", 50))
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        epoch_loss_config = loss_config_for_lambda(
            loss_config,
            scheduled_lambda_supcon(loss_config, epoch),
        )
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scheduler_step_mode,
            device,
            epoch_loss_config,
            contrastive_loss_type,
            contrastive_loss_fn,
            adaptive_kmargin_loss_fn,
            adaptive_kmargin_state,
            train=True,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer,
                scheduler,
                scheduler_step_mode,
                device,
                epoch_loss_config,
                contrastive_loss_type,
                contrastive_loss_fn,
                adaptive_kmargin_loss_fn,
                adaptive_kmargin_state,
                train=False,
            )

        if (
            adaptive_kmargin_recalibrator is not None
            and epoch >= adaptive_kmargin_config.warmup_epochs
            and (epoch - adaptive_kmargin_config.warmup_epochs) % max(1, adaptive_kmargin_config.recalibrate_interval) == 0
        ):
            calibration = adaptive_kmargin_recalibrator.recalibrate(
                model,
                val_loader,
                artifact_path=output_dir / "adaptive_kmargin" / f"epoch_{epoch:03d}_stats.json",
            )
            adaptive_kmargin_state = calibration["state"]
            adaptive_kmargin_debug = calibration["debug_stats"]

        if (
            dynamic_pair_margin_enabled
            and isinstance(contrastive_loss_fn, SubCenterPrototypeContrastiveLoss)
            and epoch >= int(dynamic_pair_margin_config.get("warmup_epochs", 20))
            and (
                epoch - int(dynamic_pair_margin_config.get("warmup_epochs", 20))
            )
            % max(1, int(dynamic_pair_margin_config.get("recompute_interval", 10)))
            == 0
        ):
            current_pair_margin_specs, dynamic_pair_margin_debug = recalibrate_dynamic_pair_margins(
                model=model,
                loader=val_loader,
                device=device,
                loss_config=config["losses"],
                contrastive_loss_fn=contrastive_loss_fn,
                current_specs=current_pair_margin_specs,
                artifact_path=output_dir / "dynamic_pair_margin" / f"epoch_{epoch:03d}_margins.json",
                epoch=epoch,
            )

        epoch_record = {
            "epoch": epoch,
            "train": {k: v for k, v in train_metrics.items() if k not in {"labels", "predictions"}},
            "val": {k: v for k, v in val_metrics.items() if k not in {"labels", "predictions"}},
        }
        if adaptive_kmargin_debug:
            epoch_record["adaptive_kmargin"] = adaptive_kmargin_debug
        if dynamic_pair_margin_debug:
            epoch_record["dynamic_pair_margin"] = dynamic_pair_margin_debug
        history.append(epoch_record)

        if selection_metric not in val_metrics:
            raise KeyError(f"Selection metric '{selection_metric}' was not found in validation metrics.")
        current_val_score = float(val_metrics[selection_metric])

        if current_val_score >= best_val_score:
            best_val_score = current_val_score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(best_state, output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if scheduler is not None and scheduler_step_mode == "epoch":
            scheduler.step()

        if epochs_without_improvement > patience:
            break

    if best_state is None:
        raise RuntimeError("Training completed without producing a checkpoint.")

    model.load_state_dict(best_state)
    test_loss_config = loss_config_for_lambda(
        loss_config,
        float(loss_config.get("lambda_supcon", 0.0)),
    )
    with torch.no_grad():
        test_metrics = run_epoch(
            model,
            test_loader,
            optimizer,
            scheduler,
            scheduler_step_mode,
            device,
            test_loss_config,
            contrastive_loss_type,
            contrastive_loss_fn,
            adaptive_kmargin_loss_fn,
            adaptive_kmargin_state,
            train=False,
        )

    confusion = confusion_matrix(test_metrics["labels"], test_metrics["predictions"])
    if bool(config["evaluation"].get("save_confusion_matrix", True)):
        np.save(output_dir / "confusion_matrix.npy", confusion)
        save_confusion_csv(output_dir / "confusion_matrix.csv", confusion)

    result = {
        "seed": seed,
        "target_env": target_env,
        "source_envs": split["source_envs"],
        "selection_metric": selection_metric,
        "best_val_score": best_val_score,
        "best_val_accuracy": max((epoch["val"]["accuracy"] for epoch in history), default=0.0),
        "train_size": len(split["train_indices"]),
        "val_size": len(split["val_indices"]),
        "test_size": len(split["test_indices"]),
        "test_metrics": {k: v for k, v in test_metrics.items() if k not in {"labels", "predictions"}},
        "shape_debug": test_metrics["shape_debug"],
        "history": history,
        "losses_enabled": {
            "contrastive_loss_type": contrastive_loss_type,
            "lambda_supcon": config["losses"].get("lambda_supcon", 0.0),
            "lambda_supcon_warmup_epochs": int(
                config["losses"].get("lambda_supcon_warmup_epochs", 0)
            ),
            "temperature": float(config["losses"].get("temperature", 0.2)),
            "include_same_domain_same_class": bool(
                config["losses"].get("include_same_domain_same_class", True)
            ),
            "supcon_positive_mode": (
                config["losses"].get("supcon_positive_mode", "all_views")
                if contrastive_loss_type == "pairwise_supcon"
                else None
            ),
            "prototype_num_subcenters": int(config["model"].get("prototype_num_subcenters", 0)),
            "lambda_pair_margin": float(config["losses"].get("lambda_pair_margin", 0.0)),
            "effective_pair_margin_weight": (
                float(config["losses"].get("lambda_supcon", 0.0))
                * float(config["losses"].get("lambda_pair_margin", 0.0))
            ),
            "prototype_pair_margins": list(config["losses"].get("prototype_pair_margins", []) or []),
            "dynamic_pair_margin": {
                "enabled": bool(dynamic_pair_margin_enabled),
                "config": dict(dynamic_pair_margin_config),
                "final_pair_margins": (
                    contrastive_loss_fn.get_pair_margins()
                    if isinstance(contrastive_loss_fn, SubCenterPrototypeContrastiveLoss)
                    else []
                ),
            },
            "adaptive_kmargin": {
                "enabled": bool(adaptive_kmargin_config.enabled),
                "lambda_akm": float(adaptive_kmargin_config.lambda_akm),
            },
        },
        "sampler_mode": resolve_sampler_mode(config, split),
    }
    save_json(output_dir / "metrics.json", result)
    return result


def aggregate_results(results: List[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped_by_env: Dict[str, List[float]] = {}
    per_seed_rows: List[Dict[str, Any]] = []

    for result in results:
        env_key = result.get("target_env") or "random_split"
        grouped_by_env.setdefault(env_key, []).append(float(result["test_metrics"]["accuracy"]))
        per_seed_rows.append(
            {
                "seed": int(result["seed"]),
                "target_env": env_key,
                "accuracy": float(result["test_metrics"]["accuracy"]),
                "precision_macro": float(result["test_metrics"]["precision_macro"]),
                "recall_macro": float(result["test_metrics"]["recall_macro"]),
                "f1_macro": float(result["test_metrics"]["f1_macro"]),
            }
        )

    def summarize(values: List[float]) -> Dict[str, float]:
        if len(values) == 1:
            return {"mean": values[0], "std": 0.0}
        return {"mean": mean(values), "std": stdev(values)}

    by_env = {env_name: summarize(values) for env_name, values in sorted(grouped_by_env.items())}
    overall_values = [row["accuracy"] for row in per_seed_rows]
    return {
        "per_seed": per_seed_rows,
        "per_held_out_environment": by_env,
        "overall_accuracy": summarize(overall_values),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    raw_config = load_config(config_path)
    config = resolve_runtime_config(raw_config, args)

    device = choose_device(config.get("device", "cpu"))
    dataset = load_dataset(config["data"], REPO_ROOT)

    output_root = (REPO_ROOT / config.get("output_root", "outputs")).resolve()
    experiment_name = config.get("experiment_name", config_path.stem)
    target_envs = choose_target_envs(config, dataset)

    all_results: List[Dict[str, Any]] = []
    for target_env in target_envs:
        for seed in config.get("seed_list", [42]):
            run_seed = int(seed)
            env_name = target_env or "random_split"
            run_dir = output_root / config["mode"] / experiment_name / f"target_{env_name}" / f"seed_{run_seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_config = copy.deepcopy(config)
            run_config["target_env"] = target_env
            run_config["seed_list"] = [run_seed]
            run_config["device_actual"] = str(device)
            save_json(run_dir / "resolved_config.json", run_config)
            result = run_single_experiment(
                dataset=dataset,
                config=run_config,
                seed=run_seed,
                target_env=target_env,
                output_dir=run_dir,
                device=device,
            )
            all_results.append(result)

    summary = aggregate_results(all_results)
    summary.update(
        {
            "mode": config["mode"],
            "experiment_name": experiment_name,
            "target_envs": [env or "random_split" for env in target_envs],
            "seed_list": [int(seed) for seed in config.get("seed_list", [42])],
            "device": str(device),
        }
    )
    save_json(output_root / config["mode"] / experiment_name / "summary.json", summary)


if __name__ == "__main__":
    main()
