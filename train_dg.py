from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dg_dataset import (
    DomainBalancedBatchSampler,
    DomainClassBalancedBatchSampler,
    build_split_indices,
    build_split_manifest,
    load_dataset,
)
from dg_losses import (
    DomainAwareSupConLoss,
    InstanceContrastiveLoss,
    SubCenterPrototypeContrastiveLoss,
    adain_mix,
)
from dg_models import PaCsiDGLite


SUPPORTED_CONTRASTIVE_LOSS_TYPES = {"pairwise_supcon", "subcenter_prototype"}


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
    instance_loss_fn: InstanceContrastiveLoss,
    train: bool,
) -> Dict[str, Any]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_supcon = 0.0
    total_narc = 0.0
    total_adain = 0.0
    all_predictions: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    shape_debug: Optional[Dict[str, List[int]]] = None
    lambda_supcon = float(loss_config.get("lambda_supcon", 0.0))
    lambda_narc = float(loss_config.get("lambda_narc", 0.0))
    lambda_adain = float(loss_config.get("lambda_adain", 0.0))

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
            narc_loss = torch.zeros((), device=device)
            adain_loss = torch.zeros((), device=device)
            supcon_views = outputs["projection"].unsqueeze(1)
            view_outputs: Optional[Dict[str, torch.Tensor]] = None

            if lambda_narc > 0.0 or lambda_supcon > 0.0:
                if model.view_builder is not None:
                    view_outputs = model.encode_antenna_views(amplitude, phase)
                    supcon_views = torch.cat([supcon_views, view_outputs["projection_views"]], dim=1)
                elif lambda_narc > 0.0:
                    raise RuntimeError(
                        "NARC was enabled, but antenna views are unavailable. "
                        "Configure model.antenna_layout or disable lambda_narc."
                    )

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
                else:
                    raise ValueError(
                        f"Unsupported contrastive_loss_type '{contrastive_loss_type}'. "
                        f"Choose from {sorted(SUPPORTED_CONTRASTIVE_LOSS_TYPES)}."
                    )

            if lambda_narc > 0.0 and view_outputs is not None:
                narc_loss = instance_loss_fn(view_outputs["projection_views"], sample_ids=sample_ids)

            if bool(loss_config.get("use_adain_style_aug", False)) and lambda_adain > 0.0:
                styled_feature = adain_mix(outputs["fused_feature"], domains=domains)
                styled_projection = F.normalize(model.projection_head(styled_feature), dim=-1)
                adain_loss = instance_loss_fn(
                    torch.stack([outputs["projection"], styled_projection], dim=1),
                    sample_ids=sample_ids,
                )

            total_batch_loss = (
                ce_loss
                + lambda_supcon * supcon_loss
                + lambda_narc * narc_loss
                + lambda_adain * adain_loss
            )

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
        total_narc += float(narc_loss.item())
        total_adain += float(adain_loss.item())

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
            "loss_narc": 0.0,
            "loss_adain": 0.0,
            "labels": [],
            "predictions": [],
            "shape_debug": shape_debug or {},
        }

    labels_np = np.concatenate(all_labels, axis=0)
    predictions_np = np.concatenate(all_predictions, axis=0)
    metrics = metric_dict(labels_np, predictions_np)
    metrics.update(
        {
            "loss_total": total_loss / max(1, len(loader)),
            "loss_ce": total_ce / max(1, len(loader)),
            "loss_supcon": total_supcon / max(1, len(loader)),
            "loss_narc": total_narc / max(1, len(loader)),
            "loss_adain": total_adain / max(1, len(loader)),
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
            prototype_pair_margins=list(config["losses"].get("prototype_pair_margins", []) or []),
        )
    else:
        contrastive_loss_fn = DomainAwareSupConLoss(
            temperature=float(config["losses"].get("temperature", 0.2)),
            include_same_domain_same_class=bool(config["losses"].get("include_same_domain_same_class", True)),
            positive_mode=str(config["losses"].get("supcon_positive_mode", "all_views")),
        )
    instance_loss_fn = InstanceContrastiveLoss(temperature=float(config["losses"].get("temperature", 0.2)))

    selection_metric = str(config["training"].get("selection_metric", "f1_macro"))
    best_val_score = -math.inf
    best_state: Optional[Dict[str, Any]] = None
    patience = int(config["training"].get("patience", 10))
    epochs = int(config["training"].get("epochs", 50))
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scheduler_step_mode,
            device,
            loss_config,
            contrastive_loss_type,
            contrastive_loss_fn,
            instance_loss_fn,
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
                loss_config,
                contrastive_loss_type,
                contrastive_loss_fn,
                instance_loss_fn,
                train=False,
            )

        epoch_record = {
            "epoch": epoch,
            "train": {k: v for k, v in train_metrics.items() if k not in {"labels", "predictions"}},
            "val": {k: v for k, v in val_metrics.items() if k not in {"labels", "predictions"}},
        }
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
    with torch.no_grad():
        test_metrics = run_epoch(
            model,
            test_loader,
            optimizer,
            scheduler,
            scheduler_step_mode,
            device,
            loss_config,
            contrastive_loss_type,
            contrastive_loss_fn,
            instance_loss_fn,
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
            "lambda_narc": config["losses"].get("lambda_narc", 0.0),
            "lambda_adain": config["losses"].get("lambda_adain", 0.0),
            "supcon_positive_mode": (
                config["losses"].get("supcon_positive_mode", "all_views")
                if contrastive_loss_type == "pairwise_supcon"
                else None
            ),
            "prototype_num_subcenters": int(config["model"].get("prototype_num_subcenters", 0)),
            "lambda_pair_margin": float(config["losses"].get("lambda_pair_margin", 0.0)),
            "prototype_pair_margins": list(config["losses"].get("prototype_pair_margins", []) or []),
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
            env_name = target_env or "random_split"
            run_dir = output_root / config["mode"] / experiment_name / f"target_{env_name}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = run_single_experiment(
                dataset=dataset,
                config=config,
                seed=int(seed),
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
