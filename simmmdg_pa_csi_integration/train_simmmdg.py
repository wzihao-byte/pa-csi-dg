from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dg_dataset import build_split_indices, build_split_manifest, load_dataset  # noqa: E402
from dg_losses import DomainAwareSupConLoss, InstanceContrastiveLoss, adain_mix  # noqa: E402
from train_dg import (  # noqa: E402
    aggregate_results,
    build_data_loader,
    build_scheduler,
    choose_device,
    choose_target_envs,
    load_config,
    metric_dict,
    resolve_runtime_config,
    resolve_sampler_mode,
    save_confusion_csv,
    save_json,
    set_seed,
)

from simmmdg_losses import SharedPrivateSeparationLoss, cross_view_translation_loss  # noqa: E402
from simmmdg_models import PaCsiSimMMDG  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated SimMMDG-inspired PA-CSI DG runner.")
    parser.add_argument("--config", required=True, help="Path to a JSON config preset.")
    parser.add_argument("--mode", choices=["random_split", "dg_loeo"], help="Override split mode.")
    parser.add_argument("--target-env", dest="target_env", help="Run only one held-out environment.")
    parser.add_argument("--device", help="Override the configured device.")
    parser.add_argument("--output-root", help="Override the configured output directory.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override the configured seed list.")
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count.")
    return parser.parse_args()


def build_model_and_optimizer(
    dataset,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[
    PaCsiSimMMDG,
    torch.optim.Optimizer,
    Optional[torch.optim.lr_scheduler.LRScheduler],
    Optional[str],
]:
    model = PaCsiSimMMDG(
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


def _mean_losses(losses: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def run_epoch(
    model: PaCsiSimMMDG,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scheduler_step_mode: Optional[str],
    device: torch.device,
    loss_config: Mapping[str, Any],
    supcon_loss_fn: DomainAwareSupConLoss,
    instance_loss_fn: InstanceContrastiveLoss,
    separation_loss_fn: SharedPrivateSeparationLoss,
    train: bool,
) -> Dict[str, Any]:
    model.train(train)

    total_loss = 0.0
    total_ce = 0.0
    total_supcon = 0.0
    total_sep = 0.0
    total_translation = 0.0
    total_narc = 0.0
    total_adain = 0.0
    all_predictions: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    shape_debug: Optional[Dict[str, List[int]]] = None

    lambda_supcon = float(loss_config.get("lambda_supcon", 0.0))
    lambda_sep = float(loss_config.get("lambda_sep", 0.0))
    lambda_translation = float(loss_config.get("lambda_translation", 0.0))
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
            sep_loss = torch.zeros((), device=device)
            translation_loss = torch.zeros((), device=device)
            narc_loss = torch.zeros((), device=device)
            adain_loss = torch.zeros((), device=device)

            view_outputs: Optional[Dict[str, torch.Tensor]] = None
            needs_views = (
                lambda_narc > 0.0 or lambda_translation > 0.0 or lambda_supcon > 0.0
            ) and model.view_builder is not None
            if needs_views:
                view_outputs = model.encode_antenna_views(amplitude, phase)
            elif lambda_narc > 0.0 or lambda_translation > 0.0:
                raise RuntimeError(
                    "A view-dependent loss was enabled, but antenna views are unavailable. "
                    "Configure model.antenna_layout or set lambda_narc/lambda_translation to 0."
                )

            if lambda_supcon > 0.0:
                supcon_views = outputs["projection"].unsqueeze(1)
                if view_outputs is not None:
                    supcon_views = torch.cat([supcon_views, view_outputs["projection_views"]], dim=1)
                supcon_loss = supcon_loss_fn(
                    supcon_views,
                    labels=labels,
                    domains=domains,
                    sample_ids=sample_ids,
                )

            if lambda_sep > 0.0:
                sep_parts = [
                    separation_loss_fn(outputs["shared_feature"], outputs["private_feature"]),
                ]
                if view_outputs is not None:
                    sep_parts.append(separation_loss_fn(view_outputs["shared_views"], view_outputs["private_views"]))
                sep_loss = _mean_losses(sep_parts, device)

            if lambda_translation > 0.0 and view_outputs is not None:
                translation_loss = cross_view_translation_loss(
                    model.shared_translator,
                    view_outputs["shared_views"],
                    detach_target=bool(loss_config.get("translation_detach_target", False)),
                )

            if lambda_narc > 0.0 and view_outputs is not None:
                narc_loss = instance_loss_fn(view_outputs["projection_views"], sample_ids=sample_ids)

            if bool(loss_config.get("use_adain_style_aug", False)) and lambda_adain > 0.0:
                styled_feature = adain_mix(outputs["fused_feature"], domains=domains)
                styled_decomposed = model.decompose_feature(styled_feature)
                adain_loss = instance_loss_fn(
                    torch.stack([outputs["projection"], styled_decomposed["projection"]], dim=1),
                    sample_ids=sample_ids,
                )

            total_batch_loss = (
                ce_loss
                + lambda_supcon * supcon_loss
                + lambda_sep * sep_loss
                + lambda_translation * translation_loss
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
        total_sep += float(sep_loss.item())
        total_translation += float(translation_loss.item())
        total_narc += float(narc_loss.item())
        total_adain += float(adain_loss.item())

        if shape_debug is None:
            shape_debug = {
                "amplitude": list(amplitude.shape),
                "phase": list(phase.shape),
                "fused_feature": list(outputs["fused_feature"].shape),
                "shared_feature": list(outputs["shared_feature"].shape),
                "private_feature": list(outputs["private_feature"].shape),
                "projection": list(outputs["projection"].shape),
            }
            if view_outputs is not None:
                shape_debug["multi_antenna_views"] = list(view_outputs["amp_views"].shape)
                shape_debug["shared_views"] = list(view_outputs["shared_views"].shape)
                shape_debug["private_views"] = list(view_outputs["private_views"].shape)

    if not all_labels:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "loss_total": 0.0,
            "loss_ce": 0.0,
            "loss_supcon": 0.0,
            "loss_sep": 0.0,
            "loss_translation": 0.0,
            "loss_narc": 0.0,
            "loss_adain": 0.0,
            "labels": [],
            "predictions": [],
            "shape_debug": shape_debug or {},
        }

    labels_np = np.concatenate(all_labels, axis=0)
    predictions_np = np.concatenate(all_predictions, axis=0)
    metrics = metric_dict(labels_np, predictions_np)
    denom = max(1, len(loader))
    metrics.update(
        {
            "loss_total": total_loss / denom,
            "loss_ce": total_ce / denom,
            "loss_supcon": total_supcon / denom,
            "loss_sep": total_sep / denom,
            "loss_translation": total_translation / denom,
            "loss_narc": total_narc / denom,
            "loss_adain": total_adain / denom,
            "labels": labels_np.tolist(),
            "predictions": predictions_np.tolist(),
            "shape_debug": shape_debug or {},
        }
    )
    return metrics


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
    supcon_loss_fn = DomainAwareSupConLoss(
        temperature=float(config["losses"].get("temperature", 0.2)),
        include_same_domain_same_class=bool(config["losses"].get("include_same_domain_same_class", True)),
    )
    instance_loss_fn = InstanceContrastiveLoss(temperature=float(config["losses"].get("temperature", 0.2)))
    separation_loss_fn = SharedPrivateSeparationLoss()

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
            supcon_loss_fn,
            instance_loss_fn,
            separation_loss_fn,
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
                supcon_loss_fn,
                instance_loss_fn,
                separation_loss_fn,
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
            supcon_loss_fn,
            instance_loss_fn,
            separation_loss_fn,
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
            "lambda_supcon": config["losses"].get("lambda_supcon", 0.0),
            "lambda_sep": config["losses"].get("lambda_sep", 0.0),
            "lambda_translation": config["losses"].get("lambda_translation", 0.0),
            "lambda_narc": config["losses"].get("lambda_narc", 0.0),
            "lambda_adain": config["losses"].get("lambda_adain", 0.0),
        },
        "sampler_mode": resolve_sampler_mode(config, split),
    }
    save_json(output_dir / "metrics.json", result)
    return result


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    raw_config = load_config(config_path)
    config = resolve_runtime_config(raw_config, args)

    device = choose_device(config.get("device", "cpu"))
    dataset = load_dataset(config["data"], REPO_ROOT)

    output_root = (REPO_ROOT / config.get("output_root", "simmmdg_pa_csi_integration/outputs")).resolve()
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
            "integration": "simmmdg_shared_private_pa_csi",
        }
    )
    save_json(output_root / config["mode"] / experiment_name / "summary.json", summary)


if __name__ == "__main__":
    main()
