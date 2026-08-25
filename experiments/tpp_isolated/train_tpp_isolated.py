from __future__ import annotations

import copy
import inspect
import math
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_dg
from adaptive_kmargin import AdaptiveKMarginConfig, AdaptiveKMarginRecalibrator
from dg_dataset import build_split_indices, build_split_manifest, load_dataset
from dg_losses import AdaptiveKMarginLoss, DomainAwareSupConLoss, SubCenterPrototypeContrastiveLoss
from tpp_models import IsolatedPaCsiDGLiteTPP


def build_split_indices_for_config(dataset, config: Mapping[str, Any], seed: int, target_env: Optional[str]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "mode": config["mode"],
        "seed": seed,
        "val_ratio": float(config["split"].get("val_ratio", 0.2)),
        "test_ratio": float(config["split"].get("test_ratio", 0.2)),
        "target_env": target_env,
    }
    signature = inspect.signature(build_split_indices)
    if "inner_validation" in signature.parameters:
        kwargs["inner_validation"] = str(config["split"].get("inner_validation", "stratified_holdout"))
    if "loso_validation_env" in signature.parameters:
        kwargs["loso_validation_env"] = config["split"].get("loso_validation_env")
    return build_split_indices(**kwargs)


def build_model_and_optimizer(
    dataset,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[
    IsolatedPaCsiDGLiteTPP,
    torch.optim.Optimizer,
    Optional[torch.optim.lr_scheduler.LRScheduler],
    Optional[str],
]:
    model = IsolatedPaCsiDGLiteTPP(
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

    scheduler, scheduler_step_mode = train_dg.build_scheduler(optimizer, training_config)
    return model, optimizer, scheduler, scheduler_step_mode


def hard_pair_report(confusion: np.ndarray, class_a: int = 1, class_b: int = 4) -> Dict[str, float]:
    def row_total(class_index: int) -> float:
        if class_index >= confusion.shape[0]:
            return 0.0
        return float(confusion[class_index, :].sum())

    def cell(true_class: int, pred_class: int) -> float:
        if true_class >= confusion.shape[0] or pred_class >= confusion.shape[1]:
            return 0.0
        return float(confusion[true_class, pred_class])

    row_a = row_total(class_a)
    row_b = row_total(class_b)
    class_a_recall = cell(class_a, class_a) / row_a if row_a else 0.0
    class_b_recall = cell(class_b, class_b) / row_b if row_b else 0.0
    a_to_b_rate = cell(class_a, class_b) / row_a if row_a else 0.0
    b_to_a_rate = cell(class_b, class_a) / row_b if row_b else 0.0
    pair_total = row_a + row_b
    pair_confusion = (cell(class_a, class_b) + cell(class_b, class_a)) / pair_total if pair_total else 0.0

    return {
        "class1_acc": class_a_recall,
        "class4_acc": class_b_recall,
        "class1_recall": class_a_recall,
        "class4_recall": class_b_recall,
        "true1_pred4_rate": a_to_b_rate,
        "true4_pred1_rate": b_to_a_rate,
        "pair_confusion_rate": pair_confusion,
        "true1_total": row_a,
        "true4_total": row_b,
        "true1_pred4_count": cell(class_a, class_b),
        "true4_pred1_count": cell(class_b, class_a),
    }


def run_single_experiment(
    dataset,
    config: Mapping[str, Any],
    seed: int,
    target_env: Optional[str],
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    train_dg.set_seed(seed)
    split = build_split_indices_for_config(dataset, config, seed, target_env)
    split_manifest = build_split_manifest(dataset, split)
    train_dg.save_json(output_dir / "split_manifest.json", split_manifest)

    batch_size = int(config["training"].get("batch_size", 32))
    num_workers = int(config["training"].get("num_workers", 0))
    sampler_mode = train_dg.resolve_sampler_mode(config, split)
    train_loader = train_dg.build_data_loader(
        dataset,
        split["train_indices"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=sampler_mode == "none",
        sampler_mode=sampler_mode,
    )
    val_loader = train_dg.build_data_loader(
        dataset,
        split["val_indices"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler_mode="none",
    )
    test_loader = train_dg.build_data_loader(
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
    if contrastive_loss_type not in train_dg.SUPPORTED_CONTRASTIVE_LOSS_TYPES:
        raise ValueError(
            f"Unsupported contrastive_loss_type '{contrastive_loss_type}'. "
            f"Choose from {sorted(train_dg.SUPPORTED_CONTRASTIVE_LOSS_TYPES)}."
        )
    dynamic_pair_margin_config = config["losses"].get("dynamic_pair_margin", {}) or {}
    dynamic_pair_margin_enabled = (
        contrastive_loss_type == "subcenter_prototype"
        and bool(dynamic_pair_margin_config.get("enabled", False))
    )
    current_pair_margin_specs = (
        train_dg.dynamic_margin_initial_specs(config["losses"])
        if dynamic_pair_margin_enabled
        else train_dg.normalize_pair_margin_specs(config["losses"].get("prototype_pair_margins", []) or [])
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
        epoch_loss_config = train_dg.loss_config_for_lambda(
            loss_config,
            train_dg.scheduled_lambda_supcon(loss_config, epoch),
        )
        train_metrics = train_dg.run_epoch(
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
            val_metrics = train_dg.run_epoch(
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
            current_pair_margin_specs, dynamic_pair_margin_debug = train_dg.recalibrate_dynamic_pair_margins(
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
    test_loss_config = train_dg.loss_config_for_lambda(
        loss_config,
        float(loss_config.get("lambda_supcon", 0.0)),
    )
    with torch.no_grad():
        test_metrics = train_dg.run_epoch(
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

    confusion = confusion_matrix(
        test_metrics["labels"],
        test_metrics["predictions"],
        labels=list(range(dataset.num_classes)),
    )
    hard_pair = hard_pair_report(confusion)
    if bool(config["evaluation"].get("save_confusion_matrix", True)):
        np.save(output_dir / "confusion_matrix.npy", confusion)
        train_dg.save_confusion_csv(output_dir / "confusion_matrix.csv", confusion)
    train_dg.save_json(output_dir / "hard_pair_report.json", hard_pair)

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
        "test_metrics": {
            **{k: v for k, v in test_metrics.items() if k not in {"labels", "predictions"}},
            **hard_pair,
        },
        "hard_pair_report": hard_pair,
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
        "sampler_mode": sampler_mode,
        "tpp": {
            "levels": list(config["model"].get("tpp_levels", [1, 2, 4])),
            "channel_levels": list(config["model"].get("channel_tpp_levels", config["model"].get("tpp_levels", [1, 2, 4]))),
            "pool_type": str(config["model"].get("tpp_pool_type", "max")),
            "channel_pool_type": str(
                config["model"].get(
                    "channel_tpp_pool_type",
                    config["model"].get("tpp_pool_type", "max"),
                )
            ),
            "mixed_max_weight": float(config["model"].get("tpp_mixed_max_weight", 0.5)),
            "preserve_global": bool(config["model"].get("tpp_preserve_global", False)),
            "pyramid_scale": float(config["model"].get("tpp_pyramid_scale", 1.0)),
        },
    }
    train_dg.save_json(output_dir / "metrics.json", result)
    return result


def aggregate_results(results: List[Mapping[str, Any]]) -> Dict[str, Any]:
    metric_names = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "class1_acc",
        "class4_acc",
        "true1_pred4_rate",
        "true4_pred1_rate",
        "pair_confusion_rate",
    ]
    per_seed_rows: List[Dict[str, Any]] = []
    grouped_by_env: Dict[str, List[Mapping[str, Any]]] = {}

    for result in results:
        env_key = result.get("target_env") or "random_split"
        test_metrics = result["test_metrics"]
        grouped_by_env.setdefault(env_key, []).append(test_metrics)
        per_seed_rows.append(
            {
                "seed": int(result["seed"]),
                "target_env": env_key,
                **{name: float(test_metrics.get(name, 0.0)) for name in metric_names},
            }
        )

    def summarize(values: List[float]) -> Dict[str, float]:
        if len(values) == 1:
            return {"mean": values[0], "std": 0.0}
        return {"mean": mean(values), "std": stdev(values)}

    def summarize_metrics(rows: List[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
        return {name: summarize([float(row.get(name, 0.0)) for row in rows]) for name in metric_names}

    overall_rows = [result["test_metrics"] for result in results]
    return {
        "per_seed": per_seed_rows,
        "per_held_out_environment": {
            env_name: summarize_metrics(rows) for env_name, rows in sorted(grouped_by_env.items())
        },
        "overall": summarize_metrics(overall_rows),
        "overall_accuracy": summarize([row["accuracy"] for row in per_seed_rows]),
    }


def main() -> None:
    args = train_dg.parse_args()
    config_path = Path(args.config).resolve()
    raw_config = train_dg.load_config(config_path)
    config = train_dg.resolve_runtime_config(raw_config, args)

    device = train_dg.choose_device(config.get("device", "cpu"))
    dataset = load_dataset(config["data"], REPO_ROOT)

    output_root = (REPO_ROOT / config.get("output_root", "outputs_tpp_isolated")).resolve()
    experiment_name = config.get("experiment_name", config_path.stem)
    target_envs = train_dg.choose_target_envs(config, dataset)

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
            train_dg.save_json(run_dir / "resolved_config.json", run_config)
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
            "tpp": {
                "levels": list(config["model"].get("tpp_levels", [1, 2, 4])),
                "channel_levels": list(config["model"].get("channel_tpp_levels", config["model"].get("tpp_levels", [1, 2, 4]))),
                "pool_type": str(config["model"].get("tpp_pool_type", "max")),
                "channel_pool_type": str(
                    config["model"].get(
                        "channel_tpp_pool_type",
                        config["model"].get("tpp_pool_type", "max"),
                    )
                ),
                "mixed_max_weight": float(config["model"].get("tpp_mixed_max_weight", 0.5)),
                "preserve_global": bool(config["model"].get("tpp_preserve_global", False)),
                "pyramid_scale": float(config["model"].get("tpp_pyramid_scale", 1.0)),
            },
        }
    )
    train_dg.save_json(output_root / config["mode"] / experiment_name / "summary.json", summary)


if __name__ == "__main__":
    main()
