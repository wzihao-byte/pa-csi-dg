from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dg_dataset import build_split_indices, load_dataset  # noqa: E402
from train_dg import build_data_loader, build_model_and_optimizer, choose_device, run_single_experiment  # noqa: E402


TARGETS = ("E1", "E2", "E3")
SEEDS = (42, 52, 62)
K_VALUES = (1, 2, 3, 4, 5)
FOCUS_PAIRS = ((1, 4), (4, 1), (0, 4))
RAW_MARGIN_GRID = (0.00, 0.05, 0.10, 0.15, 0.20)
RAW_VIOLATION_MARGINS = (0.05, 0.10, 0.15, 0.20)
QUANTILE_GRID = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
EPSILON_PROTO_VALUES = (0.50, 0.75, 1.00, 1.25)
R_VALUES = (1, 2, 4, 8)
N_MIN = 400


Pair = Tuple[int, int]
PairMargins = Dict[Pair, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and summarize LOEO K / PairMargin experiments for PA-CSI DG."
    )
    parser.add_argument(
        "--stage",
        choices=[
            "stage-a",
            "stage-a-confirm-200",
            "postprocess",
            "build-margin-tables",
            "margin-raw-40",
            "margin-raw-joint-40",
            "margin-quantile-40",
            "margin-quantile-joint-40",
            "margin-search-40",
            "build-shortlist-plan",
            "margin-shortlist-40",
            "confirm-200",
            "aggregate",
            "all",
        ],
        required=True,
    )
    parser.add_argument(
        "--base-config",
        default=str(REPO_ROOT / "configs" / "pa_csi_dg_subcenter_proto_layer1_40e_k2_nopair_e2.json"),
    )
    parser.add_argument("--output-root", default=str(REPO_ROOT / "results" / "k_margin_search"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--targets", nargs="+", default=list(TARGETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--k-values", nargs="+", type=int, default=list(K_VALUES))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--low-occupancy-threshold", type=float, default=0.05)
    parser.add_argument("--joint-top-values-per-pair", type=int, default=2)
    parser.add_argument("--confirm-top-n", type=int, default=2)
    parser.add_argument(
        "--confirm-selection-metric",
        choices=["source_val_accuracy", "source_val_macro_f1"],
        default="source_val_macro_f1",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key, "")) for key in fieldnames})


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(to_jsonable(value), sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    if value is None:
        return ""
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def fmt_float(value: float) -> str:
    return f"{float(value):.2f}"


def pair_tag(pair: Pair) -> str:
    return f"{pair[0]}to{pair[1]}"


def parse_pair_tag(value: str) -> Pair:
    left, right = value.split("to", 1)
    return int(left), int(right)


def margin_config_name(kind: str, values: Mapping[str, Any]) -> str:
    if kind == "no_margin":
        return "no_margin"
    if kind == "raw_shared":
        return f"raw_margin_shared_m_{fmt_float(float(values['m']))}"
    if kind == "raw_single":
        return f"raw_margin_pairspecific_{pair_tag(values['pair'])}_m_{fmt_float(float(values['m']))}"
    if kind == "raw_joint":
        parts = [
            f"{pair_tag(pair)}_{fmt_float(float(values['margins'][pair]))}"
            for pair in FOCUS_PAIRS
        ]
        return "raw_margin_pairspecific_" + "_".join(parts)
    if kind == "quantile_shared":
        return f"quantile_shared_q_{fmt_float(float(values['q']))}"
    if kind == "quantile_single":
        return f"quantile_pairspecific_{pair_tag(values['pair'])}_q_{fmt_float(float(values['q']))}"
    if kind == "quantile_joint":
        parts = [
            f"{pair_tag(pair)}_{fmt_float(float(values['q_values'][pair]))}"
            for pair in FOCUS_PAIRS
        ]
        return "quantile_pairspecific_" + "_".join(parts)
    raise ValueError(f"Unknown margin config kind: {kind}")


def run_dir_for(
    output_root: Path,
    target: str,
    k_value: int,
    config_name: str,
    seed: int,
) -> Path:
    return output_root / f"target_{target}" / f"K_{k_value}" / config_name / f"seed_{seed}"


def set_fixed_experiment_config(
    base_config: Mapping[str, Any],
    *,
    target: str,
    seed: int,
    k_value: int,
    epochs: int,
    pair_margins: Optional[PairMargins],
    output_root: Path,
    config_name: str,
    device: Optional[str],
    margin_kind: str,
    margin_values: Mapping[str, Any],
) -> Dict[str, Any]:
    config = copy.deepcopy(dict(base_config))
    config["experiment_name"] = f"k_margin_search_{target}_K{k_value}_{config_name}_seed{seed}_{epochs}e"
    config["mode"] = "dg_loeo"
    config["target_env"] = target
    config["output_root"] = str(output_root)
    config["seed_list"] = [int(seed)]
    if device is not None:
        config["device"] = device
    config.setdefault("split", {})
    config["split"]["val_ratio"] = 0.2
    config["split"]["test_ratio"] = 0.2

    config.setdefault("model", {})
    config["model"]["prototype_num_subcenters"] = int(k_value)
    config["model"]["projection_dim"] = 128
    if "antenna_layout" not in config["model"]:
        config["model"]["antenna_layout"] = {
            "mode": "tx",
            "num_rx": 1,
            "num_tx": 3,
            "num_subcarriers": 30,
        }

    config.setdefault("training", {})
    config["training"]["epochs"] = int(epochs)
    config["training"]["sampler"] = str(config["training"].get("sampler", "none"))
    config["training"]["selection_metric"] = str(config["training"].get("selection_metric", "accuracy"))

    config.setdefault("losses", {})
    config["losses"]["temperature"] = 0.2
    config["losses"]["contrastive_loss_type"] = "subcenter_prototype"
    config["losses"]["lambda_supcon"] = 0.1
    config["losses"]["lambda_narc"] = 0.0
    config["losses"]["lambda_adain"] = 0.0
    config["losses"]["use_adain_style_aug"] = False
    config["losses"]["include_same_domain_same_class"] = True

    if pair_margins is None:
        config["losses"]["lambda_pair_margin"] = 0.0
        config["losses"].pop("prototype_pair_margins", None)
    else:
        config["losses"]["lambda_pair_margin"] = 0.05
        config["losses"]["prototype_pair_margins"] = [
            {"true": int(pair[0]), "negative": int(pair[1]), "margin": float(margin)}
            for pair, margin in sorted(pair_margins.items())
        ]

    config.setdefault("evaluation", {})
    config["evaluation"]["save_confusion_matrix"] = True
    config["k_margin_search"] = {
        "target": target,
        "seed": int(seed),
        "K": int(k_value),
        "epochs": int(epochs),
        "margin_config": config_name,
        "margin_kind": margin_kind,
        "margin_values": serialize_margin_values(margin_values),
        "focus_pairs": [list(pair) for pair in FOCUS_PAIRS],
        "raw_margin_grid": list(RAW_MARGIN_GRID),
        "quantile_grid": list(QUANTILE_GRID),
        "lambda_sup": 0.1,
        "lambda_pair": 0.05,
        "tau": 0.2,
    }
    return config


def serialize_margin_values(values: Mapping[str, Any]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value):
            serialized[key] = list(value)
        elif isinstance(value, Mapping):
            mapped: Dict[str, Any] = {}
            for inner_key, inner_value in value.items():
                if isinstance(inner_key, tuple) and len(inner_key) == 2:
                    mapped[pair_tag(inner_key)] = inner_value
                else:
                    mapped[str(inner_key)] = inner_value
            serialized[key] = mapped
        else:
            serialized[key] = value
    return serialized


def stage_a_configs() -> Iterable[Tuple[str, Mapping[str, Any], Optional[PairMargins]]]:
    yield "no_margin", {}, None


def raw_shared_configs() -> Iterable[Tuple[str, Mapping[str, Any], PairMargins]]:
    for margin in RAW_MARGIN_GRID:
        values = {"m": float(margin)}
        yield margin_config_name("raw_shared", values), values, {pair: float(margin) for pair in FOCUS_PAIRS}


def raw_single_configs() -> Iterable[Tuple[str, Mapping[str, Any], PairMargins]]:
    for pair in FOCUS_PAIRS:
        for margin in RAW_MARGIN_GRID:
            values = {"pair": pair, "m": float(margin)}
            yield margin_config_name("raw_single", values), values, {pair: float(margin)}


def quantile_shared_configs(
    output_root: Path,
    target: str,
    k_value: int,
    seed: int,
) -> Iterable[Tuple[str, Mapping[str, Any], PairMargins]]:
    table = load_quantile_table(output_root, target, k_value, seed)
    for q_value in QUANTILE_GRID:
        margins = {pair: float(table[pair][float(q_value)]) for pair in FOCUS_PAIRS}
        values = {"q": float(q_value), "margins": margins}
        yield margin_config_name("quantile_shared", values), values, margins


def quantile_single_configs(
    output_root: Path,
    target: str,
    k_value: int,
    seed: int,
) -> Iterable[Tuple[str, Mapping[str, Any], PairMargins]]:
    table = load_quantile_table(output_root, target, k_value, seed)
    for pair in FOCUS_PAIRS:
        for q_value in QUANTILE_GRID:
            margin = float(table[pair][float(q_value)])
            values = {"pair": pair, "q": float(q_value), "margins": {pair: margin}}
            yield margin_config_name("quantile_single", values), values, {pair: margin}


def load_quantile_table(output_root: Path, target: str, k_value: int, seed: int) -> Dict[Pair, Dict[float, float]]:
    table_path = run_dir_for(output_root, target, k_value, "no_margin", seed) / "quantile_margin_table.csv"
    rows = read_csv_rows(table_path)
    if not rows:
        raise FileNotFoundError(
            f"Missing quantile table for target={target} K={k_value} seed={seed}: {table_path}"
        )
    table: Dict[Pair, Dict[float, float]] = defaultdict(dict)
    for row in rows:
        pair = (int(row["true_class"]), int(row["negative_class"]))
        table[pair][round(float(row["q"]), 2)] = float(row["margin"])
    missing = [pair for pair in FOCUS_PAIRS if pair not in table]
    if missing:
        raise ValueError(f"Quantile table {table_path} is missing focus pairs: {missing}")
    return table


def load_base_config(path: Path) -> Dict[str, Any]:
    config = read_json(path)
    config.setdefault("data", {})
    config.setdefault("model", {})
    config.setdefault("training", {})
    config.setdefault("losses", {})
    return config


def run_units(
    *,
    args: argparse.Namespace,
    dataset: Any,
    base_config: Mapping[str, Any],
    output_root: Path,
    units: Iterable[Tuple[str, int, int, str, Mapping[str, Any], Optional[PairMargins], int, str]],
) -> int:
    completed = 0
    for target, k_value, seed, config_name, margin_values, pair_margins, epochs, margin_kind in units:
        if args.max_runs is not None and completed >= args.max_runs:
            print(f"[k-margin] max-runs reached: {args.max_runs}", flush=True)
            break
        config = set_fixed_experiment_config(
            base_config,
            target=target,
            seed=seed,
            k_value=k_value,
            epochs=epochs,
            pair_margins=pair_margins,
            output_root=output_root,
            config_name=config_name,
            device=args.device,
            margin_kind=margin_kind,
            margin_values=margin_values,
        )
        run_dir = run_dir_for(output_root, target, k_value, config_name, seed)
        marker_path = run_dir / "statistics_complete.json"
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists() and marker_path.exists() and not args.force:
            print(f"[k-margin] skip complete {run_dir}", flush=True)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "config.json", config)

        if not metrics_path.exists() or args.force:
            print(
                f"[k-margin] train target={target} K={k_value} seed={seed} "
                f"epochs={epochs} config={config_name}",
                flush=True,
            )
            device = choose_device(config.get("device", "cpu"))
            result = run_single_experiment(
                dataset=dataset,
                config=config,
                seed=int(seed),
                target_env=target,
                output_dir=run_dir,
                device=device,
            )
            write_json(run_dir / "metrics.json", result)
        else:
            print(f"[k-margin] postprocess existing metrics {run_dir}", flush=True)

        postprocess_run(
            dataset=dataset,
            config=config,
            run_dir=run_dir,
            low_occupancy_threshold=float(args.low_occupancy_threshold),
        )
        completed += 1
    return completed


def collect_model_outputs(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    all_labels: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_projection: List[np.ndarray] = []
    all_view_projection: List[np.ndarray] = []
    all_true_proto_assignment: List[np.ndarray] = []

    prototypes = getattr(model, "class_prototypes", None)
    if prototypes is None:
        raise RuntimeError("Expected model.class_prototypes for K / margin statistics.")

    with torch.no_grad():
        for amplitude, phase, labels, _domains, _sample_ids in loader:
            amplitude = amplitude.to(device)
            phase = phase.to(device)
            labels = labels.to(device)
            outputs = model(amplitude, phase)
            logits = outputs["logits"]
            projection = F.normalize(outputs["projection"], dim=-1)
            predictions = logits.argmax(dim=1)

            proto_norm = F.normalize(prototypes, dim=-1)
            sims = torch.einsum("nd,ckd->nck", projection, proto_norm)
            scores = sims.max(dim=-1).values
            true_class_sims = sims[torch.arange(labels.size(0), device=device), labels]
            true_assignments = true_class_sims.argmax(dim=-1)

            view_embeddings = [projection.unsqueeze(1)]
            if getattr(model, "view_builder", None) is not None:
                view_outputs = model.encode_antenna_views(amplitude, phase)
                view_embeddings.append(F.normalize(view_outputs["projection_views"], dim=-1))
            projection_views = torch.cat(view_embeddings, dim=1)

            all_labels.append(labels.detach().cpu().numpy())
            all_predictions.append(predictions.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())
            all_scores.append(scores.detach().cpu().numpy())
            all_projection.append(projection.detach().cpu().numpy())
            all_view_projection.append(projection_views.detach().cpu().numpy())
            all_true_proto_assignment.append(true_assignments.detach().cpu().numpy())

    return {
        "labels": np.concatenate(all_labels, axis=0),
        "predictions": np.concatenate(all_predictions, axis=0),
        "logits": np.concatenate(all_logits, axis=0),
        "scores": np.concatenate(all_scores, axis=0),
        "projection": np.concatenate(all_projection, axis=0),
        "projection_views": np.concatenate(all_view_projection, axis=0),
        "true_proto_assignment": np.concatenate(all_true_proto_assignment, axis=0),
    }


def compute_basic_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> Dict[str, Any]:
    class_labels = list(range(num_classes))
    per_class_f1 = f1_score(labels, predictions, labels=class_labels, average=None, zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=class_labels)
    normalized = normalize_confusion(matrix)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "per_class_f1": {str(index): float(value) for index, value in enumerate(per_class_f1)},
        "confusion_matrix": matrix,
        "normalized_confusion": normalized,
    }


def normalize_confusion(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=np.float64), where=row_sums != 0)


def write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["true\\pred"] + [str(index) for index in range(matrix.shape[1])]
        writer.writerow(header)
        for index, row in enumerate(matrix.tolist()):
            writer.writerow([index] + row)


def d_eff_from_embeddings(embeddings: np.ndarray) -> float:
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return 0.0
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False, bias=False)
    if covariance.ndim == 0:
        return 0.0
    trace = float(np.trace(covariance))
    trace_sq = float(np.sum(covariance * covariance))
    if trace_sq <= 0.0:
        return 0.0
    return float((trace * trace) / trace_sq)


def compute_embedding_stats(
    labels: np.ndarray,
    projection: np.ndarray,
    projection_views: np.ndarray,
    num_classes: int,
) -> Dict[str, Any]:
    per_class = {}
    for class_index in range(num_classes):
        mask = labels == class_index
        per_class[str(class_index)] = d_eff_from_embeddings(projection[mask])

    per_view = {}
    view_count = projection_views.shape[1] if projection_views.ndim == 3 else 0
    for view_index in range(view_count):
        view_embeddings = projection_views[:, view_index, :]
        class_values = {}
        for class_index in range(num_classes):
            mask = labels == class_index
            class_values[str(class_index)] = d_eff_from_embeddings(view_embeddings[mask])
        per_view[str(view_index)] = {
            "overall": d_eff_from_embeddings(view_embeddings),
            "per_class": class_values,
        }

    return {
        "d_eff": d_eff_from_embeddings(projection),
        "per_class_d_eff": per_class,
        "per_view_d_eff": per_view,
    }


def compute_distortion_rows(
    split_name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    num_classes: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    values = []
    weighted_sum = 0.0
    total_count = 0
    for class_index in range(num_classes):
        mask = labels == class_index
        count = int(mask.sum())
        qhat = 0.0
        if count > 0:
            qhat = float(np.mean(1.0 - scores[mask, class_index]))
            values.append(qhat)
            weighted_sum += qhat * count
            total_count += count
        rows.append(
            {
                "split": split_name,
                "class": class_index,
                "sample_count": count,
                "Qhat_c": qhat,
            }
        )
    qhat_mean = float(np.mean(values)) if values else 0.0
    qhat_weighted = float(weighted_sum / total_count) if total_count else 0.0
    for row in rows:
        row["Qhat_mean"] = qhat_mean
        row["Qhat_weighted"] = qhat_weighted
    return rows, {"Qhat_mean": qhat_mean, "Qhat_weighted": qhat_weighted}


def compute_occupancy_rows(
    split_name: str,
    labels: np.ndarray,
    assignments: np.ndarray,
    num_classes: int,
    k_value: int,
    low_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    class_entropy_values = []
    for class_index in range(num_classes):
        mask = labels == class_index
        class_count = int(mask.sum())
        counts = np.bincount(assignments[mask], minlength=k_value).astype(np.int64) if class_count else np.zeros(k_value, dtype=np.int64)
        occupancy = counts / max(1, class_count)
        entropy = float(-np.sum(occupancy * np.log(occupancy + 1e-12))) if k_value > 0 else 0.0
        entropy_norm = float(entropy / math.log(k_value)) if k_value > 1 else 0.0
        empty_count = int(np.sum(counts == 0))
        low_count = int(np.sum(occupancy < low_threshold))
        class_entropy_values.append(entropy_norm)
        for proto_index in range(k_value):
            rows.append(
                {
                    "split": split_name,
                    "class": class_index,
                    "prototype_k": proto_index,
                    "assigned_count": int(counts[proto_index]),
                    "class_count": class_count,
                    "occupancy": float(occupancy[proto_index]),
                    "occupancy_entropy": entropy,
                    "occupancy_entropy_norm": entropy_norm,
                    "empty_prototype_count": empty_count,
                    "low_occupancy_threshold": float(low_threshold),
                    "low_occupancy_prototype_count": low_count,
                }
            )
    summary = {
        "occupancy_entropy_mean": float(np.mean(class_entropy_values)) if class_entropy_values else 0.0,
    }
    return rows, summary


def summarize_array(values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {
            "sample_count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
        }
    return {
        "sample_count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def compute_gap_rows(
    split_name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    num_classes: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for true_class in range(num_classes):
        mask = labels == true_class
        for negative_class in range(num_classes):
            if true_class == negative_class:
                continue
            gap = scores[mask, true_class] - scores[mask, negative_class]
            row = {
                "split": split_name,
                "true_class": true_class,
                "negative_class": negative_class,
                "pair": f"{true_class}->{negative_class}",
            }
            row.update(summarize_array(gap))
            for margin in RAW_VIOLATION_MARGINS:
                key = f"violation_rate_m_{fmt_float(margin)}"
                row[key] = float(np.mean(gap < margin)) if gap.size else 0.0
            rows.append(row)
    return rows


def compute_quantile_rows(source_gap_rows: Sequence[Mapping[str, Any]], labels: np.ndarray, scores: np.ndarray) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    num_classes = scores.shape[1]
    for true_class in range(num_classes):
        mask = labels == true_class
        for negative_class in range(num_classes):
            if true_class == negative_class:
                continue
            gap = scores[mask, true_class] - scores[mask, negative_class]
            for q_value in QUANTILE_GRID:
                margin = float(np.quantile(gap, q_value)) if gap.size else 0.0
                rows.append(
                    {
                        "true_class": true_class,
                        "negative_class": negative_class,
                        "pair": f"{true_class}->{negative_class}",
                        "q": float(q_value),
                        "margin": margin,
                        "sample_count": int(gap.size),
                    }
                )
    return rows


def compute_margin_violation_rows(
    split_name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    num_classes: int,
    quantile_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    margin_specs: List[Tuple[str, str, int, int, float]] = []
    for true_class in range(num_classes):
        for negative_class in range(num_classes):
            if true_class == negative_class:
                continue
            for margin in RAW_MARGIN_GRID:
                margin_specs.append(("raw_grid", f"m_{fmt_float(margin)}", true_class, negative_class, float(margin)))
    for row in quantile_rows:
        margin_specs.append(
            (
                "quantile",
                f"q_{fmt_float(float(row['q']))}",
                int(row["true_class"]),
                int(row["negative_class"]),
                float(row["margin"]),
            )
        )

    for margin_kind, margin_id, true_class, negative_class, margin in margin_specs:
        mask = labels == true_class
        gap = scores[mask, true_class] - scores[mask, negative_class]
        shortfall = np.maximum(0.0, margin - gap) if gap.size else np.array([], dtype=np.float64)
        violations = gap < margin if gap.size else np.array([], dtype=bool)
        rows.append(
            {
                "split": split_name,
                "true_class": true_class,
                "negative_class": negative_class,
                "pair": f"{true_class}->{negative_class}",
                "margin_kind": margin_kind,
                "margin_id": margin_id,
                "margin": float(margin),
                "sample_count": int(gap.size),
                "violation_rate": float(np.mean(violations)) if gap.size else 0.0,
                "average_shortfall": float(np.mean(shortfall)) if gap.size else 0.0,
                "conditional_shortfall": float(np.mean(shortfall[violations])) if np.any(violations) else 0.0,
            }
        )
    return rows


def compute_confusion_ranking_rows(
    split_name: str,
    normalized: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for true_class in range(normalized.shape[0]):
        for predicted_class in range(normalized.shape[1]):
            if true_class == predicted_class:
                continue
            rows.append(
                {
                    "split": split_name,
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "pair": f"{true_class}->{predicted_class}",
                    "directional_confusion": float(normalized[true_class, predicted_class]),
                }
            )
    rows.sort(key=lambda row: (-float(row["directional_confusion"]), int(row["true_class"]), int(row["predicted_class"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def compute_score_summary_rows(
    split_name: str,
    labels: np.ndarray,
    logits: np.ndarray,
    scores: np.ndarray,
    num_classes: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for true_class in range(num_classes):
        mask = labels == true_class
        for class_index in range(num_classes):
            logit_values = logits[mask, class_index]
            score_values = scores[mask, class_index]
            rows.append(
                {
                    "split": split_name,
                    "true_class": true_class,
                    "class": class_index,
                    "sample_count": int(mask.sum()),
                    "logit_mean": float(np.mean(logit_values)) if logit_values.size else 0.0,
                    "logit_std": float(np.std(logit_values, ddof=1)) if logit_values.size > 1 else 0.0,
                    "prototype_score_mean": float(np.mean(score_values)) if score_values.size else 0.0,
                    "prototype_score_std": float(np.std(score_values, ddof=1)) if score_values.size > 1 else 0.0,
                }
            )
    return rows


def load_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def postprocess_run(
    *,
    dataset: Any,
    config: Mapping[str, Any],
    run_dir: Path,
    low_occupancy_threshold: float,
) -> None:
    best_model_path = run_dir / "best_model.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {best_model_path}")

    target = str(config["target_env"])
    seed = int(config["seed_list"][0])
    k_value = int(config["model"]["prototype_num_subcenters"])
    batch_size = int(config["training"].get("batch_size", 32))
    num_workers = int(config["training"].get("num_workers", 0))
    device = choose_device(config.get("device", "cpu"))

    split = build_split_indices(
        dataset,
        mode="dg_loeo",
        seed=seed,
        val_ratio=float(config["split"].get("val_ratio", 0.2)),
        test_ratio=float(config["split"].get("test_ratio", 0.2)),
        target_env=target,
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

    model, _optimizer, _scheduler, _scheduler_step_mode = build_model_and_optimizer(dataset, config, device)
    model.load_state_dict(load_state_dict(best_model_path, device))

    outputs_by_split = {
        "source_val": collect_model_outputs(model, val_loader, device),
        "target_test": collect_model_outputs(model, test_loader, device),
    }
    num_classes = int(dataset.num_classes)

    metrics_by_split: Dict[str, Dict[str, Any]] = {}
    distortion_rows: List[Dict[str, Any]] = []
    occupancy_rows: List[Dict[str, Any]] = []
    gap_rows: List[Dict[str, Any]] = []
    ranking_rows: List[Dict[str, Any]] = []
    score_summary_rows: List[Dict[str, Any]] = []
    embedding_stats_by_split: Dict[str, Any] = {}
    distortion_summary: Dict[str, Any] = {}
    occupancy_summary: Dict[str, Any] = {}

    source_quantile_rows: List[Dict[str, Any]] = []
    margin_violation_rows: List[Dict[str, Any]] = []

    for split_name, outputs in outputs_by_split.items():
        metrics = compute_basic_metrics(outputs["labels"], outputs["predictions"], num_classes)
        metrics_by_split[split_name] = {
            key: value
            for key, value in metrics.items()
            if key not in {"confusion_matrix", "normalized_confusion"}
        }

        write_matrix_csv(run_dir / f"{split_name}_confusion_matrix.csv", metrics["confusion_matrix"])
        write_matrix_csv(run_dir / f"{split_name}_normalized_confusion.csv", metrics["normalized_confusion"])
        if split_name == "target_test":
            write_matrix_csv(run_dir / "confusion_matrix.csv", metrics["confusion_matrix"])
            write_matrix_csv(run_dir / "normalized_confusion.csv", metrics["normalized_confusion"])

        split_distortion_rows, split_distortion_summary = compute_distortion_rows(
            split_name,
            outputs["labels"],
            outputs["scores"],
            num_classes,
        )
        split_occupancy_rows, split_occupancy_summary = compute_occupancy_rows(
            split_name,
            outputs["labels"],
            outputs["true_proto_assignment"],
            num_classes,
            k_value,
            low_occupancy_threshold,
        )
        distortion_rows.extend(split_distortion_rows)
        occupancy_rows.extend(split_occupancy_rows)
        gap_rows.extend(compute_gap_rows(split_name, outputs["labels"], outputs["scores"], num_classes))
        ranking_rows.extend(compute_confusion_ranking_rows(split_name, metrics["normalized_confusion"]))
        score_summary_rows.extend(
            compute_score_summary_rows(split_name, outputs["labels"], outputs["logits"], outputs["scores"], num_classes)
        )
        embedding_stats_by_split[split_name] = compute_embedding_stats(
            outputs["labels"],
            outputs["projection"],
            outputs["projection_views"],
            num_classes,
        )
        distortion_summary[split_name] = split_distortion_summary
        occupancy_summary[split_name] = split_occupancy_summary

        if split_name == "source_val":
            source_quantile_rows = compute_quantile_rows(gap_rows, outputs["labels"], outputs["scores"])

    for split_name, outputs in outputs_by_split.items():
        margin_violation_rows.extend(
            compute_margin_violation_rows(
                split_name,
                outputs["labels"],
                outputs["scores"],
                num_classes,
                source_quantile_rows,
            )
        )

    write_csv(
        run_dir / "distortion_by_class.csv",
        ["split", "class", "sample_count", "Qhat_c", "Qhat_mean", "Qhat_weighted"],
        distortion_rows,
    )
    write_csv(
        run_dir / "prototype_occupancy.csv",
        [
            "split",
            "class",
            "prototype_k",
            "assigned_count",
            "class_count",
            "occupancy",
            "occupancy_entropy",
            "occupancy_entropy_norm",
            "empty_prototype_count",
            "low_occupancy_threshold",
            "low_occupancy_prototype_count",
        ],
        occupancy_rows,
    )
    write_csv(
        run_dir / "gap_stats.csv",
        [
            "split",
            "true_class",
            "negative_class",
            "pair",
            "sample_count",
            "mean",
            "std",
            "min",
            "max",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "violation_rate_m_0.05",
            "violation_rate_m_0.10",
            "violation_rate_m_0.15",
            "violation_rate_m_0.20",
        ],
        gap_rows,
    )
    write_csv(
        run_dir / "quantile_margin_table.csv",
        ["true_class", "negative_class", "pair", "q", "margin", "sample_count"],
        source_quantile_rows,
    )
    write_csv(
        run_dir / "margin_violation_stats.csv",
        [
            "split",
            "true_class",
            "negative_class",
            "pair",
            "margin_kind",
            "margin_id",
            "margin",
            "sample_count",
            "violation_rate",
            "average_shortfall",
            "conditional_shortfall",
        ],
        margin_violation_rows,
    )
    write_csv(
        run_dir / "directed_pair_confusion_ranking.csv",
        ["split", "rank", "true_class", "predicted_class", "pair", "directional_confusion"],
        ranking_rows,
    )
    write_csv(
        run_dir / "logits_or_scores_summary.csv",
        [
            "split",
            "true_class",
            "class",
            "sample_count",
            "logit_mean",
            "logit_std",
            "prototype_score_mean",
            "prototype_score_std",
        ],
        score_summary_rows,
    )

    embedding_payload = {
        "source_val": embedding_stats_by_split["source_val"],
        "target_test": embedding_stats_by_split["target_test"],
    }
    write_json(run_dir / "embedding_stats.json", embedding_payload)

    metrics_path = run_dir / "metrics.json"
    metrics_payload = read_json(metrics_path) if metrics_path.exists() else {}
    target_metrics = metrics_by_split["target_test"]
    source_metrics = metrics_by_split["source_val"]
    metrics_payload.update(
        {
            "acc": target_metrics["accuracy"],
            "macro_f1": target_metrics["macro_f1"],
            "per_class_f1": target_metrics["per_class_f1"],
            "source_val_metrics": source_metrics,
            "target_test_metrics": target_metrics,
            "distortion_summary": distortion_summary,
            "occupancy_summary": occupancy_summary,
            "embedding_stats": embedding_payload,
            "prototype_parameter_count": int(6 * k_value * 128),
            "prototype_parameter_multiplier_vs_k1": float(k_value),
            "supcon_denominator_prototype_count": int(6 * k_value),
        }
    )
    write_json(metrics_path, metrics_payload)
    write_json(run_dir / "config.json", config)
    write_json(
        run_dir / "statistics_complete.json",
        {
            "complete": True,
            "target": target,
            "seed": seed,
            "K": k_value,
            "margin_config": config["k_margin_search"]["margin_config"],
            "files": [
                "metrics.json",
                "confusion_matrix.csv",
                "normalized_confusion.csv",
                "distortion_by_class.csv",
                "prototype_occupancy.csv",
                "gap_stats.csv",
                "margin_violation_stats.csv",
                "embedding_stats.json",
                "config.json",
                "logits_or_scores_summary.csv",
            ],
        },
    )
    print(f"[k-margin] statistics saved {run_dir}", flush=True)


def stage_a_units(args: argparse.Namespace) -> Iterable[Tuple[str, int, int, str, Mapping[str, Any], Optional[PairMargins], int, str]]:
    epochs = int(args.epochs or 40)
    for target in args.targets:
        for k_value in args.k_values:
            for seed in args.seeds:
                for config_name, values, pair_margins in stage_a_configs():
                    yield target, int(k_value), int(seed), config_name, values, pair_margins, epochs, "no_margin"


def raw_margin_units(args: argparse.Namespace) -> Iterable[Tuple[str, int, int, str, Mapping[str, Any], PairMargins, int, str]]:
    epochs = int(args.epochs or 40)
    for target in args.targets:
        for k_value in args.k_values:
            for seed in args.seeds:
                for config_name, values, pair_margins in itertools.chain(raw_shared_configs(), raw_single_configs()):
                    yield target, int(k_value), int(seed), config_name, values, pair_margins, epochs, (
                        "raw_shared" if config_name.startswith("raw_margin_shared") else "raw_single"
                    )


def quantile_margin_units(output_root: Path, args: argparse.Namespace) -> Iterable[Tuple[str, int, int, str, Mapping[str, Any], PairMargins, int, str]]:
    epochs = int(args.epochs or 40)
    for target in args.targets:
        for k_value in args.k_values:
            for seed in args.seeds:
                configs = itertools.chain(
                    quantile_shared_configs(output_root, target, int(k_value), int(seed)),
                    quantile_single_configs(output_root, target, int(k_value), int(seed)),
                )
                for config_name, values, pair_margins in configs:
                    yield target, int(k_value), int(seed), config_name, values, pair_margins, epochs, (
                        "quantile_shared" if config_name.startswith("quantile_shared") else "quantile_single"
                    )


def shortlist_margin_units(
    output_root: Path,
    args: argparse.Namespace,
) -> List[Tuple[str, int, int, str, Mapping[str, Any], PairMargins, int, str]]:
    epochs = int(args.epochs or 40)
    units: List[Tuple[str, int, int, str, Mapping[str, Any], PairMargins, int, str]] = []
    plan_rows: List[Dict[str, Any]] = []

    for target in args.targets:
        for k_value in args.k_values:
            for seed in args.seeds:
                run_specs = build_shortlist_specs(output_root, str(target), int(k_value), int(seed))
                for config_name, values, pair_margins, margin_kind, selection_note in run_specs:
                    run_dir = run_dir_for(output_root, str(target), int(k_value), config_name, int(seed))
                    already_complete = (run_dir / "statistics_complete.json").exists()
                    plan_rows.append(
                        {
                            "target": str(target),
                            "K": int(k_value),
                            "seed": int(seed),
                            "margin_config": config_name,
                            "margin_kind": margin_kind,
                            "epochs": epochs,
                            "selection_source": "source_val_no_margin",
                            "selection_note": selection_note,
                            "pair_margins": {pair_tag(pair): margin for pair, margin in pair_margins.items()},
                            "already_complete": already_complete,
                            "run_dir": str(run_dir),
                        }
                    )
                    units.append(
                        (
                            str(target),
                            int(k_value),
                            int(seed),
                            config_name,
                            values,
                            pair_margins,
                            epochs,
                            margin_kind,
                        )
                    )

    write_csv(
        output_root / "aggregate" / "shortlist_plan.csv",
        [
            "target",
            "K",
            "seed",
            "margin_config",
            "margin_kind",
            "epochs",
            "selection_source",
            "selection_note",
            "pair_margins",
            "already_complete",
            "run_dir",
        ],
        plan_rows,
    )
    return units


def build_shortlist_specs(
    output_root: Path,
    target: str,
    k_value: int,
    seed: int,
) -> List[Tuple[str, Mapping[str, Any], PairMargins, str, str]]:
    no_margin_dir = run_dir_for(output_root, target, k_value, "no_margin", seed)
    if not (no_margin_dir / "statistics_complete.json").exists():
        raise FileNotFoundError(
            f"Shortlist requires completed no_margin statistics: {no_margin_dir}"
        )

    margin_rows = read_csv_rows(no_margin_dir / "margin_violation_stats.csv")
    quantile_table = load_quantile_table(output_root, target, k_value, seed)

    shared_raw_m = select_shared_raw_margin(margin_rows)
    pair_raw_margins = {
        pair: select_pair_raw_margin(margin_rows, pair)
        for pair in FOCUS_PAIRS
    }

    pair_q_values = {
        pair: select_pair_quantile_q(margin_rows, pair, pair_raw_margins[pair])
        for pair in FOCUS_PAIRS
    }
    shared_q = nearest_quantile(float(np.mean(list(pair_q_values.values()))))

    specs: List[Tuple[str, Mapping[str, Any], PairMargins, str, str]] = []

    values = {"m": float(shared_raw_m)}
    specs.append(
        (
            margin_config_name("raw_shared", values),
            values,
            {pair: float(shared_raw_m) for pair in FOCUS_PAIRS},
            "raw_shared",
            "top shared raw margin by source-val focus-pair hinge shortfall",
        )
    )

    for pair in FOCUS_PAIRS:
        margin_value = float(pair_raw_margins[pair])
        values = {"pair": pair, "m": margin_value}
        specs.append(
            (
                margin_config_name("raw_single", values),
                values,
                {pair: margin_value},
                "raw_single",
                f"top raw margin for {pair_tag(pair)} by source-val hinge shortfall",
            )
        )

    shared_quantile_margins = {
        pair: float(quantile_table[pair][shared_q])
        for pair in FOCUS_PAIRS
    }
    values = {"q": float(shared_q), "margins": shared_quantile_margins}
    specs.append(
        (
            margin_config_name("quantile_shared", values),
            values,
            shared_quantile_margins,
            "quantile_shared",
            "shared q from source-val raw-violation rates",
        )
    )

    for pair in FOCUS_PAIRS:
        q_value = float(pair_q_values[pair])
        margin_value = float(quantile_table[pair][q_value])
        values = {"pair": pair, "q": q_value, "margins": {pair: margin_value}}
        specs.append(
            (
                margin_config_name("quantile_single", values),
                values,
                {pair: margin_value},
                "quantile_single",
                f"q for {pair_tag(pair)} from source-val raw-violation rate",
            )
        )

    deduped: List[Tuple[str, Mapping[str, Any], PairMargins, str, str]] = []
    seen = set()
    for spec in specs:
        if spec[0] in seen:
            continue
        seen.add(spec[0])
        deduped.append(spec)
    return deduped


def select_shared_raw_margin(rows: Sequence[Mapping[str, Any]]) -> float:
    ranked = []
    for margin in RAW_MARGIN_GRID:
        if margin <= 0.0:
            continue
        selected = [
            row for row in rows
            if row.get("split") == "source_val"
            and row.get("margin_kind") == "raw_grid"
            and abs(float(row.get("margin", 0.0)) - float(margin)) <= 1e-8
            and (int(row["true_class"]), int(row["negative_class"])) in FOCUS_PAIRS
        ]
        if not selected:
            continue
        ranked.append(
            (
                float(np.mean([float(row["average_shortfall"]) for row in selected])),
                float(np.mean([float(row["violation_rate"]) for row in selected])),
                float(margin),
            )
        )
    if not ranked:
        return 0.10
    ranked.sort(key=lambda item: (-item[0], -item[1], -item[2]))
    return float(ranked[0][2])


def select_pair_raw_margin(rows: Sequence[Mapping[str, Any]], pair: Pair) -> float:
    selected = [
        row for row in rows
        if row.get("split") == "source_val"
        and row.get("margin_kind") == "raw_grid"
        and (int(row["true_class"]), int(row["negative_class"])) == pair
        and float(row.get("margin", 0.0)) > 0.0
    ]
    if not selected:
        return 0.10
    selected.sort(
        key=lambda row: (
            -float(row["average_shortfall"]),
            -float(row["violation_rate"]),
            -float(row["margin"]),
        )
    )
    return float(selected[0]["margin"])


def select_pair_quantile_q(
    rows: Sequence[Mapping[str, Any]],
    pair: Pair,
    raw_margin: float,
) -> float:
    selected = [
        row for row in rows
        if row.get("split") == "source_val"
        and row.get("margin_kind") == "raw_grid"
        and (int(row["true_class"]), int(row["negative_class"])) == pair
        and abs(float(row.get("margin", 0.0)) - float(raw_margin)) <= 1e-8
    ]
    violation = float(selected[0]["violation_rate"]) if selected else 0.10
    return nearest_quantile(max(min(violation, max(QUANTILE_GRID)), min(QUANTILE_GRID)))


def nearest_quantile(value: float) -> float:
    return float(min(QUANTILE_GRID, key=lambda q_value: (abs(float(q_value) - float(value)), float(q_value))))


def top_joint_units(
    output_root: Path,
    args: argparse.Namespace,
    *,
    family: str,
) -> Iterable[Tuple[str, int, int, str, Mapping[str, Any], PairMargins, int, str]]:
    epochs = int(args.epochs or 40)
    for target in args.targets:
        for k_value in args.k_values:
            top_values = select_top_single_factor_values(
                output_root=output_root,
                target=target,
                k_value=int(k_value),
                seeds=[int(seed) for seed in args.seeds],
                family=family,
                top_n=int(args.joint_top_values_per_pair),
            )
            if not top_values:
                continue
            for value_combo in itertools.product(*(top_values[pair] for pair in FOCUS_PAIRS)):
                if family == "raw":
                    raw_margins = {pair: float(value) for pair, value in zip(FOCUS_PAIRS, value_combo)}
                    values = {"margins": raw_margins}
                    config_name = margin_config_name("raw_joint", values)
                    for seed in args.seeds:
                        yield target, int(k_value), int(seed), config_name, values, raw_margins, epochs, "raw_joint"
                elif family == "quantile":
                    q_values = {pair: float(value) for pair, value in zip(FOCUS_PAIRS, value_combo)}
                    for seed in args.seeds:
                        q_table = load_quantile_table(output_root, target, int(k_value), int(seed))
                        pair_margins = {pair: float(q_table[pair][q_values[pair]]) for pair in FOCUS_PAIRS}
                        values = {"q_values": q_values, "margins": pair_margins}
                        config_name = margin_config_name("quantile_joint", values)
                        yield target, int(k_value), int(seed), config_name, values, pair_margins, epochs, "quantile_joint"
                else:
                    raise ValueError(f"Unknown family: {family}")


def select_top_single_factor_values(
    *,
    output_root: Path,
    target: str,
    k_value: int,
    seeds: Sequence[int],
    family: str,
    top_n: int,
) -> Dict[Pair, List[float]]:
    values_by_pair: Dict[Pair, Dict[float, List[float]]] = {pair: defaultdict(list) for pair in FOCUS_PAIRS}
    for seed in seeds:
        for pair in FOCUS_PAIRS:
            value_grid = RAW_MARGIN_GRID if family == "raw" else QUANTILE_GRID
            for value in value_grid:
                if family == "raw":
                    config_name = margin_config_name("raw_single", {"pair": pair, "m": float(value)})
                else:
                    config_name = margin_config_name("quantile_single", {"pair": pair, "q": float(value)})
                metrics_path = run_dir_for(output_root, target, k_value, config_name, int(seed)) / "metrics.json"
                if not metrics_path.exists():
                    continue
                metrics = read_json(metrics_path)
                source_val = metrics.get("source_val_metrics", {})
                score = float(source_val.get("macro_f1", source_val.get("accuracy", 0.0)))
                values_by_pair[pair][round(float(value), 2)].append(score)

    selected: Dict[Pair, List[float]] = {}
    for pair, score_map in values_by_pair.items():
        if not score_map:
            return {}
        ranked = sorted(
            ((float(np.mean(scores)), value) for value, scores in score_map.items()),
            key=lambda item: (-item[0], item[1]),
        )
        selected[pair] = [float(value) for _score, value in ranked[: max(1, top_n)]]
    return selected


def confirm_units(
    output_root: Path,
    args: argparse.Namespace,
    base_config: Mapping[str, Any],
    *,
    no_margin_only: bool,
) -> Iterable[Tuple[str, int, int, str, Mapping[str, Any], Optional[PairMargins], int, str]]:
    epochs = int(args.epochs or 200)
    selected = select_top_configs_for_confirmation(
        output_root=output_root,
        targets=args.targets,
        top_n=int(args.confirm_top_n),
        metric_name=str(args.confirm_selection_metric),
        no_margin_only=no_margin_only,
    )
    for target, selections in selected.items():
        for k_value, source_config_name in selections:
            for seed in args.seeds:
                source_dir = run_dir_for(output_root, target, k_value, source_config_name, int(seed))
                source_config_path = source_dir / "config.json"
                if source_config_path.exists():
                    source_config = read_json(source_config_path)
                    margin_kind = str(source_config.get("k_margin_search", {}).get("margin_kind", "no_margin"))
                    margin_values = source_config.get("k_margin_search", {}).get("margin_values", {})
                    pair_margins = pair_margins_from_config(source_config)
                else:
                    margin_kind = "no_margin"
                    margin_values = {}
                    pair_margins = None
                confirm_config_name = f"{source_config_name}_200e"
                yield target, int(k_value), int(seed), confirm_config_name, margin_values, pair_margins, epochs, margin_kind


def pair_margins_from_config(config: Mapping[str, Any]) -> Optional[PairMargins]:
    loss_config = config.get("losses", {})
    if float(loss_config.get("lambda_pair_margin", 0.0)) <= 0.0:
        return None
    specs = loss_config.get("prototype_pair_margins", []) or []
    return {
        (int(spec["true"]), int(spec["negative"])): float(spec["margin"])
        for spec in specs
    }


def select_top_configs_for_confirmation(
    *,
    output_root: Path,
    targets: Sequence[str],
    top_n: int,
    metric_name: str,
    no_margin_only: bool,
) -> Dict[str, List[Tuple[int, str]]]:
    selected: Dict[str, List[Tuple[int, str]]] = {}
    for target in targets:
        target_dir = output_root / f"target_{target}"
        candidate_scores: Dict[Tuple[int, str], List[float]] = defaultdict(list)
        if not target_dir.exists():
            selected[target] = []
            continue
        for k_dir in sorted(target_dir.glob("K_*")):
            try:
                k_value = int(k_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            for config_dir in sorted(k_dir.iterdir()):
                if not config_dir.is_dir():
                    continue
                config_name = config_dir.name
                if config_name.endswith("_200e"):
                    continue
                if no_margin_only and config_name != "no_margin":
                    continue
                if (not no_margin_only) and config_name == "no_margin":
                    pass
                for seed_dir in sorted(config_dir.glob("seed_*")):
                    metrics_path = seed_dir / "metrics.json"
                    if not metrics_path.exists():
                        continue
                    metrics = read_json(metrics_path)
                    source_val = metrics.get("source_val_metrics", {})
                    if metric_name == "source_val_accuracy":
                        score = float(source_val.get("accuracy", 0.0))
                    else:
                        score = float(source_val.get("macro_f1", 0.0))
                    candidate_scores[(k_value, config_name)].append(score)
        ranked = sorted(
            (
                (float(np.mean(scores)), len(scores), k_value, config_name)
                for (k_value, config_name), scores in candidate_scores.items()
                if scores
            ),
            key=lambda item: (-item[0], -item[1], item[2], item[3]),
        )
        selected[target] = [(k_value, config_name) for _score, _count, k_value, config_name in ranked[:top_n]]
    write_json(
        output_root / "aggregate" / ("selected_stage_a_confirm_200.json" if no_margin_only else "selected_margin_confirm_200.json"),
        {
            "selection_metric": metric_name,
            "top_n": int(top_n),
            "source_val_only": True,
            "selected": {
                target: [{"K": k_value, "margin_config": config_name} for k_value, config_name in rows]
                for target, rows in selected.items()
            },
        },
    )
    return selected


def iter_run_dirs(output_root: Path) -> Iterable[Path]:
    for target_dir in sorted(output_root.glob("target_*")):
        if not target_dir.is_dir():
            continue
        for k_dir in sorted(target_dir.glob("K_*")):
            if not k_dir.is_dir():
                continue
            for config_dir in sorted(k_dir.iterdir()):
                if not config_dir.is_dir():
                    continue
                for seed_dir in sorted(config_dir.glob("seed_*")):
                    if (seed_dir / "metrics.json").exists():
                        yield seed_dir


def run_context_from_dir(run_dir: Path) -> Dict[str, Any]:
    config_dir = run_dir.parent
    k_dir = config_dir.parent
    target_dir = k_dir.parent
    return {
        "target": target_dir.name.removeprefix("target_"),
        "K": int(k_dir.name.split("_", 1)[1]),
        "margin_config": config_dir.name,
        "seed": int(run_dir.name.split("_", 1)[1]),
    }


def aggregate_results(output_root: Path) -> None:
    aggregate_dir = output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    summary_k_rows: List[Dict[str, Any]] = []
    pair_summary_rows: List[Dict[str, Any]] = []
    margin_raw_rows: List[Dict[str, Any]] = []
    margin_quantile_rows: List[Dict[str, Any]] = []
    kmax_rows: List[Dict[str, Any]] = []
    raw_candidate_rows: List[Dict[str, Any]] = []
    quant_candidate_rows: List[Dict[str, Any]] = []
    complexity_rows: List[Dict[str, Any]] = []
    inventory_rows: List[Dict[str, Any]] = []

    for run_dir in iter_run_dirs(output_root):
        context = run_context_from_dir(run_dir)
        metrics = read_json(run_dir / "metrics.json")
        config = read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
        search_config = config.get("k_margin_search", {})
        epochs = int(search_config.get("epochs", config.get("training", {}).get("epochs", 0)))
        margin_kind = str(search_config.get("margin_kind", "unknown"))
        target_metrics = metrics.get("target_test_metrics", {})
        source_metrics = metrics.get("source_val_metrics", {})
        distortion = metrics.get("distortion_summary", {})
        occupancy = metrics.get("occupancy_summary", {})
        embedding = metrics.get("embedding_stats", {})
        source_d_eff = float(embedding.get("source_val", {}).get("d_eff", 0.0))
        target_d_eff = float(embedding.get("target_test", {}).get("d_eff", 0.0))

        inventory_rows.append(
            {
                **context,
                "epochs": epochs,
                "margin_kind": margin_kind,
                "statistics_complete": (run_dir / "statistics_complete.json").exists(),
                "run_dir": str(run_dir),
            }
        )

        if context["margin_config"] == "no_margin" or context["margin_config"] == "no_margin_200e":
            summary_k_rows.append(
                {
                    **context,
                    "epochs": epochs,
                    "acc": target_metrics.get("accuracy", metrics.get("acc", "")),
                    "macro_f1": target_metrics.get("macro_f1", metrics.get("macro_f1", "")),
                    "source_val_acc": source_metrics.get("accuracy", ""),
                    "source_val_macro_f1": source_metrics.get("macro_f1", ""),
                    "Qhat_mean": distortion.get("target_test", {}).get("Qhat_mean", ""),
                    "source_val_Qhat_mean": distortion.get("source_val", {}).get("Qhat_mean", ""),
                    "target_test_Qhat_mean": distortion.get("target_test", {}).get("Qhat_mean", ""),
                    "d_eff": source_d_eff,
                    "target_test_d_eff": target_d_eff,
                    "occupancy_entropy_mean": occupancy.get("source_val", {}).get("occupancy_entropy_mean", ""),
                    "prototype_parameter_count": metrics.get("prototype_parameter_count", ""),
                    "prototype_parameter_multiplier_vs_k1": metrics.get("prototype_parameter_multiplier_vs_k1", ""),
                    "supcon_denominator_prototype_count": metrics.get("supcon_denominator_prototype_count", ""),
                }
            )

        append_pair_summary_rows(pair_summary_rows, run_dir, context)
        append_margin_summary_rows(margin_raw_rows, margin_quantile_rows, run_dir, context, metrics, margin_kind, search_config)
        append_kmax_rows(kmax_rows, context, source_d_eff)
        append_candidate_rows(raw_candidate_rows, quant_candidate_rows, run_dir, context)
        complexity_rows.append(
            {
                **context,
                "epochs": epochs,
                "prototype_parameter_count": int(6 * int(context["K"]) * 128),
                "prototype_parameter_multiplier_vs_k1": float(context["K"]),
                "supcon_denominator_prototype_count": int(6 * int(context["K"])),
            }
        )

    write_csv(
        aggregate_dir / "summary_k_sweep.csv",
        [
            "target",
            "K",
            "seed",
            "margin_config",
            "epochs",
            "acc",
            "macro_f1",
            "source_val_acc",
            "source_val_macro_f1",
            "Qhat_mean",
            "source_val_Qhat_mean",
            "target_test_Qhat_mean",
            "d_eff",
            "target_test_d_eff",
            "occupancy_entropy_mean",
            "prototype_parameter_count",
            "prototype_parameter_multiplier_vs_k1",
            "supcon_denominator_prototype_count",
        ],
        summary_k_rows,
    )
    write_csv(
        aggregate_dir / "summary_pair_ranking.csv",
        [
            "target",
            "K",
            "seed",
            "margin_config",
            "pair",
            "true_class",
            "negative_class",
            "source_val_confusion",
            "target_confusion",
            "gap_mean",
            "gap_std",
            "p10",
            "p50",
            "p90",
        ],
        pair_summary_rows,
    )
    write_csv(
        aggregate_dir / "summary_margin_raw_grid.csv",
        [
            "target",
            "K",
            "margin_config",
            "margin_kind",
            "seed",
            "epochs",
            "acc",
            "macro_f1",
            "source_val_acc",
            "source_val_macro_f1",
            "F1_0",
            "F1_1",
            "F1_4",
            "C_1to4",
            "C_4to1",
            "C_0to4",
            "source_val_C_1to4",
            "source_val_C_4to1",
            "source_val_C_0to4",
            "focus_violation_rate_source_val",
            "focus_average_shortfall_source_val",
            "margin_values",
        ],
        margin_raw_rows,
    )
    write_csv(
        aggregate_dir / "summary_margin_quantile_grid.csv",
        [
            "target",
            "K",
            "margin_config",
            "margin_kind",
            "seed",
            "epochs",
            "acc",
            "macro_f1",
            "source_val_acc",
            "source_val_macro_f1",
            "F1_0",
            "F1_1",
            "F1_4",
            "C_1to4",
            "C_4to1",
            "C_0to4",
            "source_val_C_1to4",
            "source_val_C_4to1",
            "source_val_C_0to4",
            "focus_violation_rate_source_val",
            "focus_average_shortfall_source_val",
            "q_config",
            "raw_margins",
            "margin_values",
        ],
        margin_quantile_rows,
    )
    write_csv(
        aggregate_dir / "kmax_sensitivity.csv",
        ["target", "K", "seed", "margin_config", "d_eff", "formula", "epsilon_proto", "r", "K_max"],
        kmax_rows,
    )
    write_csv(
        aggregate_dir / "raw_margin_candidate_table.csv",
        ["target", "K", "seed", "pair", "true_class", "negative_class", "margin"],
        raw_candidate_rows,
    )
    write_csv(
        aggregate_dir / "quantile_margin_candidate_table.csv",
        ["target", "K", "seed", "pair", "true_class", "negative_class", "q", "margin", "sample_count"],
        quant_candidate_rows,
    )
    write_csv(
        aggregate_dir / "complexity_by_k.csv",
        [
            "target",
            "K",
            "seed",
            "margin_config",
            "epochs",
            "prototype_parameter_count",
            "prototype_parameter_multiplier_vs_k1",
            "supcon_denominator_prototype_count",
        ],
        complexity_rows,
    )
    write_csv(
        aggregate_dir / "run_inventory.csv",
        ["target", "K", "seed", "margin_config", "epochs", "margin_kind", "statistics_complete", "run_dir"],
        inventory_rows,
    )
    print(f"[k-margin] aggregate written {aggregate_dir}", flush=True)


def append_pair_summary_rows(rows: List[Dict[str, Any]], run_dir: Path, context: Mapping[str, Any]) -> None:
    ranking_rows = read_csv_rows(run_dir / "directed_pair_confusion_ranking.csv")
    gap_rows = read_csv_rows(run_dir / "gap_stats.csv")
    source_conf = {}
    target_conf = {}
    for row in ranking_rows:
        pair = row["pair"]
        if row["split"] == "source_val":
            source_conf[pair] = float(row["directional_confusion"])
        elif row["split"] == "target_test":
            target_conf[pair] = float(row["directional_confusion"])
    source_gaps = {row["pair"]: row for row in gap_rows if row.get("split") == "source_val"}
    for pair, gap in sorted(source_gaps.items()):
        true_class, negative_class = pair.split("->", 1)
        rows.append(
            {
                **context,
                "pair": pair,
                "true_class": int(true_class),
                "negative_class": int(negative_class),
                "source_val_confusion": source_conf.get(pair, 0.0),
                "target_confusion": target_conf.get(pair, 0.0),
                "gap_mean": gap.get("mean", ""),
                "gap_std": gap.get("std", ""),
                "p10": gap.get("p10", ""),
                "p50": gap.get("p50", ""),
                "p90": gap.get("p90", ""),
            }
        )


def append_margin_summary_rows(
    raw_rows: List[Dict[str, Any]],
    quantile_rows: List[Dict[str, Any]],
    run_dir: Path,
    context: Mapping[str, Any],
    metrics: Mapping[str, Any],
    margin_kind: str,
    search_config: Mapping[str, Any],
) -> None:
    target_metrics = metrics.get("target_test_metrics", {})
    source_metrics = metrics.get("source_val_metrics", {})
    target_f1 = target_metrics.get("per_class_f1", {})
    target_norm = matrix_from_csv(run_dir / "target_test_normalized_confusion.csv")
    source_norm = matrix_from_csv(run_dir / "source_val_normalized_confusion.csv")
    violation = focus_violation_summary(run_dir / "margin_violation_stats.csv", search_config)
    epochs = int(search_config.get("epochs", 0))
    row = {
        **context,
        "margin_kind": margin_kind,
        "epochs": epochs,
        "acc": target_metrics.get("accuracy", metrics.get("acc", "")),
        "macro_f1": target_metrics.get("macro_f1", metrics.get("macro_f1", "")),
        "source_val_acc": source_metrics.get("accuracy", ""),
        "source_val_macro_f1": source_metrics.get("macro_f1", ""),
        "F1_0": target_f1.get("0", ""),
        "F1_1": target_f1.get("1", ""),
        "F1_4": target_f1.get("4", ""),
        "C_1to4": matrix_value(target_norm, 1, 4),
        "C_4to1": matrix_value(target_norm, 4, 1),
        "C_0to4": matrix_value(target_norm, 0, 4),
        "source_val_C_1to4": matrix_value(source_norm, 1, 4),
        "source_val_C_4to1": matrix_value(source_norm, 4, 1),
        "source_val_C_0to4": matrix_value(source_norm, 0, 4),
        "focus_violation_rate_source_val": violation["violation_rate"],
        "focus_average_shortfall_source_val": violation["average_shortfall"],
        "margin_values": search_config.get("margin_values", {}),
    }
    if margin_kind.startswith("quantile"):
        row["q_config"] = extract_q_config(search_config.get("margin_values", {}))
        row["raw_margins"] = extract_raw_margins(search_config.get("margin_values", {}))
        quantile_rows.append(row)
    elif margin_kind.startswith("raw") or margin_kind == "no_margin":
        raw_rows.append(row)


def matrix_from_csv(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((6, 6), dtype=np.float64)
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        for row in reader:
            rows.append([float(cell) for cell in row[1:]])
    return np.asarray(rows, dtype=np.float64)


def matrix_value(matrix: np.ndarray, row: int, column: int) -> float:
    if matrix.shape[0] <= row or matrix.shape[1] <= column:
        return 0.0
    return float(matrix[row, column])


def focus_violation_summary(path: Path, search_config: Mapping[str, Any]) -> Dict[str, float]:
    rows = read_csv_rows(path)
    selected = []
    applied_margins = applied_focus_margins(search_config)
    for row in rows:
        pair = (int(row["true_class"]), int(row["negative_class"]))
        if row.get("split") != "source_val" or pair not in FOCUS_PAIRS:
            continue
        desired_margin = applied_margins.get(pair, 0.0)
        if abs(float(row["margin"]) - desired_margin) <= 1e-8:
            selected.append(row)
    if not selected:
        return {"violation_rate": 0.0, "average_shortfall": 0.0}
    return {
        "violation_rate": float(np.mean([float(row["violation_rate"]) for row in selected])),
        "average_shortfall": float(np.mean([float(row["average_shortfall"]) for row in selected])),
    }


def applied_focus_margins(search_config: Mapping[str, Any]) -> Dict[Pair, float]:
    margin_values = search_config.get("margin_values", {}) or {}
    margin_kind = str(search_config.get("margin_kind", "no_margin"))
    if margin_kind == "no_margin":
        return {pair: 0.0 for pair in FOCUS_PAIRS}

    applied = {pair: 0.0 for pair in FOCUS_PAIRS}
    if "m" in margin_values and "pair" not in margin_values:
        return {pair: float(margin_values["m"]) for pair in FOCUS_PAIRS}
    if "pair" in margin_values and "m" in margin_values:
        pair_value = margin_values["pair"]
        pair = (int(pair_value[0]), int(pair_value[1]))
        applied[pair] = float(margin_values["m"])
        return applied
    if "margins" in margin_values and isinstance(margin_values["margins"], Mapping):
        for key, value in margin_values["margins"].items():
            pair = parse_pair_tag(str(key))
            applied[pair] = float(value)
    return applied


def extract_q_config(margin_values: Mapping[str, Any]) -> Dict[str, Any]:
    if "q" in margin_values:
        return {"shared_q": margin_values["q"]}
    if "q_values" in margin_values:
        return margin_values["q_values"]
    return {}


def extract_raw_margins(margin_values: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(margin_values.get("margins", {})) if isinstance(margin_values.get("margins", {}), Mapping) else {}


def append_kmax_rows(rows: List[Dict[str, Any]], context: Mapping[str, Any], d_eff: float) -> None:
    for epsilon in EPSILON_PROTO_VALUES:
        k_max = int(math.floor((epsilon * epsilon * N_MIN) / d_eff)) if d_eff > 0.0 else 0
        rows.append(
            {
                **context,
                "d_eff": d_eff,
                "formula": "epsilon",
                "epsilon_proto": float(epsilon),
                "r": "",
                "K_max": k_max,
            }
        )
    for r_value in R_VALUES:
        k_max = int(math.floor(N_MIN / (r_value * d_eff))) if d_eff > 0.0 else 0
        rows.append(
            {
                **context,
                "d_eff": d_eff,
                "formula": "r",
                "epsilon_proto": "",
                "r": int(r_value),
                "K_max": k_max,
            }
        )


def append_candidate_rows(
    raw_rows: List[Dict[str, Any]],
    quantile_rows: List[Dict[str, Any]],
    run_dir: Path,
    context: Mapping[str, Any],
) -> None:
    if context["margin_config"] != "no_margin":
        return
    for pair in FOCUS_PAIRS:
        for margin in RAW_MARGIN_GRID:
            raw_rows.append(
                {
                    **context,
                    "pair": f"{pair[0]}->{pair[1]}",
                    "true_class": pair[0],
                    "negative_class": pair[1],
                    "margin": float(margin),
                }
            )
    for row in read_csv_rows(run_dir / "quantile_margin_table.csv"):
        pair = (int(row["true_class"]), int(row["negative_class"]))
        if pair not in FOCUS_PAIRS:
            continue
        quantile_rows.append(
            {
                **context,
                "pair": row["pair"],
                "true_class": int(row["true_class"]),
                "negative_class": int(row["negative_class"]),
                "q": float(row["q"]),
                "margin": float(row["margin"]),
                "sample_count": int(row["sample_count"]),
            }
        )


def postprocess_existing_runs(
    *,
    args: argparse.Namespace,
    dataset: Any,
    output_root: Path,
) -> None:
    processed = 0
    for run_dir in iter_run_dirs(output_root):
        if args.max_runs is not None and processed >= args.max_runs:
            break
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        marker_path = run_dir / "statistics_complete.json"
        if marker_path.exists() and not args.force:
            continue
        postprocess_run(
            dataset=dataset,
            config=read_json(config_path),
            run_dir=run_dir,
            low_occupancy_threshold=float(args.low_occupancy_threshold),
        )
        processed += 1


def build_margin_tables(output_root: Path) -> None:
    aggregate_results(output_root)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    base_config = load_base_config(Path(args.base_config).resolve())
    if args.device is not None:
        base_config["device"] = args.device
    dataset = load_dataset(base_config["data"], REPO_ROOT)

    if args.stage == "stage-a":
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=stage_a_units(args),
        )
        aggregate_results(output_root)
    elif args.stage == "stage-a-confirm-200":
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=confirm_units(output_root, args, base_config, no_margin_only=True),
        )
        aggregate_results(output_root)
    elif args.stage == "postprocess":
        postprocess_existing_runs(args=args, dataset=dataset, output_root=output_root)
        aggregate_results(output_root)
    elif args.stage == "build-margin-tables":
        build_margin_tables(output_root)
    elif args.stage == "margin-raw-40":
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=raw_margin_units(args),
        )
        aggregate_results(output_root)
    elif args.stage == "margin-raw-joint-40":
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=top_joint_units(output_root, args, family="raw"),
        )
        aggregate_results(output_root)
    elif args.stage == "margin-quantile-40":
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=quantile_margin_units(output_root, args),
        )
        aggregate_results(output_root)
    elif args.stage == "margin-quantile-joint-40":
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=top_joint_units(output_root, args, family="quantile"),
        )
        aggregate_results(output_root)
    elif args.stage == "margin-search-40":
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=raw_margin_units(args),
        )
        aggregate_results(output_root)
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=top_joint_units(output_root, args, family="raw"),
        )
        aggregate_results(output_root)
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=quantile_margin_units(output_root, args),
        )
        aggregate_results(output_root)
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=top_joint_units(output_root, args, family="quantile"),
        )
        aggregate_results(output_root)
    elif args.stage == "build-shortlist-plan":
        shortlist_margin_units(output_root, args)
        aggregate_results(output_root)
    elif args.stage == "margin-shortlist-40":
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=shortlist_margin_units(output_root, args),
        )
        aggregate_results(output_root)
    elif args.stage == "confirm-200":
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=confirm_units(output_root, args, base_config, no_margin_only=False),
        )
        aggregate_results(output_root)
    elif args.stage == "aggregate":
        aggregate_results(output_root)
    elif args.stage == "all":
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=stage_a_units(args),
        )
        aggregate_results(output_root)
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=raw_margin_units(args),
        )
        aggregate_results(output_root)
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=top_joint_units(output_root, args, family="raw"),
        )
        aggregate_results(output_root)
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=quantile_margin_units(output_root, args),
        )
        aggregate_results(output_root)
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=top_joint_units(output_root, args, family="quantile"),
        )
        aggregate_results(output_root)
        run_units(
            args=args,
            dataset=dataset,
            base_config=base_config,
            output_root=output_root,
            units=confirm_units(output_root, args, base_config, no_margin_only=False),
        )
        aggregate_results(output_root)
    else:
        raise AssertionError(f"Unhandled stage: {args.stage}")


if __name__ == "__main__":
    main()
