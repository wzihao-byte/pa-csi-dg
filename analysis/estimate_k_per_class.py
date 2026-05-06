"""Empirically estimate K (sub-center prototypes per class) for PA-CSI DG.

For each LOEO target checkpoint, extracts 256-d fused features for source-env
samples, then per class runs:
  - k-means + silhouette score sweep
  - GMM + BIC sweep
over K in a configurable grid. Reports per-class K* (silhouette argmax, BIC
argmin), a single recommended global K (median over classes), and a raw-input
sanity check using temporal-mean features.

Outputs go under analysis/k_estimate/.

Usage (defaults match domain_shift_analysis.py):

    python analysis/estimate_k_per_class.py
    python analysis/estimate_k_per_class.py \
        --config configs/pa_csi_dg_supcon_paper_aligned.json \
        --checkpoint-root experiment_outputs/diagnostics/outputs_diag_paper_aligned_single_seed \
        --experiment pa_csi_dg_supcon_paper_aligned --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dg_dataset import load_dataset  # noqa: E402
from dg_models import PaCsiDGLite  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "pa_csi_dg_supcon_paper_aligned.json"
DEFAULT_CHECKPOINT_ROOT = (
    REPO_ROOT
    / "experiment_outputs"
    / "diagnostics"
    / "outputs_diag_paper_aligned_single_seed"
)
DEFAULT_EXPERIMENT = "pa_csi_dg_supcon_paper_aligned"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "k_estimate"
DEFAULT_K_GRID = (1, 2, 3, 4, 5, 6, 8)


@torch.no_grad()
def extract_fused_features(
    model: PaCsiDGLite,
    amp: np.ndarray,
    phase: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    model.eval()
    parts: List[np.ndarray] = []
    n = amp.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        amp_batch = torch.from_numpy(amp[start:end]).float().to(device)
        phase_batch = torch.from_numpy(phase[start:end]).float().to(device)
        out = model.encode_features(amp_batch, phase_batch)
        parts.append(out["fused_feature"].cpu().numpy())
    return np.concatenate(parts, axis=0)


def temporal_mean_inputs(amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return np.concatenate([amp.mean(axis=1), phase.mean(axis=1)], axis=1).astype(np.float32, copy=False)


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, eps)


def sweep_k_per_class(
    features: np.ndarray,
    k_grid: Sequence[int],
    seed: int,
) -> Dict[str, object]:
    """Run silhouette + BIC sweeps for one class's feature matrix.

    Returns a dict with the K grid, the two score curves, and the chosen K* per metric.
    Silhouette is undefined for K=1 (returns NaN for that point); BIC is well-defined.
    """
    n_samples = features.shape[0]
    if n_samples < max(k_grid) + 1:
        # Truncate the grid so every K we attempt has at least 2 points per cluster.
        k_grid = tuple(k for k in k_grid if k <= max(1, n_samples // 2))
        if not k_grid:
            k_grid = (1,)

    silhouette_curve: List[float] = []
    bic_curve: List[float] = []
    for k in k_grid:
        if k <= 1 or k >= n_samples:
            silhouette_curve.append(float("nan"))
        else:
            km = KMeans(n_clusters=k, n_init=10, random_state=seed)
            labels = km.fit_predict(features)
            if len(set(labels)) < 2:
                silhouette_curve.append(float("nan"))
            else:
                silhouette_curve.append(float(silhouette_score(features, labels)))

        # BIC defined for K >= 1.
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=seed,
                reg_covar=1e-4,
                max_iter=200,
            )
            gmm.fit(features)
            bic_curve.append(float(gmm.bic(features)))
        except Exception as exc:  # pragma: no cover - defensive
            bic_curve.append(float("nan"))
            print(f"[warn] GMM failed at K={k}: {exc}")

    valid_silhouette = [(k, s) for k, s in zip(k_grid, silhouette_curve) if not np.isnan(s)]
    silhouette_argmax = max(valid_silhouette, key=lambda pair: pair[1])[0] if valid_silhouette else int(k_grid[0])
    valid_bic = [(k, b) for k, b in zip(k_grid, bic_curve) if not np.isnan(b)]
    bic_argmin = min(valid_bic, key=lambda pair: pair[1])[0] if valid_bic else int(k_grid[0])

    return {
        "n_samples": int(n_samples),
        "k_grid": list(k_grid),
        "silhouette_curve": silhouette_curve,
        "bic_curve": bic_curve,
        "silhouette_argmax_K": int(silhouette_argmax),
        "bic_argmin_K": int(bic_argmin),
    }


def aggregate_global_k(per_class: Dict[int, Dict[str, object]]) -> Dict[str, int]:
    silhouette_ks = [int(v["silhouette_argmax_K"]) for v in per_class.values()]
    bic_ks = [int(v["bic_argmin_K"]) for v in per_class.values()]
    return {
        "K_recommended_silhouette_median": int(round(median(silhouette_ks))),
        "K_recommended_bic_median": int(round(median(bic_ks))),
        "K_max_silhouette": int(max(silhouette_ks)),
        "K_min_silhouette": int(min(silhouette_ks)),
        "K_max_bic": int(max(bic_ks)),
        "K_min_bic": int(min(bic_ks)),
    }


def per_class_features(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            out[c] = features[mask]
    return out


def evaluate_target(
    config: Dict[str, object],
    dataset,
    target_env: str,
    ckpt_path: Path,
    device: torch.device,
    k_grid: Sequence[int],
    seed: int,
    batch_size: int,
) -> Dict[str, object]:
    """Build the model for this target, extract source-env fused features, and run K sweeps."""
    target_env_id = dataset.env_names.index(target_env)
    source_mask = dataset.env_ids != target_env_id

    amp = dataset.amplitude[source_mask]
    phase = dataset.phase[source_mask]
    labels = dataset.labels[source_mask]

    print(f"[info] target={target_env} source samples: {amp.shape[0]} (per class: {dict(zip(*np.unique(labels, return_counts=True)))})")

    model = PaCsiDGLite(
        input_dim=dataset.input_dim,
        num_classes=dataset.num_classes,
        model_config=config["model"],
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)

    fused = extract_fused_features(model, amp, phase, device=device, batch_size=batch_size)
    fused_norm = l2_normalize(fused)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fused_per_class = per_class_features(fused_norm, labels, dataset.num_classes)
    per_class_results: Dict[int, Dict[str, object]] = {}
    for c, feats in fused_per_class.items():
        per_class_results[c] = sweep_k_per_class(feats, k_grid, seed=seed)

    summary = aggregate_global_k(per_class_results)
    return {
        "target_env": target_env,
        "checkpoint": str(ckpt_path),
        "per_class": {str(k): v for k, v in per_class_results.items()},
        "summary": summary,
    }


def evaluate_raw_sanity(
    dataset,
    k_grid: Sequence[int],
    seed: int,
) -> Dict[str, object]:
    raw = temporal_mean_inputs(dataset.amplitude, dataset.phase)
    raw_norm = l2_normalize(raw)
    raw_per_class = per_class_features(raw_norm, dataset.labels, dataset.num_classes)
    per_class_results: Dict[int, Dict[str, object]] = {}
    for c, feats in raw_per_class.items():
        per_class_results[c] = sweep_k_per_class(feats, k_grid, seed=seed)
    return {
        "per_class": {str(k): v for k, v in per_class_results.items()},
        "summary": aggregate_global_k(per_class_results),
    }


def render_markdown(
    experiment: str,
    seed: int,
    k_grid: Sequence[int],
    fused_results: Dict[str, Dict[str, object]],
    raw_results: Dict[str, object],
    num_classes: int,
) -> str:
    lines: List[str] = []
    lines.append(f"# K estimation: {experiment} (seed {seed})\n\n")
    lines.append(f"K grid: {list(k_grid)}\n\n")

    lines.append("## Recommended global K (single int across all classes)\n\n")
    lines.append("| LOEO target | K_median (silhouette) | K_median (BIC) | K range (silhouette) | K range (BIC) |\n")
    lines.append("|---|---:|---:|---|---|\n")
    for target_env, result in fused_results.items():
        s = result["summary"]
        lines.append(
            f"| {target_env} "
            f"| {s['K_recommended_silhouette_median']} "
            f"| {s['K_recommended_bic_median']} "
            f"| {s['K_min_silhouette']}-{s['K_max_silhouette']} "
            f"| {s['K_min_bic']}-{s['K_max_bic']} |\n"
        )
    raw_s = raw_results["summary"]
    lines.append(
        f"| raw-input sanity "
        f"| {raw_s['K_recommended_silhouette_median']} "
        f"| {raw_s['K_recommended_bic_median']} "
        f"| {raw_s['K_min_silhouette']}-{raw_s['K_max_silhouette']} "
        f"| {raw_s['K_min_bic']}-{raw_s['K_max_bic']} |\n\n"
    )

    for target_env, result in fused_results.items():
        lines.append(f"## Fused features, LOEO target {target_env}\n\n")
        lines.append("| class | n | silhouette K* | BIC K* |\n|---:|---:|---:|---:|\n")
        for c in range(num_classes):
            entry = result["per_class"].get(str(c))
            if entry is None:
                continue
            lines.append(
                f"| {c} | {entry['n_samples']} "
                f"| {entry['silhouette_argmax_K']} "
                f"| {entry['bic_argmin_K']} |\n"
            )
        lines.append("\n")

    lines.append("## Raw input (temporal-mean amp+phase) sanity check\n\n")
    lines.append("| class | n | silhouette K* | BIC K* |\n|---:|---:|---:|---:|\n")
    for c in range(num_classes):
        entry = raw_results["per_class"].get(str(c))
        if entry is None:
            continue
        lines.append(
            f"| {c} | {entry['n_samples']} "
            f"| {entry['silhouette_argmax_K']} "
            f"| {entry['bic_argmin_K']} |\n"
        )
    lines.append("\n")

    return "".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Empirical K estimation for PA-CSI sub-center prototypes.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--k-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_GRID),
        help="K values to sweep (default: 1 2 3 4 5 6 8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    print(f"[info] loading dataset from {args.config.name}")
    dataset = load_dataset(config["data"], REPO_ROOT)
    env_names = list(dataset.env_names)
    print(
        f"[info] envs={env_names} num_classes={dataset.num_classes} "
        f"input_dim={dataset.input_dim} time={dataset.time_steps} "
        f"total_samples={len(dataset)}"
    )

    device = torch.device(args.device)
    fused_results: Dict[str, Dict[str, object]] = {}

    for target_env in env_names:
        ckpt_path = (
            args.checkpoint_root
            / "dg_loeo"
            / args.experiment
            / f"target_{target_env}"
            / f"seed_{args.seed}"
            / "best_model.pt"
        )
        if not ckpt_path.exists():
            print(f"[warn] missing checkpoint {ckpt_path}, skipping target {target_env}")
            continue
        print(f"[info] loading checkpoint for target {target_env}: {ckpt_path}")
        fused_results[target_env] = evaluate_target(
            config=config,
            dataset=dataset,
            target_env=target_env,
            ckpt_path=ckpt_path,
            device=device,
            k_grid=tuple(args.k_grid),
            seed=args.seed,
            batch_size=args.batch_size,
        )

    if not fused_results:
        raise RuntimeError(
            f"No checkpoints found under {args.checkpoint_root}/dg_loeo/{args.experiment}/target_*/seed_{args.seed}."
        )

    print("[info] running raw-input sanity sweep")
    raw_results = evaluate_raw_sanity(dataset, tuple(args.k_grid), seed=args.seed)

    payload = {
        "experiment": args.experiment,
        "seed": args.seed,
        "k_grid": list(args.k_grid),
        "env_names": env_names,
        "num_classes": int(dataset.num_classes),
        "fused_feature_K_per_target": fused_results,
        "raw_input_K_per_class_sanity": raw_results,
    }

    json_path = args.output_dir / f"k_per_class_{args.experiment}_seed{args.seed}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[info] wrote {json_path}")

    md_text = render_markdown(
        experiment=args.experiment,
        seed=args.seed,
        k_grid=args.k_grid,
        fused_results=fused_results,
        raw_results=raw_results,
        num_classes=dataset.num_classes,
    )
    md_path = args.output_dir / f"k_per_class_{args.experiment}_seed{args.seed}.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(md_text)
    print(md_text)
    print(f"[info] wrote {md_path}")


if __name__ == "__main__":
    main()
