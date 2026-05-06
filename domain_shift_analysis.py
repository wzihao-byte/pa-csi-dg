"""Standalone domain-shift analysis for PA-CSI DG.

Computes MMD (RBF, median heuristic), CORAL distance, centroid distance, and
optional t-SNE plots between environments E1/E2/E3, in two spaces:

  1. Raw input: temporal-mean of (amp, phase) per sample, model-free.
  2. Fused 256-d feature space, using each LOEO checkpoint.

This script imports from existing modules but does NOT modify any of them.
It writes all results to ``analysis/domain_shift/``.

Usage (defaults are fine for the local SupCon paper-aligned run):

    python domain_shift_analysis.py
    python domain_shift_analysis.py --config configs/pa_csi_dg_supcon_paper_aligned.json \
        --checkpoint-root experiment_outputs/diagnostics/outputs_diag_paper_aligned_single_seed \
        --experiment pa_csi_dg_supcon_paper_aligned --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from dg_dataset import load_dataset
from dg_models import PaCsiDGLite


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pa_csi_dg_supcon_paper_aligned.json"
DEFAULT_CHECKPOINT_ROOT = (
    REPO_ROOT
    / "experiment_outputs"
    / "diagnostics"
    / "outputs_diag_paper_aligned_single_seed"
)
DEFAULT_EXPERIMENT = "pa_csi_dg_supcon_paper_aligned"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "domain_shift"


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def median_heuristic_bandwidth(x: np.ndarray, max_samples: int = 1000) -> float:
    rng = np.random.default_rng(0)
    if x.shape[0] > max_samples:
        idx = rng.choice(x.shape[0], size=max_samples, replace=False)
        x = x[idx]
    diffs = x[:, None, :] - x[None, :, :]
    sq = (diffs ** 2).sum(axis=-1)
    iu = np.triu_indices_from(sq, k=1)
    pairwise = sq[iu]
    pairwise = pairwise[pairwise > 0]
    if pairwise.size == 0:
        return 1.0
    return float(np.sqrt(np.median(pairwise) / 2.0))


def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    sq_x = (x ** 2).sum(axis=1, keepdims=True)
    sq_y = (y ** 2).sum(axis=1, keepdims=True)
    cross = x @ y.T
    sq_dist = sq_x + sq_y.T - 2.0 * cross
    sq_dist = np.maximum(sq_dist, 0.0)
    return np.exp(-sq_dist / (2.0 * sigma * sigma))


def mmd2_rbf(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """Unbiased MMD^2 estimator with a Gaussian kernel."""
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        return float("nan")
    kxx = gaussian_kernel(x, x, sigma)
    kyy = gaussian_kernel(y, y, sigma)
    kxy = gaussian_kernel(x, y, sigma)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    term_xx = kxx.sum() / (n * (n - 1))
    term_yy = kyy.sum() / (m * (m - 1))
    term_xy = kxy.mean()
    return float(term_xx + term_yy - 2.0 * term_xy)


def coral_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Frobenius distance between feature covariance matrices, normalised by 4d^2."""
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")
    cov_x = np.cov(x, rowvar=False)
    cov_y = np.cov(y, rowvar=False)
    diff = cov_x - cov_y
    d = x.shape[1]
    return float((diff ** 2).sum() / (4.0 * d * d))


def centroid_distance(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x.mean(axis=0) - y.mean(axis=0)))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_fused_features(
    model: PaCsiDGLite,
    amp: np.ndarray,
    phase: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    model.eval()
    features: List[np.ndarray] = []
    n = amp.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        amp_batch = torch.from_numpy(amp[start:end]).float().to(device)
        phase_batch = torch.from_numpy(phase[start:end]).float().to(device)
        out = model.encode_features(amp_batch, phase_batch)
        features.append(out["fused_feature"].cpu().numpy())
    return np.concatenate(features, axis=0)


def temporal_mean_inputs(amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Concatenate temporal-mean of amplitude and phase as a model-free per-sample vector."""
    amp_mean = amp.mean(axis=1)
    phase_mean = phase.mean(axis=1)
    return np.concatenate([amp_mean, phase_mean], axis=1).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Per-environment splitting
# ---------------------------------------------------------------------------

def collect_env_arrays(dataset, env_names: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for env_id, env_name in enumerate(env_names):
        mask = dataset.env_ids == env_id
        out[env_name] = {
            "amp": dataset.amplitude[mask],
            "phase": dataset.phase[mask],
            "labels": dataset.labels[mask],
        }
    return out


def pairwise_distance_table(
    env_features: Dict[str, np.ndarray],
    env_names: List[str],
    sigma: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, a in enumerate(env_names):
        for b in env_names[i + 1 :]:
            x, y = env_features[a], env_features[b]
            out[f"mmd2[{a},{b}]"] = mmd2_rbf(x, y, sigma)
            out[f"coral[{a},{b}]"] = coral_distance(x, y)
            out[f"centroid[{a},{b}]"] = centroid_distance(x, y)
    return out


def per_env_average_distance(pair_metrics: Dict[str, float], env_names: List[str]) -> Dict[str, Dict[str, float]]:
    """For each env, average its distances against the other two envs."""
    summary: Dict[str, Dict[str, float]] = {env: {"mmd2": 0.0, "coral": 0.0, "centroid": 0.0, "count": 0} for env in env_names}
    for key, value in pair_metrics.items():
        metric_name, pair = key.split("[", 1)
        a, b = pair.rstrip("]").split(",")
        for env in (a, b):
            summary[env][metric_name] += value
            summary[env]["count"] += 1
    for env in env_names:
        n_pairs = summary[env]["count"] // 3
        if n_pairs > 0:
            summary[env]["mmd2"] /= n_pairs
            summary[env]["coral"] /= n_pairs
            summary[env]["centroid"] /= n_pairs
        summary[env].pop("count")
    return summary


def tsne_plot(
    features: np.ndarray,
    env_labels: np.ndarray,
    env_names: List[str],
    title: str,
    output_path: Path,
    perplexity: float = 30.0,
    seed: int = 0,
    max_points: int = 1500,
) -> None:
    rng = np.random.default_rng(seed)
    if features.shape[0] > max_points:
        idx = rng.choice(features.shape[0], size=max_points, replace=False)
        features = features[idx]
        env_labels = env_labels[idx]
    embedded = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca", learning_rate="auto").fit_transform(features)
    plt.figure(figsize=(6, 5))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    for env_id, env_name in enumerate(env_names):
        mask = env_labels == env_id
        plt.scatter(embedded[mask, 0], embedded[mask, 1], s=8, alpha=0.6, label=env_name, c=colors[env_id % len(colors)])
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Domain-shift diagnostics for PA-CSI DG.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Training config JSON used to build dataset/model.")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT, help="Root directory of LOEO outputs.")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="Experiment name under <root>/dg_loeo/.")
    parser.add_argument("--seed", type=int, default=42, help="Seed subdirectory to load.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write results.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples-per-env", type=int, default=800, help="Random subsample per env for distance metrics.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-tsne", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    print(f"[info] loading dataset from {args.config.name}")
    dataset = load_dataset(config["data"], REPO_ROOT)
    env_names = list(dataset.env_names)
    print(f"[info] envs={env_names} input_dim={dataset.input_dim} time={dataset.time_steps} num_classes={dataset.num_classes}")

    env_arrays = collect_env_arrays(dataset, env_names)
    rng = np.random.default_rng(0)
    env_indices: Dict[str, np.ndarray] = {}
    for env_name in env_names:
        n = env_arrays[env_name]["amp"].shape[0]
        size = min(args.samples_per_env, n)
        env_indices[env_name] = rng.choice(n, size=size, replace=False)

    # ---- 1. Raw-input distances ----
    raw_features: Dict[str, np.ndarray] = {}
    for env_name in env_names:
        idx = env_indices[env_name]
        raw_features[env_name] = temporal_mean_inputs(
            env_arrays[env_name]["amp"][idx],
            env_arrays[env_name]["phase"][idx],
        )
    pooled_raw = np.concatenate([raw_features[e] for e in env_names], axis=0)
    sigma_raw = median_heuristic_bandwidth(pooled_raw)
    print(f"[info] raw-input median-heuristic sigma = {sigma_raw:.4f}")

    raw_pair_metrics = pairwise_distance_table(raw_features, env_names, sigma_raw)
    raw_per_env = per_env_average_distance(raw_pair_metrics, env_names)

    # ---- 2. Fused-feature distances per LOEO checkpoint ----
    device = torch.device(args.device)
    model_features_per_target: Dict[str, Dict[str, np.ndarray]] = {}
    fused_pair_metrics: Dict[str, Dict[str, float]] = {}
    fused_per_env: Dict[str, Dict[str, Dict[str, float]]] = {}
    sigmas_per_target: Dict[str, float] = {}

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
        print(f"[info] loading checkpoint for target {target_env}")
        model = PaCsiDGLite(
            input_dim=dataset.input_dim,
            num_classes=dataset.num_classes,
            model_config=config["model"],
        ).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)

        env_features: Dict[str, np.ndarray] = {}
        for env_name in env_names:
            idx = env_indices[env_name]
            env_features[env_name] = extract_fused_features(
                model,
                env_arrays[env_name]["amp"][idx],
                env_arrays[env_name]["phase"][idx],
                device=device,
                batch_size=args.batch_size,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        pooled = np.concatenate([env_features[e] for e in env_names], axis=0)
        sigma = median_heuristic_bandwidth(pooled)
        sigmas_per_target[target_env] = sigma
        pair_metrics = pairwise_distance_table(env_features, env_names, sigma)
        fused_pair_metrics[target_env] = pair_metrics
        fused_per_env[target_env] = per_env_average_distance(pair_metrics, env_names)
        model_features_per_target[target_env] = env_features

    # ---- 3. Save numeric results ----
    results = {
        "experiment": args.experiment,
        "seed": args.seed,
        "samples_per_env": args.samples_per_env,
        "env_names": env_names,
        "raw_input": {
            "sigma": sigma_raw,
            "pair_metrics": raw_pair_metrics,
            "per_env_average": raw_per_env,
            "feature_dim": int(pooled_raw.shape[1]),
        },
        "fused_feature": {
            "sigmas_per_target": sigmas_per_target,
            "pair_metrics_per_target": fused_pair_metrics,
            "per_env_average_per_target": fused_per_env,
        },
    }
    results_path = args.output_dir / f"distances_{args.experiment}_seed{args.seed}.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"[info] wrote {results_path}")

    # Pretty-print summary table to stdout and a markdown file
    summary_lines: List[str] = []
    summary_lines.append(f"# Domain-shift diagnostics: {args.experiment} (seed {args.seed})\n")
    summary_lines.append(f"- envs: {env_names}\n")
    summary_lines.append(f"- samples per env: {args.samples_per_env}\n\n")

    summary_lines.append("## Raw input space (temporal-mean of amp+phase, model-free)\n")
    summary_lines.append(f"sigma_RBF = {sigma_raw:.4f}\n\n")
    summary_lines.append("| Pair | MMD^2 | CORAL | Centroid L2 |\n|---|---:|---:|---:|\n")
    pairs = [(env_names[i], env_names[j]) for i in range(len(env_names)) for j in range(i + 1, len(env_names))]
    for a, b in pairs:
        mmd_v = raw_pair_metrics[f"mmd2[{a},{b}]"]
        coral_v = raw_pair_metrics[f"coral[{a},{b}]"]
        cen_v = raw_pair_metrics[f"centroid[{a},{b}]"]
        summary_lines.append(f"| {a}-{b} | {mmd_v:.4f} | {coral_v:.4f} | {cen_v:.4f} |\n")
    summary_lines.append("\n| Env | mean MMD^2 | mean CORAL | mean centroid L2 |\n|---|---:|---:|---:|\n")
    for env in env_names:
        s = raw_per_env[env]
        summary_lines.append(f"| {env} | {s['mmd2']:.4f} | {s['coral']:.4f} | {s['centroid']:.4f} |\n")
    summary_lines.append("\n")

    for target_env, pair_metrics in fused_pair_metrics.items():
        summary_lines.append(f"## Fused 256-d feature space (LOEO target {target_env})\n")
        summary_lines.append(f"sigma_RBF = {sigmas_per_target[target_env]:.4f}\n\n")
        summary_lines.append("| Pair | MMD^2 | CORAL | Centroid L2 | involves target? |\n|---|---:|---:|---:|---|\n")
        for a, b in pairs:
            mmd_v = pair_metrics[f"mmd2[{a},{b}]"]
            coral_v = pair_metrics[f"coral[{a},{b}]"]
            cen_v = pair_metrics[f"centroid[{a},{b}]"]
            tag = "yes (target held out)" if target_env in (a, b) else "no (both seen in training)"
            summary_lines.append(f"| {a}-{b} | {mmd_v:.4f} | {coral_v:.4f} | {cen_v:.4f} | {tag} |\n")
        summary_lines.append("\n| Env | mean MMD^2 | mean CORAL | mean centroid L2 |\n|---|---:|---:|---:|\n")
        for env in env_names:
            s = fused_per_env[target_env][env]
            summary_lines.append(f"| {env} | {s['mmd2']:.4f} | {s['coral']:.4f} | {s['centroid']:.4f} |\n")
        summary_lines.append("\n")

    summary_text = "".join(summary_lines)
    summary_path = args.output_dir / f"summary_{args.experiment}_seed{args.seed}.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(summary_text)
    print(summary_text)
    print(f"[info] wrote {summary_path}")

    # ---- 4. t-SNE plots ----
    if args.skip_tsne:
        return

    raw_pooled = np.concatenate([raw_features[e] for e in env_names], axis=0)
    raw_env_labels = np.concatenate(
        [np.full(raw_features[e].shape[0], i, dtype=np.int64) for i, e in enumerate(env_names)],
        axis=0,
    )
    tsne_plot(
        raw_pooled,
        raw_env_labels,
        env_names,
        title="t-SNE of raw input (temporal-mean amp+phase)",
        output_path=args.output_dir / f"tsne_raw_input.png",
    )
    print(f"[info] wrote raw-input t-SNE")

    for target_env, env_features in model_features_per_target.items():
        pooled = np.concatenate([env_features[e] for e in env_names], axis=0)
        env_labels = np.concatenate(
            [np.full(env_features[e].shape[0], i, dtype=np.int64) for i, e in enumerate(env_names)],
            axis=0,
        )
        tsne_plot(
            pooled,
            env_labels,
            env_names,
            title=f"t-SNE of fused features (LOEO target {target_env}, {args.experiment})",
            output_path=args.output_dir / f"tsne_fused_target_{target_env}.png",
        )
        print(f"[info] wrote fused-feature t-SNE for target {target_env}")


if __name__ == "__main__":
    main()
