# PA-CSI DG Path

This repository now keeps a focused PyTorch domain-generalization path for three experiment families:

- baseline PA-CSI DG
- PA-CSI DG with supervised contrastive learning
- PA-CSI DG with subcenter prototype SupCon and optional directed pair margins

The SimMMDG, NARC, AdaIN, and bridge ablation branches have been removed from the active code and presets.

## Entry Point

Use `train_dg.py` with a JSON preset in `configs/`.

Supported split modes:

- `dg_loeo`: leave one environment out
- `random_split`: legacy-style fallback

For `dg_loeo`, the target environment is used only for testing. Validation is drawn from source environments only.

## Retained Presets

Baseline:

```bash
python train_dg.py --config configs/baseline_pa_csi_dg.json
python train_dg.py --config configs/baseline_pa_csi_dg_paper_aligned.json
```

SupCon:

```bash
python train_dg.py --config configs/pa_csi_dg_supcon.json
python train_dg.py --config configs/pa_csi_dg_supcon_paper_aligned.json
```

Subcenter SupCon:

```bash
python train_dg.py --config configs/pa_csi_dg_subcenter_proto_layer2_200e_k3_margin_e2.json
python train_dg.py --config configs/pa_csi_dg_subcenter_proto_layer2_200e_k5_margin_0to4_m020_e2.json
```

Single held-out environment:

```bash
python train_dg.py --config configs/pa_csi_dg_supcon_paper_aligned.json --target-env E2
```

## Losses

Baseline uses cross-entropy:

```text
L_total = L_ce
```

SupCon adds supervised contrastive learning over projection embeddings:

```text
L_total = L_ce + lambda_supcon * L_supcon
```

LCA-KM is an optional source-calibrated margin term over the same projection embeddings:

```text
L_total = existing losses + lambda_akm * L_akm
```

After a warm-up period, LOSO-Calibrated Adaptive K-Margin recalibrates on the source validation split only. For each source environment, it builds leave-one-source-out class prototypes from the other source environments, measures class-pair hardness, violation rate, compactness, and cross-source instability on the held-out source environment, then selects an adaptive number of hard negative class pairs per class. Cached global source prototypes and pair tables are detached before use, so gradients do not flow through calibration statistics.

The held-out target environment is never used for calibration in `dg_loeo`; target data remains test-only through the existing split builder. Recalibration artifacts are saved under each run directory in `adaptive_kmargin/`.

Enable it with `losses.adaptive_kmargin`:

```json
"adaptive_kmargin": {
  "enabled": true,
  "warmup_epochs": 10,
  "recalibrate_interval": 5,
  "beta_softmax": 10.0,
  "gap_threshold_tau": 0.0,
  "rho0": 0.65,
  "rho_min": 0.45,
  "rho_max": 0.85,
  "rho_entropy_scale": 0.15,
  "rho_instability_scale": 0.15,
  "m_min": 0.02,
  "m_max": 0.20,
  "alpha_h": 1.0,
  "beta_v": 1.0,
  "gamma_kappa": 1.0,
  "delta_u": 1.0,
  "lambda_akm": 0.1
}
```

Subcenter SupCon replaces pairwise SupCon geometry with learnable class subcenters:

```text
L_total = L_ce + lambda_supcon * L_subcenter
```

When directed pair margins are enabled, the subcenter loss adds:

```text
L_pair = mean [ margin(i,j) + score_j(x) - score_i(x) ]_+
```

where `score_c(x)` is the maximum cosine score between the sample projection and class `c` subcenters.

## Data Format

The DG path expects one amplitude file, one phase file, and one label file per environment:

```json
"env_files": {
  "E1": {"amp": "datasets/multienv/E1_amp.npy", "phase": "datasets/multienv/E1_phase.npy", "label": "datasets/multienv/E1_label.npy"},
  "E2": {"amp": "datasets/multienv/E2_amp.npy", "phase": "datasets/multienv/E2_phase.npy", "label": "datasets/multienv/E2_label.npy"},
  "E3": {"amp": "datasets/multienv/E3_amp.npy", "phase": "datasets/multienv/E3_phase.npy", "label": "datasets/multienv/E3_label.npy"}
}
```

Supported tensor shapes:

- `[N, T, D]`
- `[N, T, A, S]`, flattened internally to `[N, T, D]`

Amplitude and phase must have the same `N`, `T`, and `D`.

## Outputs

Each run writes:

- `split_manifest.json`
- `metrics.json`
- `best_model.pt`
- `confusion_matrix.npy`
- `confusion_matrix.csv`

Each preset writes an experiment-level `summary.json`.
