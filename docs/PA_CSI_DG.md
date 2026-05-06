# PA-CSI DG Upgrade

This repository now contains a separate PyTorch domain-generalization path that preserves the main PA-CSI structure while leaving the original TensorFlow script intact.

## What Changed

- Added a true DG training entry point: `train_dg.py`
- Added a PyTorch PA-CSI-DG-lite model: `dg_models.py`
- Added DG dataset loading, LOEO split logic, and domain-balanced batching: `dg_dataset.py`
- Added supervised contrastive, natural antenna-view contrastive, and optional AdaIN-style feature perturbation: `dg_losses.py`
- Added runnable JSON presets under `configs/`
- Patched `dataset.py` so it no longer runs preprocessing at import time

## Data Flow

Main classifier path:

1. Load preprocessed amplitude and phase tensors separately.
2. Apply `np.unwrap` to phase on the configured axis before tensor reshaping. The current presets match the legacy PA-CSI behavior with `phase_unwrap_axis = -1`.
3. For each stream, reduce time by the configured `time_downsample`.
4. Add Gaussian relative positional encoding.
5. Apply an MCAT-like temporal encoder.
6. Apply multi-scale temporal CNN pooling.
7. Build a channel-summary path by reshaping the feature width into groups of `channel_group_width` when possible.
8. Fuse amplitude and phase with a gated residual fusion block.
9. Classify from the fused feature.

DG extensions:

- Projection head attaches to the fused feature only.
- `L_supcon` uses projection embeddings with class supervision and source-domain labels.
- `L_narc` uses natural antenna views built from the original flattened CSI width and re-encodes them with the same shared PA-CSI encoder.
- `L_adain` is optional and applies feature-level style perturbation to the fused feature before projection.

## Losses

- `L_ce`: standard cross-entropy on classifier logits
- `L_supcon`: supervised contrastive loss on projection embeddings
- `L_narc`: instance contrastive loss over natural antenna views from the same sample
- `L_adain`: instance contrastive loss between original and AdaIN-perturbed fused features

Total objective:

`L_total = L_ce + lambda_supcon * L_supcon + lambda_narc * L_narc + lambda_adain * L_adain`

Default presets keep `lambda_adain = 0`.

## DG Protocol

Supported modes:

- `dg_loeo`: leave one environment out
- `random_split`: legacy-style baseline fallback

For `dg_loeo`:

- target environment is used only for testing
- validation is drawn only from source environments
- source environments are inferred automatically as all non-target environments
- runs are repeated over `seed_list`
- per-seed, per-held-out-environment, and aggregate metrics are written under `outputs/`
- best checkpoint selection is configurable and now defaults to `val f1_macro` rather than raw accuracy

Each run writes:

- `split_manifest.json`
- `metrics.json`
- `best_model.pt`
- `confusion_matrix.npy`
- `confusion_matrix.csv`

Each preset writes an experiment-level `summary.json`.

## Expected Data Format

The DG path currently expects one amplitude file, one phase file, and one label file per environment:

```json
"env_files": {
  "E1": {"amp": "datasets/multienv/E1_amp.npy", "phase": "datasets/multienv/E1_phase.npy", "label": "datasets/multienv/E1_label.npy"},
  "E2": {"amp": "datasets/multienv/E2_amp.npy", "phase": "datasets/multienv/E2_phase.npy", "label": "datasets/multienv/E2_label.npy"},
  "E3": {"amp": "datasets/multienv/E3_amp.npy", "phase": "datasets/multienv/E3_phase.npy", "label": "datasets/multienv/E3_label.npy"}
}
```

Supported tensor shapes:

- `[N, T, D]`
- `[N, T, A, S]` which is flattened internally to `[N, T, D]`

Amplitude and phase must have the same `N`, `T`, and `D`.

## Antenna Assumptions

NARC assumes the flattened feature width can be reshaped into:

- `num_rx x num_tx x num_subcarriers`

The current local MultiEnv presets use:

- amplitude: `data_6c_{env}.npy`
- phase: `data_angle_6c_{env}.npy`
- label: `label_6c_{env}.npy`
- sample shape: `[3000, 850, 90]`
- antenna layout: `1 x 3 x 30 = 90`

Natural views are built by zero-masking all but one receiver group, transmitter group, or link group and then reusing the same shared PA-CSI encoder. This keeps the main backbone intact and avoids synthetic raw-space augmentations.

For the provided MultiEnv arrays, there is only one receive chain preserved in the processed files, so the NARC presets use `tx` views rather than `rx` views. That is a documented deviation from a receive-antenna ARC interpretation, but the views are still natural antenna/link views rather than synthetic perturbations.

If your processed tensors do not match the configured antenna layout:

- change `model.antenna_layout` to match the real layout
- or disable `lambda_narc`

## Batch Sampling

Sampler modes:

- `none`
- `domain_balanced`
- `domain_class_balanced`

The current SupCon/NARC presets use `domain_class_balanced` so each source-domain batch has a stronger multi-class structure for supervised contrastive learning.

## Commands

Baseline DG:

```bash
python train_dg.py --config configs/baseline_pa_csi_dg.json
```

DG + SupCon:

```bash
python train_dg.py --config configs/pa_csi_dg_supcon.json
```

DG + NARC:

```bash
python train_dg.py --config configs/pa_csi_dg_narc.json
```

DG + SupCon + NARC:

```bash
python train_dg.py --config configs/pa_csi_dg_supcon_narc.json
```

DG + SupCon + NARC + AdaIN:

```bash
python train_dg.py --config configs/pa_csi_dg_supcon_narc_adain.json
```

Single held-out environment:

```bash
python train_dg.py --config configs/pa_csi_dg_supcon_narc.json --target-env E3
```

Legacy-style random split fallback:

```bash
python train_dg.py --config configs/baseline_pa_csi_dg.json --mode random_split
```

## Important Deviations

- The original repository ships a TensorFlow training path and several mixed TensorFlow/PyTorch utility files that are not reliable enough for a DG extension.
- The new DG path is a PyTorch reimplementation that preserves the PA-CSI backbone structure at the architecture level rather than trying to reuse the broken TensorFlow execution path.
- The raw MultiEnv preprocessing code in `dataset.py` still needs manual confirmation before it should be trusted for final experiments because the legacy file contains incomplete phase handling and shape assumptions.
