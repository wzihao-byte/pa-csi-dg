# SimMMDG PA-CSI Integration

This is a self-contained experimental folder for attaching SimMMDG-style shared/specific decomposition to the PA-CSI DG model. The repository body is left untouched.

The implementation references only the useful mathematical pieces from `donghao51/SimMMDG`: shared/specific feature splitting, SupCon on shared features, a distance/separation constraint for private features, and optional cross-view translation. It does not clone or vendor the full upstream project.

References:

- SimMMDG GitHub: https://github.com/donghao51/SimMMDG
- SimMMDG paper: https://arxiv.org/abs/2310.19795
- Local requirement source: `reports/weekly_report_20260420_simmmdg_focus/weekly_report_20260420_simmmdg_focus.tex`

## Main Design

The previous SupCon path used `projection_head(fused_feature)`. This folder changes only the isolated experimental path:

```text
fused_feature -> shared_feature s, private_feature p
logits = classifier(s)
projection = projection_head(s)
```

The objective is:

```text
L = L_ce + lambda_supcon * L_supcon(s) + lambda_sep * L_sep(s, p)
```

with:

```text
L_sep = mean | normalize(s)^T normalize(p) |^2
```

This matches the weekly report's emphasis: keep class-aware SupCon, but apply it only to action-shared information instead of forcing all CSI view/environment variation into one embedding.

## Important

This folder keeps only the multiview SimMMDG path.

The retained configuration assumes shared/specific decomposition across natural antenna views or an equivalent cross-view structure. Use the multiview bridge configuration as the default SimMMDG entry point in this folder.

## Commands

Synthetic smoke test:

```powershell
python simmmdg_pa_csi_integration\smoke_test.py
```

Diagnostic multiview bridge run:

```powershell
python simmmdg_pa_csi_integration\train_simmmdg.py --config simmmdg_pa_csi_integration\configs\pa_csi_dg_simmmdg_multiview_bridge.json
```

## Outputs

By default, runs write to:

```text
simmmdg_pa_csi_integration/outputs/
```

Each run preserves the existing metrics format: `split_manifest.json`, `metrics.json`, `best_model.pt`, `confusion_matrix.npy`, `confusion_matrix.csv`, and experiment-level `summary.json`.
