# SimMMDG Alignment Notes

This folder is an isolated adapter. It does not modify `train_dg.py`, `dg_models.py`, `dg_losses.py`, `dg_dataset.py`, or the root configs.

## Source Insight

The referenced SimMMDG implementation separates each modality embedding into a shared part and a modality-specific part. Its README states the same high-level structure: shared features receive supervised contrastive learning, specific features receive a distance constraint, and cross-modal translation regularizes features.

For this PA-CSI project, the weekly report makes the mapping more specific:

```text
CSI view feature = shared action semantics + view/domain-specific variation
```

The clean integration is therefore:

```text
h_i^(v) -> [s_i^(v), p_i^(v)]
```

where `h` is the fused PA-CSI feature from the existing amplitude/phase encoders, `s` is the action-shared feature, and `p` stores view/domain-specific variation.

## Implemented Objective

The retained objective is:

```text
L = L_ce(s) + lambda_supcon * L_supcon(proj(s)) + lambda_sep * L_sep(s, p)
```

The separation term follows the weekly-report form:

```text
L_sep = mean | normalize(s)^T normalize(p) |^2
```

This differs deliberately from copying the open-source code line by line. The original implementation pushes the first and second halves of an embedding apart with a negative MSE-style term. Here the squared inner-product penalty is a cleaner mathematical fit because it discourages leakage between shared and private subspaces without making the total loss unbounded below.

## SupCon Placement

The old PA-CSI path used:

```text
projection = projection_head(fused_feature)
L_supcon = SupCon(projection)
```

This adapter uses:

```text
shared, private = decomposition(fused_feature)
logits = classifier(shared)
projection = projection_head(shared)
L_supcon = SupCon(projection)
```

This keeps the existing SupCon gain, but restricts it to the intended invariant component.

## Natural-View Bridge

The `pa_csi_dg_simmmdg_multiview_bridge.json` config keeps antenna views enabled. Each natural view feature is also decomposed:

```text
h_i^(v) -> [s_i^(v), p_i^(v)]
```

Then SupCon can consume all shared projections, and the optional translation loss uses ordered pairs of natural view shared features:

```text
L_trans = mean || normalize(T(s_i^a)) - normalize(s_i^b) ||_2
```

This retained path keeps the antenna multiview prior explicit instead of treating it as a hidden byproduct of the contrastive setup.

## Files

- `simmmdg_models.py`: shared/private PA-CSI model adapter.
- `simmmdg_losses.py`: orthogonality separation and optional cross-view translation loss.
- `train_simmmdg.py`: independent training entry point adapted from the existing DG runner.
- `configs/pa_csi_dg_simmmdg_multiview_bridge.json`: diagnostic natural-view bridge config.
- `smoke_test.py`: synthetic forward/loss sanity check.
