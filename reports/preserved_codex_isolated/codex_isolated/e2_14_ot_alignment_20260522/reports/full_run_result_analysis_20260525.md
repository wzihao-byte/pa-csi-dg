# Full-Run Result Analysis: E2 1/4 OT Alignment

## Run Status
- Run id: `full_20260524_retry1`.
- Completed: Phase 1 baseline, Phase 2 unfiltered OT, Phase 2 prefiltered OT, aggregation, and full report shell.
- Summaries are under `analysis/full_20260524_retry1/`.
- Outputs are under `outputs_full_20260524_retry1/`.

## Main Paired Result
- Baseline target `min_f1_1_4`: mean 0.5888, std 0.0509, n=3.
- Unfiltered OT target `min_f1_1_4`: mean 0.5433, std 0.0488, n=3.
- Unfiltered OT paired delta: mean -0.0454, n=3.
- Prefiltered OT target `min_f1_1_4`: mean 0.5504, std 0.0321, n=3.
- Prefiltered OT paired delta: mean -0.0383, n=3.

Seed-wise unfiltered OT only improved seed 52 by +0.0171 and worsened seeds 42 and 62 by -0.0620 and -0.0914. Prefiltered OT followed the same pattern: only seed 52 improved slightly.

## Source-Val Check
- Baseline source-val `min_f1_1_4`: mean 0.9696.
- Unfiltered OT source-val `min_f1_1_4`: mean 0.9508.
- Prefiltered OT source-val `min_f1_1_4`: mean 0.9589.

This is not a catastrophic source collapse, but OT did not preserve source-heldout performance perfectly.

## Geometry Check
Baseline final target 1/4 centroid distances:
- seed 42: 0.1668
- seed 52: 0.1453
- seed 62: 0.0356

Unfiltered OT final target 1/4 centroid distances:
- seed 42: 0.0891
- seed 52: 0.0390
- seed 62: 0.0177

Prefiltered OT final target 1/4 centroid distances:
- seed 42: 0.0518
- seed 52: 0.0189
- seed 62: 0.0299

OT did not improve the target 1/4 separation symptom. It compressed the geometry relative to the source-only baseline, so this is not a case where geometry improved but F1 was capped by manual-review label/semantic noise.

## Transport Diagnostics
- Source transported mass stayed near the requested 1:1 source prior.
- Target selected predicted-class mass was often imbalanced, especially toward predicted class 1.
- Target entropy was low in the final OT plans, indicating confident pseudo-label structure rather than useful uncertainty correction.
- Class 1/4 transition counts were substantial in multiple runs, but they did not translate into lower bidirectional confusion or better `min_f1_1_4`.

## Decision
- The primary success criterion is not met: unfiltered OT does not improve target `min_f1_1_4` consistently and the paired mean is negative.
- The mechanism-success fallback is not met: target 1/4 geometry did not improve.
- The prefiltered run also does not support a data-quality/noise-aware filtering success story; it is slightly less negative than unfiltered, but still below baseline on average.

## Next-Step Gate
Do not automatically continue to OT sensitivity, CORAL, or conditional-MMD ablations from this run. The current evidence points to representation/alignment failure for this OT formulation, not to a promising alignment mechanism needing small hyperparameter tuning.

Reasonable follow-up, if explicitly requested, is representation re-learning or a diagnostic experiment that changes pseudo-label construction/selection, not stronger OT alignment.
