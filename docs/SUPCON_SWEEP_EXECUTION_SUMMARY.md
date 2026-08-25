# SupCon Sweep Execution Summary

## Scope

This work added auditable infrastructure for a staged PA-CSI-DG SupCon sweep and then ran the selected configuration through final leave-one-environment-out evaluation.

The active dataset split has three environments only: `E1`, `E2`, and `E3`. Any earlier mention of `E4` was incorrect for this PA-CSI-DG setup.

## Implemented Infrastructure

- Added CLI overrides in `train_dg.py` for SupCon lambda, temperature, positive-pair mode, same-domain same-class inclusion, batch size, sampler, experiment name, contrastive loss type, lambda warmup, pair margin weight, and prototype subcenters.
- Saved a per-run `resolved_config.json` before training, including the actual target environment, single seed, and resolved device.
- Added logging-only SupCon diagnostics:
  - `loss_supcon_weighted`
  - `loss_supcon_to_ce`
  - positive-pair coverage metrics
- Added sweep scripts, summarizers, aggregate analysis, documentation, and a lightweight CLI override smoke test.

## Staged Sweep Decisions

The first-stage sweeps used `target_env=E2` and seed 42 as screening runs. Target test metrics in these stages were diagnostic only.

Selected final configuration:

```text
contrastive_loss_type = pairwise_supcon
lambda_supcon = 0.05
lambda_supcon_warmup_epochs = 0
temperature = 0.2
batch_size = 32
sampler = domain_class_balanced
include_same_domain_same_class = true
supcon_positive_mode = global_class_instance_views
target_env = all
seeds = 42, 52, 62
epochs = 200
```

Warmup did not show a clear validation benefit, so the final run kept no warmup.

## Final Evaluation Results

Final output root: `outputs_supcon_final_eval`

Summary artifacts:

- `supcon_final_eval_per_run.csv`
- `supcon_final_eval_aggregate.csv`
- `analysis_supcon_final_eval/SUPCON_FINAL_EVAL_RESULTS.md`

Aggregate result over 9 runs:

| Scope | Test Acc. | Macro-F1 | Best Val Acc. | SupCon/CE |
|---|---:|---:|---:|---:|
| Overall | 0.8643 +/- 0.0252 | 0.8444 +/- 0.0221 | 0.9983 | 0.1521 |
| E1 | 0.8949 +/- 0.0091 | 0.8694 +/- 0.0116 | 0.9978 | 0.1656 |
| E2 | 0.8544 +/- 0.0043 | 0.8271 +/- 0.0124 | 0.9989 | 0.1708 |
| E3 | 0.8436 +/- 0.0153 | 0.8369 +/- 0.0142 | 0.9983 | 0.1200 |

## Comparison To Earlier Canonical Ablation

Earlier canonical paper-aligned LOEO ablation results from the April report:

| Method | Mean Acc. | Mean Macro-F1 | E1 Acc. | E2 Acc. | E3 Acc. |
|---|---:|---:|---:|---:|---:|
| CE baseline, seed 42 | 0.8787 | 0.8515 | 0.9007 | 0.8547 | 0.8807 |
| SupCon only, seed 42 | 0.9027 | 0.8820 | 0.9147 | 0.8953 | 0.8980 |
| Current selected SupCon, 9 runs | 0.8643 | 0.8444 | 0.8949 | 0.8544 | 0.8436 |

The selected sweep configuration did not improve over the earlier canonical SupCon-only anchor. It also did not improve over the earlier CE baseline on the overall mean. The strongest practical conclusion is therefore negative: the new selected configuration is auditable and stable, but it does not reproduce the old SupCon-only advantage under the final multi-seed evaluation.

## Next Step

Return to the earlier `pa_csi_dg_supcon_paper_aligned` anchor and isolate the source of the regression. The first differences to check are sampler, batch size, positive-pair mask/mode, lambda, config version, code version, and validation/selection protocol.
