# SupCon Batch/Sampler Sweep

## Purpose

This second-round sweep checks whether pairwise SupCon is limited by batch composition. It keeps the SupCon setup fixed and sweeps only batch size and sampler mode for one or two selected lambda values.

This round adds diagnostics and run infrastructure only. It does not change model architecture, dataset split logic, antenna-view logic, SupCon loss math, or optimizer behavior.

## Fixed Settings

Keep these fixed unless explicitly overridden:

- `losses.temperature = 0.2`
- `losses.include_same_domain_same_class = true`
- `losses.supcon_positive_mode = "all_views"`
- `losses.contrastive_loss_type = "pairwise_supcon"`
- `target_env = E2`
- `seed = 42`
- `epochs = 80`

## Swept Settings

Default sweep grid:

- `losses.lambda_supcon`: `0.05, 0.1`
- `training.batch_size`: `8, 16, 32`
- `training.sampler`: `none, domain_balanced, domain_class_balanced`

## Run Commands

Dry run:

```bash
DRY_RUN=1 bash scripts/run_supcon_batch_sampler_sweep.sh
```

Actual run:

```bash
LAMBDAS="0.05 0.1" \
BATCH_SIZES="8 16 32" \
SAMPLERS="none domain_balanced domain_class_balanced" \
TARGET_ENV=E2 \
SEEDS="42" \
EPOCHS=80 \
DEVICE=cuda:0 \
bash scripts/run_supcon_batch_sampler_sweep.sh
```

Summarize:

```bash
python scripts/summarize_supcon_batch_sampler_sweep.py --output-root outputs_supcon_batch_sampler_sweep
```

## Interpreting Positive-Pair Diagnostics

Low `supcon_frac_anchors_with_positive` means the batch/sampler is not giving enough valid positives for SupCon.

If higher batch size improves SupCon diagnostics but not target accuracy, SupCon may be statistically better supported but too strong or poorly weighted.

If `domain_class_balanced` improves positive availability and source validation also improves, use it for later temperature or mask sweeps.

Do not repeatedly optimize hyperparameters against target test metrics. For reporting, choose settings based on source validation behavior, then rerun broader evaluation with multiple seeds and held-out environments.
