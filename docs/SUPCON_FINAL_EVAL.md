# SupCon Final Evaluation

## Purpose

This final evaluation runs a selected SupCon configuration across all held-out target environments and multiple seeds after settings have been chosen primarily from source validation behavior. This is not another hyperparameter search round.

Do not continue changing hyperparameters after looking at final target test results.

## Default Settings

- `target_env = all`
- `seeds = 42 52 62`
- `epochs = 200`
- `losses.lambda_supcon = 0.05`
- `losses.temperature = 0.2`
- `training.batch_size = 16`
- `training.sampler = "domain_class_balanced"`
- `losses.include_same_domain_same_class = true`
- `losses.supcon_positive_mode = "all_views"`
- `losses.lambda_supcon_warmup_epochs = 0`

## Run Commands

Dry run:

```bash
DRY_RUN=1 bash scripts/run_supcon_final_eval.sh
```

Actual run:

```bash
LAMBDA_SUPCON=0.05 \
TEMPERATURE=0.2 \
BATCH_SIZE=16 \
SAMPLER=domain_class_balanced \
INCLUDE_SAME_DOMAIN_SAME_CLASS=1 \
SUPCON_POSITIVE_MODE=all_views \
LAMBDA_SUPCON_WARMUP_EPOCHS=20 \
TARGET_ENV=all \
SEEDS="42 52 62" \
EPOCHS=200 \
DEVICE=cuda:0 \
bash scripts/run_supcon_final_eval.sh
```

Aggregate:

```bash
python scripts/aggregate_dg_final_results.py --output-root outputs_supcon_final_eval
```

Optional baseline comparison:

```bash
python scripts/aggregate_dg_final_results.py \
  --output-root outputs_supcon_final_eval \
  --baseline-root outputs_baseline_final_eval
```

## Reporting Guidance

Report mean +/- std across target environments and seeds.

Report per-target-env results in addition to the overall aggregate.

Do not claim improvement based on a single seed or a single held-out environment.

Do not continue changing hyperparameters after looking at final target test results.
