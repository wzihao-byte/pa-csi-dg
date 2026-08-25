# SupCon Lambda Warmup Sweep

## Purpose

This fifth-round sweep tests whether gradually increasing `losses.lambda_supcon` stabilizes early training and improves source-validation behavior for pairwise SupCon.

This is the first SupCon tuning round that intentionally changes training-time scheduling. It does not change model architecture, dataset logic, DG split logic, SupCon mathematical form, positive-pair mask logic, optimizer type, or subcenter prototype logic.

## Schedule

Use optional config key:

```json
"losses": {
  "lambda_supcon_warmup_epochs": 20
}
```

The scheduled training lambda is:

```text
effective_lambda_supcon = target_lambda_supcon * min(1.0, epoch / warmup_epochs)
```

`epoch` is 1-indexed. If `lambda_supcon_warmup_epochs` is missing or `<= 0`, the existing behavior is preserved exactly and `effective_lambda_supcon = target_lambda_supcon`.

Validation inside each epoch uses the same effective lambda as that training epoch. Final test evaluation logs the full target lambda.

## Fixed Defaults

- `losses.contrastive_loss_type = "pairwise_supcon"`
- `losses.temperature = 0.2`
- `losses.include_same_domain_same_class = true`
- `losses.supcon_positive_mode = "all_views"`
- `training.batch_size = 16`
- `training.sampler = "domain_class_balanced"`
- `target_env = E2`
- `seed = 42`
- `epochs = 100`

Swept defaults:

- `losses.lambda_supcon`: `0.05, 0.1, 0.2`
- `losses.lambda_supcon_warmup_epochs`: `0, 10, 20, 40`

## Run Commands

Dry run:

```bash
DRY_RUN=1 bash scripts/run_supcon_warmup_sweep.sh
```

Actual run:

```bash
LAMBDAS="0.05 0.1 0.2" \
WARMUP_EPOCHS="0 10 20 40" \
TARGET_ENV=E2 \
SEEDS="42" \
EPOCHS=100 \
DEVICE=cuda:0 \
bash scripts/run_supcon_warmup_sweep.sh
```

Summarize:

```bash
python scripts/summarize_supcon_warmup_sweep.py --output-root outputs_supcon_warmup_sweep
```

## Interpretation

If high lambda without warmup performs poorly but high lambda with warmup improves validation, the issue is likely early-stage optimization instability.

If warmup does not help and high lambda still hurts validation, the contrastive objective may be too strong or positives may be noisy.

If `loss_supcon_to_ce` remains `> 1.0` late in training, consider lowering lambda or increasing temperature.

Do not tune by repeatedly optimizing target test metrics. Choose settings from source validation behavior before broader held-out evaluation.
