# SupCon Temperature Sweep

## Purpose

This third-round sweep tunes `losses.temperature` for pairwise SupCon after selecting candidate lambda and batch/sampler settings. It keeps lambda, batch size, sampler, positive mask, and positive mode fixed while varying temperature.

This round adds run infrastructure only. It does not change model architecture, dataset logic, DG split logic, antenna-view logic, optimizer logic, SupCon loss formula, or subcenter prototype logic.

## Fixed Settings

Keep these fixed unless explicitly overridden:

- `losses.contrastive_loss_type = "pairwise_supcon"`
- `losses.include_same_domain_same_class = true`
- `losses.supcon_positive_mode = "all_views"`
- `losses.lambda_supcon`: selected candidates, default `0.05, 0.1`
- `training.batch_size = 16`
- `training.sampler = "domain_class_balanced"`
- `target_env = E2`
- `seed = 42`
- `epochs = 80`

Temperature grid:

```text
0.07, 0.1, 0.2, 0.3
```

## Run Commands

Dry run:

```bash
DRY_RUN=1 bash scripts/run_supcon_temperature_sweep.sh
```

Actual run:

```bash
LAMBDAS="0.05 0.1" \
TEMPERATURES="0.07 0.1 0.2 0.3" \
BATCH_SIZE=16 \
SAMPLER=domain_class_balanced \
TARGET_ENV=E2 \
SEEDS="42" \
EPOCHS=80 \
DEVICE=cuda:0 \
bash scripts/run_supcon_temperature_sweep.sh
```

Summarize:

```bash
python scripts/summarize_supcon_temperature_sweep.py --output-root outputs_supcon_temperature_sweep
```

## Interpretation

Lower temperature makes the contrastive softmax sharper and emphasizes hard negatives more. Higher temperature smooths the contrastive distribution and can stabilize training.

If validation accuracy is unstable at low temperature, prefer `0.2` or `0.3`.

If SupCon loss decreases but accuracy does not improve, inspect `loss_supcon_to_ce`.

If `loss_supcon_to_ce > 1.0`, the contrastive objective may be overpowering CE.

Do not repeatedly select hyperparameters using held-out target test accuracy. For reporting, choose settings from source validation behavior and then rerun broader held-out evaluation.
