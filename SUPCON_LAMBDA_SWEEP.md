# SupCon Lambda Sweep

## Purpose

This first-round sweep isolates the effect of `losses.lambda_supcon` in the pairwise SupCon setup. It adds run infrastructure and diagnostics only; it does not change model architecture, dataset logic, DG split logic, antenna-view logic, SupCon implementation, subcenter prototype logic, or adaptive/dynamic margin behavior.

## Fixed Round-One Settings

Keep these settings fixed while sweeping lambda:

- `losses.temperature = 0.2`
- `losses.include_same_domain_same_class = true`
- `losses.supcon_positive_mode = "all_views"`
- `losses.contrastive_loss_type = "pairwise_supcon"`
- `target_env = E2`
- `seed = 42`
- short budget such as `80` epochs

Lambda grid:

```text
0.0, 0.025, 0.05, 0.1, 0.2, 0.4
```

## Run Commands

Dry run:

```bash
DRY_RUN=1 bash scripts/run_supcon_lambda_sweep.sh
```

Actual run:

```bash
TARGET_ENV=E2 \
SEEDS="42" \
EPOCHS=80 \
DEVICE=cuda:0 \
bash scripts/run_supcon_lambda_sweep.sh
```

Summarize:

```bash
python scripts/summarize_supcon_lambda_sweep.py --output-root outputs_supcon_lambda_sweep
```

## Interpreting `loss_supcon_to_ce`

`loss_supcon_to_ce` is the weighted SupCon contribution divided by CE:

```text
loss_supcon_to_ce < 0.05  => SupCon is probably too weak
0.1 to 0.5               => reasonable first search range
> 1.0                    => SupCon may be overpowering CE
```

Use this ratio as a diagnostic, not as a direct performance claim.

## Reporting Discipline

Do not repeatedly use target test metrics to hand-pick hyperparameters. For real reporting, choose settings based on source validation behavior, then rerun with `target_env=all` and multiple seeds.
