# Subcenter Prototype SupCon Sweep

## Purpose

This optional later-round sweep evaluates subcenter prototype SupCon after pairwise SupCon lambda, temperature, batch composition, and positive-mask behavior are understood.

This round is backward-compatible and does not alter pairwise SupCon behavior. It uses existing config keys:

- `losses.contrastive_loss_type = "subcenter_prototype"`
- `losses.lambda_supcon`
- `losses.lambda_pair_margin`
- `model.prototype_num_subcenters`

## Swept Values

Default grid:

- `lambda_supcon`: `0.05, 0.1, 0.2`
- `lambda_pair_margin`: `0.0, 0.02, 0.05, 0.1`
- `prototype_num_subcenters`: `2, 3, 5`

Fixed defaults:

- `losses.temperature = 0.2`
- `training.batch_size = 16`
- `training.sampler = "domain_class_balanced"`
- `target_env = E2`
- `seed = 42`
- `epochs = 100`

## Effective Pair-Margin Weight

The logged diagnostic is:

```text
effective_pair_margin_weight ~= lambda_supcon * lambda_pair_margin
```

This is logging-only. It does not change the subcenter prototype loss computation.

## Run Commands

Dry run:

```bash
DRY_RUN=1 bash scripts/run_subcenter_supcon_sweep.sh
```

Actual run:

```bash
LAMBDAS="0.05 0.1 0.2" \
LAMBDA_PAIR_MARGINS="0.0 0.02 0.05 0.1" \
PROTOTYPE_NUM_SUBCENTERS="2 3 5" \
TARGET_ENV=E2 \
SEEDS="42" \
EPOCHS=100 \
DEVICE=cuda:0 \
bash scripts/run_subcenter_supcon_sweep.sh
```

Summarize:

```bash
python scripts/summarize_subcenter_supcon_sweep.py --output-root outputs_subcenter_supcon_sweep
```

## Interpretation

If increasing `lambda_pair_margin` has no effect, the effective weight may be too small or the configured margins may not be active.

If large `lambda_supcon` and large `lambda_pair_margin` hurt validation, the prototype/margin objective may be overpowering CE.

If `k=5` improves training but not validation, subcenters may be overfitting source-domain structure.

Do not pick subcenter hyperparameters by repeatedly optimizing target test metrics. Use source validation behavior first, then run final multi-seed, multi-target evaluation once.
