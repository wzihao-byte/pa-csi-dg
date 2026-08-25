# SupCon Positive-Mask Sweep

## Purpose

This fourth-round sweep evaluates positive-pair definitions for pairwise SupCon. It checks whether cross-domain-only positives improve DG behavior compared with including same-domain same-class positives.

This round adds scripts, summarization, documentation, and diagnostics only. It does not change model architecture, dataset logic, leave-one-environment-out split logic, optimizer logic, CE loss, SupCon loss formula, or subcenter prototype logic.

## Swept Settings

- `losses.include_same_domain_same_class`: `true`, `false`
- `losses.supcon_positive_mode`: `all_views`, `global_class_instance_views`

## Fixed Defaults

- `losses.contrastive_loss_type = "pairwise_supcon"`
- `losses.lambda_supcon = 0.05`
- `losses.temperature = 0.2`
- `training.batch_size = 16`
- `training.sampler = "domain_class_balanced"`
- `target_env = E2`
- `seed = 42`
- `epochs = 80`

## Run Commands

Dry run:

```bash
DRY_RUN=1 bash scripts/run_supcon_positive_mask_sweep.sh
```

Actual run:

```bash
LAMBDAS="0.05 0.1" \
TEMPERATURES="0.1 0.2" \
BATCH_SIZE=16 \
SAMPLER=domain_class_balanced \
TARGET_ENV=E2 \
SEEDS="42" \
EPOCHS=80 \
DEVICE=cuda:0 \
bash scripts/run_supcon_positive_mask_sweep.sh
```

Summarize:

```bash
python scripts/summarize_supcon_positive_mask_sweep.py --output-root outputs_supcon_positive_mask_sweep
```

## Interpretation

`include_same_domain_same_class = true` is usually more stable because it uses all same-class positives, including same-domain same-class samples.

`include_same_domain_same_class = false` is more domain-invariance-oriented because it excludes same-domain same-class positives and depends more heavily on cross-domain same-class availability.

`supcon_positive_mode = all_views` applies a stronger contrastive constraint across global and antenna views.

`supcon_positive_mode = global_class_instance_views` is a more conservative class-level constraint and can help if `all_views` is too strong.

If `supcon_frac_anchors_with_positive` drops sharply when `include_same_domain_same_class=false`, the sampler or batch structure may not provide enough cross-domain positives.

Use source validation behavior for hyperparameter choice. Do not repeatedly optimize on target test metrics.
