# SupCon Sweep Analysis

## Purpose

Use `scripts/analyze_supcon_sweeps.py` to aggregate all SupCon sweep outputs into one per-run CSV, grouped summary CSVs, a Markdown report, and optional plots.

This script only reads existing outputs. It does not run training, modify configs, or change model/loss behavior.

## Inputs

By default, the script scans these roots when they exist:

- `outputs_supcon_lambda_sweep`
- `outputs_supcon_batch_sampler_sweep`
- `outputs_supcon_temperature_sweep`
- `outputs_supcon_positive_mask_sweep`
- `outputs_supcon_warmup_sweep`
- `outputs_supcon_final_eval`
- `outputs_subcenter_supcon_sweep`

Each run directory should contain `metrics.json`; `resolved_config.json` is loaded when available.

To choose roots explicitly:

```bash
python scripts/analyze_supcon_sweeps.py \
  --output-roots outputs_supcon_lambda_sweep outputs_supcon_temperature_sweep \
  --out-dir analysis_supcon_sweeps
```

## Outputs

Default output directory:

```text
analysis_supcon_sweeps/
```

Main files:

- `all_supcon_runs.csv`
- `by_lambda.csv`
- `by_temperature.csv`
- `by_batch_sampler.csv`
- `by_positive_mask.csv`
- `by_warmup.csv`
- `by_subcenter.csv`
- `SUPCON_SWEEP_REPORT.md`

If `matplotlib` is available, simple plots are written to:

```text
analysis_supcon_sweeps/plots/
```

## Reading The Report

Use validation-driven sections first, especially top configurations by best validation accuracy and the grouped summaries.

The test-accuracy ranking is diagnostic only. Do not repeatedly tune hyperparameters based on held-out target test results.

After selecting settings from source validation behavior, run final multi-seed, multi-target evaluation once and report mean +/- std plus per-target-env results.
