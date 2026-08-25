# C3 Recovery Experiment

## Objective

Recover walking (`C3`) without giving back the useful `C1`/`C4` hard-pair improvement from TPP `{1,2,4}`.

The existing `{1,2,4}` result is the negative control:

- mean C3 recall delta: `-0.342`
- mean C3 F1 delta: `-0.246`
- mean C1 F1 delta: `+0.121`
- mean C1/C4 pair-confusion reduction: about `22%`

Most new C3 errors go to C0 and C4. The experiment therefore tests whether fixed-bin TPP has replaced the shift-invariant global detector with source-domain timing features.

## Phase 1: Causal Architecture Screen

Keep the data split, CE loss, sampler, optimizer, scheduler, model width, TPP levels, and seed matching unchanged. Change one pooling property at a time.

| Candidate | Change from TPP `{1,2,4}` | Hypothesis |
|---|---|---|
| `tpp_c3_residual025_l124` | output = global max + `0.25 * TPP` | A weak local correction restores shift-invariant C3 detection while retaining C1 timing information. |
| `tpp_c3_residual050_l124` | output = global max + `0.50 * TPP` | A stronger local correction gives a better C1/C3 tradeoff. |
| `tpp_c3_mixed050_l124` | each bin = `0.5 * max + 0.5 * average` | Sustained walking needs bin occupancy/energy, not only the largest local response. |

Screen on seeds 52 and 62 across E1/E2/E3. These seeds are intentionally used first because their CE baselines learn C3 strongly; seed 42 already has weak C3 on E2/E3 and is less diagnostic of recovery.

```powershell
python experiments/tpp_isolated/run_tpp_c3_recovery_matrix.py --stage screening
```

This schedules 18 runs: 3 candidates x 3 targets x 2 seeds.

Analyze completed or partial results with:

```powershell
python experiments/tpp_isolated/analyze_tpp_c3_recovery.py
```

## Phase 1 Gates

A candidate advances only if the screening trend satisfies all diagnostic gates:

- mean macro-F1 delta versus CE baseline >= `-0.01`
- mean C3 F1 delta >= `-0.05`
- mean C1 F1 delta >= `+0.05`
- relative C1/C4 pair-confusion reduction >= `10%`
- mean C0 precision delta >= `-0.03`
- E1 pair confusion improves on both screening seeds

The analyzer reports diagnostic gate status but will not mark a candidate promotion-ready until all nine target/seed combinations are complete.

## Confirmation

Run only the best Phase 1 candidate over the full seed matrix:

```powershell
python experiments/tpp_isolated/run_tpp_c3_recovery_matrix.py `
  --stage confirmation `
  --experiments residual025
```

Replace `residual025` with `residual050` or `mixed050` if that candidate wins. Completed screening runs are reused, so confirmation adds only seed 42.

Do not select the winner from overall accuracy alone. Use the full per-class report and require E1 improvement in at least two of three seeds.

## Phase 2: Training Robustness, Only After an Architecture Winner

Apply these one at a time to the winning Phase 1 architecture:

1. `domain_class_balanced` sampler, to test whether C0 prevalence amplifies C3-to-C0 errors.
2. AdamW with weight decay `1e-4`, keeping dropout at `0.1`.
3. Dropout `0.2`, keeping optimizer and weight decay matched to the Phase 1 winner.
4. Joint amplitude/phase temporal shift augmentation of up to 5% of the window.

Only combine changes after their isolated effects are known. A class-balanced or regularized model is accepted only if it improves C3 without reducing the Phase 1 winner's C1 gain or hard-pair reduction below the gates.

## Deferred Resolution Study

TPP `{1,2,4,8}` is deferred until C3 passes the recovery gates. In the current implementation it increases total parameters by about 28.6% and more than doubles pooling-projection parameters, so an uncontrolled `{1,2,4,8}` run would confound temporal resolution with capacity.

If tested later, use a parameter-matched projection and compare:

- temporal levels `{1,2,4,8}`, channel levels `{1,2,4}`
- the winning global-residual or mixed-pooling mechanism
- the same three targets and three seeds

## Required Outputs

The analyzer writes:

- `analysis_tpp_c3_recovery/c3_recovery_per_run.csv`
- `analysis_tpp_c3_recovery/c3_recovery_per_class.csv`
- `analysis_tpp_c3_recovery/c3_recovery_summary.csv`
- `analysis_tpp_c3_recovery/C3_RECOVERY_REPORT.md`

The report includes all six classes, C3-to-C0/C4 flows, E1 seed consistency, gate status, and final promotion readiness.
