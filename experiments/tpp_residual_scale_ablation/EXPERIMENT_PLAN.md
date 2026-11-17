# H11 L1248 residual-scale sensitivity ablation

## Question

How sensitive is the completed Factorial-S32 H11 L1248 system to the
residual coefficient in

`h = g + alpha * t`?

The sealed `alpha=0.25` H11 L1248 matrix is the reference.  Four new values
are evaluated: `0.125`, `0.5`, `0.75`, and `1.0`.

## Locked matrix

| Field | Value |
|---|---|
| Arm | H11 only (TPP + SupCon) |
| TPP levels | `[1,2,4,8]` on temporal and channel paths |
| Residual scale | `0.125`, `0.5`, `0.75`, `1.0` |
| Reference scale | sealed `0.25` results; not rerun |
| Pooling | max, preserve global |
| SupCon | pairwise, lambda `0.05`, temperature `0.2` |
| Views | global plus three Tx views, fully coupled |
| Carrier | batch 32, domain-class-balanced |
| Training | 200 epochs, 150 steps/epoch, 30,000 optimizer steps |
| Seeds | 42, 52, 62 |
| LOEO targets | E1, E2, E3 |
| New jobs | 4 scales x 3 targets x 3 seeds = 36 |

Only `experiment_name`, `job_id`, `model.tpp_pyramid_scale`,
`factorial.classifier_representation`, and `protocol.version` may differ
from the paired sealed `alpha=0.25` H11 L1248 configuration.

## Execution and pairing

Jobs are ordered by target, seed, then scale.  After the four new scales in
each target-by-seed cell complete, a fail-closed paired barrier requires:

- identical train, validation, and target split hashes;
- the same ordered 960,000 source sampler draws as the sealed 0.25 run;
- 200 epochs and exactly 30,000 optimizer steps;
- 100% SupCon positive-anchor coverage;
- source-validation checkpoint selection and strict reload;
- target access only after checkpoint freeze;
- exactly 3,000 ordered target predictions;
- exactly 4,786,294 trainable parameters for every scale.

Completed 0.25 bundles and outputs are read-only references.  New artifacts
are append-only under `outputs_tpp_residual_scale_ablation/h11_l1248_v1`.
The runner never automatically restarts, skips, or repairs a failed job.

## Analysis

Each new scale is paired against 0.25 over the same nine target-by-seed
cells.  The report includes overall and per-environment macro-F1, all six
classes, paired bootstrap 95% intervals, exact two-sided sign-flip tests,
and Holm correction across the four comparisons.  The complete five-point
response curve is reported; the maximum observed target score is not used
as a confirmatory hyperparameter-selection claim.

This is a post-hoc sensitivity study on repeatedly inspected E1-E3 target
data.  H11-only results describe sensitivity of the combined system and do
not isolate the TPP main effect or the SupCon-by-TPP interaction.
