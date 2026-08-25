# H11 TPP level-8 ablation

## Question

Does adding the level-8 temporal pyramid bin improve the completed
Factorial-S32 H11 carrier?

The reference is the sealed nine-cell H11 matrix with effective TPP levels
`[1,2,4]`.  The candidate changes only the effective TPP levels to
`[1,2,4,8]`.  SupCon, residual scale, model carrier, data, split, sampler,
optimization, checkpoint selection, and delayed target evaluation stay fixed.

## Locked comparison

| Field | Reference | Candidate |
|---|---|---|
| Arm | H11 | H11 |
| TPP levels | `[1,2,4]` | `[1,2,4,8]` |
| Channel TPP levels | inherit `[1,2,4]` | inherit `[1,2,4,8]` |
| Pooling | max | max |
| Preserve global | true | true |
| Residual scale | 0.25 | 0.25 |
| SupCon | pairwise, lambda 0.05, temperature 0.2 | unchanged |
| Views | global plus three Tx views, fully coupled | unchanged |
| Carrier | batch 32, domain-class-balanced | unchanged |
| Training | 200 epochs, 150 steps/epoch | unchanged |
| Seeds | 42, 52, 62 | unchanged |
| LOEO targets | E1, E2, E3 | unchanged |

The candidate has 4,786,294 trainable parameters.  The reference has
3,721,334, so level 8 adds 1,064,960 parameters (+28.62%).

## Execution and pairing

There are nine new jobs, ordered by target and then seed.  Every completed
candidate job must match its sealed reference H11 cell on:

- train, validation, and target split hashes;
- all 960,000 ordered source sampler draws;
- 200 epochs and 30,000 optimizer steps;
- 100% SupCon positive-anchor coverage;
- source-validation checkpoint selection and strict reload;
- target access only after checkpoint freeze;
- exactly 3,000 ordered target predictions.

The existing Factorial-S32 bundle and outputs are read-only references.  This
experiment writes to a separate bundle and output root.

## Analysis

The primary estimand is the paired target macro-F1 difference
`L1248 - L124` over the nine target-by-seed cells.  The report also includes
paired accuracy, per-environment macro-F1, per-class F1, falling/turning pair
confusion, a paired bootstrap interval, and an exact sign-flip test.

The experiment is a mechanism study on the repeatedly inspected E1-E3 data.
It does not replace confirmation on new environments or subjects.
