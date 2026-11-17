# H10 TPP level-8 ablation

## Question

Does adding the level-8 temporal pyramid bin improve the completed
Factorial-S32 H10 carrier without SupCon?

The reference is the sealed nine-cell H10 matrix with effective TPP levels
`[1,2,4]`. The candidate changes only the effective TPP levels to
`[1,2,4,8]`. The classifier, residual scale, data, split, sampler,
optimization, checkpoint selection, and delayed target evaluation remain
fixed. SupCon remains disabled.

## Locked comparison

| Field | Reference | Candidate |
|---|---|---|
| Arm | H10 | H10 |
| TPP levels | `[1,2,4]` | `[1,2,4,8]` |
| Pooling | max | max |
| Preserve global | true | true |
| Residual scale | 0.25 | 0.25 |
| SupCon | off | off |
| Carrier | batch 32, domain-class-balanced | unchanged |
| Training | 200 epochs, 150 steps/epoch | unchanged |
| Seeds | 42, 52, 62 | unchanged |
| LOEO targets | E1, E2, E3 | unchanged |

The candidate has 4,786,294 trainable parameters. The reference has
3,721,334, so level 8 adds 1,064,960 parameters (+28.62%).

H00 and H01 are not rerun because TPP is disabled in those arms; a TPP-level
setting has no effect on their model. Their sealed Factorial-S32 results remain
the correct common-carrier controls. The completed H11-L1248 matrix is reused
as the SupCon-on counterpart.

Every candidate cell must match its H10 reference on split hashes, all 960,000
ordered sampler draws, 200 epochs, 30,000 optimizer steps, checkpoint rule,
and 3,000 ordered target predictions.

