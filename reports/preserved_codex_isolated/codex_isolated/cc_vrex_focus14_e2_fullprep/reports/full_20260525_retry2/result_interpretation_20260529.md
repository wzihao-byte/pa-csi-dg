# CC-VREx Full Validation Result

Run id: `full_20260525_retry2`

## Status
- Baseline seeds 42/52/62 completed.
- Proposed CC-VREx seeds 42/52/62 completed.
- Aggregation completed.
- Success gate: failed.

## Mean Metrics
| Case | Accuracy | Macro-F1 | min_f1_1_4 |
|---|---:|---:|---:|
| Baseline | 0.8779 | 0.8529 | 0.5996 |
| CC-VREx | 0.8339 | 0.8001 | 0.5530 |
| Delta | -0.0440 | -0.0528 | -0.0465 |

## Seed-Wise Delta
| Seed | Delta accuracy | Delta macro-F1 | Delta min_f1_1_4 |
|---:|---:|---:|---:|
| 42 | -0.0010 | +0.0038 | +0.0194 |
| 52 | -0.1447 | -0.1793 | -0.1708 |
| 62 | +0.0137 | +0.0170 | +0.0119 |

## Interpretation
CC-VREx showed small positive class 1/4 effects on seeds 42 and 62, but seed 52 collapsed badly enough that the matched 3-seed mean is negative across accuracy, macro-F1, min-class F1, and `min_f1_1_4`.

This validates the earlier smoke signal as unstable rather than robust. CC-VREx should not be treated as a successful fix for E2 class 1/4 collapse under this configuration.

## Decision
Stop this CC-VREx configuration as a main candidate. Keep the result as evidence that source-only class-conditional environment-risk regularization can help some seeds but is not reliable enough in the current setup.
