# SupCon Final Evaluation Results

Generated: 2026-06-13 13:29:43 +09:00

## Configuration

- lambda_supcon=0.05
- lambda_supcon_warmup_epochs=0
- temperature=0.2
- batch_size=32
- sampler=domain_class_balanced
- supcon_positive_mode=global_class_instance_views
- include_same_domain_same_class=true
- target_env=all -> E1, E2, E3; seeds=42 52 62; epochs=200
- expected final runs=9

## Per-Run Results

| target | seed | best val acc | test acc | test F1 | SupCon/CE | pos frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 | 42 | 0.9983 | 0.8967 | 0.8632 | 0.1442 | 1.000 |
| E1 | 52 | 1.0000 | 0.9030 | 0.8827 | 0.2067 | 1.000 |
| E1 | 62 | 0.9950 | 0.8850 | 0.8621 | 0.1459 | 1.000 |
| E2 | 42 | 0.9992 | 0.8580 | 0.8307 | 0.1936 | 1.000 |
| E2 | 52 | 0.9992 | 0.8497 | 0.8132 | 0.1668 | 1.000 |
| E2 | 62 | 0.9983 | 0.8557 | 0.8372 | 0.1521 | 1.000 |
| E3 | 42 | 0.9992 | 0.8297 | 0.8238 | 0.1181 | 1.000 |
| E3 | 52 | 0.9992 | 0.8410 | 0.8350 | 0.0865 | 1.000 |
| E3 | 62 | 0.9967 | 0.8600 | 0.8519 | 0.1554 | 1.000 |

## Aggregate Results

| scope | target | runs | mean acc | std acc | mean F1 | std F1 | mean val acc | mean SupCon/CE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | all | 9 | 0.8643 | 0.0252 | 0.8444 | 0.0221 | 0.9983 | 0.1521 |
| target_env | E1 | 3 | 0.8949 | 0.0091 | 0.8694 | 0.0116 | 0.9978 | 0.1656 |
| target_env | E2 | 3 | 0.8544 | 0.0043 | 0.8271 | 0.0124 | 0.9989 | 0.1708 |
| target_env | E3 | 3 | 0.8436 | 0.0153 | 0.8369 | 0.0142 | 0.9983 | 0.1200 |

## Notes

- These are final held-out target results for the selected configuration, not another hyperparameter search round.
- Do not use these target metrics to hand-pick another SupCon hyperparameter setting.
