# Why We Abandon the E2 Multi-Center Plan

## Conclusion

The class 1 vs. class 4 confusion on E2 is not primarily a model architecture problem. The evidence shows that, in the current representation, target-domain class 1 and class 4 collapse into a much tighter shared region than they do in the source domains. Source class 1 and class 4 are well separated under K=1, but E2 target class 1/4 centroids shrink to roughly one quarter of the source separation. At the same time, warmup k-means splits are unstable across seeds, and the trained multi-center runs collapse prototype occupancy back toward a single active center.

Therefore, the following directions should be stopped:

- multi-center geometry / sub-center prototype repair
- occupancy floor loss tuning
- class 1<->4 asymmetric margin grids, including m04-style margin grids
- chasing single-seed historical K3 gains

The research question should be reframed as: why do E2 target class 1 and class 4 collapse in the embedding space, and is the root cause label noise or domain shift?

## Evidence Sources

The report uses the following artifacts:

- `outputs_geometry_balance_e2_200e/aggregate/geometry_multiseed_selection.json`
- `outputs_k3_current_baseline_distribution/aggregate/k3_current_baseline_distribution.json`
- `analysis/data_issue_diagnostics/data_issue_diagnostics_report.json`
- `analysis/data_issue_diagnostics/kmeans_init_counts.csv`
- `analysis/data_issue_diagnostics/proxy_k1_kmeans_cross_seed_overlap.csv`

## 1. K=3 Is Not Significantly Better Than K=1

Under the current codebase, E2, 200 epochs, and seeds 42/52/62, K3 and K1 are statistically tied.

| Metric | K3 - K1 mean delta | std | t(2) | p-value |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | -0.0033 | 0.0153 | -0.3768 | 0.7425 |
| Macro-F1 | -0.0030 | 0.0138 | -0.3819 | 0.7393 |

Mathematical interpretation: if the E2 1/4 issue were caused by stable, learnable intra-class submodes, K3 should provide reproducible gains over K1. Instead, the paired deltas are near zero and the p-values are far above 0.05. Extra prototype capacity is not contributing reproducible discriminative information.

## 2. Target Class 1/4 Centroids Collapse

K1 seed 42 centroid cosine distances:

| Pair | Cosine distance |
| --- | ---: |
| source class 1 <-> source class 4 | 1.2177 |
| target class 1 <-> target class 4 | 0.2722 |
| source class 1 <-> target class 1 | 0.2625 |
| source class 4 <-> target class 4 | 0.0481 |

Mathematical interpretation:

- Source class 1 and class 4 are well separated: distance 1.2177.
- Target class 1 and class 4 are much closer: distance 0.2722.
- The target distance is only 22.4% of the source distance, or a 4.47x compression.
- Source class 4 and target class 4 are strongly aligned: distance 0.0481.
- Source class 1 and target class 1 are still closer than source class 1 and source class 4, but target class 1 lies near the collapsed 1/4 target region.

The same pattern holds across seeds:

| Seed | source 1<->4 | target 1<->4 | target 1 closer to source 4 | target 4 closer to source 1 |
| --- | ---: | ---: | ---: | ---: |
| 42 | 1.2177 | 0.2722 | 40.75% | 17.75% |
| 52 | 1.2045 | 0.1528 | 41.75% | 31.25% |
| 62 | 1.2540 | 0.2076 | 31.00% | 38.00% |

This is the strongest reason to abandon the multi-center plan. The problem is not that the source class needs more internal centers. The problem is that E2 target class 1/4 embeddings are collapsed into a shared region. Source-side multi-center structure or source-side margin engineering cannot reliably unfold a collapsed target representation.

## 3. Warmup K-Means Does Not Find Stable Submodes

If class 1 or class 4 contained two stable submodes, warmup k-means should produce similar splits across seeds. Instead, minority fractions drift heavily:

| Variant | Class | Seed 42 | Seed 52 | Seed 62 | Range |
| --- | ---: | ---: | ---: | ---: | ---: |
| geom_balance | 1 | 40.00% | 0.78% | 26.09% | 39.22 pp |
| geom_balance | 4 | 7.97% | 13.13% | 34.69% | 26.72 pp |
| geom_balance_div | 1 | 31.56% | 0.16% | 30.16% | 31.41 pp |
| geom_balance_div | 4 | 0.31% | 6.72% | 27.97% | 27.66 pp |

Mathematical interpretation: a real two-mode structure should produce comparable split ratios across seeds. Here the same class can move from almost K=1 behavior, such as 0.16% minority, to a large minority split such as 31.56%. This is consistent with k-means cutting a continuous distribution at seed-dependent positions, not recovering stable substructure.

A K1 embedding proxy gives the same qualitative signal, though it should be treated as supporting evidence rather than primary evidence because K1 itself encourages single-center structure:

| Class | Seed pair | Minority-cluster Jaccard |
| ---: | --- | ---: |
| 1 | 42-52 | 0.0000 |
| 1 | 42-62 | 0.0000 |
| 1 | 52-62 | 0.0621 |
| 4 | 42-52 | 0.2145 |
| 4 | 42-62 | 0.0327 |
| 4 | 52-62 | 0.0691 |

## 4. Training Dynamics Collapse the Multi-Center Structure

Both `geom_balance` and `geom_balance_div` aborted at epoch 100 for all three seeds because source-val active occupancy dropped below 0.05.

Mathematical interpretation: if two sub-centers were discriminatively useful, cross-entropy and prototype gradients should maintain both centers because they would attract different samples. Instead, active occupancy collapses. The optimization process is saying that the initialized two-center structure is redundant for the learned decision rule.

This agrees with the warmup k-means instability: the proposed multi-center structure is not a stable, useful data structure.

## 5. Misclassifications Are Not Low-Confidence Boundary Cases

K1 target pair diagnostics:

| Direction | Counts by seed | Mean count | Mean frac max_conf < 0.5 | Mean closer to predicted source centroid |
| --- | --- | ---: | ---: | ---: |
| 1 -> 4 | 157 / 146 / 85 | 129.33 | 0.85% | 97.27% |
| 4 -> 1 | 61 / 123 / 150 | 111.33 | 0.00% | 93.30% |

Source 5-nearest-neighbor evidence is consistent:

| Seed | Direction | Mean predicted-label neighbors in source 5-NN |
| ---: | --- | ---: |
| 42 | 1 -> 4 | 4.60 / 5 |
| 52 | 1 -> 4 | 4.82 / 5 |
| 62 | 1 -> 4 | 4.87 / 5 |
| 42 | 4 -> 1 | 4.75 / 5 |
| 52 | 4 -> 1 | 4.89 / 5 |
| 62 | 4 -> 1 | 4.55 / 5 |

Mathematical interpretation: low-confidence mistakes would suggest boundary uncertainty. These mistakes are high-confidence, closer to the predicted source centroid, and surrounded by predicted-class source neighbors. The model is not merely uncertain at the decision boundary. In the learned embedding geometry, these target samples genuinely look like the predicted class.

## 6. Confusion Asymmetry Is Not Stable Across Seeds

K1 class 1<->4 directional counts:

| Seed | 1 -> 4 | 4 -> 1 |
| ---: | ---: | ---: |
| 42 | 157 | 61 |
| 52 | 146 | 123 |
| 62 | 85 | 150 |

Mathematical interpretation: a stable asymmetric boundary problem should preserve its direction across seeds. Seed 62 reverses the direction. Therefore, single-seed asymmetry, such as 1->4 being much larger than 4->1, is not a reliable engineering target. An asymmetric margin grid would likely chase seed noise rather than a stable defect.

## 7. Ensemble Mismatch Is Concentrated in Class 1/4

K1 unanimous high-confidence mismatches across three independent seeds:

| True class | Total | Unanimous mismatch | Fraction | Main predicted class |
| ---: | ---: | ---: | ---: | --- |
| 0 | 1200 | 1 | 0.08% | 4 |
| 1 | 400 | 77 | 19.25% | 75 -> 4 |
| 2 | 400 | 16 | 4.00% | 14 -> 1, 2 -> 4 |
| 3 | 400 | 0 | 0.00% | - |
| 4 | 400 | 50 | 12.50% | 50 -> 1 |
| 5 | 200 | 1 | 0.50% | 0 |

Mathematical interpretation: classes 0, 3, and 5 have a unanimous high-confidence mismatch baseline of 0-0.5%. Class 1 and class 4 reach 19.25% and 12.50%, roughly 25-40x higher than the clean-class baseline. The abnormality is concentrated in class 1/4, not distributed across all classes.

Important caveat: this metric does not distinguish label noise from covariate shift. It only says that, under the geometry learned from source domains, these E2 samples consistently land in the opposite-class region. Possible causes include:

- systematic E2 label errors;
- E2 class 1/4 physical CSI patterns overlapping with the opposite source class;
- subject, environment, hardware, or time-specific collection shift.

In all cases, source-side multi-center repair is the wrong next step.

## Overall Mathematical Decision

The multi-center plan assumes that class 1/4 contains stable, reproducible, discriminatively useful intra-class submodes, and that K>1 prototypes can capture those submodes to reduce 1<->4 confusion.

The evidence rejects this assumption:

| Evidence | Mathematical meaning | What it rejects |
| --- | --- | --- |
| K3-K1 accuracy delta = -0.0033, p = 0.7425 | No significant K3 gain | Extra centers have no reproducible value |
| target 1<->4 distance = 0.27 vs source 1<->4 = 1.22 | Target class 1/4 collapse | The issue is not insufficient source centers |
| warmup minority fractions drift heavily across seeds | K-means split is unstable | No stable submode |
| epoch-100 occupancy collapse | Gradients do not maintain two centers | Sub-centers are not classification-useful |
| mistakes are high-confidence and closer to predicted centroids | Errors match embedding geometry | Not boundary hesitation |
| unanimous mismatch is much higher for class 1/4 | The anomaly is class-pair specific | Not general model instability |

The multi-center direction should therefore be formally abandoned. The problem should be reframed from "fix source class 1/4 boundaries or intra-class structure" to "diagnose and repair E2 target class 1/4 representation collapse."

## Recommended Next Step

The next step should be manual review, not another training run. Sample 20 unanimous high-confidence mismatch cases from `analysis/data_issue_diagnostics/k1_ensemble_target_unanimous_highconf_mismatches.csv`, balanced between class 1->4 and class 4->1. Review raw CSI and collection metadata.

Decision split:

- If raw CSI or metadata shows that these samples genuinely resemble the predicted class, prioritize E2 label audit, relabeling, or removal of corrupted samples.
- If raw CSI visually matches the ground truth but the embedding maps it to the predicted class, treat the problem as representation-level domain shift and move to target alignment.

If the path is domain alignment, suggested priority is:

1. target pseudo-label + prototype refinement;
2. class-1/4 targeted MMD or CORAL alignment;
3. DANN-style domain discriminator.

The success metric should also change. Do not optimize multi-center occupancy. Instead, directly track whether target class 1/4 centroid distance recovers from the current 0.15-0.27 range toward 0.7-1.0 or higher, and whether 1<->4 confusion decreases accordingly.

