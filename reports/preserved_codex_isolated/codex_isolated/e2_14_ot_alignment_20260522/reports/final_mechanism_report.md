# Final Mechanism Report: E2 Class 1/4 Collapse

## Question 1: Is E2 1/4 collapse an alignable shift?
Fill from Phase 0 manual review plus Phase 0.5 probe summary.

## Phase 0.5 Probe Summary
```json
{
  "source_lda_projection:source_val:auc": {
    "mean": 1.0,
    "std": 0.0,
    "n": 3
  },
  "source_lda_projection:source_val:class_mean_gap": {
    "mean": 2.3171518643697104,
    "std": 0.43346927376082806,
    "n": 3
  },
  "source_lda_projection:source_val:fisher_ratio": {
    "mean": 649.3329380486704,
    "std": 284.5469739479371,
    "n": 3
  },
  "source_lda_projection:source_val:hist_overlap": {
    "mean": 0.0,
    "std": 0.0,
    "n": 3
  },
  "source_lda_projection:target_E2:auc": {
    "mean": 0.7196395833333333,
    "std": 0.056732977324884255,
    "n": 3
  },
  "source_lda_projection:target_E2:class_mean_gap": {
    "mean": 0.7765912810961405,
    "std": 0.3206099794660572,
    "n": 3
  },
  "source_lda_projection:target_E2:fisher_ratio": {
    "mean": 0.3358076733431838,
    "std": 0.19136895006584884,
    "n": 3
  },
  "source_lda_projection:target_E2:hist_overlap": {
    "mean": 0.6108332842686999,
    "std": 0.07229856030946963,
    "n": 3
  },
  "source_logistic_projection:source_val:acc": {
    "mean": 0.9989583333333333,
    "std": 0.0018042195912176063,
    "n": 3
  },
  "source_logistic_projection:source_val:auc": {
    "mean": 1.0,
    "std": 0.0,
    "n": 3
  },
  "source_logistic_projection:target_E2:acc": {
    "mean": 0.6675,
    "std": 0.03520031960082186,
    "n": 3
  },
  "source_logistic_projection:target_E2:auc": {
    "mean": 0.7042229166666666,
    "std": 0.05817356584412581,
    "n": 3
  },
  "source_pca_logistic_fused_feature:source_val:acc": {
    "mean": 0.9989583333333333,
    "std": 0.0018042195912176063,
    "n": 3
  },
  "source_pca_logistic_fused_feature:source_val:auc": {
    "mean": 1.0,
    "std": 0.0,
    "n": 3
  },
  "source_pca_logistic_fused_feature:target_E2:acc": {
    "mean": 0.6895833333333333,
    "std": 0.049154306355937254,
    "n": 3
  },
  "source_pca_logistic_fused_feature:target_E2:auc": {
    "mean": 0.7360427083333333,
    "std": 0.04119691545612311,
    "n": 3
  },
  "source_pca_logistic_raw_temporal_mean:source_val:acc": {
    "mean": 0.853125,
    "std": 0.008267972847076838,
    "n": 3
  },
  "source_pca_logistic_raw_temporal_mean:source_val:auc": {
    "mean": 0.8947786458333333,
    "std": 0.015136420527976192,
    "n": 3
  },
  "source_pca_logistic_raw_temporal_mean:target_E2:acc": {
    "mean": 0.5879166666666666,
    "std": 0.002602082499332673,
    "n": 3
  },
  "source_pca_logistic_raw_temporal_mean:target_E2:auc": {
    "mean": 0.5388729166666667,
    "std": 0.006530042751263846,
    "n": 3
  }
}
```

## Question 2: Does OT improve representation geometry?
Use `centroid_distance_trajectory.csv` and `transport_diagnostics.csv` from matched OT runs.

## Question 3: Does geometry translate to F1?
Baseline summary:
```json
{
  "completed_seed_count": 1,
  "matched_baseline_seed_count": 0,
  "target_min_f1_1_4": {
    "mean": 0.6577777777777778,
    "std": 0.0,
    "n": 1
  },
  "source_val_min_f1_1_4": {
    "mean": 0.904191616766467,
    "std": 0.0,
    "n": 1
  },
  "target_c1_to_c4": {
    "mean": 30.0,
    "std": 0.0,
    "n": 1
  },
  "target_c4_to_c1": {
    "mean": 171.0,
    "std": 0.0,
    "n": 1
  },
  "paired_deltas": [],
  "paired_delta_target_min_f1_1_4": {
    "mean": 0.0,
    "std": 0.0,
    "n": 0
  }
}
```
Unfiltered OT summary:
```json
{
  "completed_seed_count": 1,
  "matched_baseline_seed_count": 1,
  "target_min_f1_1_4": {
    "mean": 0.41134751773049644,
    "std": 0.0,
    "n": 1
  },
  "source_val_min_f1_1_4": {
    "mean": 0.888888888888889,
    "std": 0.0,
    "n": 1
  },
  "target_c1_to_c4": {
    "mean": 41.0,
    "std": 0.0,
    "n": 1
  },
  "target_c4_to_c1": {
    "mean": 269.0,
    "std": 0.0,
    "n": 1
  },
  "paired_deltas": [
    {
      "seed": 42,
      "delta_target_min_f1_1_4": -0.24643026004728136,
      "delta_source_val_min_f1_1_4": -0.015302727877578026,
      "delta_c1_to_c4": 11.0,
      "delta_c4_to_c1": 98.0
    }
  ],
  "paired_delta_target_min_f1_1_4": {
    "mean": -0.24643026004728136,
    "std": 0.0,
    "n": 1
  }
}
```
Prefiltered OT summary:
```json
{
  "completed_seed_count": 1,
  "matched_baseline_seed_count": 1,
  "target_min_f1_1_4": {
    "mean": 0.5003408316291752,
    "std": 0.0,
    "n": 1
  },
  "source_val_min_f1_1_4": {
    "mean": 0.841823056300268,
    "std": 0.0,
    "n": 1
  },
  "target_c1_to_c4": {
    "mean": 13.0,
    "std": 0.0,
    "n": 1
  },
  "target_c4_to_c1": {
    "mean": 217.0,
    "std": 0.0,
    "n": 1
  },
  "paired_deltas": [
    {
      "seed": 42,
      "delta_target_min_f1_1_4": -0.1574369461486026,
      "delta_source_val_min_f1_1_4": -0.062368560466198986,
      "delta_c1_to_c4": -17.0,
      "delta_c4_to_c1": 46.0
    }
  ],
  "paired_delta_target_min_f1_1_4": {
    "mean": -0.1574369461486026,
    "std": 0.0,
    "n": 1
  }
}
```

## Interpretation Checklist
- Unfiltered OT improves target `min_f1_1_4` with no source-val collapse: pending.
- Target prediction histogram avoids collapse: pending.
- Transport entropy and transported mass avoid degeneracy: pending.
- Prefiltered vs unfiltered gap indicates data-quality/noise-aware ceiling: pending.
- Label/semantic ceiling vs representation/alignment failure: pending.
