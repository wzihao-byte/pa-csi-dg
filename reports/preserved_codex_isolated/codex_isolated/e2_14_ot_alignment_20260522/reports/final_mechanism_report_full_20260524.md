# Final Mechanism Report: E2 Class 1/4 Collapse

Summary source: `C:\Users\howie\Desktop\pa-csi-dg\codex_isolated\e2_14_ot_alignment_20260522\analysis\full_20260524_retry1`.

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
  "completed_seed_count": 3,
  "matched_baseline_seed_count": 0,
  "target_min_f1_1_4": {
    "mean": 0.588757624333577,
    "std": 0.05088822960849028,
    "n": 3
  },
  "source_val_min_f1_1_4": {
    "mean": 0.9695960882554502,
    "std": 0.013228185485906718,
    "n": 3
  },
  "target_c1_to_c4": {
    "mean": 93.33333333333333,
    "std": 23.24507116214819,
    "n": 3
  },
  "target_c4_to_c1": {
    "mean": 152.66666666666666,
    "std": 42.39496825489239,
    "n": 3
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
  "completed_seed_count": 3,
  "matched_baseline_seed_count": 3,
  "target_min_f1_1_4": {
    "mean": 0.543337147962597,
    "std": 0.04881257877591985,
    "n": 3
  },
  "source_val_min_f1_1_4": {
    "mean": 0.9508372035685954,
    "std": 0.00793800951652138,
    "n": 3
  },
  "target_c1_to_c4": {
    "mean": 101.0,
    "std": 4.58257569495584,
    "n": 3
  },
  "target_c4_to_c1": {
    "mean": 167.0,
    "std": 30.04995840263344,
    "n": 3
  },
  "paired_deltas": [
    {
      "seed": 42,
      "delta_target_min_f1_1_4": -0.06198038616851609,
      "delta_source_val_min_f1_1_4": -0.042429994343945565,
      "delta_c1_to_c4": -5.0,
      "delta_c4_to_c1": 24.0
    },
    {
      "seed": 52,
      "delta_target_min_f1_1_4": 0.017137523814701683,
      "delta_source_val_min_f1_1_4": -0.005169787213130883,
      "delta_c1_to_c4": 33.0,
      "delta_c4_to_c1": -33.0
    },
    {
      "seed": 62,
      "delta_target_min_f1_1_4": -0.0914185667591258,
      "delta_source_val_min_f1_1_4": -0.00867687250348792,
      "delta_c1_to_c4": -5.0,
      "delta_c4_to_c1": 52.0
    }
  ],
  "paired_delta_target_min_f1_1_4": {
    "mean": -0.04542047637098007,
    "std": 0.05614070857715599,
    "n": 3
  }
}
```
Prefiltered OT summary:
```json
{
  "completed_seed_count": 3,
  "matched_baseline_seed_count": 3,
  "target_min_f1_1_4": {
    "mean": 0.5504279521994828,
    "std": 0.03211580295233858,
    "n": 3
  },
  "source_val_min_f1_1_4": {
    "mean": 0.9588820084864575,
    "std": 0.019618355095371947,
    "n": 3
  },
  "target_c1_to_c4": {
    "mean": 72.66666666666667,
    "std": 55.148284953689476,
    "n": 3
  },
  "target_c4_to_c1": {
    "mean": 193.0,
    "std": 31.22498999199199,
    "n": 3
  },
  "paired_deltas": [
    {
      "seed": 42,
      "delta_target_min_f1_1_4": -0.07117224880382778,
      "delta_source_val_min_f1_1_4": -0.04802028632470101,
      "delta_c1_to_c4": -82.0,
      "delta_c4_to_c1": 104.0
    },
    {
      "seed": 52,
      "delta_target_min_f1_1_4": 0.022387935297811135,
      "delta_source_val_min_f1_1_4": 0.009823300389338097,
      "delta_c1_to_c4": 1.0,
      "delta_c4_to_c1": 5.0
    },
    {
      "seed": 62,
      "delta_target_min_f1_1_4": -0.06620470289626612,
      "delta_source_val_min_f1_1_4": 0.00605474662838501,
      "delta_c1_to_c4": 19.0,
      "delta_c4_to_c1": 12.0
    }
  ],
  "paired_delta_target_min_f1_1_4": {
    "mean": -0.03832967213409425,
    "std": 0.05264161868025301,
    "n": 3
  }
}
```

## Interpretation Checklist
- Unfiltered OT improves target `min_f1_1_4` with no source-val collapse: pending.
- Target prediction histogram avoids collapse: pending.
- Transport entropy and transported mass avoid degeneracy: pending.
- Prefiltered vs unfiltered gap indicates data-quality/noise-aware ceiling: pending.
- Label/semantic ceiling vs representation/alignment failure: pending.
