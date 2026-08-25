# SupCon Sweep Report

This report aggregates SupCon sweep outputs across available result roots.

Source validation behavior should drive hyperparameter selection. Target test rankings below are diagnostic only and should not be used for repeated hand-tuning.

## Scan Summary

- Total runs found: 44
- Roots scanned: outputs_supcon_lambda_sweep, outputs_supcon_batch_sampler_sweep, outputs_supcon_temperature_sweep, outputs_supcon_positive_mask_sweep, outputs_supcon_warmup_sweep
- Missing roots: None

## Top Configurations By Best Validation Accuracy

| experiment_name | contrastive_loss_type | lambda_supcon | temperature | batch_size | sampler_mode | mean_best_val_accuracy | mean_test_accuracy | num_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pa_csi_dg_supcon_lam0050_bs32_domain_class_balanced_tau020 | pairwise_supcon | 0.0500 | 0.2000 | 32.0000 | domain_class_balanced | 1.0000 | 0.8630 | 1.0000 |
| pa_csi_dg_supcon_lam0100_warm040_tau020_bs32_domain_class_balanced | pairwise_supcon | 0.1000 | 0.2000 | 32.0000 | domain_class_balanced | 1.0000 | 0.8490 | 1.0000 |
| pa_csi_dg_supcon_lam0200_bs32_domain_balanced_tau020 | pairwise_supcon | 0.2000 | 0.2000 | 32.0000 | domain_balanced | 1.0000 | 0.8990 | 1.0000 |
| pa_csi_dg_supcon_lam0200_bs32_domain_class_balanced_tau020 | pairwise_supcon | 0.2000 | 0.2000 | 32.0000 | domain_class_balanced | 1.0000 | 0.8757 | 1.0000 |
| pa_csi_dg_supcon_lam0200_bs32_none_tau020 | pairwise_supcon | 0.2000 | 0.2000 | 32.0000 | none | 1.0000 | 0.8863 | 1.0000 |
| pa_csi_dg_supcon_lam0200_warm000_tau020_bs32_domain_class_balanced | pairwise_supcon | 0.2000 | 0.2000 | 32.0000 | domain_class_balanced | 1.0000 | 0.8727 | 1.0000 |
| pa_csi_dg_supcon_lam0200_warm010_tau020_bs32_domain_class_balanced | pairwise_supcon | 0.2000 | 0.2000 | 32.0000 | domain_class_balanced | 1.0000 | 0.8860 | 1.0000 |
| pa_csi_dg_supcon_lam0050_bs32_none_tau020 | pairwise_supcon | 0.0500 | 0.2000 | 32.0000 | none | 0.9992 | 0.8803 | 1.0000 |
| pa_csi_dg_supcon_lam0050_tau020_bs32_domain_class_balanced | pairwise_supcon | 0.0500 | 0.2000 | 32.0000 | domain_class_balanced | 0.9992 | 0.8917 | 1.0000 |
| pa_csi_dg_supcon_lam0050_tau020_bs32_domain_class_balanced_incSame0_global_class_instance_views | pairwise_supcon | 0.0500 | 0.2000 | 32.0000 | domain_class_balanced | 0.9992 | 0.8490 | 1.0000 |


## Top Configurations By Test Accuracy (Diagnostic Only)

These rows are for diagnostics only. Do not use target test rankings for repeated hyperparameter selection.

| experiment_name | contrastive_loss_type | lambda_supcon | temperature | batch_size | sampler_mode | mean_test_accuracy | mean_best_val_accuracy | num_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pa_csi_dg_supcon_lam0400_tau020 | pairwise_supcon | 0.4000 | 0.2000 | 16.0000 | domain_class_balanced | 0.9207 | 0.9983 | 1.0000 |
| pa_csi_dg_supcon_lam0200_bs32_domain_balanced_tau020 | pairwise_supcon | 0.2000 | 0.2000 | 32.0000 | domain_balanced | 0.8990 | 1.0000 | 1.0000 |
| pa_csi_dg_supcon_lam0050_tau020 | pairwise_supcon | 0.0500 | 0.2000 | 16.0000 | domain_class_balanced | 0.8937 | 0.9983 | 1.0000 |
| pa_csi_dg_supcon_lam0200_bs16_domain_class_balanced_tau020 | pairwise_supcon | 0.2000 | 0.2000 | 16.0000 | domain_class_balanced | 0.8923 | 0.9992 | 1.0000 |
| pa_csi_dg_supcon_lam0050_tau020_bs32_domain_class_balanced | pairwise_supcon | 0.0500 | 0.2000 | 32.0000 | domain_class_balanced | 0.8917 | 0.9992 | 1.0000 |
| pa_csi_dg_supcon_lam0200_bs8_domain_balanced_tau020 | pairwise_supcon | 0.2000 | 0.2000 | 8.0000 | domain_balanced | 0.8913 | 0.9975 | 1.0000 |
| pa_csi_dg_supcon_lam0200_bs16_none_tau020 | pairwise_supcon | 0.2000 | 0.2000 | 16.0000 | none | 0.8903 | 0.9992 | 1.0000 |
| pa_csi_dg_supcon_lam0100_warm000_tau020_bs32_domain_class_balanced | pairwise_supcon | 0.1000 | 0.2000 | 32.0000 | domain_class_balanced | 0.8867 | 0.9983 | 1.0000 |
| pa_csi_dg_supcon_lam0200_bs32_none_tau020 | pairwise_supcon | 0.2000 | 0.2000 | 32.0000 | none | 0.8863 | 1.0000 | 1.0000 |
| pa_csi_dg_supcon_lam0200_warm010_tau020_bs32_domain_class_balanced | pairwise_supcon | 0.2000 | 0.2000 | 32.0000 | domain_class_balanced | 0.8860 | 1.0000 | 1.0000 |


## Lambda Sweep Summary

| contrastive_loss_type | lambda_supcon | mean_best_val_accuracy | mean_test_accuracy | num_runs |
| --- | --- | --- | --- | --- |
| pairwise_supcon | 0.2000 | 0.9990 | 0.8812 | 14.0000 |
| pairwise_supcon | 0.1000 | 0.9985 | 0.8629 | 5.0000 |
| pairwise_supcon | 0.0500 | 0.9984 | 0.8679 | 22.0000 |
| pairwise_supcon | 0.4000 | 0.9983 | 0.9207 | 1.0000 |
| pairwise_supcon | 0.0000 | 0.9975 | 0.8663 | 1.0000 |
| pairwise_supcon | 0.0250 | 0.9967 | 0.8520 | 1.0000 |


## Temperature Sweep Summary

| contrastive_loss_type | lambda_supcon | temperature | mean_best_val_accuracy | mean_test_accuracy | num_runs |
| --- | --- | --- | --- | --- | --- |
| pairwise_supcon | 0.0500 | 0.3000 | 0.9992 | 0.8517 | 1.0000 |
| pairwise_supcon | 0.2000 | 0.2000 | 0.9990 | 0.8812 | 14.0000 |
| pairwise_supcon | 0.1000 | 0.2000 | 0.9985 | 0.8629 | 5.0000 |
| pairwise_supcon | 0.0500 | 0.0700 | 0.9983 | 0.8690 | 1.0000 |
| pairwise_supcon | 0.0500 | 0.1000 | 0.9983 | 0.8627 | 1.0000 |
| pairwise_supcon | 0.0500 | 0.2000 | 0.9983 | 0.8690 | 19.0000 |
| pairwise_supcon | 0.4000 | 0.2000 | 0.9983 | 0.9207 | 1.0000 |
| pairwise_supcon | 0.0000 | 0.2000 | 0.9975 | 0.8663 | 1.0000 |
| pairwise_supcon | 0.0250 | 0.2000 | 0.9967 | 0.8520 | 1.0000 |


## Batch/Sampler Sweep Summary

| contrastive_loss_type | lambda_supcon | batch_size | sampler_mode | mean_best_val_accuracy | mean_test_accuracy | num_runs |
| --- | --- | --- | --- | --- | --- | --- |
| pairwise_supcon | 0.2000 | 32.0000 | domain_balanced | 1.0000 | 0.8990 | 1.0000 |
| pairwise_supcon | 0.2000 | 32.0000 | none | 1.0000 | 0.8863 | 1.0000 |
| pairwise_supcon | 0.2000 | 32.0000 | domain_class_balanced | 0.9995 | 0.8757 | 5.0000 |
| pairwise_supcon | 0.0500 | 32.0000 | none | 0.9992 | 0.8803 | 1.0000 |
| pairwise_supcon | 0.2000 | 16.0000 | domain_balanced | 0.9992 | 0.8693 | 1.0000 |
| pairwise_supcon | 0.2000 | 16.0000 | domain_class_balanced | 0.9992 | 0.8815 | 2.0000 |
| pairwise_supcon | 0.2000 | 16.0000 | none | 0.9992 | 0.8903 | 1.0000 |
| pairwise_supcon | 0.0500 | 32.0000 | domain_class_balanced | 0.9986 | 0.8648 | 13.0000 |
| pairwise_supcon | 0.1000 | 32.0000 | domain_class_balanced | 0.9985 | 0.8676 | 4.0000 |
| pairwise_supcon | 0.0500 | 16.0000 | domain_balanced | 0.9983 | 0.8693 | 1.0000 |


## Positive-Mask Sweep Summary

| include_same_domain_same_class | supcon_positive_mode | lambda_supcon | temperature | mean_best_val_accuracy | mean_test_accuracy | mean_supcon_frac_anchors_with_positive | num_runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | global_class_instance_views | 0.2000 | 0.2000 | 0.9994 | 0.8758 | 1.0000 | 4.0000 |
| 0.0000 | global_class_instance_views | 0.0500 | 0.2000 | 0.9992 | 0.8490 | 1.0000 | 1.0000 |
| 1.0000 | all_views | 0.0500 | 0.3000 | 0.9992 | 0.8517 | 1.0000 | 1.0000 |
| 1.0000 | all_views | 0.2000 | 0.2000 | 0.9989 | 0.8834 | 1.0000 | 10.0000 |
| 1.0000 | global_class_instance_views | 0.1000 | 0.2000 | 0.9985 | 0.8676 | 1.0000 | 4.0000 |
| 1.0000 | global_class_instance_views | 0.0500 | 0.2000 | 0.9985 | 0.8673 | 1.0000 | 5.0000 |
| 1.0000 | all_views | 0.0500 | 0.0700 | 0.9983 | 0.8690 | 1.0000 | 1.0000 |
| 1.0000 | all_views | 0.0500 | 0.1000 | 0.9983 | 0.8627 | 1.0000 | 1.0000 |
| 1.0000 | all_views | 0.1000 | 0.2000 | 0.9983 | 0.8443 | 1.0000 | 1.0000 |
| 1.0000 | all_views | 0.4000 | 0.2000 | 0.9983 | 0.9207 | 1.0000 | 1.0000 |


## Warmup Sweep Summary

| contrastive_loss_type | lambda_supcon | lambda_supcon_warmup_epochs | mean_best_val_accuracy | mean_test_accuracy | num_runs |
| --- | --- | --- | --- | --- | --- |
| pairwise_supcon | 0.1000 | 40.0000 | 1.0000 | 0.8490 | 1.0000 |
| pairwise_supcon | 0.2000 | 10.0000 | 1.0000 | 0.8860 | 1.0000 |
| pairwise_supcon | 0.0500 | 40.0000 | 0.9992 | 0.8613 | 1.0000 |
| pairwise_supcon | 0.2000 | 20.0000 | 0.9992 | 0.8770 | 1.0000 |
| pairwise_supcon | 0.2000 | 0.0000 | 0.9990 | 0.8824 | 11.0000 |
| pairwise_supcon | 0.0500 | 0.0000 | 0.9984 | 0.8690 | 19.0000 |
| pairwise_supcon | 0.1000 | 0.0000 | 0.9983 | 0.8655 | 2.0000 |
| pairwise_supcon | 0.1000 | 20.0000 | 0.9983 | 0.8527 | 1.0000 |
| pairwise_supcon | 0.2000 | 40.0000 | 0.9983 | 0.8673 | 1.0000 |
| pairwise_supcon | 0.4000 | 0.0000 | 0.9983 | 0.9207 | 1.0000 |


## Subcenter Summary

_No rows available._


## Recommended Next Action

Placeholder: choose candidate settings from source validation behavior, then run or inspect final multi-seed, multi-target evaluation once.
