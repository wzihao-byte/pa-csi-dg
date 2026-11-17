# SupCon × Residual-TPP Paired Margin Analysis

This report uses the locked 36-run formal-interim factorial and all 108,000 target predictions. 
The independent analysis unit is the paired `(target, seed)` block (9 blocks). The signed probability margin is 
`p(true) - max p(false)`; it is positive exactly when the prediction is correct. All effects are paired within block.

## Four-arm summary

| arm | interpretation | macro-F1 | accuracy | balanced margin | margin SNR |
|---|---|---:|---:|---:|---:|
| Y00 | global-basis / CE | 0.6196 ± 0.0770 | 0.7148 ± 0.0564 | 0.2601 | 0.3077 |
| Y01 | global-basis / SupCon | 0.6392 ± 0.0822 | 0.7311 ± 0.0604 | 0.2954 | 0.3549 |
| Y10 | residual-TPP / CE | 0.7243 ± 0.0814 | 0.7951 ± 0.0530 | 0.4623 | 0.6935 |
| Y11 | residual-TPP / SupCon | 0.7171 ± 0.0464 | 0.7908 ± 0.0338 | 0.4365 | 0.6019 |

## Paired factorial effects

| effect | Δ macro-F1 | 95% CI | p | Δ balanced margin | Δ margin SNR |
|---|---:|---:|---:|---:|---:|
| TPP_without_SupCon | 0.1047 ± 0.0308 | [0.0765, 0.1329] | 9.987e-05 | 0.2022 | 0.3858 |
| TPP_with_SupCon | 0.0779 ± 0.0761 | [0.0315, 0.1243] | 0.0063 | 0.1410 | 0.2470 |
| SupCon_without_TPP | 0.0196 ± 0.0464 | [-0.0219, 0.0612] | 0.2913 | 0.0354 | 0.0472 |
| SupCon_with_TPP | -0.0072 ± 0.0638 | [-0.0463, 0.0319] | 0.6685 | -0.0258 | -0.0916 |
| interaction | -0.0268 ± 0.0863 | [-0.0869, 0.0332] | 0.3161 | -0.0612 | -0.1388 |
| joint_vs_control | 0.0975 ± 0.0525 | [0.0752, 0.1198] | 3.912e-05 | 0.1764 | 0.2942 |

## Target-specific macro-F1 effects

| target | effect | mean Δ macro-F1 | SD |
|---|---|---:|---:|
| E1 | TPP_without_SupCon | 0.1106 | 0.0291 |
| E1 | TPP_with_SupCon | 0.0181 | 0.0568 |
| E1 | SupCon_without_TPP | 0.0249 | 0.0207 |
| E1 | SupCon_with_TPP | -0.0676 | 0.0336 |
| E1 | interaction | -0.0925 | 0.0520 |
| E1 | joint_vs_control | 0.0431 | 0.0365 |
| E2 | TPP_without_SupCon | 0.1085 | 0.0415 |
| E2 | TPP_with_SupCon | 0.1502 | 0.0589 |
| E2 | SupCon_without_TPP | 0.0011 | 0.0518 |
| E2 | SupCon_with_TPP | 0.0428 | 0.0598 |
| E2 | interaction | 0.0417 | 0.1002 |
| E2 | joint_vs_control | 0.1512 | 0.0245 |
| E3 | TPP_without_SupCon | 0.0951 | 0.0318 |
| E3 | TPP_with_SupCon | 0.0654 | 0.0548 |
| E3 | SupCon_without_TPP | 0.0329 | 0.0683 |
| E3 | SupCon_with_TPP | 0.0032 | 0.0470 |
| E3 | interaction | -0.0297 | 0.0593 |
| E3 | joint_vs_control | 0.0982 | 0.0175 |

## Class-level joint-versus-control effect

| class | mean Δ F1 | SD | mean Δ margin |
|---|---:|---:|---:|
| 0 no_movement | 0.0468 | 0.0212 | 0.0227 |
| 1 falling | 0.1318 | 0.1033 | 0.1947 |
| 2 walking | 0.0886 | 0.0585 | 0.2190 |
| 3 sit_stand | 0.2236 | 0.1752 | 0.3328 |
| 4 turning | 0.0823 | 0.1282 | 0.2760 |
| 5 pick_up_pen | 0.0119 | 0.1694 | 0.0133 |

## Empirical check of the sufficient-condition proxy

Using Y01 as the global-SupCon reference and `(Y11-Y01)/0.25` as the TPP treatment-effect proxy, 
`B/A > C/V` holds in **36/54** target-seed-class cells. 
The empirical margin SNR increases in **31/54** cells.

This is a factorial treatment-effect proxy because Y01 and Y11 are separately trained models. It is not an exact 
within-checkpoint decomposition of the residual tensor. The raw cell statistics are in `margin_condition_proxy.csv`.

## Interpretation guardrail

A positive joint-versus-control contrast does not prove synergy. Synergy requires a positive interaction and the joint arm 
to outperform both single-component arms. Target-specific effects must be reported because the formal study stopped for 
harm on E1 in the joint-versus-TPP contrast.
