# TPP C3 Recovery Report

Candidates are matched to the CE baseline by target environment and seed.
A candidate is promotion-ready only after the full 3-target x 3-seed matrix is complete and every gate passes.

## Acceptance Gates

- mean macro-F1 delta >= -0.010
- mean C3 F1 delta >= -0.050
- mean C1 F1 delta >= +0.050
- relative C1/C4 pair-confusion reduction >= 10.0%
- mean C0 precision delta >= -0.030
- E1 pair confusion improves in at least 2 seeds

## Experiment Summary

| experiment | runs | complete | delta macro F1 | delta C1 F1 | delta C3 F1 | delta C3 recall | pair reduction | E1 wins | diagnostic | promote |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| tpp_c3_mixed050_l124 | 6 | no | -0.0917 | 0.0409 | -0.4290 | -0.5479 | 0.1210 | 1/2 | FAIL | FAIL |
| tpp_c3_residual025_l124 | 6 | no | 0.0103 | 0.1099 | -0.0991 | -0.1521 | 0.1788 | 2/2 | FAIL | FAIL |
| tpp_c3_residual050_l124 | 6 | no | -0.0689 | 0.0616 | -0.4140 | -0.5433 | 0.2040 | 1/2 | FAIL | FAIL |
| tpp_l124 | 9 | yes | -0.0194 | 0.1209 | -0.2464 | -0.3422 | 0.2408 | 1/3 | FAIL | FAIL |

## Mean Per-Class F1 Delta

### tpp_c3_mixed050_l124

| class | baseline F1 | candidate F1 | delta F1 | delta recall | delta precision |
|---|---:|---:|---:|---:|---:|
| C0 no movement | 0.9708 | 0.9128 | -0.0580 | 0.0083 | -0.1123 |
| C1 falling | 0.5859 | 0.6269 | 0.0409 | 0.0821 | -0.0202 |
| C2 sit/stand | 0.9020 | 0.9063 | 0.0043 | 0.0117 | -0.0053 |
| C3 walking | 0.9510 | 0.5220 | -0.4290 | -0.5479 | -0.1486 |
| C4 turning | 0.6652 | 0.5896 | -0.0757 | -0.0983 | -0.0017 |
| C5 pick up pen | 0.9924 | 0.9596 | -0.0328 | -0.0317 | -0.0334 |

### tpp_c3_residual025_l124

| class | baseline F1 | candidate F1 | delta F1 | delta recall | delta precision |
|---|---:|---:|---:|---:|---:|
| C0 no movement | 0.9708 | 0.9622 | -0.0086 | 0.0085 | -0.0226 |
| C1 falling | 0.5859 | 0.6958 | 0.1099 | 0.1500 | 0.0267 |
| C2 sit/stand | 0.9020 | 0.9453 | 0.0432 | 0.0725 | 0.0067 |
| C3 walking | 0.9510 | 0.8518 | -0.0991 | -0.1521 | -0.0152 |
| C4 turning | 0.6652 | 0.6804 | 0.0152 | -0.0367 | 0.0568 |
| C5 pick up pen | 0.9924 | 0.9938 | 0.0014 | 0.0067 | -0.0041 |

### tpp_c3_residual050_l124

| class | baseline F1 | candidate F1 | delta F1 | delta recall | delta precision |
|---|---:|---:|---:|---:|---:|
| C0 no movement | 0.9708 | 0.9174 | -0.0533 | 0.0035 | -0.0987 |
| C1 falling | 0.5859 | 0.6475 | 0.0616 | 0.0521 | 0.0297 |
| C2 sit/stand | 0.9020 | 0.9423 | 0.0402 | 0.0658 | 0.0028 |
| C3 walking | 0.9510 | 0.5370 | -0.4140 | -0.5433 | -0.1232 |
| C4 turning | 0.6652 | 0.6348 | -0.0304 | -0.0229 | -0.0340 |
| C5 pick up pen | 0.9924 | 0.9746 | -0.0178 | -0.0033 | -0.0318 |

### tpp_l124

| class | baseline F1 | candidate F1 | delta F1 | delta recall | delta precision |
|---|---:|---:|---:|---:|---:|
| C0 no movement | 0.9469 | 0.9246 | -0.0223 | 0.0144 | -0.0545 |
| C1 falling | 0.5549 | 0.6758 | 0.1209 | 0.1508 | 0.0341 |
| C2 sit/stand | 0.8884 | 0.9353 | 0.0469 | 0.0742 | 0.0074 |
| C3 walking | 0.8056 | 0.5592 | -0.2464 | -0.3422 | -0.0636 |
| C4 turning | 0.6358 | 0.6440 | 0.0082 | -0.0278 | 0.0352 |
| C5 pick up pen | 0.9938 | 0.9699 | -0.0239 | -0.0156 | -0.0311 |

