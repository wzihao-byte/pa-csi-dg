# E2 Method/Result Inventory

## Scope
整理当前工作树里可复查的 E2 输出，重点是 class 1/4 collapse。机器汇总表：

- `analysis/e2_seed_level_results.csv`: 每个 E2 seed/run 一行。
- `analysis/e2_grouped_results.csv`: 按 experiment 聚合。
- `analysis/e2_grouped_results.json`: 同一聚合的 JSON 版本。

## Baseline Rule
`baseline` 和 `baseline + SupCon` 是 clean line，必须保留。后续所有方法只能和自己的 matched baseline 比较；不同配置、sampler、selection metric、epoch budget、prototype K 或 run family 不能混成一个统一 baseline。

判断层级：

- Clean anchor: CE baseline 或 source-only SupCon baseline，不含 OT、CC-VREx、pair margin、dynamic/adaptive margin、geometry-balance loss。
- Matched full result: 同 target/env/seeds/config family 的 full 多 seed 对比，可以做主要结论。
- Smoke/single-seed: 只用于筛选方向，不能 claim success。
- Aborted/debug: 记录原因，不计入性能胜负。

## Clean Baseline Line

| Anchor | Status | Seeds | Target acc | Macro-F1 | min_f1_1_4 | Note |
|---|---:|---:|---:|---:|---:|---|
| `baseline_pa_csi_dg_paper_aligned` | clean CE baseline | 42 | 0.8547 | 0.8304 | 0.5950 | paper-aligned diagnostic single seed |
| `pa_csi_dg_supcon_paper_aligned` | clean baseline+SupCon | 42 | 0.8953 | 0.8763 | 0.6970 | cleanest single-seed SupCon anchor found |
| `control_new_kall1` | clean K=1 source-only control | 42/52/62 | 0.8821 mean | 0.8582 mean | 0.6170 mean | full 3-seed control for geometry-balance cycle |
| `e2_14_phase1_k1_source_only` | clean K=1 source-only control | 42/52/62 | 0.8508 mean | 0.8302 mean | 0.5888 mean | matched full baseline for OT cycle |

Interpretation: SupCon itself is not the problem. The clean CE -> SupCon single-seed move improved E2 and class 1/4 balance. Later failures are failures of added mechanisms or changed protocols, not evidence against baseline+SupCon.

## Method Results

### Adaptive K-Margin / LCAKM
- `pa_csi_dg_supcon_paper_aligned_adaptive_kmargin_e2`, seed 42, 200e:
  - acc 0.8980, macro-F1 0.8765, `min_f1_1_4` 0.6574.
- This is not a clean win over the clean single-seed SupCon anchor (`min_f1_1_4` 0.6970).
- Status: single-seed non-win; do not continue as main line.

### Dynamic Margin
- `dynamic_margin_e2`, seed 42, 200e: acc 0.8753, macro-F1 0.8491, `min_f1_1_4` 0.5854.
- `dynamic_margin_v2_e2`, seed 42, 200e: acc 0.8863, macro-F1 0.8636, `min_f1_1_4` 0.6417.
- Neither beats the clean single-seed SupCon anchor on class 1/4.
- Status: not a main-line success.

### Pair Margin / Subcenter Margin
- K1 pair-margin smoke:
  - matched baseline `min_f1_1_4` 0.5368.
  - pair-margin `min_f1_1_4` 0.3654.
  - acc rose, but class 1 collapsed harder.
- K-margin/search variants include single-seed results around `min_f1_1_4` 0.61-0.65, and one layer1 K3 margin run at 0.6758.
- These are smoke/single-seed and do not beat the clean SupCon anchor robustly.
- Status: K1 pair-margin stopped; broader margin variants are not validated.

### Geometry Balance
- Full 3-seed `geom_balance` and `geom_balance_div` attempts aborted.
- Abort reason: warmup geometry collapse, especially class 1.
- Matched control `control_new_kall1` completed and remains a clean source-only anchor.
- Status: method failed operational gate; do not continue as-is.

### CC-VREx Focus 1/4
- `lambda=0.2`, no-geometry smoke, seed 42:
  - matched baseline `min_f1_1_4` 0.4237.
  - CC-VREx `min_f1_1_4` 0.6592.
  - acc 0.7840 -> 0.8577, macro-F1 0.7683 -> 0.8396.
- `lambda=0.1`, no-geometry smoke, seed 42:
  - matched baseline `min_f1_1_4` 0.5331.
  - CC-VREx `min_f1_1_4` 0.5744.
- The geometry-enabled CC-VREx smoke aborted during warmup.
- Status: only unverified positive candidate. It is not a success claim until full matched seeds 42/52/62 are run against the same baseline family.

### OT Alignment
- Full matched run `full_20260524_retry1`, seeds 42/52/62:
  - baseline `min_f1_1_4`: 0.5888 mean.
  - unfiltered OT `min_f1_1_4`: 0.5433 mean, paired delta -0.0454.
  - prefiltered OT `min_f1_1_4`: 0.5504 mean, paired delta -0.0383.
- Only seed 52 improved slightly; seeds 42/62 worsened.
- Diagnostics showed geometry compression rather than useful target 1/4 separation.
- Status: main OT alignment failed. Do not continue ordinary OT sensitivity/ablation from this result.

## Current Conclusions

1. Clean baseline and clean baseline+SupCon should stay as the reference line.
2. No recent added method has produced a reliable matched full success.
3. OT is a confirmed full-run failure for this formulation.
4. Geometry balance and geometry-enabled variants hit operational collapse.
5. Margin variants are mostly smoke/single-seed and did not establish a robust improvement over clean SupCon.
6. CC-VREx is the only remaining unverified positive signal, but it is still smoke-only.

## Next Discipline

- Every future table should start with the matched clean baseline row.
- Do not compare a method to a weaker baseline from a different run family to claim a win.
- Keep `baseline + SupCon` as the clean anchor unless a direct matched rerun disproves it.
- Treat future work as either:
  - full validation of the CC-VREx candidate, or
  - a new representation/pseudo-label diagnostic line.
- Do not resume OT, geometry balance, or K1 pair-margin without a new mechanism reason.
