# MMFi PA-CSI-DG Weekly Report

Period: 2026-05-25 to 2026-05-31

## 1. 本周目标

本周主要目标不是继续调 PA-CSI-DG，而是重新建立 MMFi 的可信 segment-level 实验协议。重点检查 sample unit、segment/window 定义、split 泄漏、duration confound、subject shift、packet/phase 表示，以及 small-model sanity gates。

## 2. 已完成工作

1. 建立独立工作区：

```text
C:/Users/howie/Desktop/mmfi-pa-csi-dg-workspace/
```

原 `pa-csi-dg` 仓库只作为参考，MMFi 相关探索没有混入原项目。

2. 核对 MMFi 数据源：

```text
raw root: D:/MMFi_Dataset
segment CSV: D:/datasets/mmfi/raw_drive/MMFi_action_segments.csv
```

抽查确认当前使用的 WiFi-CSI 文件和之前下载/解压的部分 MMFi 文件一致。

3. 完成 inventory 和 segment length 检查：

```text
p50 = 19 frames
p75 = 24 frames
p95 = 32 frames
p99 = 38 frames
```

因此暂时不使用 T=64/96，第一版采用 T=32。

4. 生成第一版 segment-level feature cache：

```text
feature: P0 packet mean + A1 log1p amplitude + phase none + T32
shape: [N, 32, 342]
segments: 16447
```

5. 生成并验证 split manifests：

```text
S0 label-shuffle
S2 same-env segment-disjoint
S3 same-env subject-disjoint
S4a/S4b/S4c
S5b strict LOEO, target E01-E04
```

已检查 sample、segment、subject、target leakage。

6. 完成 small baselines：

```text
M_length duration-only
M_static static amplitude
M0 handcrafted amplitude stats
M1 TemporalCNN
```

7. 完成 A1+delta ablation：

```text
feature: concat(A1_log1p_resampled, delta_time_A1_log1p_resampled)
shape: [N, 32, 684]
```

8. 完成 weighted A1+delta TCN 的 S5b 3-seed strict LOEO：

```text
seeds: 42, 43, 44
target-env mean accuracy = 0.1407 +/- 0.0186
target-env mean macro-F1 = 0.1021 +/- 0.0114
```

9. 补充 sanity gates：

S0 label-shuffle 通过：

```text
weighted A1+delta TCN:
  test accuracy = 0.0313
  test macro-F1 = 0.0298
```

S2/S3 显示明显 subject shift：

```text
S2 E01 segment-disjoint macro-F1 = 0.9553
S3 E01 subject-disjoint macro-F1 = 0.2496

S2 E03 segment-disjoint macro-F1 = 0.8518
S3 E03 subject-disjoint macro-F1 = 0.0857
```

## 3. 关键发现

1. 当前 pipeline 没有明显 label/split 泄漏。S0 label-shuffle 后 val/test 接近随机。

2. MMFi 的 CSI/action 信号在同环境、segment-disjoint 下非常可学，说明数据、label、loader、模型训练路径不是主要问题。

3. 一旦 subject-disjoint，性能大幅下降。subject shift 是当前最大障碍之一。

4. S5 strict LOEO 实际是：

```text
unseen environment + unseen subject group
```

不是纯 environment DG。

5. Duration confound 很强。length-only baseline 在 S5 target accuracy 上可达到约 0.17-0.22，而当前 CSI TCN accuracy 约 0.14。

6. A1+delta 明显优于原始 P0/A1，但仍不足以支撑 PA-CSI-DG 或 SupCon 结论。

## 4. 当前问题

1. strict LOEO 表现很差：

```text
best small-model S5b macro-F1 mean = 0.1021
best small-model S5b accuracy mean = 0.1407
```

这只是略高于随机，远不是可接受的最终结果。

2. subject shift 严重。S2 很高但 S3 急剧下降，说明模型学到大量 subject-specific pattern。

3. duration / segment length confound 还没有控制住。

4. 当前 delta 是 fixed-T 派生 delta，不是 raw segment 上的 converter-level delta。

5. packet 维信息还没有充分利用，目前主结果仍基于 P0 packet mean。

6. phase V1/V2 还没进入可信实验，phase-only gate 未完成。

7. 还没有进入 full PA-CSI-DG 或 SupCon 的条件。

## 5. 下周建议

优先继续 protocol 和 preprocessing，不建议直接上 full PA-CSI-DG。

建议顺序：

```text
1. 做 duration/length-controlled evaluation。
2. 实现 converter-level raw A1_delta。
3. 跑 P1 packet mean/std/diff 和 P3a packet dynamics。
4. 做 per-class confusion / failure analysis。
5. 做 subject-ID predictability diagnostic。
6. 再决定是否进入 phase V1 和 full PA-CSI-DG。
```

当前结论：

```text
pipeline 可用，但 MMFi strict DG 任务尚未解决。
当前失败点主要是 subject shift + duration confound + P0 feature 表示不足。
```
