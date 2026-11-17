# 修订证据审计

本报告由 `python analysis/revision_evidence.py` 从冻结记录重算。H/Y 是不同开发协议，不合并结果，也不以跨协议差值归因。

- H：36 个目标混淆矩阵重算 macro-F1。
- Y：36 份目标预测（108,000 行）核对 SHA256、样本身份、标签、分组、混淆矩阵和 macro-F1。
- 每环境的三个 seed 只描述固定测试集上的优化波动；没有环境总体 p 值或置信区间。

| Protocol | Arm | E1 mean ± seed SD | E2 | E3 | Equal-environment mean |
|---|---|---|---|---|---|
| H_historical | H00 | 86.01 ± 1.77 | 85.44 ± 0.66 | 83.30 ± 1.70 | 84.92 |
| H_historical | H01 | 85.63 ± 1.54 | 84.36 ± 1.88 | 83.69 ± 1.40 | 84.56 |
| H_historical | H10 | 88.80 ± 1.31 | 87.86 ± 1.13 | 89.11 ± 3.19 | 88.59 |
| H_historical | H11 | 89.54 ± 1.26 | 89.99 ± 0.70 | 91.23 ± 1.76 | 90.26 |
| Y_grouped | Y00 | 71.50 ± 2.22 | 56.75 ± 3.27 | 57.62 ± 4.03 | 61.96 |
| Y_grouped | Y01 | 74.00 ± 3.99 | 56.86 ± 1.92 | 60.91 ± 3.19 | 63.92 |
| Y_grouped | Y10 | 82.57 ± 1.52 | 67.60 ± 2.10 | 67.13 ± 5.22 | 72.43 |
| Y_grouped | Y11 | 75.81 ± 2.03 | 71.88 ± 4.34 | 67.45 ± 3.24 | 71.71 |

| Protocol | Environment | TPP with CE (pp) | Contrastive on TPP (pp) | Interaction (pp) |
|---|---|---:|---:|---:|
| H_historical | E1 | +2.79 | +0.73 | +1.12 |
| H_historical | E2 | +2.42 | +2.13 | +3.21 |
| H_historical | E3 | +5.82 | +2.12 | +1.73 |
| H_historical | equal_environment_mean | +3.68 | +1.66 | +2.02 |
| Y_grouped | E1 | +11.06 | -6.76 | -9.25 |
| Y_grouped | E2 | +10.85 | +4.28 | +4.17 |
| Y_grouped | E3 | +9.51 | +0.32 | -2.97 |
| Y_grouped | equal_environment_mean | +10.47 | -0.72 | -2.68 |

## 停止记录与解释

Y 的保存判定为 `stop`；触发项为 `target_specific_harm`。
该停止规则是固定三环境、给定模型假设下的内部决策，不是未知环境总体的显著性检验。保持原停止记录，不补 seed 寻找正结果。
两协议同时改变了 phase、split、结构、loss/view、batch 与训练步数；不能把分数差归因于其中一项。TPP 相对各自对照的方向一致，正交互并不跨协议保持。

## 身份核验与后续证据

- 全量 9000 行原始信号映射已核验。另行核对 Y 的 36 个冻结 split：样本集合哈希一致，真实 subject 和 acquisition 分组不交叉。
- 这是事后身份核验；原 manifest 仍保持 inferred_group_block_holdout，未回写历史记录，也未改变开发环境的性质。
- 时间戳精度、相位处理仍需独立验证。原始路径可重放不等于物理校准；关闭 unwrap 也不构成校准。
- 同新协议的常规小模型 ERM、模态对照、常规容量控制、外部 DG 基线。
- 冻结方法之后的新目标环境。

Y 运行登记平均耗时：2.36 h/run；不是新模型的保证预算。
