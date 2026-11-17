# MultiEnv 数据来源与处理审计

## 原始资料与软件定义

采集说明见 [Alsaify 等原始数据论文](https://doi.org/10.1016/j.dib.2020.106534)、[Mendeley v1](https://data.mendeley.com/datasets/v38wjmz6f6/1)、[原作者 MAT/DAT 仓库](https://github.com/lcsig/Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments)。物理采集采用 Intel 5300、1Tx/3Rx、每链路 30 个子载波、2.4 GHz 信道 3、20 MHz，名义速率 320 packets/s，时间戳时钟 1 MHz。E1/E2 为 LOS，E3 为 NLOS；参与者依次为 S01–S10、S11–S20、S21–S30。

用户代码的 `mode=tx, num_rx=1, num_tx=3, num_subcarriers=30` 定义**伪 Tx 特征组**：0–29、30–59、60–89 维逐组保留，其余置零。软件含义以代码与实际 run 为准。原始 MAT 的 Rx 维度与软件伪 Tx 命名分别说明，保留历史切片行为。

## 全量映射：9000/9000 通过

原始仓库固定在 `5337147c5156948b13e6b05a50b79529614acfbe`。重放版本为 `legacy-replay-v3-timestamp-text15`。全部 9000 个原始 MAT 文件唯一匹配到 3×3000 个处理后索引，并核对全部 850×90 幅度、相位元素和活动标签。容差为 1e-10，实际最大幅度误差 0.0、相位误差 0.0。

首帧幅度只定位候选；完整数组相等、标签一致与一对一索引覆盖才构成匹配。E1 存在 S01,S10,S02,… 的历史目录顺序，映射依赖原始信号，不依赖猜测的数字排序。所有缓存原始文件及逐条匹配哈希保留。

已生成 3 份 subject sidecar，覆盖 30 个原始 subject、3000 个 subject–experiment–trial acquisition 分组。分组 ID 取 `E:S:C:T`，把同一采集试次的不同 activity 保守放在一起；它不是额外的独立样本计数。原始资料中的身份声明与信号对应关系可核验，但不会由此追溯验证现场招募记录。

sidecar 与 amp/phase/label 文件 SHA256 绑定，并通过现有 loader 的读取、分组及哈希校验函数。生成新协议时重新计算了全部 9 个实际数组文件的哈希。`array_audit.json` 是此前仅检查数组的记录；新增身份结论见 `verified_subject_audit.json`。

## 两条初始差异的原因与边界

初始 90 条抽查中有 88 条直接重放成功。两条差异分别为 E1_S03_C04_A01_T09（索引 1108）和 E1_S05_C04_A01_T16（索引 1715）。其 MAT 时间戳接近 2^64，float64 已损失精度；按 15 位十进制有效数字表示后，重放结果与历史幅度和相位全部元素相同。此前 88/90 记录保存在 `raw_mapping_v2_snapshot/`。

公开仓库中的全部 9000 份原始 DAT 均已独立解析并与对应 MAT 核对：packet ID 和复数 CSI 全部一致，准备过程无失败。DAT 的原生 uint32 时间戳生成的重采样数组仍不等于历史数组；这说明“恢复历史处理”与“按原生时钟重建输入”需要区分。15 位表示是已验证的历史数值等价路径，没有找回历史 CSV、导出软件或转换日志。MAT 异常数值的具体产生环节尚未确认。

全量重放中，有 241 条记录的时间戳数值因 15 位表示而变化：E1=208, E2=33, E3=0。重放时间戳非单调记录数为 0。精度变化集中于部分采集记录，不据此声称时间已校准，也不由相关性推断模型分数的成因。

涉及的原始 subject 及文件数为：S03=128, S05=80, S18=3, S20=30。原始时间戳重建应保留这一分布，用于统一预处理后的敏感性比较。

## 原生时间重建敏感性

9000 条原生 DAT 已分别编译为 `native_time_legacy_merge` 和 `native_time_linear` 两套 float32 幅度输入；原样本顺序、标签、subject/acquisition 划分和 source/deferred-target 边界均保留。前者在原生相对时间上复现历史 merge 规则，后者在同一原生时间上作 850 点线性插值。两者都是输入敏感性对照，不代表完整物理时间校准。

历史输入与原生时间 merge 在 5256/9000 条上逐元素相同，3744 条存在大于 1e-5 的绝对差异；逐条相对 L2 误差的中位数为 0，95 分位数为 0.002289，最大值为 0.449348。历史输入与原生时间线性插值在全部 9000 条上均有差异，相对 L2 误差中位数为 0.029546、95 分位数为 0.054591、最大值为 0.448329；逐条 Pearson 相关系数中位数为 0.995974，最小值为 0.418203。原生 merge 与原生 linear 也在全部 9000 条上不同，相对 L2 误差中位数为 0.029297、95 分位数为 0.048388、最大值为 0.158467。

这里的 3744 条与前述 241 条不是同一统计量：241 只记录历史 MAT 时间戳经过 15 位表示后数值发生变化；3744 比较的是包含时间表示与时间原点变化的完整 float32 模型输入。不能把两者等同，也不能把任何性能变化归因于其中单一环节。

原生记录时长的环境中位数都约为 4.000 s，但范围并不固定：E1 为 2.996739–5.006024 s，E2 为 2.990493–5.905944 s，E3 为 2.996685–5.007438 s。有效 packet rate 中位数分别为 313.764、309.009、313.350 packet/s；单次最大 packet gap 分别为 0.106144、2.015493、0.059378 s。线性插值会跨越 E2 的 2.015493 s 间隙，因此只能作为预先定义的敏感性对照，不能解释为恢复连续运动或固定采样率。

## 已冻结 H/Y 的身份核验

Y 的 36 个原始 split manifest 保留 `inferred_group_block_holdout` 与当时 sidecar 缺失状态。此次对每份冻结 manifest 的文件哈希、全部 train/val/test 样本集合哈希、类别支持进行核对，再接入全量原始身份：三种集合的 subject 与 acquisition 均不交叉。

历史 Y 汇总 manifest 的源样本键按 `_manifest_identifier_sets` 使用拼接后 loader 的数字 ID；另一些训练诊断通过 `_stable_sample_maps` 使用环境内索引。审计按实际代码恢复源环境偏移再比对哈希，同时核对原 indices 哈希。新 R manifest 统一用 `ENV:local_processed_index`，编号约定在各自记录中明确区分。

这是**事后确认原有 split 的身份隔离**，不改写当时记录，也不把 E1–E3 改称从未参与开发的测试环境。每个外层 fold 保持源训练 4800 条、源验证 1200 条、目标 3000 条。

| 环境 | 原验证块 | 已验证受试者 |
|---|---|---|
| E1 | B01 / B07 | E1:S10 / E1:S07 |
| E2 | B00 / B06 | E2:S11 / E2:S17 |
| E3 | B00 / B04 | E3:S21 / E3:S25 |

H 的 36 个冻结 split 也已按原代码的分层随机规则恢复，全部 train/val/test 样本 ID 哈希一致。每个 run 的源训练与源验证共有 20 个 subject，共有 822–874 个 acquisition 分组。目标的 subject 和 acquisition 隔离保持成立。这个结果说明 H 的源验证存在身份相关性；H/Y 同时改变多个因素，仍不能用分数差量化其中某一因素的因果效果。

## 相位与时间的解释

- 历史重放对 wrapped phase 做算术平均，可能受 ±π 边界影响，不属于物理校准。
- H 在 float64 上沿末轴 unwrap；Y 关闭 loader unwrap；信号诊断另有时间轴处理。这些流程分别记录。
- 850 帧是历史分箱与补齐的输出长度，不等于固定 850/320 秒。图中的频率保留 cycles/processed frame。
- 当前 `Baha_et_al` 还保留调试 exit、750 帧调用与注释的 phase 导出。重放验证相关数值处理，不冒称恢复了完整导出工程。
- 新 R 代码明确采用历史幅度数组的 float32 输入，不读取 phase，不自动更正历史时间。物理时间重建若实施，必须单独版本化并重新训练全部受影响比较。

## 复现与证据

```powershell
python analysis/verify_multienv_raw_mapping.py --limit 0 --workers 6
python analysis/audit_multienv_timestamps.py
python -m experiments.revision_validation.prepare_verified_protocol
python analysis/audit_h_subject_splits.py
```

主要结果：`evidence/raw_mapping/summary.json`、`timestamp_precision_audit.json`、`subject_sidecars/`、`verified_subject_audit.json`、`h_subject_split_audit.json`、`evidence/native_amp_full_summary.json`、`evidence/native_amp_dataset_manifest.json` 和 `evidence/native_reconstruction_v1/`。新协议文件在 `experiments/revision_validation/verified_protocol/`，source manifest 仅包含源数据路径与样本，目标信息单独保存。

查询与核验日期：2026-09-04。新的完整基线、版本化相位对照和未参与开发的新环境仍是后续工作。
