"""Write source-grounded revision reports after provenance/smoke gates complete."""
from pathlib import Path
import json
import re
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/revision"
EVIDENCE = DOCS / "evidence"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    audit = read(EVIDENCE / "verified_subject_audit.json")
    h = read(EVIDENCE / "h_subject_split_audit.json")
    timestamps = read(EVIDENCE / "replay_timestamp_records.json")
    assert audit["all_9000_verified"] and audit["Y_run_count"] == h["runs_checked"] == 36
    changed = Counter(row["raw_path"].split("/")[1] for row in timestamps if row["timestamp_text15_changed_values"])
    changed_subjects = Counter(re.search(r"_(S\d+)_", row["raw_path"]).group(1) for row in timestamps if row["timestamp_text15_changed_values"])
    text = f"""# MultiEnv 数据来源与处理审计

## 原始资料与软件定义

采集说明见 [Alsaify 等原始数据论文](https://doi.org/10.1016/j.dib.2020.106534)、[Mendeley v1](https://data.mendeley.com/datasets/v38wjmz6f6/1)、[原作者 MAT/DAT 仓库](https://github.com/lcsig/Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments)。物理采集采用 Intel 5300、1Tx/3Rx、每链路 30 个子载波、2.4 GHz 信道 3、20 MHz，名义速率 320 packets/s，时间戳时钟 1 MHz。E1/E2 为 LOS，E3 为 NLOS；参与者依次为 S01–S10、S11–S20、S21–S30。

用户代码的 `mode=tx, num_rx=1, num_tx=3, num_subcarriers=30` 定义**伪 Tx 特征组**：0–29、30–59、60–89 维逐组保留，其余置零。软件含义以代码与实际 run 为准。原始 MAT 的 Rx 维度与软件伪 Tx 命名分别说明，保留历史切片行为。

## 全量映射：9000/9000 通过

原始仓库固定在 `{audit['source_commit']}`。重放版本为 `{audit['verifier_version']}`。全部 9000 个原始 MAT 文件唯一匹配到 3×3000 个处理后索引，并核对全部 850×90 幅度、相位元素和活动标签。容差为 1e-10，实际最大幅度误差 {audit['max_amplitude_error']:.1f}、相位误差 {audit['max_phase_error']:.1f}。

首帧幅度只定位候选；完整数组相等、标签一致与一对一索引覆盖才构成匹配。E1 存在 S01,S10,S02,… 的历史目录顺序，映射依赖原始信号，不依赖猜测的数字排序。所有缓存原始文件及逐条匹配哈希保留。

已生成 3 份 subject sidecar，覆盖 {audit['subject_count']} 个原始 subject、{audit['acquisition_count']} 个 subject–experiment–trial acquisition 分组。分组 ID 取 `E:S:C:T`，把同一采集试次的不同 activity 保守放在一起；它不是额外的独立样本计数。原始资料中的身份声明与信号对应关系可核验，但不会由此追溯验证现场招募记录。

sidecar 与 amp/phase/label 文件 SHA256 绑定，并通过现有 loader 的读取、分组及哈希校验函数。生成新协议时重新计算了全部 9 个实际数组文件的哈希。`array_audit.json` 是此前仅检查数组的记录；新增身份结论见 `verified_subject_audit.json`。

## 两条初始差异的原因与边界

初始 90 条抽查中有 88 条直接重放成功。两条差异分别为 E1_S03_C04_A01_T09（索引 1108）和 E1_S05_C04_A01_T16（索引 1715）。其 MAT 时间戳接近 2^64，float64 已损失精度；按 15 位十进制有效数字表示后，重放结果与历史幅度和相位全部元素相同。此前 88/90 记录保存在 `raw_mapping_v2_snapshot/`。

两份原始 DAT 的 uint32 时间戳、所有 packet ID 与复数 CSI 也已独立解析核对。DAT 与 MAT 的全部 CSI 相同，DAT 时间戳重采样却不等于历史数组；这说明“恢复历史处理”与“恢复物理时间”需要区分。15 位表示是已验证的数值等价处理路径，没有找回历史 CSV、导出软件或转换日志。MAT 异常数值的具体产生环节尚未确认。

全量重放中，有 {audit['timestamp_precision_changed_files']} 条记录的时间戳数值因 15 位表示而变化：{', '.join(f'{env}={changed[env]}' for env in ('E1','E2','E3'))}。重放时间戳非单调记录数为 {audit['nonmonotonic_replay_timestamp_files']}。精度变化集中于部分采集记录，不据此声称时间已校准，也不由相关性推断模型分数的成因。

涉及的原始 subject 及文件数为：{', '.join(f'{subject}={count}' for subject,count in sorted(changed_subjects.items()))}。原始时间戳重建应保留这一分布，用于统一预处理后的敏感性比较。

## 已冻结 H/Y 的身份核验

Y 的 36 个原始 split manifest 保留 `inferred_group_block_holdout` 与当时 sidecar 缺失状态。此次对每份冻结 manifest 的文件哈希、全部 train/val/test 样本集合哈希、类别支持进行核对，再接入全量原始身份：三种集合的 subject 与 acquisition 均不交叉。

历史 Y 汇总 manifest 的源样本键按 `_manifest_identifier_sets` 使用拼接后 loader 的数字 ID；另一些训练诊断通过 `_stable_sample_maps` 使用环境内索引。审计按实际代码恢复源环境偏移再比对哈希，同时核对原 indices 哈希。新 R manifest 统一用 `ENV:local_processed_index`，编号约定在各自记录中明确区分。

这是**事后确认原有 split 的身份隔离**，不改写当时记录，也不把 E1–E3 改称从未参与开发的测试环境。每个外层 fold 保持源训练 4800 条、源验证 1200 条、目标 3000 条。

| 环境 | 原验证块 | 已验证受试者 |
|---|---|---|
"""
    for env in ("E1", "E2", "E3"):
        selected = [row for row in audit["blocks"] if row["environment"] == env and row["validation"]]
        text += f"| {env} | {' / '.join(row['block_id'] for row in selected)} | {' / '.join(row['subject_id'] for row in selected)} |\n"
    lo, hi = h["shared_source_acquisition_count_range"]
    text += f"""
H 的 36 个冻结 split 也已按原代码的分层随机规则恢复，全部 train/val/test 样本 ID 哈希一致。每个 run 的源训练与源验证共有 {h['shared_source_subject_count_range'][0]} 个 subject，共有 {lo}–{hi} 个 acquisition 分组。目标的 subject 和 acquisition 隔离保持成立。这个结果说明 H 的源验证存在身份相关性；H/Y 同时改变多个因素，仍不能用分数差量化其中某一因素的因果效果。

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

主要结果：`evidence/raw_mapping/summary.json`、`timestamp_precision_audit.json`、`subject_sidecars/`、`verified_subject_audit.json`、`h_subject_split_audit.json`。新协议文件在 `experiments/revision_validation/verified_protocol/`，source manifest 仅包含源数据路径与样本，目标信息单独保存。

查询与核验日期：2026-09-03。新的完整基线、物理相位校准和未参与开发的新环境仍是后续工作。
"""
    (DOCS / "DATA_PROVENANCE_AUDIT.md").write_text(text, encoding="utf-8")

    smoke_folder = EVIDENCE / "amplitude_source_smoke"
    summary = read(smoke_folder / "summary.json")
    assert summary["arm_runs"] == 12 and summary["source_only"]
    reports = [read(smoke_folder / f"{env}_{arm}.json") for env in ("E1", "E2", "E3") for arm in ("R00", "R01", "R10", "R11")]
    text = f"""# 新幅度单模态对照：源数据 smoke

已完成 3 个外层 fold × 4 个实验臂，共 {summary['total_optimizer_updates_executed']} 次优化更新。每次均只读取两个源环境的幅度文件，目标信号、目标标签和相位文件均未进入 smoke。所有 loss、梯度有限，SupCon 每个 anchor 均有正样本。这是实现与资源核验，不是完整基线训练，没有目标性能结果。

四臂 R00/R01 为全局 MLP + CE/CE+global-only SupCon，R10/R11 为 L124 residual TPP + CE/CE+global-only SupCon。沿用幅度编码器，另用统一 256 维表征层；这是独立 R 候选模型，不继承 H/Y 分数。

参数总数均为 1,878,142；同池化的 CE/SupCon 臂初始状态完全相同，跨池化的 182 个公共状态张量也完全相同。池化全局维度为 d 时，TPP 的线性投影有 7d²+d 个参数；全局 MLP 的 d→7d/2→d 两层使用第一层无 bias、第二层有 bias，参数量与矩阵 MACs 相同。函数族和初始残差范数仍不同，不称为对所有混淆因素的完全控制。

| Fold | Arm | 有梯度的参数 | 16 条推理中位 ms | 更新中位秒 | GPU 峰值 MiB |
|---|---|---:|---:|---:|---:|
"""
    for r in reports:
        text += f"| {r['target_name_only']} | {r['arm']} | {r['gradient_activity'][-1]['with_gradient_tensor']:,} | {r['inference_batch_latency_ms_median']:.2f} | {r['training_step_seconds_median']:.3f} | {r['peak_gpu_allocated_bytes']/2**20:.1f} |\n"
    text += f"""
硬件为 {summary['gpu']}，PyTorch {summary['torch_version']}。四臂单样本推理的卷积、线性层、注意力与位置嵌入合计 MACs 均为 {reports[0]['inference_counted_macs']['total_counted_macs']:,}。该统计不含池化比较、softmax、归一化、激活、bias 加法与数据移动，不当作完整 FLOPs。GPU 峰值包含 smoke 预载的幅度 batch。

CE 臂保留训练用 projection head，但它没有梯度；表中“有梯度参数”指梯度张量存在的参数数目，未声称每个标量梯度都非零。原始记录另存非零梯度张量覆盖量和实际残差/global 范数。

已核验每个 fold 的 200 epoch、60,000 更新采样日程：每批 16 条、每源 8 条、每源包含全部六类、批内样本不重复。日程使用私有 RNG，与模型初始化、dropout 和实验臂执行顺序无关。日程核验不表示完成了这 60,000 次训练。

smoke 使用固定 λ=.025 的完整权重以覆盖对比路径；完整训练入口另实现 6000 步线性 warmup、连续指数学习率衰减、每 epoch 完整源验证 macro-F1 选择、相同分数保留最早 checkpoint、失败 attempt 保留。短 smoke 吞吐排除了数据读取、完整验证和 checkpoint 写入，不能直接当作整轮训练保证时间。

下一步冻结完整 R 比较和源域调参预算，再开展同协议的完整训练。当前数据仍是开发数据；新 R 代码不改变 Y 的停止决定。

```powershell
python -m experiments.revision_validation.run_source_smoke --target all --steps 30
python -m experiments.revision_validation.train_source --target E1 --arm R11 --seed 401 --epochs 1 --output outputs_revision_validation/pilot_E1_R11_401/attempt_001
```
"""
    pilot_path = EVIDENCE / "source_pilot_verification.json"
    if pilot_path.exists():
        pilot = read(pilot_path)
        assert pilot["all_predictions_and_metrics_equal"] and not pilot["target_data_loaded"]
        text += f"\n另已完成 E1/R11/seed-401 的 1 epoch 训练入口试跑（{pilot['optimizer_updates_executed']} 更新），严格恢复保存的 checkpoint，并重算全部 {pilot['source_prediction_rows_verified']} 条源验证预测；预测与指标完全一致。试跑证据见 `evidence/source_pilot_verification.json`，原输出保存在 `outputs_revision_validation/pilot_E1_R11_401/attempt_001/`。这项试跑也不纳入完整基线结果。重复执行命令时需使用新的 attempt 目录。\n"
    (DOCS / "SOURCE_SMOKE_REPORT.md").write_text(text, encoding="utf-8")
    print("Updated provenance and source-smoke reports")


if __name__ == "__main__":
    main()
