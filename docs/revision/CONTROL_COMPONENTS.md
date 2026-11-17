# 扩展对照：冻结协议与执行状态

2026-09-04。以下组件已实现并通过源数据前向/反向及针对性检查，另以 `r_controls_v1` 冻结为 81 个正式作业。等待队列已经启动；它先等核心 R 的 36 个源训练作业与冻结评估全部完成，再顺序训练这些对照。尚未形成完整对照结果。

代码：`experiments/revision_validation/control_components.py`。验证脚本：`analysis/verify_revision_controls.py`。证据：`evidence/control_component_qa.json`。

| 对照 | 已实现的定义 | 参数或解释边界 |
|---|---|---|
| P1-only | 两条 pooling 路径仅保留 level 1；保留全局 residual | 1,478,782 参数；不能与 L124 的差异全部归因于局部信息 |
| 去 P1 | 两条路径改为 levels 2、4 | 1,811,582 参数；同时减少投影参数，需连同常规匹配控制解释 |
| Attention pooling | 每个卷积分支独立学习时间权重，再经常规 MLP 进入全局 residual | 恰为 1,878,142 参数；动态 scorer 增加实际 MACs，不宣称等总计算量 |
| 三种 mask | hybrid、standard all-view、hybrid 中从分母移除同类非正对 | 相同 global/auxiliary 两次 forward，保留历史 pseudo-Tx 三段切片 |
| Source CORAL | 同一次混合源 batch forward 后计算两个源域表征的均值与协方差差异 | 公式参照 DomainBed，保持本项目 joint-batch/BatchNorm 约定 |
| Cross-domain Mixup | 两个源域的有向混合，使用按 seed/step 独立的 Beta(0.2,0.2) RNG | 固定 alpha=.2；同核心更新预算，另计实际训练资源 |
| 时间组件检查 | 循环平移 85/170 帧、85 帧窗内打乱、十个 85 帧块重排 | 已做解析/实现 QA，但没有把这四项分别配置为正式训练臂 |

所有 pooling 对照的公共初始化均与 R10 相同。Attention 的 scorer 共 d 个参数、两个无 bias 的 MLP 矩阵共 7d² 个参数，合计与 TPP 的 7d²+d 相同。各卷积分支的有效时间长度不同，因此分别计算权重，不强行拼接不同长度的卷积输出。

mask 的手工可核算例子为四个样本、每样本四视图、标签 0/0/1/1：hybrid 有 52 个有向正对和 240 个分母对；all-view 为 112/240；移除同类非正对后为 52/180。两种原生 mask 的 loss 与现有 DomainAwareSupConLoss 数值完全一致。三种 mask 间比较只改 mask；与 V1 global-only 比较还改变视图和 BatchNorm 接触的输入分布，不能写成纯 mask 效应。

CORAL 通过同分布零惩罚、整体平移的解析值和数值梯度检查。参考实现固定为 [DomainBed commit b93c22a](https://github.com/facebookresearch/DomainBed/blob/b93c22a1cfc3b2428398272c1a116c8de1f4139e/domainbed/algorithms.py)，默认强度依据同版本的 [超参数登记](https://github.com/facebookresearch/DomainBed/blob/b93c22a1cfc3b2428398272c1a116c8de1f4139e/domainbed/hparams_registry.py)。本地保存来源和许可证；不将整个 DomainBed 训练流程称为已经复现。

冻结包在 `D:/pa-csi-dg-revision-runtime/studies/r_controls_v1`，bundle manifest SHA-256 为 `64c8032078d9a4dfa531b94e89fec9d9caf675b750bcabcb7881f76468a1aa92`；代码、六份来源/目标 manifest、模型计数与 QA 均纳入哈希。九类训练配置为 P1、NO_P1、ATTENTION、MASK_HYBRID、MASK_ALL、MASK_REMOVE、CORAL、MIXUP、TEMPORAL_AUG，各含三个环境和每环境三个种子；均为 200×300 更新、batch 16、最早最优源 macro-F1、零新增搜索。正式 TEMPORAL_AUG 仅按 seed 和 update 的独立 RNG 为每个训练样本从整数区间 [-85,85] 均匀抽取循环平移量，验证样本不增强。前表其余固定平移/打乱组件不冒充正式训练结果。

扩展训练入口 `extended_engine.py` 的核心 SupCon loss 与一次 Adam 更新经过独立公式比对，全部模型状态完全一致。实际 CORAL CPU 两轮连续训练与暂停/恢复轨迹的模型、Adam、RNG、history 和 1200 条源预测完全一致。九类目标在实际源 batch 上反向传播均为有限值。冻结评估器的负门控在缺失首个已完成状态时拒绝执行；所有 81 个源作业核验完成前不读取目标信号。详细证据在 `evidence/r_controls_v1/`。

训练入口在每作业开始前用单独模型测量推理和反向资源，然后重建正式训练模型并重置随机状态。四视图增加的计算必须按实测报告。相位处理和等初始化双流容量组件已另行通过 QA，见 PHASE_CONTROL_PROTOCOL.md；其完整数据准备、训练入口、冻结矩阵与正式运行仍为独立待执行项目。物理相位校准不能由 unwrap 或幅度重建替代。

核心第 96 轮的状态文件 I/O 失败曾使原扩展等待进程按门槛退出。失败已归档并核验核心恢复；当前由外部协调器等待核心进程成功退出，再启动相同冻结扩展代码。它不在核心运行时轮询可变 JSON，未改动任何扩展配置、训练函数或目标评估门槛。
