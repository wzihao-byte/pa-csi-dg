# Widar3.0 外部数据集确认协议

2026-09-24 复核：取消边界不变，Widar 训练和目标评估均未启动；下述冻结设计、来源及准备步骤只作历史证据。已完成的 MultiEnv H18 也不构成独立跨数据集确认。

状态（2026-09-10）：用户已明确取消 Widar 实验。等待进程及依赖 Widar 的旧汇总/发布链已经停止，完整训练和目标评估均未开始。此前完成的公开数据选择、6749 条非空库存逐条核算（6748 valid + 1 具名 signal exclusion）、编译、真实源 QA、36-job 冻结 bundle 和两项负门控继续作为取消前的可审计准备记录保留，不构成性能证据，也不得恢复训练或写成跨数据集确认。取消记录为 `evidence/widar_cancelled_20260910.json`。

## 预登记数据范围

候选来自 [Widar3.0 官方项目页](https://tns.thss.tsinghua.edu.cn/widar3.0/index.html) 指向的 [清华公开数据分享](https://cloud.tsinghua.edu.cn/d/2760bb9557ca4d09a74d/) 中的 CSI 文件；源压缩包仍由数据提供方托管，本项目保存选择清单、成员元数据、哈希、处理规则和派生数组，不把第三方源压缩包重新包装成自有数据。固定使用三个房间、每个房间三名受试者、接收端 r1、位置/朝向/重复编号 1–5，以及六个各房间共有手势：Push&Pull、Sweep、Clap、Slide、Draw-O (Horizontal)、Draw-Zigzag (Horizontal)。选择依据是官方 README 元数据和公开压缩包可用性，没有读取模型输出。

| 房间 | 受试者 | 非空库存 | 有效信号 | 作为源房间时的验证受试者 |
|---|---|---:|---:|---|
| W1 | user5, user10, user11 | 2250 | 2250 | user11 |
| W2 | user1, user2, user6 | 2250 | 2250 | user6 |
| W3 | user7, user8, user9 | 2249 | 2248 | user9 |

清单共有 6749 条非空记录。另有 `20181211/user9/user9-1-1-1-1-r1.dat` 在官方压缩包内为零字节，按预先写定的规则未进入非空库存；因此库存中 class 0 有 1124 条，其余五类各 1125 条。v2 又确认库存绑定的非空成员 `20181211/user7/user7-2-2-3-5-r1.dat` 长度和 ZIP CRC 正确，但 Intel 5300 record length 无法解码，登记为唯一 signal exclusion（raw SHA256 `898a4ab4…f1f9b`），不补选样本。最终有效数据的 class 0/1 各 1124 条，其余四类各 1125 条；房间之间没有重复受试者。

v1 在 3440 条记录已写入并哈希后停止：`20181130_user5_10_11.zip` 的公开签名 URL 过期返回 HTTP 403；上述非空异常成员的长度与 CRC 匹配冻结清单，但内部 DAT record length 无法由 Intel 5300 格式解码，时间戳入口也独立判为截断/无效。v1 及日志保留。v2 不改变幅度/时间处理，对 v1 成功项逐个重算哈希后 hard-link 或复制。按实现对成功首读记 0、对读前已发生的失败次数累计，`network_sessions_refreshed` 在两份需重试的压缩包中分别为 6 和 5，其余四份为 0；这表示合计 11 次重试性会话刷新，不是 6 个样本替换。完成 summary SHA256 为 `88a8cea2…f0b2`；编译 manifest SHA256 为 `d174c7e3…0769`，16 个列出的产物均已重新核验。

## 固定划分与训练矩阵

每次留一房间为延迟目标，另外两个房间为源。每个源房间仅把预登记的一名受试者放入源验证，其余两名进入源训练。实际冻结折为：W1 目标时 train=2999、val=1499、test=2250；W2 目标时 train=2999、val=1499、test=2250；W3 目标时 train=3000、val=1500、test=2248。三折的 sample ID、raw path、subject 和 acquisition 在 train/validation/test 间均为零交叉；源验证大小也与 36 个冻结 job 配置一致。

训练使用与核心 R 相同的 R00/R01/R10/R11 架构、Adam、学习率日程、batch=16、200 轮×每轮 300 次更新和固定 SupCon 强度，不新增超参数搜索。W1 seeds=701–703，W2=801–803，W3=901–903；四臂同目标、同 seed 配对，共 36 个作业。checkpoint 只按源验证 macro-F1 选择最早最优轮次，不做基于性能的提前停止或补种子。

冻结设计原本要求在全部 36 个源训练作业完成且哈希、checkpoint、源预测和指标都通过验证前，评估入口不得构造目标信号 loader，并等待 MultiEnv 36+81+63 作业完成。用户取消发生在该等待阶段，因此这些训练/评估门没有被触发；现行流程不得把冻结计划解释为已执行结果。

## 结果解释边界

这一段只记录取消前的设计意图：若执行，本来是在另一公开数据集内重新训练并比较固定方法效应，而不是直接任务迁移。由于实验已取消，论文不能给出 Widar 分数、跨数据集复现结论或独立确认宣称。不同手势标签及 room--subject 混杂仍作为解释为何准备记录不能替代性能证据的限制保留。

## 冻结与启动门槛

1. **已通过**：6749 条非空 DAT 全部核算为 6748 valid + 1 具名 exclusion；逐条原始/幅度/record 哈希通过，失败版本保留且没有补样本。
2. **已通过**：`widar_native_amplitude_v2` 数组和六份 source/deferred-target manifest 编译完成，房间、subject、acquisition、类别与实际样本数复核通过。
3. **已通过**：四臂真实源 batch 的目标函数、初始化、有限梯度及两条 CPU 中断恢复轨迹通过，QA 明确 `target_data_loaded=false`。
4. **已通过**：代码、manifest、数据摘要、成本、36 个配置、零搜索预算和解释边界已冻结；bundle SHA256 为 `2a232569…fdbb0`。提前目标评估和提前源训练均以非零状态拒绝，且未留下 target/prerequisite gate、failure 或 runner lock。
5. **已取消**：用户在输入/模态研究完成前取消 Widar；等待进程已退出，训练子进程为零，源训练和延迟目标评估不得启动。

当前实现入口：`experiments/revision_validation/widar_study_design.py`、`widar_study_qa.py`、`freeze_widar.py`、`widar_study_engine.py`。v2 的 36 个正式配置、唯一 job ID、统一 1,878,142 参数量、错误 fold 拒绝、真实源输入/目标函数、配对初始化和 CPU 恢复均已通过；这些检查没有读取目标信号，也不是完整训练结果。预登记清单保存在 `evidence/widar_inventory_v1/`，冻结 bundle 位于 `D:/pa-csi-dg-revision-runtime/studies/r_widar_v2/`。
