# 已有 H-S 队列核查

2026-10-14 16:15 日本时间。范围：只读既有运行和进程，不改变作业、配置、检查点或训练状态。机器记录见 [existing_queue_snapshot.json](existing_queue_snapshot.json)，可重核脚本见 [audit_existing_queue.py](audit_existing_queue.py)。

现有队列仍为 **25／54 验收完成，剩余 29 次**。当前作业 `HS_E2_P_REMOVE_42` 为 `attempt_002`；runner PID 18948 和训练进程 PID 22804 在核查时均存在。`training_status.json` 于 16:13:58 记录 10,300／30,000 optimizer updates；`source_progress.json` 于 16:14:49 记录 epoch 69、10,350 updates，两者均为 `target_opened=false`。

`queue/status.json` 的时间为 13:42:34，来自进入此作业的队列写入。冻结 `bundle/code/run.py` 在每个作业开始和完成时更新队列状态，冻结 `train_factorial.py` 分别更新逐 100 step 状态和源验证进度。因此，不能把较旧的队列时间解释为停止训练或失去心跳。最新日志文件写入时间为 16:15:14。快照同时记录采集起止时间，活跃作业文件可在此期间继续更新。

| 臂 | 已验收／预定 | 当前状态 | 可作结论的范围 |
|---|---:|---|---|
| G10，global MLP＋CE | 9／9 | complete | 全部三个环境与三个种子 |
| G11，global MLP＋原 hybrid | 9／9 | complete | 全部三个环境与三个种子 |
| P_ALL，all-view 同类正例 | 4／9 | partial | E1 三种子与 E2/42；总体均值为空 |
| P_REMOVE，排除未选同类分母项 | 3／9 | partial | E1 三种子；总体均值为空 |
| N_H11，native historical-style AP | 0／9 | partial | 尚无已验收结果 |
| N_D11，native differential AP | 0／9 | partial | 尚无已验收结果 |

独立审计核对冻结bundle的109个代码／配置／登记成员，并对25个已验收attempts检查全部completion manifest成员的长度和SHA256、completion/seal绑定、配置、初始状态与采样记录、源sample/subject/acquisition互斥关系。读取checkpoint字节计算哈希，没有反序列化或运行模型。对75,000条保存预测重算六类macro-F1和accuracy，最大绝对差为**1.4211×10⁻¹⁴ 个百分点**。另核对18个历史H-S10/H-S11引用的配置、指标及预测哈希。此审计未独立replay检查点选择，也未重读原始数组或执行推理。

G10/G11 与已封存 H-S10/H-S11 可组成完整的等参数四臂比较。下表引用当前 `analysis/results.json` 和冻结 `bundle/historical_references.json`；均值先按运行计算 macro-F1，再按种子与环境等权平均。所有模型实例化参数量为 4,786,294，包含共同的训练投影头。

| 臂 | E1 macro-F1 | E2 | E3 | 环境等权均值 |
|---|---:|---:|---:|---:|
| G10，global MLP＋CE | 86.3478 | 84.6395 | 84.8479 | 85.2784 |
| G11，global MLP＋hybrid | 86.2188 | 79.7741 | 82.6555 | 82.8828 |
| H-S10，TPP＋CE | 88.0664 | 88.9086 | 87.0853 | 88.0201 |
| H-S11，TPP＋hybrid | 89.3812 | 90.8028 | 89.3140 | 89.8327 |

全部九个配对增量保存在快照中。TPP 相对 global MLP 的 CE 增量为 +2.7417 pp，hybrid 增量为 +6.9499 pp；global MLP 的 hybrid 增量为 −2.3956 pp。描述性交互为 +4.2082 pp（+6.9499306999−2.7417021043）。这组对照说明所测试的 TPP 与此 global MLP 控制的差异，不证明一般物理域不变性，也不把九个配对块称为九个独立环境。

本次验收只推进当前队列的证据登记。2026-10-08 S1 的 **315 次 unique runs／945,000 条预测**仍是原独立包统计；当前 25 次／75,000 条不直接追加到那个包。新正文与补充材料集成时，必须重新建立去重索引、代码／配置／预测哈希绑定与可移植核验脚本。partial 正对规则不能用于倒选 WiMANS 主规则；MultiEnv 目标参与过历史开发的局限仍然成立。

来源：`D:/pa-csi-dg-revision-runtime/studies/h_main_controls_20261008_v2/{analysis,queue,bundle,runs}`，具体文件路径和哈希均在快照中。原冻结队列保持运行，本任务未新开训练。
