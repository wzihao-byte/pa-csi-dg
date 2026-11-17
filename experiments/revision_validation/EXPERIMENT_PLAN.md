# 后续修订实验：执行顺序与开跑条件

**2026-09-10 范围更新：用户取消 Widar。** 现行 R 实验只保留 MultiEnv 的
36 core、81 controls、63 inputs（共 180 次）及 117 个 source 分析。
旧 216 次及其 Widar 完成门不再适用。另接原 H10/H11 L1248 的 18 次分组验证补充实验，
见 `experiments/h_grouped_l1248/EXPERIMENT_PLAN.md`；它保留原 H 的参数与 checkpoint 规则，
不套用本文件 R 实验的 batch、macro-F1／earliest-tie 等设置。

状态（2026-09-24）：R core36、controls81、inputs63 的完整 source 训练和目标评估均已核验，117 source-only 分析已完成；所有 R 结果仅作技术审计。独立 H18 与 H/Y 写作/PDF 已完成，发布终态见 COMPLETION_AUDIT。下述日期段落保留预登记执行历史。恢复检查、固定零搜索预算及目标评估门槛见 `docs/revision/CORE_STUDY_PROTOCOL.md`。不得与已冻结 H/Y 记录混写。E1–E3 明确属于开发环境。

## A. 数据与共同协议

1. **已完成**：解决初始 2/90 差异，全量 9000 行 raw→processed 唯一匹配且误差为零；30 subject、3000 acquisition 分组与 3 份 sidecar 已保存，实际数组哈希重新核对。241 条时间戳数值经精度等价表示变化，不能称为完成物理时间校准。
2. **输入重建与冻结已完成**：9000 份 DAT 的 packet ID 与复数 CSI 均和 MAT 一致；两套幅度及历史/直接/差分三套相位 float32 数组已独立版本化并保留原样本顺序、标签和身份划分。历史幅度与原生 merge 有 3744 条大于 1e-5 的差异，与原生 linear 的全部 9000 条不同；E2 最大 packet gap 为 2.015493 s，所以 linear 仅作为敏感性对照。相位实现不跨链路 unwrap，共轭差分也不称为完整校准。七类 63 作业已通过源输入/恢复 QA、冻结、全部训练及目标评估核验。
3. **划分已完成**：外层保留环境；内层沿用 Y 的原有行集合，接入已验证 subject/acquisition 并确认不交叉。source manifest 与 deferred target 分开，固定完整样本 ID 与类别支持。新 ID 使用 ENV:环境内索引，历史 Y manifest 的 loader 偏移按实际代码恢复后核对。
4. checkpoint 采用源验证 macro-F1，tie 取最早；固定 optimizer、总更新数、batch、采样器与调参候选预算。Y 的身份隔离已由全量信号及冻结 sample/indices 哈希事后确认；其旧 manifest 的 inferred-block 字段仍保留。

## B. 最小核心比较

在共同协议上先跑以下 2×2，骨干、模态、池化路径、训练更新数保持一致：

| Arm | 表征 | 目标 |
|---|---|---|
| R00 | 常规 global 控制 | CE |
| R01 | 同一 global 控制 | global-only SupCon + CE |
| R10 | 明确定义的 residual TPP | CE |
| R11 | 同一 residual TPP | global-only SupCon + CE |

R 为独立候选模型标签，其完整结果保存在技术审计中，不替代 H 证据，也不重新纳入本 H/Y 写作范围。当前实现采用 amplitude-only、同一幅度骨干与 256 维表征；global 控制为常规 d→7d/2→d MLP，与 L124 的 7d→d 投影同参数、同矩阵 MACs，四臂总参数均为 1,878,142。smoke 另记录梯度参与参数、计数范围明确的 MACs、实测延迟与初始残差范数；等参数不等于函数族和残差强度相同。

H 的自由 residual 是可合法研究的原模型，只需准确解释 alpha；不强迫为了措辞改模型。若研究受控范数 residual，需另立版本，明确 affine、归一化维度、epsilon、近零行为，单独比较，不能继承 H11=90.26%。

## C. 按问题增加单因素控制

| 问题 | 必要对照 | 不能得出的结论 |
|---|---|---|
| 局部信息还是 P1/容量 | P1-only、去 P1、常规参数匹配 global、attention pooling | 仅同参数数量不能证明局部信息是唯一原因 |
| 伪 Tx/mask 的作用 | 同四视图下 H hybrid、standard all-view SupCon、同类非正对从分母移除；另列 V1 global-only | 不从 H vs Y 总分差归因于 mask |
| 模态的作用 | amplitude-only、phase-only、dual-stream；各自匹配核心结构 | Y 的 amplitude-path TPP 不等于 amplitude-only 模型 |
| 位置与动态机制 | 时间平移、局部打乱、temporal augmentation；固定规则及强度 | 不能只根据混淆矩阵解释时序因果机制 |
| 通用 DG 竞争力 | 同骨干 ERM、CORAL、Mixup/域平衡正则等 2–3 个代表方法 | 不拿其他论文不同数据/划分的数值作直接胜负比较 |

机制指标同时保存实际 residual/global 范数、近零分母数、类间/类内散度、源环境条件散度及 CE/SC 梯度。先按 run/环境汇总，时间点与 seed 都不冒充环境重复。

2026-09-04：P1、去 P1、attention、三种四视图 mask、CORAL、Mixup 与时间增强共九类正式对照已冻结为 `r_controls_v1` 的 81 个作业。独立恢复入口 QA 和延后目标评估的负门控通过，等待队列已启动；其依赖是核心 36 个源作业及冻结评估全部通过。各作业完整训练预算仍为 200×300 更新，零新增搜索。模态与原生时间处理另立版本，不修改这两套运行中的代码快照。

同日，原生 merge/linear 幅度、历史/直接/差分相位及 TWIN_AA/TWIN_AP 共七类正式输入对照冻结为 `r_inputs_v1` 的 63 个作业，bundle SHA256 为 `4c49208ea104e05f18f338501342dcdc73aa03fe836c95cc1a56dcd0479d6619`。真实源数据 QA、双流 CPU 中断恢复和两项负门控通过；协调器只在核心与扩展均完成评估后启动。这是冻结时状态；63 项输入研究现已完成。

## D. 调参、计算与停止

先跑源数据 smoke 检查 shapes、loss/mask、step 数、参数、前向/反向、时间和显存。方法比较共用源域选择预算；不能给新方法大搜索而不给基线。

已完成 12 次 30 步源 smoke，以及一个完整 epoch（300 更新）的源训练入口试跑；恢复 checkpoint 后 1200 条源验证预测完全一致。smoke 的批量 16 推理中位约 14.7–15.7 ms，优化更新中位约 .084–.092 秒；完整 epoch 试跑耗时约 26.4 秒（该计时从完成来源加载后开始）。这些短运行用于代码和资源核验，不作为目标准确率结果，也不保证长时训练耗时。

新 R 的单 epoch 试跑约 26.4 秒，36-run × 200 epoch 按该吞吐约 53 GPU 小时，尚不含加载、checkpoint I/O、失败重跑和额外对照，也不保证长时训练维持该速度。正式训练现已全部完成，实际资源数据保存在技术审计；上述 smoke 外推只作历史估计。四臂均不新增超参数搜索，固定 λ=0.025；详见冻结记录。

已完成 Y 的 stop 保持有效，不继续补 seed 来稀释 E1 退化。新矩阵预先记录停止规则及完整运行登记，保留失败、重跑和偏离说明。

## E. 独立确认

完成开发选择后，冻结模型、数据处理、源选择规则和分析；使用未参与开发的新环境。第二数据集若重新训练，称跨数据集复现；只有固定模型直接迁移才称直接 transfer。无法取得新环境时，论文继续定位回顾性研究。

2026-09-10：用户取消 Widar 实验；训练和目标评估从未开始。此前完成的 6749 条非空库存核算、6748 valid + 1 具名 signal exclusion、编译、QA、负门和 36-job 冻结 bundle 只作为取消前准备记录保留，不进入现行性能证据、run 总数或独立确认宣称。现行补充证据改为原 H10/H11 L1248 的 18-run grouped source train/validation 协议；E1--E3 已用于开发，所以该补充也不构成独立跨数据集确认。

## 完成标准

每项实验必须有冻结配置、样本/分组身份、参数与计算量、source-only 选择记录、完整 run、checkpoint/预测哈希和复算结果，才可在论文中标“完成”。此计划没有代替任何新实验结果。

全计划还必须依次通过 `docs/revision/COMPLETION_AUDIT.md` 中的现行正式证据、表图/叙述、逐页 PDF、本地发布和终态文档同步门。最后由 `analysis/verify_multienv_revision_completion.py` 复核 180 个 MultiEnv R、117 个 source-only 分析、18 个 grouped-H 补充 run、198 个现行作业的采样覆盖、180+18+72 个 checkpoint、发布目录/ZIP 及 12 个终态文档哈希；只有成功生成 `docs/revision/evidence/final_multienv_completion_audit.json` 才构成技术完成证据。提前调用已按预期拒绝且未生成该文件。
