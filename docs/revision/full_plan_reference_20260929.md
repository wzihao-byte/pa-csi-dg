# 原始论文修改计划与导师意见（对话存档）

来源：论文修改方案，conversationId 6ab356ac-e1e0-83e8-a580-d15c4a3e8bad；2026-09-29读取。作为任务参考，实验事实仍须核查代码和存储证据。

## 导师意见

根据修改建议给我一个新的论文修改方案Dear Zihao,

Thank you for the manuscript. The manuscript currently reads more like a detailed experimental audit than a focused research paper. Please address the following points.

1. **Clarify the central research contribution.**

The strongest supported finding is that the evaluated TPP modules outperform their respective controls, whereas the additional benefit of contrastive learning depends on the configuration. Please organize the introduction, experiments, and conclusion around this finding. Clearly distinguish established results from hypotheses and questions that remain unresolved.

2. **Provide comparisons under a common protocol.**

H and Y differ simultaneously in source splitting, phase processing, pooling placement, control architecture, contrastive objective, batch size, and checkpoint selection. Therefore, neither the performance gap nor the reversal of the contrastive effect can be attributed to one factor.

Please prioritize rerunning the H architecture with subject-disjoint source training and validation while retaining its other settings. Comparing H10 and H11 would assess the sensitivity of the contrastive benefit to the split. All four arms would be needed to assess the pooling–contrastive interaction under the revised split.

3. **Separate the contribution of temporal information from additional model capacity.**

TPP increases the H parameter count by approximately 72%. Although Y matches parameter counts, its nonlinear global-basis control is unconventional.

Please include a conventional global-pooling-plus-MLP control with a comparable parameter count under the same training protocol. The R study described in Appendix C could help address this issue, but its design alone is not evidence; the recognition results must be reported. A control using only a transformed global descriptor would also help distinguish local-bin information from the additional global transformation.

4. **Address the preprocessing issues experimentally.**

The arithmetic averaging of wrapped phase, feature-axis unwrapping, and timestamp-dependent resampling are substantive issues. Please specify the complete processing sequence from raw CSI to the model input, including the axes used for averaging and unwrapping, interpolation, and missing-data handling.

In particular, explain whether feature-axis unwrapping crosses receive-link boundaries and why that operation is appropriate. Compare amplitude-only input, amplitude with historical phase, and amplitude with a justified alternative phase representation under matched conditions. Preparing corrected inputs or verifying sample identities does not establish their recognition performance or the usefulness of phase.

5. **Evaluate the hybrid positive-pair rule directly.**

The current hybrid objective leaves different-instance, same-class pairs involving masked views in the denominator without selecting them as positives. This may create competition with the intended class-level alignment.

Under an otherwise identical model and view configuration, please compare:

- The current hybrid rule.
- Same-class positives across all views.
- A rule that excludes these unselected same-class pairs from the denominator.

The corresponding studies mentioned in Appendix C would be useful if their results are available. Do not use the H–Y comparison to rank the loss definitions, because other settings also change.

6. **Make the architecture description reproducible.**

Table 2 and the symbolic definition of (F\_{\mathrm{gate}}) are insufficient to reconstruct the model. Please provide the MCAT structure, attention axes and dimensions, convolution settings, normalization and dropout, and the actual gated-fusion equations.

Also show the intermediate tensor shapes and specify the pooling axis in the channel-context path. This is necessary to determine whether pooling on that path should be described as temporal pooling.

7. **Strengthen the classwise analysis.**

Table 7 compares H00 with H11, changing both pooling and contrastive training. Please include H10 so that their classwise contributions can be examined separately.

For Y, provide the Y10/Y11 classwise scores and confusion matrices for E1, where adding SupCon reduces macro-F1 by 6.76 percentage points. Identifying which activities deteriorate would make the discussion more informative. The signal plots and source-gradient measurements should remain descriptive evidence, rather than explanations of the target-performance changes.

8. **Shorten and refocus the manuscript.**

Please reduce repeated statements about what the experiments do not establish. Keep the limitations that directly affect interpretation in the main text, particularly prior use of the target environments, correlated source validation in H, and preprocessing concerns. Move detailed development histories and execution records to supplementary material.

For studies in Appendix C, clearly distinguish completed experiments with reportable results from planned or incomplete work. Include results that resolve the main questions; descriptions of experimental plans cannot support the conclusions.

Please also replace “pseudo-Tx” with “receive-link feature groups” in the main figures, define LayerNorm clearly in Equation (7), and distinguish counts of positive performance differences from contrastive positive-pair counts. Report the two small residual coefficients as giving approximately equal performance rather than emphasizing negligible decimal differences. Finalize the publication statements and remove editing annotations before submission.

The highest priorities are the common-protocol, capacity-matched TPP comparison, the preprocessing and modality checks, and the matched comparison of contrastive positive rules. These would substantially strengthen the scientific conclusions.

Best regards,\
Ohtsuki

[User attached 2 files; file contents were not included]

## 完整修改计划

这轮修改建议按 **“补齐关键对照实验＋重构论文主线”** 来推进，而不是继续增加局限性说明。导师最关心的不是你的实验记录是否足够完整，而是：**在控制模型容量、输入处理和训练条件之后，TPP 的收益是否仍然成立；对比学习为什么在某些配置下有效、在另一些配置下无效。**这也是 Word 批注开头明确提出的研究方向。:chatgpt-content-reference{index="0"}

我已经整理成一份 **12 页的可执行修改方案**，包括实验矩阵、固定变量、结果验收标准、正文结构和逐条回复导师的要求：

:chatgpt-content-reference{index="18"}[下载完整论文修改方案（Word）](sandbox:/mnt/data/CSI_Revision_Plan_20260923.docx)

下面是这次修订的主要方案。

---

## 一、先重新确定论文的核心贡献

### 1. 不再围绕“完整模型达到 90.26%”组织全文

现有结果中，最稳定的发现是 **TPP 模块相对各自对照有收益，而额外加入对比学习的收益依赖配置**：

| 现有比较 | 已报告结果 | 当前能够支持的判断 |
|---|---:|---|
| H：TPP＋CE 相对 global＋CE | 平均提高 **3.68 个百分点** | 历史 H 配置下，完整 TPP 模块优于原始全局池化对照 |
| H：TPP＋hybrid 相对 TPP＋CE | 平均提高 **1.66 个百分点** | hybrid 在 H 的 TPP 配置下有额外收益 |
| Y：TPP＋CE 相对 global-basis＋CE | 平均提高 **10.47 个百分点** | Y 配置下，TPP 优于这一等参数对照 |
| Y：TPP＋SupCon 相对 TPP＋CE | 平均降低 **0.72 个百分点** | SupCon 并非稳定带来额外收益 |

以上增量采用稿件中基于未舍入结果计算的配对差值，而不是重新相减表中已舍入的均值。:chatgpt-content-reference{index="1"}

但这些结果还不能直接变成“时间信息带来了提升”：H 增加 TPP 时也增加了约 72% 的参数；Y 虽然匹配了参数量，其 global-basis 对照又不是常规的 global pooling＋MLP。:chatgpt-content-reference{index="2"}

### 2. 建议把全文改成三个研究问题

**RQ1：在共同协议和可比参数量下，时间分区池化是否优于常规全局池化？**

进一步区分：收益来自局部区间信息，还是来自额外的全局特征变换？

**RQ2：在时间轴和预处理明确的情况下，相位是否提供超出幅度输入及额外编码器容量的识别信息？**

**RQ3：在模型和四视图构造完全相同的条件下，正样本与分母规则如何影响识别表现？**

这样，论文的贡献会从“把几个模块组合起来并记录实验”转为：

> **通过受控比较，检验时间分区信息、相位输入与对比配对规则的实际贡献及适用条件。**

建议题目可以暂定为：

**Temporal-Region Pooling and Contrastive Pair Selection for Wi-Fi CSI Activity Recognition: A Controlled Study**

“Temporal”这一表述仍需经过池化轴核查；不能在尚未确认 channel-context 路径的池化维度前，把所有池化都称为时间池化。

---

## 二、第一步不是重新训练，而是清点已有实验产物

**“正文没有结果表”和“实验没有完成”不是同一件事。**

你的附录 C 已经描述了 R 四臂、三种 masked-view 规则、输入对照和 grouped H 补充，但当前正文没有提供足以支撑结论的识别结果。尤其是 R，稿件记录了源训练任务和后续目标加载步骤，却没有目标识别结果表。:chatgpt-content-reference{index="3"}

建议先建立以下清单：

| 实际可用状态 | 下一步动作 | 是否需要重新训练 |
|---|---|---|
| 有完整配置、样本身份及预测结果 | 重算 macro-F1、类别 F1、混淆矩阵和配对差值 | 通常不需要 |
| 有冻结 checkpoint、源验证选模记录，但没有目标预测 | 按原配置执行评估并保存逐样本预测 | 通常不需要 |
| 有训练日志，但 checkpoint 或配置对应关系不完整 | 先核对能否恢复可靠对应关系 | 视核对结果决定 |
| 只有设计、准备记录或未完成任务 | 补齐缺失实验 | 需要 |

**这一步要优先检查 R、MASK_HYBRID／MASK_ALL／MASK_REMOVE、输入实验和 H10/H11 grouped supplement。**这些正是导师点名需要结果的实验，而不是另起一批与核心问题无关的扩展。附录中列出的其他源侧诊断不能替代目标识别结果。:chatgpt-content-reference{index="4"}

---

## 三、补充实验方案：分成五个研究包

### 实验 A：把“容量匹配的 TPP 比较”作为主结果

建议优先采用附录 C 的 R 设计作为共同协议框架，完成以下四臂：

| 实验臂 | 表征模块 | 训练目标 | 主要用途 |
|---|---|---|---|
| R*00 | 常规 global pooling＋MLP | CE | 容量匹配基线 |
| R*01 | 同一个 global pooling＋MLP | CE＋full-view SupCon | 检查对比学习在全局表征上的收益 |
| R*10 | residual TPP | CE | 检查 TPP 的独立收益 |
| R*11 | 同一个 residual TPP | CE＋full-view SupCon | 检查对比学习在 TPP 上的额外收益 |

这里用 **R*** 表示本次冻结后的正式版本，避免与历史 R 中可能不同的输入或代码版本混淆。

共同协议建议以已有 R 设置为起点：主体不重叠的源训练／验证划分、三个留出环境、每环境三个种子、相同批次安排、优化器和更新预算，以及相同的源验证 macro-F1 选模规则。原稿已经记录了 R 的具体训练参数，因此应优先核验和复用，而不是再重新搜索一套最优超参数。:chatgpt-content-reference{index="5"}

**这个实验必须同时控制三件事：**

- **参数量**：匹配实际参与推理的可训练容量，不能靠未使用的层凑参数。
- **模块以外的配置**：输入、骨干、分类器、采样、训练步数和选模规则保持相同。
- **版本**：历史幅度输入和原生时间重建输入分别编号，不能混进同一个四臂矩阵。

在主比较之外，再加入一个 **global-only residual control**：只对全局描述符做额外学习变换，不使用局部区间描述符。例如，其设计形式可以是：

\[
h_{\mathrm{global}}
=\operatorname{Dropout}\left(g+\alpha\,\operatorname{MLP}(g)\right),
\]

而 TPP 分支使用：

\[
h_{\mathrm{TPP}}
=\operatorname{Dropout}\left(g+\alpha(Wz_{\mathrm{TPP}}+b)\right).
\]

这些是建议的对照形式，最终论文必须与实际实现一致。若采用附录中的 P1 控制，还需单独报告参数量；**P1 的支持性比较不能取代常规等容量 MLP 主对照。**

**验收标准：**四臂识别结果、实际参数量及配对差值全部可报告。只有两臂时，可以讨论 TPP 的 CE 收益，不能完整讨论 pooling–contrastive interaction。

---

### 实验 B：把预处理问题变成可检验的输入比较

这部分不能只补一段“历史处理存在问题”。

原稿已经承认：历史处理包括 wrapped phase 的算术平均、H 沿最终特征轴的额外 unwrap，以及受时间戳精度影响的重采样。样本映射成功只能证明数据对应关系，不能证明这种相位输入有识别价值。:chatgpt-content-reference{index="6"}

#### 先补完整处理链

建议把原始 CSI 到模型输入的处理写成一张独立流程表，至少明确：

| 处理环节 | 必须说明的内容 |
|---|---|
| 原始 CSI | 原始复数数组形状、接收链路与子载波排列 |
| 时间戳 | 时间单位、异常与回绕处理、重复时间处理 |
| 幅度与相位提取 | 在重采样前还是之后计算；各自的处理顺序 |
| 平均与 unwrap | 沿时间、子载波还是链路维度操作 |
| 插值与缺失数据 | 目标网格、插值方法、长缺口、端点和异常值策略 |
| 最终输入 | 如何得到 \(850\times90\)，后续是否还有标准化与相邻帧平均 |

尤其要核对：

> 若 H 确实直接对按三个接收链路拼接的 90 列执行最终轴 unwrap，该操作就会跨越第 29→30、59→60 列的链路边界。必须检查实际代码，说明这种跨边界操作的依据；不能仅因为历史上如此实现就把它视为合理处理。

替代相位可以优先检查附录 A 已准备的原生时间相位和接收链路差分相位。但它们仍然是**候选表示**，不是已经验证的物理校准结果；识别是否受益必须由实验回答。:chatgpt-content-reference{index="7"}

#### 再做模态对照

建议在固定 TPP＋CE 的条件下比较：

| 输入臂 | 输入与结构 | 要回答的问题 |
|---|---|---|
| A | 幅度，单编码器 | 幅度基线 |
| A＋A | 幅度分别进入两个独立编码器 | 增加第二个编码器本身能带来多少收益 |
| A＋历史式相位 | 双编码器，使用明确记录的历史相位处理方式 | 历史相位是否提供有效信息 |
| A＋替代相位 | 双编码器，使用有依据的替代相位表示 | 替代表示是否优于历史处理和容量对照 |

**A＋A 很重要。**否则 A＋phase 高于 A，仍不能区分“相位信息”与“多了一套编码器”。

这里还有一个必须避免的新问题：**不能把原生时间网格上的幅度和旧时间网格上的历史相位，仅按行索引直接配对。**

建议将输入明确区分为历史版本和原生时间版本。在原生网格上重建“历史式相位处理”时，要另设版本名，不能称它与原始历史数组完全相同；原始历史幅度＋历史相位保留为独立的处理链敏感性比较。

另外，建议在 **global-MLP／TPP 两种 CE 模型**上比较历史幅度与原生时间幅度，检查 TPP 的结论是否依赖历史重采样。这比只给一个“修正后输入”的高分更有解释力。

---

### 实验 C：直接比较三种 contrastive positive-pair 规则

这组实验只改变 **正样本与分母的选择规则**，不改变模型或视图构造。

| 规则 | 正样本 | 未被选为正样本的“不同实例、同类别、涉及 masked view”配对 |
|---|---|---|
| Hybrid | 同实例跨视图；不同实例同类的 full–full | 保留在分母 |
| All-view same-class | 所有视图间的同类别配对，排除自身 | 作为正样本，并保留在分母 |
| Remove-unselected | 正样本集合与 Hybrid 相同 | 从分母排除 |

现有 H 的确使用第一种规则，而 Y 只使用 full-view SupCon，不包含 H 的 masked views。因此，**不能用 H 的 masked-pair 问题解释 Y 的性能下降，更不能据 H/Y 排名损失定义。**:chatgpt-content-reference{index="8"}

建议固定同一个 TPP 模型和同一种输入，并确保三组实验具有相同的：

**full＋三个 masked views、前向调用、投影头、初始化、采样、CE 作用位置、温度、损失权重、warm-up 与选模规则。**

在训练之前，先用小型人工标签批次验证三种 mask：哪些配对为正样本、哪些只进入分母、哪些被排除，必须与公式完全一致。

**验收标准：**报告每环境识别结果、配对差值和类别变化。损失值、正样本数量及源梯度指标可以辅助描述，但不能替代识别结果。

---

### 实验 D：H 的主体不重叠桥接实验，只改 source split

这组实验必须与新的共同协议区分开。

**不要在修改 source split 的同时，顺便修正 H 的相位、batch size 或 checkpoint selection。**否则又会形成一个新的 H/Y 式多因素比较。

建议记为 **H-S**，保留历史 H 的架构、输入处理、四路径 TPP、levels 1/2/4/8、batch 32、训练预算、hybrid loss、无 warm-up，以及“源验证 accuracy 最优、并列取最后”的选模规则，只改变源训练／验证样本成员，使主体不重叠。历史 H/Y 的这些差异在原稿表 4 中已经列明。:chatgpt-content-reference{index="9"}

| 完成范围 | 运行规模：3 环境×3 种子 | 能回答的问题 |
|---|---:|---|
| H-S10、H-S11 | 18 个单元 | TPP 条件下，对比学习收益对源划分是否敏感 |
| H-S00、H-S01、H-S10、H-S11 | 36 个单元 | 进一步比较完整 pooling–contrastive interaction |

资源有限时，先保证两臂完成并如实限制结论；不能只有 H-S10/H-S11，却声称验证了四臂交互。

同时继续区分：**源训练／验证相关性不等于目标样本进入训练**。桥接实验也不能把已经参与过开发的 E1–E3 重新变成“从未接触过的测试环境”。:chatgpt-content-reference{index="10"}

---

### 实验 E：补类别分析，而不是再增加信号解释图

这一部分优先从现有预测结果恢复，通常不应先考虑新训练。

**H：把 Table 7 改成 H00／H10／H11。**

| 类别 | H00 | H10 | H11 | H10−H00 | H11−H10 |
|---|---:|---:|---:|---:|---:|
| 六个活动分别列出 | 已有 | 待导出 | 已有 | TPP 条件变化 | 额外 contrastive 变化 |

现有“falling 和 turning 贡献约 82%”对应的是 **H11−H00 的联合变化**，不能直接当成 TPP 单独作用。加入 H10 后，才能拆分两部分。:chatgpt-content-reference{index="11"}

**Y：补 E1 的 Y10／Y11 六类 F1、混淆矩阵和混淆差值图。**

这里要回答具体问题：下降集中在哪些活动？是某个类别召回下降，还是某类错误吸收了其他类别？不能在没有类别结果前预设答案。导师特别要求分析的正是 E1 的 **−6.76 个百分点**。:chatgpt-content-reference{index="12"}

主指标继续采用“先计算各运行指标，再平均种子和环境”的口径；可视化用的汇总混淆矩阵不能替代主表的计算方式。

---

## 四、方法部分：从“介绍模块”改成“足以复现”

当前 Table 2 提供了一些输出维度，但 gated fusion 仍被写成一个抽象的 \(F_{\mathrm{gate}}\)，不足以重建模型。:chatgpt-content-reference{index="13"} :chatgpt-content-reference{index="14"}

建议新增一张 **逐层张量形状表**，并补齐下列内容：

| 模块 | 需要从实际代码确认并写入论文的内容 |
|---|---|
| MCAT | 全称、子模块顺序、层数、残差连接 |
| Attention | token 对应哪条轴、输入与内部维度、head 数量 |
| Convolution | 输入／输出通道、kernel、stride、padding、激活 |
| Normalization／Dropout | 位置、归一化维度、具体参数与概率 |
| Channel-context | 接收链路平均前后的形状、转置操作、最终池化轴 |
| Gated fusion | 两个 288 维描述符如何变成 256 维表示的真实公式 |
| Projection head | 层结构、激活、归一化和输出维度 |

特别需要先核实 channel-context 路径：

> **沿时间轴池化，才能按该操作描述为 temporal pooling；沿特征或子载波轴池化，就需要修改术语。**

若要进一步声称“时间分区信息有效”，而当前模块同时改变了非时间路径，建议补充固定 context 路径、只改变 temporal 路径的比较。任何因此产生的架构改动都应设为新版本并重新计算参数量。

**不要从原论文或常见实现中补一个看起来合理的 gate 公式。要以实际用于实验的冻结代码为准。**

---

## 五、正文结构：让研究问题决定结果顺序

建议重组如下：

| 部分 | 新版重点 |
|---|---|
| Introduction | 提出三个研究问题；说明需要区分容量、输入处理和配对规则 |
| Related Work | 聚焦 CSI 表征、池化与对比学习，删减与主实验无直接关系的铺陈 |
| Methods | 完整预处理、可复现架构、池化对照、三种正样本规则 |
| Experimental Protocol | 数据划分、输入版本、训练预算、选模与统计口径 |
| Results | **容量匹配 → 预处理／模态 → 配对规则 → H-S → 类别分析** |
| Discussion and Conclusion | 回答研究问题，保留直接影响解释的限制 |

H/Y 历史结果仍然保留，但从“整篇论文的主要证明”调整为**研究动机和历史参照**；Y 的不利结果不能删除。

详细开发过程、执行日志、重复训练差异、取消的 Widar 准备、较长的源梯度和信号诊断移到补充材料。正文集中保留三类关键限制：**目标环境此前参与过开发、H 的源验证相关性、历史预处理问题。**这与导师要求的精简方向一致。:chatgpt-content-reference{index="15"}

摘要和结论最后再写，等关键实验结果出来后确定措辞。不要现在就预写“TPP 在等容量条件下显著更优”或“修正相位提高了泛化”。

---

## 六、实际执行顺序与完成标准

建议按以下顺序推进，而不是先把 27 页正文全部润色一遍：

| 阶段 | 主要产出 | 完成标准 |
|---|---|---|
| 1. 结果清点 | 已有预测、checkpoint、配置与缺失实验清单 | 分清“缺报告”“缺评估”“缺训练” |
| 2. 方法冻结 | 输入处理链、张量形状、池化轴、共同协议 | 不再存在影响主要比较的隐含配置 |
| 3. 核心实验 | 容量、输入／模态、三种配对规则；H-S 可并行 | 每个预定比较都有识别结果 |
| 4. 分析整理 | 配对差值、H10 类别结果、Y E1 错误变化 | 数字、标签及汇总口径一致 |
| 5. 正文与回复信 | 重写摘要、结果、讨论及逐条回复 | 每项结论都对应实际表格或图 |

按三个环境、三个种子计算，一个实验臂是九个运行单元。**这表示实验矩阵规模，不等于必须新增这么多训练任务。**已有且版本匹配的结果应复用；原稿没有报告的结果应先恢复。

本轮建议暂缓 CORAL、MixUp、更多 temporal augmentation、全量超参数搜索和新数据集扩展。它们不能替代导师点名的三类关键对照。

提交前还要统一清理：

- 图中的 **pseudo-Tx → receive-link feature groups**；Equation (7) 明确写 LayerNorm，并说明维度与参数。
- 性能改善次数写成 **counts of positive paired differences**，与 contrastive positive pairs 区分。
- \(\alpha=0.125\) 和 \(0.25\) 均报告为约 **90.26%**、表现相近，不突出微小小数差异；“平均提高 5.34 个百分点”和“九个配对比较均提高”分开陈述。
- 单位、作者、基金、伦理与数据使用声明、代码公开标识均由实际情况确认，删除编辑批注与待确认文字。:chatgpt-content-reference{index="16"} :chatgpt-content-reference{index="17"}

**这次修订完成的标准，不是重新得到一个 90.26% 的结果，而是让每个关键结论都有同协议、可复现、能够区分替代解释的实验支持。**即使等容量对照削弱了 TPP 收益，或替代相位没有改善识别，也应据此收窄结论；这样的论文会比继续强调历史最高分更有说服力。

