**WiMANS 双流跨房间实验与 IoT Journal 和 OJ-COMS 投稿计划**

日期：2026 年 10 月 14 日，日本时间。主投 IEEE Internet of Things Journal，后续备选 IEEE Open Journal of the Communications Society。两刊共用实验、图表和复现材料，分别调整论文重点和模板。

**本轮推进与协议修订**

本轮已完成固定 1,782 条原始信号的取得和存储 v2 全量输入验收、独立九类适配实现与 CPU QA、三折真实源样本的轻量前向／fresh reload、作者基线审核及 optimizer 修复原型。已生成 36 个四臂候选和 9 个 CORAL 源搜索候选；新正式训练尚未开始。完整源训练 QA、作者表示适配和执行冻结仍待完成。当前证据入口为 [进度快照](progress_snapshot.json) 与 [执行说明](PREPARATION_EXECUTION.md)。

| 工作项 | 本轮完成内容 | 验收边界与下一步 |
|---|---|---|
| 已有队列 | 25／54 accepted attempts 的封存文件、配置、身份和全部 75,000 条新预测复算；原 runner 仍在工作 | 不改变原 54 次范围；partial 臂不生成总体结论；见 [队列核验](EXISTING_QUEUE_STATUS.md) |
| H-S 等参数四臂 | 原 H-S10／11 与新 G10／11 各九次已齐，公共协议及预测依据复核 | 已可用于主表；全队列完成数仍为 25，历史两臂不重复计入新队列 |
| 原始输入 | 全部 1,782 MAT 的身份／版本／字节通过；全量解码及幅相矩阵哈希通过，27 个房间×类别代表记录独立重建通过；无排除或替换 | 存储 v2 数值输入验收完成，物理导出来源仍未确认；见 [原始验收](RAW_ACQUISITION_QA.md)、[完整输入验收](prepared_input_verification.json) 与 [输入核验](INPUT_AXIS_AUDIT.md) |
| 轴与置换 | 标准 CSI Tool 约定已复核，作者固定 tree 无 MAT 导出源，不能确认物理映射 | 按预设替代路径登记 [storage_order_v2](axis_storage_protocol.json)，不再把软件首轴索引 0 写成已确认的物理 Tx0；不再应用 perm |
| 九类实现 | 独立 adapter／模型／源训练与离线评价接口；12 项 CPU 实现检查通过；三折真实源 batch1 前向及 fresh strict reload 通过；45 个候选配置完成绑定核验 | 轻量前向不替代 batch32／masked96 完整源短训或授权冻结；见 [实现核验](IMPLEMENTATION_QA.md)、[候选与源 shape 核验](candidate_configs_qa.json) |
| MA 作者基线 | 固定作者 commit，实际目标函数和原实现 CPU 验证完成；neck optimizer 修复原型的 4 项作者原类 CPU 测试通过 | 方法条件性合格，完整执行仍未合格；作者原始表示／H-S adapter、源预算／验证／恢复及 code-v7 分歧尚需登记；见 [作者基线审核](AUTHOR_BASELINE_AUDIT.md) 与 [改编原型](../../../experiments/arcfi_hs_20261014/ADAPTATION_PLAN.md) |
| 两刊正文 | 共用章节结构、主协议表、相关工作比较表及方法段落已建立 | WiMANS 与尚未完成的基线／成本测量不写成结果；见 [正文结构](MANUSCRIPT_STRUCTURE.md) |

storage_order_v2 在任何 WiMANS 目标模型评价前登记，不改变子集、主体分组、类序、训练预算、相位运算和辅助视图列切片。原始物理导出来源仍 unresolved；新协议仅固定存储索引及其解释边界，后续若取得导出证据，另存核验，不根据目标成绩倒选输入。

采用 MultiEnv 与 WiMANS 两套数据评价幅度和相位双流中的残差 TPP 与多视图 SupCon。WiMANS 使用 5 GHz 单人九类子集，在固定主体分组的源验证条件下做三折留一房间测试。核心新增正式训练为 72 次；已有队列尚余 29 次，另计。数据划分及存储 v2 输入质量核验已完成；完整源训练 QA、基线适配和执行配置冻结仍是正式训练的前置步骤。

研究问题是：等参数条件下保留局部时间位置是否有价值；指定多视图训练与 TPP 的组合收益是否在第二数据集延续；额外辅助前向与对比损失梯度各自如何影响结果；预测可靠性需要多少计算成本。所有实验同时保留支持和不支持这些问题的结果。

**现有实验复用与当前队列**

| 内容 | 2026 年 10 月 14 日核查状态 | 后续动作 |
|---|---|---|
| 历史 H、H-S、R、Y 与时间和模态支持控制 | 最新稿与 S1 已有相应记录 | 按各自代码和协议复用，保持负结果及重训差异的解释边界 |
| H-S 等参数 global MLP 的 CE 与 hybrid 两臂 | 各 9 次完整完成 | 与既有 TPP 两臂组成等参数四臂主表，无需重跑 |
| H-S all-view 规则 | 4／9 完成 | 原队列剩 5 次 |
| H-S 排除未选同类分母项规则 | 3／9 完成 | 原队列剩 6 次 |
| H-S native historical-style 与 differential AP | 各 0／9 完成 | 原队列剩 18 次 |

当前固定执行范围见 [EXECUTION_PLAN.txt](../../../experiments/h_main_controls_20261008/EXECUTION_PLAN.txt)。核查结果入口为 `D:/pa-csi-dg-revision-runtime/studies/h_main_controls_20261008_v2/analysis/results.json`。25／54 是已验证完成数；队列的时间戳可能是当前作业的启动时间，不应据其推断训练心跳。新结果进入补充包之前重新核验，不把旧包的 315 次运行和 945,000 条预测直接扩大为已验收的新统计。

**WiMANS 数据子集**

只使用作者标注中 `wifi_band=5` 且 `number_of_users=1` 的全部记录；固定九类顺序为 `nothing, walk, rotation, jump, wave, lie_down, pick_up, sit_down, stand_up`，类别编号为 0 至 8。唯一非空的 `user_i_activity` 字段确定主体身份。原始文件采用 `wifi_csi/mat/act_group_instance.mat`，不使用视频或作者幅度缓存构造相位。

| 房间 | 记录数 | acquisition group | 每类记录数 |
|---|---:|---|---:|
| classroom | 594 | 19 至 24 | 66 |
| meeting_room | 594 | 55 至 60 | 66 |
| empty_room | 594 | 91 至 96 | 66 |
| 合计 | 1,782 | 18 个组，每组 99 条 | 198 |

身份及标签清单为 [subset_manifest.csv](subset_manifest.csv)，源标注为 [annotation_source.csv](annotation_source.csv)。[metadata_verification.json](metadata_verification.json)记录作者仓库 commit、文件哈希、分布和划分核验；重建脚本复用该 commit，不跟随后续 main 分支变化。

筛选阶段曾在内存检查三个房间各一条原始样本，分别为 `act_19_1`、`act_55_1`、`act_91_1`。CSI 均为 complex128，堆叠形状分别为 `[2825,3,3,30]`、`[2945,3,3,30]`、`[2959,3,3,30]`，均为有限复数；这证明原始相位可取得，不代表完整子集已完成质量验收或相位物理校准。该筛选记录来自前一轮检查，正式输入准备须再次核对保存字节和哈希。

原始 MAT SHA256：

```text
act_19_1 6cb50b8bfee290083bcdaf4cb23db3c82f8972cd4d18a83a04f0b075ce49f6f3
act_55_1 1f965487d0a495f2f609966a8bfd7cda95e4637b722c616c4c25705c2baa77da
act_91_1 a3a465e6d2e139d13491713b6746e44cecd85367df226a8ea4c67371945cfc5d
```

取得选定 MAT 即可；完整发布包含视频和多人数据，体积不能当作本子集的下载量。准备时记录数据版本、每个文件的长度和哈希、数据许可及引用。损坏文件只按预声明的解码或数值规则排除，逐条记录；不按模型结果排除、替换或补样本。

**三折划分**

| 目标房间 | 两个源房间的训练主体 | 两个源房间的验证主体 | 目标房间测试主体 | train／validation／test |
|---|---|---|---|---|
| classroom | 3、4、5、6 | 1、2 | 1 至 6 | 792／396／594 |
| meeting_room | 1、2、5、6 | 3、4 | 1 至 6 | 792／396／594 |
| empty_room | 1、2、3、4 | 5、6 | 1 至 6 | 792／396／594 |

每类对应 88／44／66 条。相同目标折的所有方法与种子 42、52、62 共用成员、样本顺序及配对采样约定。完整身份见 [fold_manifest.json](fold_manifest.json)。源训练和验证的主体、采集组和样本互不交叉；目标房间包含在源中出现过的主体，因此主结果只称跨房间泛化，不称同时跨未见主体。

WiMANS 与 MultiEnv 类别不同。在 WiMANS 的源房间重新训练九类分类器，不能把实验解释为原六类分类器的直接迁移。MultiEnv 目标环境已参与历史开发；WiMANS 的预处理和研究方法在读取目标模型成绩前固定，第二数据集证据不改变旧结果的回顾性属性。原始文件的身份和数值质量审计可以覆盖目标记录，但必须记录这些访问；模型开发、调参和检查点选择仅使用源划分。

**双流输入与实现约定**

原 v1 计划要求确认物理 Tx 索引 0 与三个 Rx 的导出顺序。实际固定作者仓库没有 MAT 导出源，`perm` 字段不能证明是否已应用。现按 v1 已预留的替代路径登记 storage_order_v2：从每包存储 shape `[3,3,30]` 取 `csi[0,:,:]`，按 C-order 展为 90 列，保留第二轴三个连续 30 列组，不再次应用 `perm`。这一选择是软件存储索引，不声称首轴的物理 Tx0 或第二轴的物理 Rx 顺序已确认。完整输入 QA 的通过只适用于已登记存储协议；物理导出核验仍 unresolved，见 [输入轴核验](INPUT_AXIS_AUDIT.md)。

幅度定义为选定复数 CSI 的绝对值。依据记录内 uint32 时间戳构造相对时间，处理回绕和重复时间，使用 850 个包含端点的记录内等间隔时间点。幅度在重复时间算术平均后线性插值。每段记录的有效时间间隔依实际时长变化，不声称固定物理采样率，也不把插值称为恢复缺失动作信息。保存重复时间、包数、时长、最大缺口和零幅度统计。主输入不新增拟合归一化或 RSSI／AGC 校准；若源 QA 需要改变表示，应先登记新版，不从目标数据拟合处理参数。

为验证当前方法，主相位链预声明为历史风格：`angle` 取 float64 包角度；重复时间对 wrapped angle 算术平均；在相同 850 点时间网格上线性插值；最后按 90 列特征轴执行 `unwrap(axis=-1)`，再转 float32。它延续 [历史风格控制定义](../../../analysis/prepare_native_histstyle_phase_20260929.py)，会跨越 29→30、59→60 边界，且保留非圆周角度插值，属于软件表示约定而非物理校准。loader 不得再次 unwrap。源数值 QA 不通过时修改方案并新建版本，在目标模型评价前重新固定；不能按目标分数改选相位链。

三组辅助视图保持连续 0 至 29、30 至 59、60 至 89 列的同步幅相遮蔽语义。历史 `mode=tx,num_rx=1,num_tx=3,num_subcarriers=30` 仍是三个 pseudo-Tx 软件特征组，不代表历史采集有三个物理 Tx。WiMANS MAT 元数据为 `Ntx=Nrx=3`，本协议取一个存储首轴切片；硬件元数据和未确认物理映射的软件选择分别说明。

保留 H 的独立幅相编码器、四路径 1／2／4／8 残差 TPP、0.25 scale、门控融合和历史 hybrid 正对规则。正对包含同一原始样本的视图及同类别 full/full，不限制只能跨源域，不吸引所有同类 masked 视图。正在执行的 P_ALL／P_REMOVE 目标结果不决定 WiMANS 主规则。差分相位或改进规则只能进入另列的前瞻对照。

实现一个独立的 manifest 驱动 WiMANS adapter。现有 MultiEnv 严格分组逻辑含 3,000 条／环境、300 条／组假设，不能强行复用。模型分类头和评估改为九类，重新计算四臂及控制的参数量；不能沿用六类总数。

**固定训练和评价规则**

- 种子 42、52、62；batch 32，两个源域各 16 条，类别平衡约定与采样序列在适配后固定。
- 30,000 optimizer updates，按 200 个虚拟 epoch × 150 updates 组织验证。它不是对 792 条训练记录普通遍历 200 次。
- Adam，初始学习率 0.001、weight decay 0；逐更新学习率 `0.001 × 0.9^(step/10000)`；无梯度裁剪、无新增早停。
- hybrid 的权重 0.05、temperature 0.2、无 warm-up；CE 仅使用完整输入。完整 batch 32 先编码，三个 masked views 合成 batch 96 后编码。
- 按源验证 accuracy 选择最后一个并列最佳 checkpoint，与 H 约定一致；冻结并 reload 后统一评价目标，不使用目标 loader 长度决定训练预算或 scheduler。
- 主报告 macro-F1，辅以 accuracy、九类 F1、混淆矩阵、逐房间均值和种子 SD、全部配对差与最差房间。每运行先算 F1 再平均，不从合并混淆矩阵反算平均 F1。
- 四臂交互为 `(TPP_SC−TPP_CE)−(Global_SC−Global_CE)`。九个房间和种子组合不当作九个独立环境；描述性交互不等于物理域不变性的证明。

**核心训练矩阵**

| 编号 | 数据与实验臂 | 各臂次数 | 小计 | 预期输出 |
|---|---|---:|---:|---|
| W00 | WiMANS AP、等参数 global MLP＋CE | 9 | 9 | 全局表征基线 |
| W01 | WiMANS AP、等参数 global MLP＋hybrid SC | 9 | 9 | 全局表征的多视图增量 |
| W10 | WiMANS AP、TPP＋CE | 9 | 9 | 等参数时间表征增量 |
| W11 | WiMANS AP、TPP＋hybrid SC | 9 | 9 | 联合模型与交互 |
| WC | WiMANS AP、global MLP＋CE＋CORAL | 9 | 9 | 同骨干通用 DG 比较 |
| MC | MultiEnv H-S AP、global MLP＋CE＋CORAL | 9 | 9 | 主协议通用 DG 比较 |
| MA | MultiEnv H-S、作者 CSI-DG 方法 | 9 | 9 | 方法条件性合格；执行改编和源 QA 通过后才冻结 |
| ME | MultiEnv H-S TPP＋CE，保留同 masked 辅助前向但 SC 权重 0 | 9 | 9 | 区分曝光与对比梯度 |
| 合计 | 每臂三目标环境 × 三种子 | | **72** | 不含当前队列剩余 29 次和搜索运行 |

四臂所有公共层保持配对初始化和共同采样。global MLP 采用 [H 等容量实现](../../../experiments/h_main_controls_20261008/control_components.py)替换四个 TPP correction；保留卷积、残差、dropout，并核对九类配置下参数一致。ME 必须真的执行辅助 forward，保留 BatchNorm 更新和随机数调用顺序；现有 `lambda_supcon > 0` 分支会跳过 masked 前向，不能靠把旧配置权重改为 0 完成该实验。

CORAL 在联合源 batch 的融合 feature 上计算两个源域的特征均值与无偏协方差差：均值平方差的元素平均加协方差平方差的元素平均，再乘 gamma。复用的是 [现有 source_coral 公式](../../../experiments/revision_validation/control_components.py)，不是 R 的训练记录，也不宣称完整 DomainBed 或 CSI 专门方法复现。

MA 优先资格审核 [ARC-Fi 作者实现](https://github.com/KaoruMiyazono/UniCrossFi)中的全标签 `UniCrossFi_dg` 路径；引用 [ARC-Fi v7](https://arxiv.org/abs/2310.06328v7)，明确仓库旧命名。核验实际损失、输入模态、主体和采集成员、标签预算、验证身份以及 target 访问。当前作者训练器的 scheduler 依赖 test loader 长度，必须改为源数据定义的预算；披露所做适配，不称原程序逐字复现。必要的作者表征处理可以保留，但同样使用原始信号身份，不能混入作者论文异协议分数。

上述 Adam 与结构约定适用于同骨干主模型、CORAL 和 ME。作者方法可以保留论文规定的优化器与网络；其正式预算默认同为 30,000 更新，在资格审核时确认该预算适用性，并在目标评价前固定、披露全部差异。若需要改变作者方法预算，先更新版本和成本表，不暗中把其训练规则写成与主模型完全相同。

如 ARC-Fi 资格审核失败，先记录具体不一致，另行固定第二个基线。MixStyle 可作为同骨干备用对照，须明确为通用 DG 改编，不能冒充 CSI 作者方法；仅完成两个通用基线时，先进 CSI 方法比较仍是投稿内容的实质缺口。不能按目标测试表现选择替代方法。

**源调参与可选扩展预算**

核心四臂不搜索参数。CORAL gamma 候选固定为 0.1、1、10，每折一个单独搜索种子 20261014，选择源验证结果最好的候选；平局选较小 gamma，目标不评价。两数据集最多各 9 次，共 18 次搜索训练。作者基线最多三组事先登记的配置，每折一个搜索种子，最多另 9 次；不隐藏在 72 次正式运行中。作者原型必须先通过资格审核才能确定这三组配置。总搜索上限 27 次。

| 可选内容 | 追加正式训练 | 何时需要 |
|---|---:|---|
| WiMANS AA 双幅度流，TPP＋hybrid SC | 9 | 要在第二数据集声称相位提供超过第二编码器的额外信息时；与 W11 完全匹配，仅替换第二流输入 |
| WiMANS 作者 CSI-DG 方法 | 9 | 资格审核通过后，加强第二数据集的先进方法竞争比较 |
| WiMANS differential AP，最终模型 | 9 | 检查完整相位处理链依赖；与主链同幅度、同网格、同身份，结果不用于倒选主链 |
| 未见房间和主体联合验证 | 18 | 每折源训练主体 1、2；源验证主体 3、4；目标主体 5、6，396／396／198 条；仅最终模型与预选基线 |

可选 AA 控制保留双编码器和相同参数、初始化、视图、采样及预算；单幅度流或测试时相位置零不能替代它。未见主体扩展另列结果，不改变主协议，也不把主协议的全部 594 条目标记录改名为未见主体测试。可选内容是否纳入在其目标评价前固定。

**不需要重新训练的测量**

在相同输入、硬件和精度下测量所有主要模型的 batch 1 CPU／GPU p50、p95 延迟、吞吐、峰值内存、推理图参数和 MACs；预热后多次计时，GPU 计时同步。预处理、采集窗口等待和网络推理分开，训练投影头与辅助视图不计入推理图。硬件基准安排在现有 GPU 训练队列结束后的独占时段，以免相互干扰。

固定合成输入退化：0、10%、20%、30% 的时间帧同步幅相遮蔽，以及三个连续特征组各一次遮蔽；由 sample ID 和固定 seed 9664 生成同样的掩码，所有方法共享。遮蔽应用于已完成预处理的输入，不称物理丢包或真实天线断线。强度不按目标结果改变。报告均值与最差房间、逐类退化、低性能和失败个例。已有 MultiEnv／R 源表征和负结果可复用；WiMANS 诊断可从完整 checkpoint 再提取，不能以 t-SNE 单图证明域不变。

**执行顺序与验收**

1. 当前 H-S 队列继续按原范围完成；并行完成 WiMANS 元数据登记、基线源代码资格审核和两刊正文结构调整。
2. 按固定子集获取原始 MAT，核对版本和许可；处理 CSI 导出顺序、时间戳、幅相同网格、非有限值、重复身份、损坏排除。输出原始文件与处理数组的哈希和 QA 报告。
3. 完成独立 WiMANS adapter 与九类模型的源 QA，检查公共初始化、采样、四臂参数、辅助前向语义、checkpoint reload 与中断恢复。只用源数据做短训练测速；据实测给出 GPU 工时，不按旧数据直接推算新数据时间。
4. 固定数据、代码、处理版本、正式与搜索配置、完整作业清单和目标访问规则。先完成源搜索、选定配置，再跑正式源训练；目标模型成绩在配置及 checkpoint 固定后统一产生。新旧源划分、模型和支持实验分开建档。
5. 完成主矩阵后重新计算所有预测与配对差，核对文件身份和九类指标；保留所有预定种子及失败／重试，未满九次的臂只标 partial，不计算总体结论。
6. 完成效率与退化测量，更新主文、两刊版本和补充包；各表绑定运行、配置、代码及预测哈希，重新统计 unique runs 和预测数。保持 MultiEnv 历史局限与 WiMANS 跨房间结论范围。

阶段 2 的存储 v2 完整输入处理已验收，MA 方法已完成源码审核和 optimizer 修复原型，但完整表示／数据／训练适配与源 QA 尚未通过；九类模型还须完成 batch32／masked96 的真实源短训、源验证与独占 GPU 测速，随后冻结。45 个 WiMANS 候选配置并非已可执行的正式冻结包；WC 正式配置须待三 gamma 的源搜索后生成，MC／ME／MA 的独立执行适配仍待完成。本轮原始下载和输入处理已结束，未启动新增正式训练或修改原队列；原失败／恢复和所有字节记录均保留。

**论文与补充材料交付**

| 交付 | 内容 |
|---|---|
| 正文方法与贡献 | 复用现有公式和结构图；明确局部时间表示、指定多视图规则、两个数据集的证据与条件性收益 |
| 相关工作比较表 | 输入模态、监督、目标访问、视图、时间表示、作者代码与必要改编；文献表格分数不混进实测表 |
| 主协议表 | H／H-S／R／Y／WiMANS 分列数据、划分、输入处理、模型、预算及源选择 |
| 主结果表和图 | H-S 等参数四臂；WiMANS 五臂；竞争方法；全部配对增量与四臂交互；逐环境和逐类结果 |
| 系统测量 | 采集到分类流程、成本表、退化曲线和失败案例；未有真实部署时不称已建报警系统 |
| 补充材料 | 全部运行和预测、身份清单、负结果、R／Y 支持控制、处理链、作者方法改编、搜索预算与源结果、代码和数据许可、复现脚本 |
| 版本与发布检查 | 重建证据包、指标复算、引用和图文一致、PDF 编译与逐页检查；按真实作者信息和稿件状态补齐投稿材料 |

[IoT Journal](https://ieee-iotj.org/guidelines-for-authors/)版本突出智能环境的感知需求、第二数据集可靠性和部署成本，采用 IEEE 双栏及 150 至 250 词的单段摘要。八个发表页是超页费起点而非硬性篇幅上限。

[OJ-COMS](https://www.comsoc.org/publications/journals/ieee-ojcoms/policies-guidelines/)版本突出无线感知方法、公平比较及完整可复现证据，使用 IEEE Open Journals 模板，并提供供审稿的图形摘要与独立图文件。两套为同项研究的顺序投稿备选版本。

**来源与复核入口**

- [WiMANS 作者仓库](https://github.com/huangshk/WiMANS)、[作者数据说明](https://github.com/huangshk/WiMANS/blob/main/dataset/README.md)、[作者幅度预处理](https://github.com/huangshk/WiMANS/blob/main/benchmark/wifi_csi/preprocess.py)。
- [WiMANS 作者论文](https://arxiv.org/abs/2402.09430)，正式 ECCV 2024；主子集统计以本目录固定 commit 的标注为准。
- [CSI Tool 复数解码](https://github.com/dhalperi/linux-80211n-csitool-supplementary/blob/master/matlab/read_bfee.c)、[接收天线置换代码](https://github.com/dhalperi/linux-80211n-csitool-supplementary/blob/master/matlab/read_bf_file.m)。这些说明标准工具，不单独证明 WiMANS MAT 导出时执行过哪些步骤。
- [当前 H11 配置](../../../configs/published_h11_l1248_config.json)、[TPP 等参数实现](../../../experiments/h_main_controls_20261008/control_components.py)、[相位链对照](../../../experiments/h_main_controls_20261008/phase_inputs.py)。
- [元数据核验](metadata_verification.json)、[子集成员](subset_manifest.csv)、[三折成员](fold_manifest.json)。
- [机器可读计划和 72 个正式作业条目](protocol_plan.json)。这些条目列明研究条件，不是可直接启动的训练配置。
