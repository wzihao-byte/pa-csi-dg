# 可复现性发布清单

本清单落实审稿意见中对 split ID、采样序列、配置、源码版本、逐 seed 预测、运行清单和预处理脚本的要求。它区分已经保存的历史 H/Y 证据、已完成且只作技术审计的 R 系列证据以及投稿前必须由作者提供的公开标识。任何一项只有通过对应哈希门槛后才能标为完成。

2026-09-24：实验、技术证据及 24 页 PDF 已批准，本地目录与 ZIP 全成员已核验（270 checkpoints、198 作业采样覆盖）。终态判定见 `COMPLETION_AUDIT.md`；包内文档是构建时快照，最终 12 文档批准和总审计是另行绑定 release manifest 的构建后记录。

## 1. 当前证据边界

| 范围 | 当前状态 | 发布内容 | 不能宣称的内容 |
|---|---|---|---|
| 历史 H | 36 个 run 已核验 | 冻结配置、split manifest、混淆矩阵、逐样本目标预测、指标、checkpoint SHA-256、采样序列全局哈希、训练代码 | 历史运行没有保存完整逐批索引列表，不能把事后生成的列表称为原始记录 |
| 历史 Y | 36 个 run、108000 条目标预测已核验 | 冻结生成协议、逐 run 配置、source/target 预测、split、history、metrics、checkpoint SHA-256、首批 ID 和逐 epoch 序列哈希、训练代码 | H/Y 不能合并为单因素实验；旧 manifest 的身份字段不回写 |
| 正式 R 核心 | 36/36 source 与 target 已完成 | 包含全部 source/target 预测、history、events、配置、source/deferred-target manifest、checkpoint SHA-256 | 不恢复已撤回的 R 写作结论 |
| 正式控制 | 81/81 source 与 target 已完成 | 同上，另含 P1、attention、mask、CORAL、Mixup、随机循环平移配置 | 组件 QA 不算完整训练结果 |
| 正式输入 | 63/63 source 与 target 已完成 | 同上，另含原生时间、直接/差分相位和双流输入 manifest | 输入敏感性不等于完整物理校准 |
| H grouped L1248 | 18/18 run、9/9 pair 已核验 | 完整冻结 sampler、sealed attempts、严格重载及预测、所有 signed 差 | 训练/验证 membership 同时变化，非完整因子设计 |
| Widar3.0 | 6749 条非空库存已核算为 6748 valid + 1 具名 signal exclusion；用户随后取消实验，训练/目标评估未开始 | 只保留成员清单、唯一排除、派生数组/QA/负门/冻结 bundle 和取消记录 | 不包含任何 Widar 性能、重新训练 run 或独立确认宣称；不得恢复训练 |

既有 `evidence/release_preflight_v5.json` 仍证明 72 个历史 H/Y checkpoint 可逐文件复核及当时已完成 R run 的采样/哈希正确；其中 Widar 规模只说明取消前冻结状态。现行发布器另要求 36/81/63 的 180 个 MultiEnv R、18 个 original-H grouped-L1248 补充 run、117 个 source 分析全部完成，并独立核对 H 补充的 9 套 paired schedule 和 18 个 sealed checkpoint。

## 2. 正式发布门槛

本地发布生成器为 `analysis/build_multienv_release_bundle_v3.py`。只有以下条件同时成立才会生成目录：

1. `verified_multienv_results.json` 报告 180 个完成的 MultiEnv R 目标评估、117 个 source-only 机制分析及独立协议下 18 个完成的 grouped-H run，并明确 Widar 已取消。
2. core、controls、inputs 三个冻结 bundle 分别包含 36、81、63 个 run；每个 source run、target result 和 prediction 文件均存在且通过记录哈希。H 补充另核 18 个 sealed attempt、3000 条逐 run 预测、9 套 paired sampler 与 QA provenance。
3. 三个 R target gate 均已通过，H 补充另行核验 checkpoint 冻结后的目标访问；每个保存的 source-best checkpoint 重新加载后复现 source predictions。
4. 技术表图与 `verified_multienv_results.json` 绑定；论文只包含经 `verify_hy_writing_scope.py` 核验的 H/Y 与 H18 片段，R 素材只作技术审计。无 population-level p 值或置信区间。
5. 摘要、结果、讨论、结论和审稿回复信已逐项对照正式数值；叙述批准记录同时绑定正式证据、数值摘录、`manuscript.tex`、`revision_body.tex` 和回复信的哈希，且无未解决数值宣称。随后最终论文双遍编译，未定义引用/文献数为 0、overfull box 为 0，并完成逐页视觉检查；最终 PDF 哈希与 QA 记录一致。
6. 发布目录不存在。生成器不覆盖旧版本，也不会发布到网络。

负门控已单独调用正式发布器验证：在 `verified_formal_results.json` 和最终 PDF QA 尚不存在时，命令以非零状态退出，且没有创建发布目录。记录见 `evidence/release_gate_negative_qa_v1.json` 和对应日志；它只证明提前发布被拒绝，不是正式完成证明。

发布成功后执行现行终态总门 `analysis/verify_multienv_revision_completion.py`。同步 12 个计划、完成审计、回复、取消协议、H 协议和论文文件后，写 `evidence/final_multienv_document_sync_approval.json`，逐文件哈希同时绑定现行证据与 release manifest。总门重新读取 117 项 source-only 完成记录、叙述及逐页 PDF 批准、发布目录每个文件、ZIP 的全部成员字节及清单、180 个 R + 18 个 grouped-H + 72 个历史 checkpoint，并确认 sampler 索引覆盖全部 198 个现行作业。只有全部吻合才生成 `evidence/final_multienv_completion_audit.json`；提前调用已按预期拒绝，见 `evidence/final_multienv_completion_negative_qa_v1.json`。

## 3. 完整采样序列

新 R 系列采样器的输入只包括冻结 source manifest 中的 train 行、seed 和 epoch。正式发布生成器会：

1. 对每个唯一 train 行集合和 seed 重建所有 200×300×16 个零基索引；
2. 将每个 epoch 的 `<i8` 字节序列 SHA-256 与训练时 `history.json` 中的哈希逐项比较；
3. 仅在全部相等时写入压缩 NPZ；
4. 在 `sampler_schedule_index.json` 中记录其对应 run、train 行集合哈希、逐 epoch 哈希和 NPZ 哈希。

索引相对于发布包内 source manifest 的 `train` 列表，因此不依赖原机器绝对路径。H/Y 的原始运行没有保存同等形式的完整索引列表；发布包保留其实际保存的哈希、seed、split 和生成代码，并明确这一历史限制。

## 4. 源码与版本身份

当前 Git HEAD 为 `45c0cd3553b2df2db8b8cda375bf0062c20a53af`，但工作区包含大量本轮新增文件，因此该提交号不能单独标识修订版本。正式发布采用两层身份：

- 每个正式研究 bundle 自带冻结代码快照和 `bundle_manifest.json`；
- 总发布包以 `release_manifest.json` 列出每个文件的相对路径和 SHA-256，并记录文件总数与总字节数，该清单是本地修订状态的权威身份。

投稿前应将这一确切目录发布到稳定仓库或档案库，再把公开 commit、release URL 或 DOI 写入论文。不得提前填写不存在的标识，也不得把旧 HEAD 当作本次发布版本。

## 5. 数据与模型文件边界

- MultiEnv 原始数据使用公开 DOI；Widar3.0 原始压缩包仍由原提供方托管，本项目不重新发布。
- 发布包包含 9000 条 MultiEnv 映射、现行训练/预测证据及修订文档中的 Widar 取消记录；取消前的 Widar 清单和派生准备证据不构成性能数据，也不冒充源数据所有者重新发布压缩包。
- 紧凑发布包不重复复制数 GB 的训练/resume checkpoints；`checkpoint_index.json` 和逐 run 记录保存 source-best checkpoint SHA-256 与字节数。生成发布包前，历史 H/Y 的 72 个索引值必须再次与实际 checkpoint 文件逐一吻合；若期刊要求下载权重，应另存 checkpoint deposit，并逐项与该索引核对。
- 逐 seed 的 source/target predictions、run config、history、events 和结果属于紧凑发布包的必需文件，不能只发布汇总表。

## 6. 投稿前作者动作

- [ ] 上传由正式生成器产生且重新核验过的目录或归档。
- [ ] 填写公开仓库 commit、release URL 或 DOI，替换 `AUTHOR_TO_SUPPLY_BEFORE_SUBMISSION`。
- [ ] 如公开模型权重，核对 checkpoint deposit 的全部 SHA-256；如不公开，在 Data Availability 中准确说明仅提供哈希与运行记录。
- [ ] 核对数据提供方许可、作者姓名单位、通信信息、资助、贡献和利益冲突。
- [ ] 最后一次从公开版本下载并运行 manifest 校验，确认它与投稿 PDF 引用的是同一发布版本。

公开上传、创建 release 或 DOI 属于外部发布动作，需作者明确执行或授权；本计划只生成可审阅的本地版本化包。

2026-09-24 打包纠正：v7 中间包因遗漏正文图文件，并复制第三方 Git 对象库及 raw MAT 缓存而主动停止，部分目录与停止日志保留。新版 v8/v3 打包器排除上述缓存，保留来源清单与验证哈希，补齐批准稿的图文件/模板并从包内源码隔离编译、逐页比较渲染哈希。
