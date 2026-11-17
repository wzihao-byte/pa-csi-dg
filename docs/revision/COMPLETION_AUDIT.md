# 修订计划完成审计

状态：**实验、H/Y 叙述、24 页 PDF 及本地发布均已核验；总体验收以终态审计记录为准。**

本自动修订链的交付范围为原 H/Y 与已核验的 original-H grouped L1248 补充。
180 个 R run（core 36、controls 81、inputs 63）及 117 个 source-only 分析均已完成并保留为技术审计；不把 R 分数、表格或结论重新写入这份论文、周报或回复信。另行编辑的 `review_20260924` 审阅稿是独立工作副本，不由本链替换，也不属于本次 H/Y PDF 批准。
Widar 在训练前被用户取消；准备、失败版本、冻结 bundle 和取消记录保留，不启动训练，不声称性能或跨数据集确认。

| 技术门 | 当前证据与判定 |
|---|---|
| 原 H/Y 与输入身份 | 历史 72 run、9000 raw→processed 映射与冻结身份记录保留；历史源训练/验证共享身份及物理校准限制不隐去 |
| R core / controls / inputs | 36/81/63 全部 source 训练和延迟目标评估已完成；三个 target gate、checkpoint 和全部预测结果核验通过；仅作技术审计 |
| source-only 分析 | 117/117 完成，`formal_source_analysis_v3/completion.json` 封存，`target_data_loaded=false` |
| 原 H grouped L1248 | 18/18 sealed run、9/9 pair；全部 signed 差保留，记录在 `h_grouped_sealed_runs/` 与 `h_grouped_verified_pairs/` |
| 正式技术证据 | v6 已完成；`multienv_revision_v2/manifest.json` 与正式结果逐项绑定 |
| 素材和写作范围 | R 全部技术素材保留；论文只纳入 H-only `formal_results.tex`，由 `verify_hy_writing_scope.py` 核验 9 对 signed 差、36 项历史比较及递归 LaTeX 输入 |
| 叙述批准 | v7 `narrative_approval.json` 绑定证据、数值、摘要/正文和回复信；无未解决数值宣称 |
| 最终 PDF | v7 已完成 24 页逐页视觉批准；引用与 overfull 均为 0，哈希与工作区 PDF 一致 |
| 本地发布 | v8 本地目录和 ZIP 全成员哈希均已核验；未外部发布 |
| 文档同步和终态总门 | 12 份当前文档的批准绑定 release manifest；最终验证器成功记录为唯一终态证明 |
| Widar | 用户取消于训练前；无训练/目标评估结果，不恢复队列 |

H 补充于 2026-09-21 02:21:39 UTC 完成 18/18 run，9/9 配对核验通过。
每个 run 的冻结配置、200 epoch、30,000 次更新、完整连续采样、source-validation accuracy 的 last-tied-best 选点、CPU strict checkpoint reload、3,000 个目标预测及封存树均已核验。
E1/E2/E3 等权、先按每个 seed 汇总后，H10/H11 macro-F1 为 88.02%/89.83%，配对差 +1.81 pp，seed SD 0.85 pp；9 对中 8 对为正，E1/seed42 为 −1.8352 pp。全部 9 对 signed 差及 36 项 grouped−historical 指标差已纳入 H 片段。
这一结果只描述固定分组下的训练随机性；训练与验证 membership 同时改变，不能隔离验证选点效应，也不能据两臂估计完整 2×2 interaction。

权威技术证据为 `evidence/multienv_revision_v2/verified_multienv_results.json`，SHA-256 `d1111889335d130d19ea2ab2b97f5d0a8fefd87217a8548d819ac86a2ebd9071`。
v5 汇总器因旧恢复路径失效而失败，其日志保留；v6 正式证据、素材和 H-only 片段已完成。v6 论文等待进程的中断也保留；v7 论文流程已完成。
冻结 H 科学代码和 2026-09-17 WDDM 运行门均未修改，完成的训练和分析进程不再启动。

摘要、结果、讨论、结论与 `REVIEW_RESPONSE_DRAFT.md` 已按 H/Y 范围核对。
最终 PDF 为 24 页，隔离双遍编译后未定义引用/文献 0、overfull 0；16 处 underfull 和 2 处 hyperref 元数据警告已结合逐页视觉检查接受，无裁切或遮挡。全部 24 页均逐页检查并签署 hash-bound visual approval。
`evidence/manuscript_multienv_final_qa.json` 绑定工作区 `paper/sensors_manuscript/manuscript.pdf`，PDF SHA-256 `a03ab7f6e80d7458d92bde73ebd355e639267a03acc47893a8d8a4b1d184c541`。

2026-09-24 本地发布核验完成：`D:/pa-csi-dg-revision-runtime/releases/pa_csi_dg_multienv_revision_v2` 及其 ZIP 已生成，13317 个清单文件和 ZIP 全部文件成员逐字节 SHA-256 通过。
Release manifest SHA-256 为 `a032223ed056026b7516ff8104d4acb6b499fa986a2b917d732908eca0c4ce2b`；归档 SHA-256 为 `ec9ea351ed0bc699958128e36e8c8472c72c5ac00060a4e4f7b8910b902222cf`。
六个正文图文件与模板依赖已纳入，包内源码隔离编译后的 24 页渲染哈希与逐页批准完全相同；原始缓存和第三方 Git 对象库不重打包，来源清单与哈希保留。v7 中间包及停止记录保留。
打包器逐文件核对 180 R + 18 grouped-H + 72 历史 H/Y = 270 个 checkpoint；采样索引覆盖全部 198 个当前研究作业。
归档保留完整 R 技术审计，论文仍按 H/Y 与 H18 范围；未向外部上传。包内文档是构建时快照，当前 12 份文档由 `evidence/final_multienv_document_sync_approval.json` 另行绑定至同一 release manifest。
本轮技术完成的权威判定为 `evidence/final_multienv_completion_audit.json` 的 `status=completed`；验证器会再次核对所有终态门后写入该记录。为避免循环哈希，审计通过后不再改写这 12 份已批准文档。作者信息、声明、公开标识和投稿审核仍待作者确认。

尚未解决的科学边界保持明确：E1–E3 已参与开发，没有独立跨数据集确认；不作总体 p 值或置信区间推断；保留 Y 的退化与 stop；H/Y 不合并为单因素实验。原时间戳转换日志、物理时间/相位校准、masked-view 标签保持性、各 pooling path 的因果归因和无匹配实测的推理延迟优势均不据现有证据宣称解决。
`mode=tx, num_rx=1, num_tx=3, num_subcarriers=30` 经 `AntennaViewBuilder` 实现与配置核对，表示三个连续 30 维 pseudo-Tx 软件特征组，不表示三根物理发射天线。
作者姓名/单位、通信、资助、贡献、利益冲突、数据许可、公开仓库/release/DOI 和最终投稿审核仍需作者确认；本地批准不等于公开发布或投稿完成。

完整早期审计和逐日运行历史保留在 `archive/status_before_h18_release_20260924/docs/revision/COMPLETION_AUDIT.md`。旧 v3/v4/v5 进程身份仅作历史证据，不恢复它们。

2026-09-24 打包纠正：v7 中间包因遗漏正文图文件，并复制第三方 Git 对象库及 raw MAT 缓存而主动停止，部分目录与停止日志保留。新版 v8/v3 打包器排除上述缓存，保留来源清单与验证哈希，补齐批准稿的图文件/模板并从包内源码隔离编译、逐页比较渲染哈希。
