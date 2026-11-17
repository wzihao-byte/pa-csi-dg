**准备与复核入口**

工作目录为 `C:/Users/howie/Desktop/pa-csi-dg`；信号与运行日志保存在 D 盘 `D:/pa-csi-dg-revision-runtime/data/wimans_20261014`。本轮准备流程只下载、处理输入及执行 CPU 实现检查，没有开 GPU 或新增正式训练。原 H-S 队列继续其既定 54 次范围。

下载只取固定 1,782 个 MAT，Kaggle 版本 1、作者标注固定 commit，数据许可元数据已保存。初次 query 接口的 198 个 404 单独保留于 `acquisition/attempts`；路径式单文件接口按相同身份、版本恢复缺失 receipt，已接收文件均先核对长度与 SHA256。中断在 MAT 写入后／receipt 写入前时，以同版本再下载的字节比对恢复，已有字节不静默替换。

存储顺序 v2 已独立登记，物理导出来源仍未确认。每个处理记录保存包数、时长、重复时间、回绕、最大间隙、零幅度和 perm 统计，以及原始与幅相矩阵哈希。完整数组为 float32 `[1782,850,90]`；相位 loader 不再次 unwrap。数值失败不导致静默排除、替换或补样本，完整输入门禁保持关闭直至全部通过。

当前作业状态见 runtime `workflow_status.json`、`recovery_status.json`、`acquisition/status.json`、`processed_storage_order_v2/input_audit.json`，标准输出和错误日志均保留。以下是复核／恢复命令；已有下载正在运行时，acquisition 会拒绝另一下载 owner 覆盖状态。

```powershell
python docs/revision/wimans_iotj_ojcoms_20261014/build_metadata.py
python docs/revision/wimans_iotj_ojcoms_20261014/refresh_progress.py
python -m unittest experiments.wimans_20261014.test_acquisition experiments.wimans_20261014.test_raw_io -v
python -m experiments.wimans_20261014.qa
python -m experiments.wimans_20261014.acquisition --workers 2
python -m experiments.wimans_20261014.prepare_inputs
```

`build_metadata.py` 在已有 v2 登记时只验证历史元数据、v1 快照、当前 v2 协议及其哈希，不重新覆盖已推进的协议。`prepare_inputs` 逐条校验原始 receipt，续接与当前处理代码相同且矩阵哈希匹配的已完成记录；若版本或代码不同则拒绝，须另存版本。当前 v2 正式处理前的预览与源代码旧版本均另行保留。

完整处理验收后，可生成 36 个四臂正式候选和 9 个 WC 源搜索候选：

本轮完整验收已通过，上述 45 个候选已在 `candidate_configs_v1` 生成，见 [配置核验](candidate_configs_qa.json)。三折 `source_shape_qa_v1` 已完成真实源样本 batch1 前向与 fresh strict reload，目标信号读取为 0；正式执行锁仍为 `candidate_pre_freeze`。

```powershell
python -m experiments.wimans_20261014.prepare_configs generate --subset-manifest docs/revision/wimans_iotj_ojcoms_20261014/subset_manifest.csv --fold-manifest docs/revision/wimans_iotj_ojcoms_20261014/fold_manifest.json --processed-dir D:/pa-csi-dg-revision-runtime/data/wimans_20261014/processed_storage_order_v2 --output-dir D:/pa-csi-dg-revision-runtime/data/wimans_20261014/candidate_configs_v1
```

候选不是正式冻结配置。真实源 batch1 前向／fresh strict reload 的 `source-shape-qa` 只验收输入适配，不授权冻结。完整 W11 batch32＋masked batch96 的两步训练／396 条源验证，以及独占 GPU 测速另行验收；未据合成 CPU 输入估算 GPU 工时。CORAL 的三候选仅比较源验证，搜索种子 20261014 禁止目标模型评分。

正式 `freeze` 要求本折真实源短训 receipt、完整输入、代码／配置／身份／采样哈希一致；WC 的正式种子配置另需源 gamma 选择 receipt。`train-source` 与 `evaluate-target` 是独立命令；后者须在完整 30,000 更新、检查点冻结与 fresh strict reload 后才能打开目标信号。当前未冻结或启动新增正式作业。

H-S 已完成的等参数主表在 [hs_four_arm_per_run.csv](hs_four_arm_per_run.csv)、[配对表](hs_four_arm_paired.csv)、[逐房间统计](hs_four_arm_by_room.csv) 与 [来源绑定](hs_four_arm_provenance.json)。其中 18 条是新队列 G10／11、18 条是历史 H-S10／11；不能将这 36 条全部算作本轮新增训练。既有补充包的去重统计尚未更新。
