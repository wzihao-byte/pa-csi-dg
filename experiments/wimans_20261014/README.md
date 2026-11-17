# WiMANS 独立源训练适配

这是 PLAN 阶段 3 的前置实现，尚未冻结。`IMPLEMENTATION_QA.md/json` 仅记录 CPU 合成输入检查；正式训练必须先通过完整输入 QA、真实源 W11 短训练和执行冻结。当前 H-S 队列不受修改。

`frozen_models` 从 H/H-S 已执行包保存骨干、TPP 和等参数／hybrid 组件，来源见 `MODEL_PROVENANCE.json`。九类模型共 4,787,065 参数。WiMANS adapter 用清单 sample ID 划分，读取幅相 float32 `[1782,850,90]`、int64 标签及 `sample_ids.json`；相位已处理，loader 不再 unwrap。存储轴 fallback 的边界由输入审计和哈希绑定的注册协议定义。

在仓库根目录执行：

```powershell
python -m experiments.wimans_20261014.qa
python -m experiments.wimans_20261014.prepare_configs generate --subset-manifest docs/revision/wimans_iotj_ojcoms_20261014/subset_manifest.csv --fold-manifest docs/revision/wimans_iotj_ojcoms_20261014/fold_manifest.json --processed-dir <complete_processed_dir> --output-dir <new_candidate_config_dir>
python -m experiments.wimans_20261014.training source-shape-qa --config <candidate_for_fold> --output-dir <new_source_shape_qa_dir>
python -m experiments.wimans_20261014.training source-qa --config <W11_candidate_for_fold> --output-dir <new_source_qa_dir> --device cpu --updates 2
python -m experiments.wimans_20261014.prepare_configs freeze --candidate <candidate_config> --source-qa <real_source_qa.json> --output <new_frozen_config>
python -m experiments.wimans_20261014.training train-source --config <frozen_config> --run-dir <new_run_dir> --device cuda:0
python -m experiments.wimans_20261014.training evaluate-target --run-dir <completed_source_run_dir> --device cuda:0
```

`source-shape-qa` 仅在完整 numerical QA 后读取一条真实源训练记录，做 batch 1、eval mode、850×90 CPU 前向，不更新 BN、不输出目标指标、不授权冻结。完整 `source-qa` 则执行实际 batch32 hybrid 训练及396条源验证；CPU attention 的 full32+aux96 图可能占数十 GB host 内存。该量为结构估计，尚无实测峰值；原 GPU 队列运行期间不宜直接启动完整 CPU hybrid QA 造成 host 内存／分页压力。

生成器先生成四臂 36 个正式候选和 WC 的 9 个源搜索候选。搜索仅用 20261014 seed，gamma 为 0.1／1／10；`select_coral` 接口从三个完整源搜索 run 的源验证 accuracy 选择 gamma，平局取小值。WC 的九个正式配置须随后用选定 gamma 构建，并携带 selection receipt 冻结；候选不能直接运行。MultiEnv MC／MA／ME 的正式适配不由这个 WiMANS adapter 宣称完成。

训练按 200 个虚拟 epoch、每个 150 更新提交 `latest.pt`，保存完整 Adam 状态、全部 RNG、history 和最佳权重。`--resume` 必须保留数据、代码、配置、完整采样轨迹、运行库版本与 device；从上个完整 epoch 恢复，重放中断时尚未提交的 epoch。完成 30,000 更新后写 `best_model.pt`，取源验证 accuracy 最后并列最佳，strict reload 并记录 checkpoint receipt。训练命令不会执行目标评价；离线目标命令验证冻结文件及模型后才读取目标信号，搜索 runs 永不评价目标。

显式 `models.objective(..., masked_forward_exposure=True)` 支持零 SC 辅助曝光的机制 QA，会保留实际辅助 forward／BN／RNG。它不是已经完成的 MultiEnv ME 正式实验。
