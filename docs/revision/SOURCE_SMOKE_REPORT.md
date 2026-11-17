# 新幅度单模态对照：源数据 smoke

已完成 3 个外层 fold × 4 个实验臂，共 360 次优化更新。每次均只读取两个源环境的幅度文件，目标信号、目标标签和相位文件均未进入 smoke。所有 loss、梯度有限，SupCon 每个 anchor 均有正样本。这是实现与资源核验，不是完整基线训练，没有目标性能结果。

四臂 R00/R01 为全局 MLP + CE/CE+global-only SupCon，R10/R11 为 L124 residual TPP + CE/CE+global-only SupCon。沿用幅度编码器，另用统一 256 维表征层；这是独立 R 候选模型，不继承 H/Y 分数。

参数总数均为 1,878,142；同池化的 CE/SupCon 臂初始状态完全相同，跨池化的 182 个公共状态张量也完全相同。池化全局维度为 d 时，TPP 的线性投影有 7d²+d 个参数；全局 MLP 的 d→7d/2→d 两层使用第一层无 bias、第二层有 bias，参数量与矩阵 MACs 相同。函数族和初始残差范数仍不同，不称为对所有混淆因素的完全控制。

| Fold | Arm | 有梯度的参数 | 16 条推理中位 ms | 更新中位秒 | GPU 峰值 MiB |
|---|---|---:|---:|---:|---:|
| E1 | R00 | 1,779,454 | 14.80 | 0.084 | 400.9 |
| E1 | R01 | 1,878,142 | 14.66 | 0.091 | 402.0 |
| E1 | R10 | 1,779,454 | 15.26 | 0.087 | 403.1 |
| E1 | R11 | 1,878,142 | 15.20 | 0.091 | 402.8 |
| E2 | R00 | 1,779,454 | 15.72 | 0.086 | 401.7 |
| E2 | R01 | 1,878,142 | 14.88 | 0.091 | 403.3 |
| E2 | R10 | 1,779,454 | 15.35 | 0.089 | 403.5 |
| E2 | R11 | 1,878,142 | 15.32 | 0.092 | 404.2 |
| E3 | R00 | 1,779,454 | 14.82 | 0.087 | 401.7 |
| E3 | R01 | 1,878,142 | 14.93 | 0.091 | 403.3 |
| E3 | R10 | 1,779,454 | 15.25 | 0.088 | 403.5 |
| E3 | R11 | 1,878,142 | 15.38 | 0.092 | 404.2 |

硬件为 Quadro RTX 5000，PyTorch 2.6.0+cu124。四臂单样本推理的卷积、线性层、注意力与位置嵌入合计 MACs 均为 673,774,664。该统计不含池化比较、softmax、归一化、激活、bias 加法与数据移动，不当作完整 FLOPs。GPU 峰值包含 smoke 预载的幅度 batch。

CE 臂保留训练用 projection head，但它没有梯度；表中“有梯度参数”指梯度张量存在的参数数目，未声称每个标量梯度都非零。原始记录另存非零梯度张量覆盖量和实际残差/global 范数。

已核验每个 fold 的 200 epoch、60,000 更新采样日程：每批 16 条、每源 8 条、每源包含全部六类、批内样本不重复。日程使用私有 RNG，与模型初始化、dropout 和实验臂执行顺序无关。日程核验不表示完成了这 60,000 次训练。

smoke 使用固定 λ=.025 的完整权重以覆盖对比路径；完整训练入口另实现 6000 步线性 warmup、连续指数学习率衰减、每 epoch 完整源验证 macro-F1 选择、相同分数保留最早 checkpoint、失败 attempt 保留。短 smoke 吞吐排除了数据读取、完整验证和 checkpoint 写入，不能直接当作整轮训练保证时间。

下一步冻结完整 R 比较和源域调参预算，再开展同协议的完整训练。当前数据仍是开发数据；新 R 代码不改变 Y 的停止决定。

```powershell
python -m experiments.revision_validation.run_source_smoke --target all --steps 30
python -m experiments.revision_validation.train_source --target E1 --arm R11 --seed 401 --epochs 1 --output outputs_revision_validation/pilot_E1_R11_401/attempt_001
```

另已完成 E1/R11/seed-401 的 1 epoch 训练入口试跑（300 更新），严格恢复保存的 checkpoint，并重算全部 1200 条源验证预测；预测与指标完全一致。试跑证据见 `evidence/source_pilot_verification.json`，原输出保存在 `outputs_revision_validation/pilot_E1_R11_401/attempt_001/`。这项试跑也不纳入完整基线结果。重复执行命令时需使用新的 attempt 目录。
