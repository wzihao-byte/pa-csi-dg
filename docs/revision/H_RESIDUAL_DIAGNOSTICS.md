# 历史 H 模型的实际残差

18 个 H10/H11 保存模型均严格加载且哈希一致。每 run 使用两个源环境，每环境每类固定随机抽 4 条（NumPy seed 20260903），共 48 条；范围含源 train/validation。仅作事后模型诊断，不用于选模型，不推断未见环境。H phase 输入按原 loader 先在 float64 上沿末轴 unwrap，再转 float32。

逐路径 hook 重建全局特征和 alpha(Wz+b)，与实际 forward 输出核对一致。alpha=0.25，epsilon=1e-12。下表范围跨同一 arm 的 9 个 run；没有合并 H/Y 样本。

部分 H10 相位路径的全局 ReLU 特征范数为零，含 epsilon 的比值因此很大，不能解释成正常残差相对强度。原始比值保留在 CSV，下表仅对非零全局特征计算 run 中位数；若整 run 全零则该 run 不贡献中位数。零全局范数数量单列。

| Arm | 模态 | 路径 | 非零全局样本的 run 中位数范围 | 每 run 零全局范数数 / 48 |
|---|---|---|---|---|
| H10 | amplitude | temporal | 0.891–2.402 | 0–0 |
| H10 | amplitude | channel | 0.720–3.924 | 0–0 |
| H10 | phase | temporal | 0.888–2.576 | 0–28 |
| H10 | phase | channel | 0.426–0.890 | 0–48 |
| H11 | amplitude | temporal | 1.104–2.646 | 0–0 |
| H11 | amplitude | channel | 1.615–4.466 | 0–0 |
| H11 | phase | temporal | 0.831–2.485 | 0–22 |
| H11 | phase | channel | 0.670–1.385 | 0–0 |
