# MA optimizer 修复原型

此目录只实现作者全标签 `UniCrossFi_dg` 的 fresh-model optimizer 覆盖修复，不含 MultiEnv 数据 adapter、搜索/正式训练 runner 或目标评价。MA 整体执行资格仍未通过。

依据 [作者审核](../../docs/revision/wimans_iotj_ojcoms_20261014/AUTHOR_BASELINE_AUDIT.md)固定 commit `f31e655a8414415de58659c0af7d7dba6281a0c1`。作者 `contrast_methods.py:2202` 的两组 optimizer 仅包含 encoder 与 classifier；328,320 个 neck 参数有梯度但未更新。本原型读取并校验该类、2D ResNet 和 Adam helper 的已登记 SHA256，通过 AST 隔离加载真实类，避开无关的 UDA/数据/W&B driver；没有改作者源码。作者代码 MIT 许可随固定源码保留。

修复仅在任何 update 前进行：feature Adam 使用 `feature_extractor + neck`，classifier Adam 保留原对象及其 scheduler；feature cosine scheduler 绑定新 optimizer，保留同一显式源更新 horizon、eta_min 和 Adam defaults。每个 trainable parameter 必须由两组 optimizer 恰好拥有一次，缺失、重复或 foreign parameter 均拒绝。原 `update` 清零 feature optimizer 后也清掉 neck 梯度。

保留原类的 512→512→128 neck、归一化 embedding 后分类头、两个自然比值视图及随机 feature view、all-view CE、类别/实例对比混合和 detached 动态倍率；没有改成论文概括式或 full-only CE。直接输入约定为 `[B,2 ratio views,2 amplitude/phase,time,30 subcarriers]`。实际 raw→ratio 表示没有在此原型实现。

CPU 示例默认 Adam lr 0.003、weight decay 0.0003、temperature 0.07、beta 0.5、scale_cap 6，对应审核中的候选 B。lr 0.001 的候选 A 和 tau 0.10 的候选 C 仍是事先建议；没有完成源搜索，不能称为选定参数。`source_update_horizon=30000` 是独立传入的源预算，完全没有 target loader/长度参数；正式预算适用性仍须源 QA、成本及版本固定。原 code-score/paper-score 天线选择、conjugation/Fourier 时间重采样与 v7 文本的分歧保持未确认，建议保留作者 code 并在表示适配时登记，不据目标分数选择版本。

复验：`python -m unittest experiments.arcfi_hs_20261014.test_optimizer_adapter -v`。四项 CPU 检查使用真实作者 2D ResNet/类和合成输入，核验修复前后所有 model state、源码方法和推理概率完全相等；修复后原 `update` 确实更新 neck，gradient 清零，全部参数恰一次覆盖；重复 ownership、迟修和源码 hash drift fail closed。它们不证明真实原始表示、H-S 成员、源 batch32、checkpoint/resume 或完整算法训练已验收。

后续仍须：按同 raw identity 构建作者表示；固定 score/tie/ratio/time 数值边界；H-S 源成员和采样；真正 source-only 30,000 更新/验证/中断恢复；源选择与完整预测、配置、源码/checkpoint 绑定。没有上述验收之前，不启动 MA 正式训练，也不将该原型称为 ARC-Fi v7 原程序完全复现。
