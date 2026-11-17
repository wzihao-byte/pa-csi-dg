# WiMANS 独立 adapter 与九类训练实现 QA

本报告仅验收 CPU 合成输入的实现契约，未运行 GPU、正式训练或目标模型评价。真实源 QA 与正式冻结尚未完成。

本次 12 项检查，失败 0，错误 0；状态：passed。机器可读证据见 `IMPLEMENTATION_QA.json`。

- 三折以 sample ID 清单驱动：792／396／594；检查主体、采集组、房间及九类成员，没有复用 MultiEnv 的 3000／300 大小假设。
- 九类 TPP 与 global MLP 各 4,787,065 参数，三种子下 358 个公共 state 张量逐个相等；global 替换保持 CPU RNG 状态。850×90 全输入的 CPU eval 输出为九类有限 logits。
- 独立固定采样 RNG 每个虚拟 epoch 重建源域／类别池，每 batch 两源各 16：每类一次加七类第二次；完整轨迹为 200×150=30,000 个 batch。每折／种子完整轨迹哈希列于 JSON。
- 三辅助视图同步保留 0–29、30–59、60–89 列，其余置零；这沿用 pseudo-Tx 软件视图语义，不能推断为三个物理发射天线。
- 检查历史 hybrid 的实例身份正对与同类 full/full，分母保留未选同类 masked 对。CE 仅完整输入；显式零 SC exposure API 实际执行辅助 forward，BN 计数／running state 与 hybrid 相同，RNG 相同而投影头不获 CE 梯度。MultiEnv ME 仍须独立适配。
- 验证 CORAL 的特征均值差加无偏协方差差公式及梯度；九类 macro-F1 固定包含九类，最后并列最佳 checkpoint 选择经检查。
- CPU Adam、BatchNorm、dropout 的权重／优化器／RNG checkpoint 恢复后，下一个更新的 loss 与全部 state 完全相同。actual runner 以轻量合成数据模拟首 epoch 中断，epoch0 checkpoint 存在且 resume 重放 RNG 一致；每 150 更新提交完整恢复点，重放尚未提交的 epoch。失败和重试各保存独立 attempt receipt 与实际耗时，正式完整执行仍待真实源验收。
- `train-source` 与 `evaluate-target` 为独立命令；目标打开前要求完整 30,000 更新、冻结 receipt、checkpoint hash 与 strict reload。20261014 源搜索 seed 禁止目标模型评价；候选配置、partial／preview 输入、未解决的原轴协议均拒绝正式训练。
- 注册 `storage_order_v2` 可在完整输入 QA 通过且协议文件哈希匹配后通过输入门禁：存储首轴索引 0、次轴全组、C-order，不重用 perm；其物理 Tx/Rx 导出来源仍未确认，门禁不会将其变为物理 Tx0 已核验。

实现入口：`experiments/wimans_20261014/{adapter,models,metrics,training,prepare_configs,qa}.py`；骨干快照来源及改编见该目录 `MODEL_PROVENANCE.json`。

尚需完整子集数值验收、每折真实源 W11 短训练与 reload、独占 GPU 测速、source-only CORAL 搜索和执行冻结。候选配置不是已冻结可执行训练包；没有据合成输入推算 GPU 工时，也没有生成正式目标分数。
