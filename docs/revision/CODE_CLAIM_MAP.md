# 论文表述—代码证据对照

| 表述 | 对应实现/记录 | 修订结论 |
|---|---|---|
| 三个 Tx 视图 | dg_models.py:AntennaViewBuilder；configs/published_h11_l1248_config.json | 软件 pseudo-Tx 三段 30 维，保留切片语义；不等于三根物理 Tx |
| SupCon 吸引全部同类 | dg_losses.py:domain_aware_supcon_positive_mask、masked_contrastive_loss | H 仅 full/full 跨实例同类为正，同实例跨视图为正；Y global-only 是另一目标 |
| CE 与 SupCon 权重 | H 发布配置；Y resolved_config；train_joint.py | CE batch mean；SC 先正样本 mean 再有效 anchor mean；H λ=.05，无 warmup；Y λ=.025，6000 步 warmup |
| 0.25 约束弱残差 | experiments/tpp_isolated/tpp_models.py:IsolatedTemporalPyramidPool | g+.25(Wz+b)，自由 W/b，可吸收系数；实测见 H/Y residual 表 |
| TPP 不增加容量混淆 | H 参数记录；joint_models.py | H 参数不同；Y 参数相同但函数族不同，不能都称为容量完全控制 |
| P1 已去掉 | 上述两个池化模块 | H/Y residual descriptor 均含 P1；正文明确保留 |
| 训练仅使用幅度 | joint_models.py:JointPaCsiDGLite | Y 是幅度路径采用 TPP，phase 模态仍存在且使用 global pooling |
| 相位已校准 | dg_dataset.py:load_env_npy_triplets；dataset.py:merge_timestamp；analysis/native_phase_processing.py | 历史 loader unwrap、历史 wrapped-phase 分箱和新原生时间相位是三个流程。新差分相位通过 `H_a·conj(H_b)` 消除共享复相位项，但不恢复 AoA/ToF，也不代替 RCO、非线性误差等硬件校准 |
| 241 条时间戳变化等同于 3744 条输入变化 | analysis/audit_multienv_timestamps.py；analysis/audit_native_reconstruction.py | 241 只统计历史 MAT 时间戳的 15 位表示变化；3744 比较历史数组与含时间表示、时间原点变化的原生 merge float32 输入，二者不可互换 |
| Y 的 subject 隔离 | Y split_manifest.json；verified_subject_audit.json | 旧字段 inferred_group_block_holdout/sidecars=false 保留；全量原始身份与 36 份冻结样本哈希已事后确认 subject/acquisition 不交叉 |
| Y 样本编号相同就应直接连接 | dg_dataset.py 的 running_sample_id；train_joint.py 的 _manifest_identifier_sets / _stable_sample_maps | 汇总 manifest 的源编号包含 loader 拼接偏移；部分诊断与新 R 编号使用环境内索引。按实际函数恢复偏移并核对哈希 |
| H 的源验证隔离 subject | train_factorial.py:source_split；h_subject_split_audit.json | 原分层随机 split 的全部 ID 哈希吻合；每 run 源训练/验证共享 20 个 subject、822–874 个 acquisition，目标身份隔离保持 |
| 新 R 已成为完整性能基线 | experiments/revision_validation/ | 模型与源训练入口已实现；smoke/pilot 单独记录，不能充当完整 2×2 结果 |
| grouped 结果降低证明泄漏 | 两协议 config/split/history | 多因素共同改变，不能从 H−Y 差直接识别原因 |
| 热图与统计都用 240 条 | analysis/visualize_csi_signal_by_class.py | 修订版统一选样；旧热图 cap=120 存在环境顺序截断 |
| 850/320 秒 | 原论文采集说明；merge_timestamp 重放 | 320 packets/s 是原始名义速率；处理后帧数不能直接换算时长 |
| projection head 导致 CE 参数更少 | dg_models.py、tpp_models.py、joint_models.py | CE 模型也实例化 projection head，只是未参与 CE 损失；表报告全模型参数 |
| 梯度机制尚未测量 | train_joint.py:run_gradient_diagnostic；runner_utils.py | Y 已每 5 epoch 保存；此次汇总现有记录，而非臆造缺失或启动重复训练 |

证据优先级按要回答的问题区分：软件定义看实际 run 的实现/配置；物理硬件看原始采集资料；处理后数组是否与原始数据一致靠信号重放核验。任何一类证据不能替代另一类。
