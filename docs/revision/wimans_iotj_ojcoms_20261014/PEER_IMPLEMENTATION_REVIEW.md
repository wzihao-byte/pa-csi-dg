# WiMANS 实现独立复核

日期：2026-10-14，日本时间。对应 [机器记录](peer_implementation_review.json)保存本次审核的源文件 SHA256、合成 CPU 检查和独立 raw access 登记。审核未修改实现文件、启动正式训练或使用 GPU。

三折种子 42 的全部 30,000 个 batch 已独立逐项核对：每 batch 32 条、每源 16 条、每源九类各 1 或 2 条，索引全部来自源训练成员。九类指标、完整 batch 后辅助 3B 的 hybrid 正对、full-only CE，以及 joint-source 融合 feature 的均值/无偏协方差 CORAL 符合所读代码。ME 的零对比权重辅助前向 API 存在，但 WiMANS 包并不等于 MultiEnv ME 正式适配已经完成。

训练更新预算与 source 验证选择不读取目标 loader；目标评价是另一个调用，搜索种子被禁止评价目标，严格 reload 后才建立 test dataset。存储顺序 v2 保留物理轴未确认的边界，不以登记软件切片证明物理 Tx/Rx 次序。

独立审核提出的具体缺口已由实现负责人修正：

- 存储 fallback 门增加 exact slicing、C-order、相位/no-reperm/no-extra-unwrap 及全部 1,782 条 numerical QA 检查；正确协议通过，index 1 的错误协议被拒绝。
- processed audit 的成员/划分哈希绑定到当前 manifest。
- CORAL 逐候选 frozen/config/checkpoint 绑定与三个 gamma 相同输入/划分核对；本次分别确认错误 fingerprint 和不同 data/fold 候选无法选入正式配置。
- 首个虚拟 epoch 前保存 epoch 0 模型/optimizer/RNG checkpoint，解决首次 150 步内中断没有恢复边界的问题；真实 loop 重放测试由 adapter 负责人另行记录。
- raw rename 后 receipt 尚未写入的中断，可通过同版本重新下载比对原字节后补 receipt；本次使用 fake ZIP 验证相同字节恢复。
- 时间戳及 Nrx/Ntx/perm 在 cast 前验证 finite/integral/range；分数时间戳由原来静默截断改为拒绝。
- acquisition 的 running 状态增加 owner PID/create-time 活跃检查并封存旧 attempt，避免异常退出后的状态永久阻止恢复。

此次复核不批准正式执行。完整原始输入、源 batch 32 短训练、真实 checkpoint/reload 与冻结配置仍应由执行门验收；本记录不是新的模型结果。

一次单独的原始访问需纳入记录：审核用临时假文件测试旧 `acquire_one` 的即时拒绝时，root 的并行修复使该调用进入新增 redownload 分支，实际从 Kaggle version 1 将 `wifi_csi/mat/act_19_2.mat` 的 ZIP 读取到内存。真实下载与假文件不相等，helper 拒绝并保留假字节；真实 payload 未保存、未做 MAT/CSI 数值解码、未输入模型，也未产生目标成绩。此访问属于 raw identity/hash 恢复核查；下载 SHA256 和精确事件时刻没有捕获，不能补猜。后续 peer probe 已用本地 fake ZIP mock 网络，未重复读取真实数据。详情见机器记录的 `separate_raw_access_disclosure`。
