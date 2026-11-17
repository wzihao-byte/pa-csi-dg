# MA 作者 CSI-DG 基线资格审核

审核日期：2026-10-14，日本时间。机器可读证据为 [author_baseline_audit.json](author_baseline_audit.json)。

**结论：全标签 `UniCrossFi_dg` 在目标函数层面具备源域 DG 比较资格，作者原训练程序不满足本计划的冻结执行条件。** 保留 MA 为可适配候选；现阶段不启动正式训练，也不能称为 ARC-Fi v7 的逐字复现。必须先登记表示、优化器和预算差异，完成真实源数据 QA，再决定冻结。这里的“条件性可适配”不等于基线失败、已完成复现或已有性能结果。

本次重新读取未修改的作者源码和 v7 主来源；不以旧审稿描述代替实现。本次只获取源码及论文 HTML，读取既有成员/采样元数据，并在 CPU 上用合成输入核验原类。未加载真实源样本或目标信号，未读取或生成目标分数，未访问 GPU，未修改原作业队列。

## 1. 固定来源与版本边界

作者仓库当前 `main` 的 commit 为 `f31e655a8414415de58659c0af7d7dba6281a0c1`，commit 时间为 2025-12-20 14:10:21 +08:00。本次独立 clone 存于 `work/wimans_20261014/author_baseline_audit/UniCrossFi`，审核结束时工作树干净。逐文件长度、SHA256、Git tree 和本地路径均登记在 JSON。

| 固定对象 | SHA256 |
|---|---|
| `models/contrast_methods.py` | `40a392f83bf4d68553b9682c64b7278d80efb3b1653f0e921d64119397d5e2a0` |
| `models/sdaresnet.py` | `377fbce2277182ac1820637a57a783ee3076197b47446dfce40bf0ce9c1cad2c` |
| `util/optimizer.py` | `b12db0f2f1f9e45eaa7ce9638677fcb7591e97b9d9c967c2dd080cb82a5baf2d` |

论文使用 [ARC-Fi v7，2026-06-16](https://arxiv.org/abs/2310.06328v7)，内容核查使用其[完整 HTML](https://arxiv.org/html/2310.06328v7)。论文以 SSDG 为主，同时报告全标签 DG；SSDG 的无标签样本来自源域，训练阶段不允许目标域访问。其表示采用天线比值和静态路径处理，幅相信息进入网络。上述是论文规定，后文各项实现证据分别来自固定 commit。

作者仓库仍用 UniCrossFi 命名，且 commit 早于 v6/v7。未发现 v7 专属 tag、逐表运行配置、原始成员和存储结果的可验证绑定。因此可以写“ARC-Fi / UniCrossFi 作者公开全标签 DG 实现的适配”，不能据此认证其恰好生成了 v7 各表成绩；不得把文献成绩放入本地公平比较表。代码为 [MIT 许可](https://github.com/KaoruMiyazono/UnicrossFi/blob/f31e655a8414415de58659c0af7d7dba6281a0c1/LICENSE)，分发改编时保留许可及版权。

## 2. 实际全标签目标函数与网络

审核对象为 [UniCrossFi_dg，源码 2168–2433 行](https://github.com/KaoruMiyazono/UnicrossFi/blob/f31e655a8414415de58659c0af7d7dba6281a0c1/models/contrast_methods.py#L2168)，不把 `UniCrossFi_semidg` 或 UDA 路径混入 MA。

- 两个自然视图来自同一记录的两个 Rx 比值信号；共享 2D ResNet，BasicBlock 层数 `[2,2,2,2]`，每视图两个输入通道，输出 512 维。实际导入的是 `sdaresnet.py`，不能根据仓库另一个 `resnet.py` 误记成 1D 骨干。
- neck 为 `Linear(512,512) → ReLU → Linear(512,128)`；L2 归一化后同时供对比损失与六类线性分类器使用。分类器实际接在 128 维 embedding 上，不是直接接 512 维 encoder feature。
- Widar 分支产生三训练视图：两次自然输入编码，以及第二视图 feature 的一次随机均值/标准差混合。测试用两个自然视图的分类概率平均；不生成随机风格测试视图。CSIDA 分支是三自然视图加两风格视图，不能静默换分支。

记所有训练视图平均交叉熵为 `L_CE`，实际代码为：

```text
L_con = beta × SupCon(emb, y) + (1 − beta) × SupCon(emb)
s = min(scale_cap, detach(L_con) / (detach(L_CE) + 1e-6))
L_total = L_CE + s × L_con
```

这保留一个依赖当前 loss 的 detached 动态倍率；不能用论文概括式或作者 `loss_dict` 中的 `loss_ce + loss_con` 当作实际反向目标。倍率按源码执行，不能根据注释改成倒数。`SupConLoss` 还乘 `temperature / base_temperature`，其中 base 为 0.07；变 temperature 同时改变这一缩放。正式日志应分别保存 `L_CE`、有监督/实例对比、`s`、`L_total` 和逐视图 CE。[损失实现](https://github.com/KaoruMiyazono/UnicrossFi/blob/f31e655a8414415de58659c0af7d7dba6281a0c1/models/contrast_methods.py#L1097)

有标签调用的类别相等 mask 跨**所有视图组合**展开，去除自身比较；无标签调用使用实例相等 mask。不同源域不是正对成立的必要条件。`domain_label`、`idx`、`epoch` 和 `loader` 不参与该全标签类的 loss 或 feature style randomization。风格伙伴是同 batch `randperm`，均值/标准差在最终 512 维特征上计算，并以随机 alpha 插值；并不保证风格伙伴来自不同域。保留作者代码时必须如实描述，不能凭方法名称声称每次都是跨域 AdaIN。

本研究的历史 H 是全输入加三个零遮蔽 pseudo-Tx 软件组，且跨实例同类正对限于 full/full。MA 的物理 Rx 比值视图及 all-view 类正对与 H 不同。历史 `mode=tx,num_rx=1,num_tx=3` 不能解释成三个物理发射天线。

## 3. 优化器覆盖缺口：已用 CPU 验证

[源码 2202–2205 行](https://github.com/KaoruMiyazono/UnicrossFi/blob/f31e655a8414415de58659c0af7d7dba6281a0c1/models/contrast_methods.py#L2202)只创建 encoder 和 classifier 的 optimizer；neck 的四个参数 tensor 没有进入任何 optimizer。`update` 也只清零这两个 optimizer。于是 neck 有梯度、保持随机初始化权重，并且其梯度跨 update 累积。

本次使用原始 `UniCrossFi_dg`、`SupConLoss`、真实 `sdaresnet.ResNet` 和作者 Adam helper，隔离加载类，执行**一个合成 CPU update**。输入为 `[4,2,2,64,30]`；它只验证类行为，不代表完整 800 点源数据训练。

| CPU 核验 | 结果 |
|---|---:|
| 六类模型总参数 | 11,502,470 |
| 两组 optimizer 覆盖参数 | 11,174,150 |
| 未被优化的 neck 参数 | 328,320 |
| neck 有非零梯度 | 是 |
| optimizer step 后 neck 权重变化 | 无 |
| 两个 optimizer 清零后 neck 梯度仍保留 | 是 |
| 推理等于两自然视图概率平均 | 是 |
| 对推理结果再做 softmax 会改变概率 | 是 |

可复验脚本和输出存于 `work/wimans_20261014/author_baseline_audit/cpu_probe.py`、`cpu_probe.json`，其脚本哈希及源码哈希也包含在机器报告中。CPU 2 threads，PyTorch 2.6.0+cu124；所有 tensor 明确在 CPU，未查询或初始化 CUDA。

建议 MA 使用单独版本名 `MA_authorcode_DG_corrected_optimizer_v1`，将 neck 加入 feature optimizer 并按 update 清零，保留其他表示、网络和 loss 语义。必须披露此修复及上述代码差异，不能称原作者程序完全复现。若决定忠实保留固定随机 neck，应独立登记为作者公开代码固定 neck 变体；该语义决策须在目标评价前完成，不把它混成三组超参数搜索或按目标成绩选实现。目前这项正式决策未冻结。

## 4. 输入表示与原始信号身份

[作者 Widar 处理脚本](https://github.com/KaoruMiyazono/UnicrossFi/blob/f31e655a8414415de58659c0af7d7dba6281a0c1/data/TD_CSI_process_widar.py#L260)实行为：

1. Intel DAT 解码为 `[packet,30,Rx=3,Tx=1]`，解码器应用 Rx `perm`；使用 `get_scaled_csi` 的逐包 RSSI/AGC/noise 缩放并做 complex conjugation。
2. 天线分数为 `mean_subcarrier(mean_time(abs(H))/(var_time(abs(H))+1e-8))`。`argmin` 作为参考，`argmax` 和剩下一个 Rx 作为两个分子；分别除以 `reference+1e-8`。
3. 对每个 i，用 `searchsorted(timestamp, timestamp[i]+80000)` 找最早后续包，做复数比值差分。时间单位依调用为微秒，即 80 ms。
4. 用 `scipy.signal.resample` 在**样本索引**上将每个复数序列 Fourier resample 到 800 点；时间戳参与差分，不参与最后的均匀物理时间重采样。
5. 对差分复数取 `abs` 与 `angle`；每条记录在两个比值视图共同范围内将幅度 min/max 归一化到 `[0,pi]`，相位保持 wrapped angle，无 unwrap，保存 `[2,2,30,800]`。

作者 driver 先把磁盘 batch 的末两轴调成 `[B,2,2,T,30]`，类的 `decompose` 再调回 `[B,2,2,30,T]`。新 adapter 应明确直接输入接口，避免重复 transpose 或复用 H 的 unwrap。推理头已经输出 probability，可靠性评估不能再次 softmax；原 driver 中再次 softmax 对 argmax accuracy/F1 不改变类别，但会改变置信度/NLL/ECE。

**与 v7 不一致的部分已查明：** v7 Eq.(4) 写的是 `mean(var/mean)` 并取最小参考，源码是 `mean(mean/(var+eps))` 并取最小；对一般多子载波也不能当作同一排序。v7 叙述时间戳重采样，代码最终用 Fourier 索引重采样。以上只说明实现和文字不同，不推断哪个生成了论文分数。MA 候选建议优先保留公开代码的 score、conjugation 和重采样，按“作者代码改编”报告；若改成论文公式则另建版本。

MultiEnv DAT 固定来源 commit 为 `5337147c5156948b13e6b05a50b79529614acfbe`。本地 `analysis/audit_multienv_timestamps.py` 明确核对 `Nrx=3,Ntx=1`、按标准 reader 的 `perm` 恢复轴；既有 [timestamp_precision_audit.json](../evidence/timestamp_precision_audit.json)保存两个 E1 记录的 DAT/MAT 全部复数及包号相等证据。本次读取此存储证据，没有重新解码，也不据两个记录认证全部 MA 输入已经验收。这提供物理 Rx 的代码和样本依据；全部新表示仍需逐记录核验，不重新解释 H 的 pseudo-Tx 软件 convention。

新表示须从同一原始 DAT 或已验证等价 MAT 构造，与 H-S 的 `raw_path`、`raw_sha256` 一一相连；不要从历史插值幅度/相位逆构造复数再做比值。作者原处理分支筛选 Widar 六手势，不能把该类筛选套到 MultiEnv；保留本地六类编号和全部 H-S 成员。

尚需源 QA 登记 score 相等时的确定性处理、参考近零比例、`+1e-8` 后的非有限值、uint32 回绕/重复时间和短序列无 80 ms 差分等边界。作者原 `.pop()` 遇 argmax 与 argmin 相同时的行为不能被静默“修好”；如需规则应建立预处理版本。预处理的缩放/方向/800 点与主模型输入不同，必须分开披露；它们不改变成员资格，但不能称双方输入表示完全相同。

## 5. 监督预算、源验证与 target 访问

`method=UniCrossFi_dg, ratio=null` 时，`LiSADataset.label_flags` 全为 true，所有源训练标签用于 CE 与类别对比。该类不产生 pseudo-label，也不需要无标签目标 batch。`task_setting` 的默认字符串虽然为 `UDA`，实际路由依据 `method` 进入 DG 类，不能用这个字符串把全标签 loss 误判成 UDA。[dataset 路由](https://github.com/KaoruMiyazono/UnicrossFi/blob/f31e655a8414415de58659c0af7d7dba6281a0c1/data/dataset.py)、[标签标记](https://github.com/KaoruMiyazono/UnicrossFi/blob/f31e655a8414415de58659c0af7d7dba6281a0c1/data/utils.py#L1148)

| 核查项 | 作者公开 Widar DG | MA 必要条件 |
|---|---|---|
| 源训练成员 | 按 filename / 留出域分支扫描 | 复用 H-S 固定 identity manifest |
| 源验证 | 源样本随机 20%，seed 0；未按主体/采集分组 | 4 主体、400 acquisition、1,200 记录；与训练互斥 |
| 标签预算 | `ratio=null` 的全部源标签 | 每折 train 4,800、validation 1,200；目标标签只最终评分 |
| target 训练张量 | `UniCrossFi_dg.update` 未使用 | source runner 无 target 对象/loader 参数 |
| target 身份元数据 | 训练前扫描目标文件路径与标签 | 分开评价 manifest；源 runner 不扫描目标目录 |
| scheduler | `100 × max(len(train_loader),len(test_loader))` | 源预声明 30,000 更新的 cosine horizon |
| epoch loss 日志分母 | `max(len(train_loader),len(test_loader))` | 实际源更新数；不传 test loader |
| checkpoint | source validation accuracy，`>=`，reload 后 test | 保留 source accuracy 末个并列最佳；冻结再 reload 评价 |
| 搜索 driver | Optuna 每 trial 的 train 末尾仍执行 test | 搜索只保存源结果，禁止 trial 目标评价 |

这里 scheduler 与 loss 分母依赖 target 长度，是严格协议不通过的具体理由；不据此声称 DG loss 使用了目标样本。Widar 的 per-record normalization 不拟合跨目标总体统计。CSIDA 的同名 DG 路径则在进入训练前载入全部 target 数组，并独立用目标总体 min/max 归一化；该 sibling loader 不适合直接拿来当 MultiEnv adapter。[训练 driver](https://github.com/KaoruMiyazono/UnicrossFi/blob/f31e655a8414415de58659c0af7d7dba6281a0c1/algorithm/train_dg.py#L155)、[CSIDA 处理](https://github.com/KaoruMiyazono/UnicrossFi/blob/f31e655a8414415de58659c0af7d7dba6281a0c1/data/CSIDA.py#L242)

本次重算 H-S 三折 train/val 的 `sample_id`、`subject_id`、`acquisition_id`、`raw_path` 交集均为零，源成员中也没有目标环境；逐种子采样文件字节哈希与冻结配置相符。成员表的 actual distribution 是不平衡的（train 类计数 1,920/640/640/640/640/320），不能把 4,800 个源记录说成每类等量。

| 目标 | 固定 source manifest SHA256 | train / val |
|---|---|---:|
| E1 | `bf15bf0b217f29edffefafb353f1047dab1bd191933b68f0e6b5e1a1fe4a63bd` | 4,800 / 1,200 |
| E2 | `669bcfe646ff0ac7f5bbd80468da9f686c6ed577c4f147ec26d3bf37aa4eda67` | 4,800 / 1,200 |
| E3 | `768c4afb169d40ae16f950b1dcff20807348e5ab7b96a8e13c5bfb2e79312b7d` | 4,800 / 1,200 |

每折源训练 16 主体、1,600 acquisition，验证 4 主体、400 acquisition；各具体主体、原始路径来源和 42/52/62 采样哈希在 JSON。以上是现有 H-S 的核验，不是新 author adapter 已验收。MultiEnv 历史 target 已参与开发的事实仍保留，不将新 MA 源选择重写成完全前瞻数据集证据。

## 6. 可审阅的源训练接口与配置候选

建议新建独立 runner，训练函数只接受源对象：

```python
train_source(
    model, source_train_manifest, source_val_manifest,
    sampler_schedule, optimizer_updates=30000,
    validation_every_updates=150,
)
# freeze selected configuration + checkpoint hashes, then a separate invocation:
evaluate_frozen(selected_checkpoint, test_manifest)
```

训练/搜索函数不接受 `test_loader`、目标长度、目标总体统计或目标成绩。使用既有 H-S 32 样本 schedule（两个源环境各 16）并保持序列，按 200 个虚拟 epoch × 150 更新验证；不把作者原 80 epoch 和 128 batch 等价化成这一预算。两组 Adam 保留作者默认 betas，修复版本的 feature optimizer 覆盖 encoder+neck，classifier 另组；source-only `CosineAnnealingLR(T_max=30000,eta_min=0)` 每更新后 step，无早停，无新增梯度裁剪。最终 source accuracy 取末个并列最佳。

候选保留代码网络、all-view CE/对比、动态 multiplier、随机风格混合、code-score 比值表示和 T=800。T=800 与 H 的 T=850 属明确表示差异，不影响六类标签或记录身份。以下三组是**待登记建议**，不是正式冻结配置；依据作者 [Widar DG 脚本](https://github.com/KaoruMiyazono/UnicrossFi/blob/f31e655a8414415de58659c0af7d7dba6281a0c1/bash/UnicrossFi/UniCrossFi_dg_widar.sh)的参数区间与保守学习率提出，不依任何目标成绩。

| 候选 | Adam lr | weight decay | temperature | beta | scale cap |
|---|---:|---:|---:|---:|---:|
| A | 0.001 | 0.0003 | 0.07 | 0.5 | 6 |
| B | 0.003 | 0.0003 | 0.07 | 0.5 | 6 |
| C | 0.003 | 0.0003 | 0.10 | 0.5 | 6 |

每折搜索种子 20261014，最多三组/折、共 9 个 source-only 搜索；按选中 checkpoint 的源验证 accuracy 取最好，候选平局按 A、B、C 顺序。选定后使用 42/52/62 正式三折，共 9 次，不把搜索计入正式 72 次。`beta=0.5` 对应一般 CLI 默认；作者 Optuna wrapper 的 beta 默认又是 0.1，缺少逐运行证据时不能混用默认值宣称“论文固定超参数”。

保存模型、两组 optimizer/scheduler、随机数状态、完整采样位置和 attempt；中断恢复核验后才允许续跑。推理 probability 直接导出，预测记录绑 sample ID / raw identity / configuration / checkpoint / source hash。报告总参数、实际优化参数、训练三个 view 与推理两个自然 view 的 encoder 次数和独立工时。作者骨干与主模型不同，无等容量宣称；合成 CPU 参数数不替代将来真实固定配置的系统测量。

## 7. 未解决事项与完成门槛

1. **代码与论文实验绑定未确认。** 作者实际主体、采集成员、处理调用、各表 hyperparameter 来源和存储结果未公开绑定；不猜测其随机分割是否等于 v7 各表分割。
2. **改编决策尚未登记。** trainable neck 修复、code-score 天线选择及 T=800/Fourier 表示需在版本配置里明确。保留 code semantics 与改为 paper semantics 是不同版本，不根据目标成绩选择。
3. **本地表示尚未构建。** 全部 H-S raw identity 的 byte/hash 与轴/相位方向、逐包缩放、80 ms 时差、数值边界、无损样本成员 QA 未完成。源 QA 若发现需改变协议，先登记新版；目标 numerical audit 可独立留访问记录，不能用于模型调参。
4. **执行适配尚未验收。** 30,000 update source-only budget、真实源数据短 smoke、完整参数覆盖、probability 导出、最佳 checkpoint reload、resume 和 wall-time 成本待核验。本次合成 CPU 检查不是这些门槛的替代。

若上述门槛完成，MA 可成为“同身份/全源标签/源验证选择”的作者代码 CSI-DG 比较；如无法保持成员或只能依目标统计调参，再记录资格失败并预先登记替代。现有审核未提供性能结果，也未触发 MixStyle 替换。

复验入口：在仓库根运行 `python work/wimans_20261014/author_baseline_audit/build_audit_json.py` 重算源码哈希、语法、源成员交集和采样哈希；`cpu_probe.py` 为另行的合成 CPU 检查。二者均不调用作者训练 driver。
