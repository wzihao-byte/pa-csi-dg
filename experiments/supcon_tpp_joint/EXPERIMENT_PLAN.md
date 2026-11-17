# SupCon + TPP 联合实验计划（预注册草案）

## 0. 结论先行

本项目的首选联合模型固定为：

\[
h_a=g_a+0.25t_a,\qquad h_p=g_p,
\]

其中：

- amplitude 分支的 temporal path 与 channel path 都使用 residual TPP，levels 为 `[1,2,4]`；
- phase 分支保留原始 global pooling，不使用 TPP；
- SupCon 只作用于每个样本的 global fused embedding，不构造 3 个链路 masked views；
- 主实验用参数量匹配、所有 projection blocks 均有效的 `global-basis control` 与真实 TPP 做严格的 \(2\times2\) 因子比较。

在没有完成核心四组之前，不能根据“联合组分数最高”直接声称协同。真正的交互项是：

\[
I_{e,s}=Y_{11,e,s}-Y_{10,e,s}-Y_{01,e,s}+Y_{00,e,s}.
\]

只有联合组同时优于两个单独组，且 \(I\) 的均值和置信区间满足预注册标准，才能称为“协同”；否则最多称为“可以共存”或“加性收益”。

---

## 1. 本计划要回答什么

### 1.1 主问题

在相同数据、切分、参数量、batch、sampler、训练步数和 checkpoint 规则下：

1. amplitude residual TPP 是否提升性能；
2. global-only SupCon 是否提升性能；
3. 两者联合是否优于任一单独组件；
4. 联合收益是正交互、简单相加，还是相互干扰。

### 1.2 能声称的泛化范围

原始采集协议说明三个环境使用互不重叠的受试者：

- E1：实验室 LOS，S1–S10；
- E2：走廊 LOS，S11–S20；
- E3：隔木墙 NLOS，S21–S30。

但处理后的 `.npy` 没有保存 subject 元数据；连续 300 条和重复标签模板只能证明“连续组结构”，不能单独证明真实身份。正式运行前必须从原始文件名/采集日志生成 subject sidecar，并把映射与数据 SHA256 绑定。只有 sidecar 校验通过后，outer LOEO 才能表述为“新环境 + 新受试者”的联合域偏移：

> joint unseen-environment and unseen-subject generalization

若无法取得 sidecar，全文必须降级为 `inferred-group-block holdout` 和“新环境 + 未见连续组”，不能声称 unseen-subject。无论 sidecar 是否存在，都不能解释成纯环境泛化；纯环境泛化需要同一批受试者跨环境采集的新数据。

---

## 2. 数据协议先冻结

### 2.1 类别语义

所有配置、图表和报告统一使用正确语义：

| Label | 正确语义 | 每环境样本数 |
|---:|---|---:|
| 0 | No movement | 1200 |
| 1 | Falling | 400 |
| 2 | Walking | 400 |
| 3 | Sitting down / standing up | 400 |
| 4 | Turning | 400 |
| 5 | Picking up a pen | 200 |

label 2/3 的旧名称写反，旧报告只能保留数值，必须重新解释语义。

### 2.2 Outer test

对 E1、E2、E3 分别做 target LOEO：

- 在本计划后续流程中，target 环境全部 3000 样本只在锁定配置后用于内部复验；
- 另外两个环境作为 source；
- 筛选阶段必须设置 `evaluation.evaluate_target=false`，代码不得执行 target inference。

E1–E3 的 target 结果在历史实验中已经被反复查看，并影响了候选范围；因此这里的 target 不是 fresh confirmatory test。`evaluate_target=false` 只能防止本计划继续泄漏，不能恢复历史独立性。真正独立确认见第 11 节。

### 2.3 Source validation

不使用当前逐样本 stratified holdout，也不使用“拿一个完整 source 环境做 val”的环境级 LOSO。后者会让训练只剩一个源域，破坏跨域 SupCon 的实验条件。

sidecar 校验通过时使用 `subject_grouped_holdout`；否则实现名称和报告名称都必须改为 `inferred_group_block_holdout`：

- 当前数组每环境 3000 样本；
- 每 300 个连续样本先定义为一个匿名 group block；
- sidecar 进一步验证每个 block 是否唯一对应一个真实 subject；
- 每个 source 环境 8 个 group blocks 训练、2 个验证；
- train/val/test 的 group 集合两两不重叠；通过 sidecar 后才能进一步称 subject 集合不重叠。

固定 F0 验证块：

| 环境 | F0 validation blocks |
|---|---|
| E1 | B01, B07 |
| E2 | B00, B06 |
| E3 | B00, B04 |

任一 target 下的切分规模都保持：

| Split | Groups（sidecar 通过后为 subjects） | Samples | Label counts |
|---|---:|---:|---|
| train | 16 | 4800 | `[1920,640,640,640,640,320]` |
| val | 4 | 1200 | `[480,160,160,160,160,80]` |
| test | 10 | 3000 | `[1200,400,400,400,400,200]` |

固定：

```json
{
  "inner_validation": "subject_grouped_holdout",
  "subject_block_size": 300,
  "subjects_per_env": 10,
  "subject_fold": 0,
  "split_seed": 20260714
}
```

若 sidecar 门禁失败，上述 `inner_validation` 必须改为 `inferred_group_block_holdout`；其余显式 F0 表和样本规模不变。

F0 显式表是唯一权威输入，运行时直接读取表内 block IDs，禁止再根据 seed 随机生成另一套 F0。`split_seed=20260714` 仅作为该预注册表的 provenance，不参与运行时抽样。manifest 必须保存实际 validation block IDs、完整样本索引哈希和 sidecar 哈希。model seed 不得影响 split；四个实验臂在同一个 `(target, model_seed)` block 内共享完全相同的 manifest。

### 2.4 Phase 预处理

正式实验固定 `unwrap_phase=false`。运行前必须同时断言 raw config、resolved config 和 dataloader audit 均未执行隐藏 unwrap；四个正式 arms 及所有新基线都按这一协议重跑。

当前 `phase_unwrap_axis=-1` 会在扁平的 90 维特征轴上跨链路 unwrap，并在相邻时间点制造大量人为跳变，不能继续作为正式输入。旧 `unwrap_phase=true` 输出不能与新协议合并。时间轴 unwrap、分链路子载波 unwrap、sin/cos circular encoding 都属于独立的 phase preprocessing 研究，不在本次 \(2\times2\) 内继续扫参，避免再引入一个未控制因素。

---

## 3. 参数量匹配的核心模型

### 3.1 为什么需要 null control

现有 TPP-l124 比原 CE 模型多约 33.4% 参数。直接比较原 CE 与 TPP，无法区分收益来自时间分段还是额外 projection 容量。

对 amplitude 的每个 pooling path 定义：

- global feature：\(g\in\mathbb{R}^d\)；
- TPP feature：\(z_T=[P_1,P_2,P_4]\in\mathbb{R}^{7d}\)；
- global-only null feature：\(z_0=[\phi_1(g),\ldots,\phi_7(g)]\in\mathbb{R}^{7d}\)。

两者都使用同一个形状的 projection 和 residual：

\[
h=g+0.25Wz.
\]

`global-basis control` 的定义在 sweep 前完全锁定。先令

\[
u=\operatorname{LayerNorm}(g;\epsilon=10^{-5},\text{affine}=False),\qquad
c=\operatorname{clip}(u,-3,3),
\]

再逐元素计算七个固定、无参数的 basis：

\[
\begin{aligned}
\phi_1(c)&=c, & \phi_2(c)&=c^2, & \phi_3(c)&=c^3/9,\\
\phi_4(c)&=\tanh(c), & \phi_5(c)&=\sin(\pi c/3), &
\phi_6(c)&=\cos(\pi c/3),\\
\phi_7(c)&=\operatorname{sign}(c)\sqrt{|c|+10^{-6}}.
\end{aligned}
\]

每个 outer fold 只用 F0 source-train 校准：按环境各取 sample ID 最小的 256 条，使用固定初始化 seed 9001、eval mode、关闭 dropout。对每个 amplitude pooling path 和每个 basis，在 `batch × feature` 上计算标量 \(\mu_i,\sigma_i\)，实际输入为 \((\phi_i-\mu_i)/(\sigma_i+10^{-6})\)。把七列展平后的 source-train basis design matrix 做 SVD：相对容差 `1e-6` 下必须 rank=7，condition number 必须不大于 `1e4`。

使用相同初始化的 projection，在同一 source-train calibration set 上计算一次性缩放

\[
\kappa=\frac{\operatorname{RMS}(Wz_T)}
{\operatorname{RMS}(Wz_0)+10^{-8}},
\]

并令 control 为 \(h=g+0.25\kappa Wz_0\)。要求 \(0.25\le\kappa\le4\)，且未参与校准的 source-train smoke batch 上，control/TPP 初始 residual RMS 比在 `[0.8,1.25]`。所有 sample IDs、\(\mu_i,\sigma_i,\kappa\)、SVD 结果、source 文件与索引哈希写入 calibration artifact；同一 outer fold 的所有 arms/seeds 复用它，target 文件不得被加载进校准过程。

该 control 不含局部时间 bins，却让 projection 的七个权重块在一般输入上都能影响输出，并与真实 TPP 保持相同的可训练参数量和 projection 形状。任何函数、常数或校准门槛变化都必须提升 protocol version，并发生在首次 target inference 之前。

该控制仍不能让两个模型的有效假设空间完全相同。因此 `Y10-Y00` 解释为“多尺度时间局部统计相对全局非线性统计的贡献”，不能仅归因于绝对时间位置。若要单独声称 position/order 的贡献，需要第 10 节所述的顺序空控制。phase 分支在所有臂中都保持 global pooling。

### 3.2 核心四组

| Arm | Temporal factor | SupCon factor | 解释 |
|---|---|---:|---|
| `Y00` | global-basis control | 0 | 参数量匹配的 CE control |
| `Y01` | global-basis control | 1 | SupCon only |
| `Y10` | amplitude residual TPP | 0 | TPP only |
| `Y11` | amplitude residual TPP | 1 | TPP + SupCon |

旧的原始 CE 模型只作为描述性 anchor；它不进入交互项主检验。若论文需要声称“超过原始架构”，再用完全相同新协议补跑 legacy CE，而不能复用旧输出。

### 3.3 SupCon 定义

正式 SupCon 必须是真正的 `global_only`：

- 每个样本只有一个 fused projection，\(V=1\)；
- 不调用 `encode_antenna_views()`；
- 不使用当前 `global_class_instance_views`，因为它仍把同类 local views 留在分母中形成假负样本；
- temperature 固定为 0.2；
- `include_same_domain_same_class=true`；
- lambda 使用 20 epoch 线性 ramp，最大值由 source-only 筛选确定。

---

## 4. 所有正式组共享的训练配置

| Item | Locked value |
|---|---|
| epochs | 200 |
| samples / source domain / batch | 8 |
| batch size | 16（E1–E3 outer folds 均有两个 sources） |
| optimizer steps / epoch | 300 |
| total optimizer steps | 60,000 |
| sampler | `domain_class_balanced` |
| optimizer | Adam |
| learning rate | 0.001 |
| weight decay | 0 |
| scheduler | step exponential decay, 0.9 / 10,000 steps |
| early stopping | disabled; all arms run 200 epochs |
| checkpoint metric | source-val `f1_macro` |
| checkpoint tie | earlier epoch wins |
| TPP | max, `[1,2,4]`, preserve global, scale 0.25 |
| SupCon temperature | 0.2 |
| SupCon warm-up | 20 epochs |

选择每个 source domain 取 8 个样本的理由：在 E1–E3 outer folds 中总 batch 为 16；当前 sampler 会在每个环境内遍历全部 6 类后再补 2 个样本。因此每个类别在两个环境中都至少出现一次，每个 global anchor 都有跨域同类正样本，同时避免 all-view TPP 的显存与二次相似度开销。

同一 `(target, seed)` 的四组还必须共享：

- backbone 初始化；
- train batch 顺序；
- train/val/test indices；
- 数据预处理；
- optimizer、scheduler 和 checkpoint 规则。

模型初始化和 sampler 各自使用显式 generator。TPP/null-control 新模块使用独立固定子 seed，避免“模块创建顺序不同”改变 backbone 初始化。

---

## 5. 阶段 0：代码与协议门禁

任何正式训练前必须完成以下实现。

### 5.1 数据与切分

在 `dg_dataset.py` 中：

1. 优先从原始文件名/采集日志生成 namespaced subject sidecar，并绑定数组 SHA256；
2. 在没有 sidecar 时只生成 `inferred_group_ids`，例如 `E1:B00`，不得把它写成已验证 subject ID；
3. 严格校验每环境 10 个 300-sample blocks，且每 block 标签数为 `[120,40,40,40,40,20]`；
4. 实现 `subject_grouped_holdout` 和显式降级的 `inferred_group_block_holdout`；
5. manifest 记录实际 block IDs、样本/group/subject overlap audit、索引哈希、sidecar 状态和数据文件 SHA256；
6. model seed 的变化不得改变 split。

### 5.2 模型与 loss

1. 实现 amplitude-only TPP；
2. 实现 `global-basis control`，并校准其初始 residual RMS 与 TPP control 在同一数量级；
3. 实现显式 `supcon_view_scope=global_only`；
4. 保留四组完全相同的 projection head 与参数结构；
5. 每 5 epochs 记录 CE 与 SupCon 对各参数块的梯度范数和余弦。

梯度诊断定义：

\[
r_b=\frac{\lambda\lVert\nabla_bL_{SupCon}\rVert}
{\lVert\nabla_bL_{CE}\rVert+\epsilon},\qquad
c_b=\cos(\nabla_bL_{CE},\nabla_bL_{SupCon}).
\]

只在同时收到两种 loss 梯度的共享参数块上解释 \(r_b,c_b\)：amplitude encoder、phase encoder、fusion 和共享 backbone。SupCon projection head 的 CE 梯度为零、classifier 的 SupCon 梯度为零，二者只记录原始 norm，不计算梯度比。梯度比是诊断量，不单独作为 lambda 的硬淘汰依据；CE 梯度接近零时必须同时报告分子、分母，不能只报告比值。

### 5.3 训练与输出

1. 增加 `evaluation.evaluate_target=false`；筛选时 target 文件不得进入 forward、basis calibration、数据统计或 phase input audit，也不得生成 target predictions；
2. 始终保存每个样本的 `sample_id, group_id, true, pred, probability`；只有 sidecar 通过时才额外保存真实 `subject_id`，否则该字段必须为 null；
3. 保存六类 precision/recall/F1、完整 confusion matrix 和 C1/C4 pair confusion；
4. 保存 resolved config、git commit、数据 fingerprint、parameter count、峰值显存、秒/epoch；
5. 四组统一走一个训练入口，不再分别维护 CE/SupCon runner 与 TPP runner 的训练循环。

### 5.4 必须通过的测试

- train/val/test sample 与 group overlap 都为空；sidecar 通过后 subject overlap 也必须为空；
- 每个 source 环境为 8 train groups + 2 val groups；
- 三个 split 的样本数、类别数与上表完全一致；
- 四个 arm 可训练参数量完全相同；global-basis calibration matrix 达到 rank 7、condition number 不大于 `1e4`，且七个 projection blocks 均有非零输入/梯度；
- global-basis 与 TPP 的初始 residual RMS 比值在预注册容差 `[0.8,1.25]` 内；
- `global_only` 不执行任何 antenna-view forward；
- batch 16 中每个域均包含六类，global anchor 的正样本覆盖率为 100%；
- 同一 target 内四组的 resolved config 只允许 temporal factor 与 SupCon factor 不同；跨 target 只额外允许 source/target manifest、calibration artifact 和已记录的 fold-specific lambda 不同；
- `evaluate_target=false` 时无 target inference artifact；
- resolved config 断言 `unwrap_phase=false`，dataloader audit 确认输入没有隐藏 unwrap；
- RTX 5000 16 GB 上四组均通过 10-epoch smoke test，无 OOM/NaN；
- 相同 seed 重复 smoke 的 batch IDs 与初始化 hash 相同。

门禁未通过时，不开始 sweep。

---

## 6. 阶段 1：source-only lambda 筛选

旧结果足以淘汰 standard TPP、residual-0.5、mixed pooling、batch8+random sampler、antenna all-views 和较大的默认 SupCon 权重；正式筛选只保留两个小 lambda：

\[
\lambda_{max}\in\{0.01,0.025\}.
\]

lambda 必须在每个 outer target fold 内独立嵌套选择，绝不允许一个 fold 的 source 数据决定另一个 fold 的 lambda：

| Outer target | 只允许用于筛选的 sources | Screen seed |
|---|---|---:|
| E1 | E2 + E3 | 81 |
| E2 | E1 + E3 | 82 |
| E3 | E1 + E2 | 83 |

每个 outer fold 都执行相同的 6-run source-only 矩阵：

| Run | Arm | Lambda |
|---:|---|---:|
| 1 | Y00 | 0 |
| 2 | Y10 | 0 |
| 3 | Y01 | 0.01 |
| 4 | Y01 | 0.025 |
| 5 | Y11 | 0.01 |
| 6 | Y11 | 0.025 |

共 `3 folds × 6 = 18` 个筛选 runs。全部设置 `evaluate_target=false`，训练 200 epochs，其余配置与正式实验相同。选择过程由脚本逐 fold 自动执行，任何指标和选择都不能跨 fold 传播。

为避免把 `converged=finite` 或固定的 `gradient_exploded=false` 当作占位结论，训练健康状态在运行前按以下 source-only 规则锁定：

1. `finite_training`：逐 epoch 审计 source-train、source-val 和每 5 epochs 梯度诊断中的全部数值；任一 loss、metric 或 gradient 为 NaN/Inf 即失败。实际联合 loss 反向传播后、optimizer step 前，同时用一次参数扫描计算全参数梯度 (L_2) norm；该扫描与 finite 检查合并，不做第二次全参数遍历。
2. `converged`：同时满足（a）最佳 source-val macro-F1 不低于均匀六分类机会水平 (1/6)；（b）最后 20 epochs 的 source-train CE 中位数不高于最初 20 epochs 中位数的 95%。这是“明显不收敛”门禁，不用于在已收敛模型之间排名。
3. `gradient_exploded`：以前 5 epochs 的逐 epoch 梯度 (L_2) 中位数之中位数 (g_0) 为参考，定义 (T_g=\max(100,100g_0))。若逐 epoch 梯度 norm 的 p95 连续至少 3 epochs 超过 (T_g)，或任一步导致该 epoch 的最大 norm 达到 (10^6)，则判为爆炸。只有 1–2 epochs 的有限尖峰保留为报警，不单独淘汰。
4. `screen_metrics.health_evidence` 必须保存完整的 200 点 train-CE、source-val macro-F1、梯度 median/p95/max 序列、finite audit 与上述阈值。`run_screen.py` 对照 hash-bound `resources.history` 重算，`select_lambda.py` 再从 screen results 独立重算三个布尔值；任一不一致均 fail closed。

上述阈值写入 protocol lock。它们不使用 target 数据，不根据 18 个 screen 的结果回调，也不把 CE/SupCon 梯度比 (r_b) 当作硬门禁。

硬淘汰条件只包括：

- NaN、非有限 loss/gradient、持续梯度爆炸或明显不收敛；
- global anchor 正样本覆盖率低于 99%；
- Y11 source-val macro-F1 比 Y10 低超过 2 pp；
- Y11 的 C3 sit/stand F1 比 Y10 低超过 3 pp。

梯度比 \(r_b\) 与余弦 \(c_b\) 按第 5 节定义，在 Y01/Y11 的共享参数块上每 5 epochs 记录。它们用于解释和异常报警，不因某个固定比值区间单独淘汰 lambda。

对每个 outer fold 和 lambda 定义唯一评分：

\[
S(\lambda)=\frac12\left[(F_{Y01,\lambda}-F_{Y00})+(F_{Y11,\lambda}-F_{Y10})\right],
\]

其中 \(F\) 是该 fold 的最佳 source-val macro-F1。只在通过硬门槛的 lambda 中，选择距 \(\max S\) 不超过 0.5 pp 的最小值。若某个 outer fold 没有任何 lambda 通过，则停止整个正式实验，不读取该 fold 的 target test。

正式 2×2 在每个 target 使用该 target 自己选出的 lambda，不要求三个 target 的最终数值相同。由于每个 fold 只用一个 screen seed 选择一次，后续 CI 明确条件于这一次筛选结果和固定的 fold-specific lambda；它只覆盖正式训练的初始化随机性，不覆盖超参数筛选本身的随机性。若要估计完整选择算法的方差，必须在每个正式 seed 内重新执行嵌套筛选，本计划不作这一更强声明。

---

## 7. 阶段 2：正式 2×2

### 7.1 第一批 36 runs

每个 `(target, seed)` 都完整运行四个 arm。不同 target 使用不同的初始化 seeds，避免同一个“弱 seed”跨环境形成伪相关：

| Target | Initial seeds | Extension seeds |
|---|---|---|
| E1 | 101, 102, 103 | 104, 105 |
| E2 | 201, 202, 203 | 204, 205 |
| E3 | 301, 302, 303 | 304, 305 |

第一批：

\[
3\ targets\times3\ seeds\times4\ arms=36\ runs.
\]

配置在首个锁定的 target inference 前完全冻结。旧的 seeds 42/52/62 不进入本次预指定内部统计，因为它们已经参与过方案选择。

### 7.2 36-run 中期规则

中期只允许做预注册的安全性/无效性检查，不允许改配置，也不允许因为结果好而提前宣布成功。

满足任一项则停止，不扩到 60：

- Y11 的 sit/stand F1 相对 Y00 的单侧 95% 上界仍低于 `-2 pp`；
- 任一 target 的 seed 均值中，\(D_S\) 或 \(D_T\) 低于 `-5 pp`，且对应单侧 90% CI 上界仍小于 0；
- 三个 target 等权平均的 \(D_S\) 或 \(D_T\)，其单侧 90% CI 上界低于预注册的最小实用效应 `+1 pp`。

36-run 中期模型使用 target 的 sum-to-zero coding，残差自由度为 6。这里不使用未明确定义的 conditional power，也不比较数据内随机选出的 `max(Y01,Y10)` 点估计。

若未触发无效性停止，则四组一起补 seeds 104/105、204/205、304/305：

\[
3\ targets\times5\ seeds\times4\ arms=60\ runs.
\]

不能只给表现好的 arm 补 seed。

---

## 8. 指标、统计和最终判定

### 8.1 主指标

主指标只有一个：target test macro-F1。

E1–E3 使用预先指定的内部复验分析层级，不称独立确认性检验：

1. 先检验 Y11 是否同时优于 Y01 与 Y10（IUT）；
2. 只有第 1 层通过，才检验交互项 \(I\)；
3. Y11 对 Y00、C3、walking 和 hard-pair 属于安全 guardrail 或次要机制结果。

层级外不再做新的显著性声明。accuracy、其余 per-class 指标和 embedding 几何都是探索性结果，不能反过来选择配置。

所有有方向的最终判定使用单侧 95% confidence bound（等价于双侧 90% CI 的相应端点，\(\alpha=0.05\)）；同时为透明度完整报告双侧 95% CI。第 7 节中期 futility 使用预先写明的单侧 90% 上界。

### 8.2 “联合优于单独使用”

定义：

\[
D_S=Y_{11}-Y_{01},\qquad D_T=Y_{11}-Y_{10}.
\]

两项必须同时满足：

- 平均差值至少 `+1 pp`；
- 配对单侧 95% 下界大于 0。

这是 alpha=0.05 的 intersection-union 判定，不需要为这两个组成检验再做 Bonferroni。缺一项时不能说联合优于两个单组件，也不能进入交互项的预指定检验。

### 8.3 “真正协同”

只有第 8.2 节通过后才检验：

\[
\overline I\ge1\text{ pp},\qquad LCB^{one\text{-}sided}_{95\%}(I)>0.
\]

若 Y11 胜过 Y01/Y10，但交互项 CI 包含 0，则结论只能是“可共存并产生加性收益”，不能称为 synergy。

### 8.4 数据集特定安全指标

还必须同时报告以下 guardrail/次要结果：

1. 次要基线结果：Y11 相对 Y00 的 macro-F1 均值至少 `+1 pp`，单侧 95% 下界大于 0；
2. 安全 guardrail：C3 sit/stand F1 中，Y11 相对 Y00 的单侧 95% 下界大于 `-2 pp`；
3. 描述性机制 guardrail：Y11 相对 Y10 的 C3 F1 平均恢复至少 `+5 pp`；
4. 描述性安全 guardrail：C2 walking F1 中，Y11 不得比 Y10 平均下降超过 `2 pp`；
5. falling/turning 双向混淆率

\[
q=\frac{CM_{1,4}+CM_{4,1}}{N_1+N_4}
\]

作为次要机制结果，相对 Y00 至少下降 `2.5` 个绝对百分点，且差值单侧 95% 上界小于 0；
6. target 稳健性 guardrail：分别计算 Y11-Y01 与 Y11-Y10；任一 target 的 seed 均值都不得低于 `-2 pp`。不使用数据内 `max` 选择 comparator。

六类 F1 的探索性显著性使用 BH-FDR 0.05，不用未经校正的多重 p 值。

### 8.5 置信区间

所有差值先在相同 `(target, seed)` 内配对。完成 60 runs 后，对 15 个优化 blocks 拟合：

\[
d_{e,s}=\mu+\gamma_e+\epsilon_{e,s},
\]

target 使用 sum-to-zero/effect coding；\(\mu\) 明确定义为三个 target 的等权 marginal mean。条件于 E1/E2/E3 的主 CI 使用 t 分布，残差自由度为 12。36-run 中期同一模型的残差自由度为 6。

另外报告：

- 每个 target 的 seed 均值与 CI；
- 把三个 target 均值视为三个点的 target-block CI（df=2），只显示环境异质性；
- 若 sidecar 通过，以真实 subject 为顶层配对 cluster 做敏感性 bootstrap；否则只能以 inferred group block 为 cluster。一次抽中的 cluster 必须同时保留四种方法和全部 seeds 的对应预测，seed 作为固定测试集上的优化重复单独处理，不能把 `subject × seed` 展开成独立样本。

不能把 15 个 seed runs 解释成 15 个独立环境或 15 组独立受试者；同一 target 的 10 个测试 groups 被所有 seeds 重复预测。seed CI 只解释固定测试集上的优化稳定性。数据只有三个环境，因此对“更广泛环境总体”的推断仍然很弱。

---

## 9. 资源预算

| Stage | Required runs | Epochs/run | Purpose |
|---|---:|---:|---|
| 10-epoch smoke | 4 | 10 | 显存、速度、确定性、数值稳定性 |
| Nested lambda screen | 18 | 200 | 每个 outer fold 独立 source-only 选择 |
| Core interim | 36 | 200 | 正式四臂、3 seeds/target |
| Core extension | +24 | 200 | 达到 5 seeds/target |
| E4/E5 independent confirmation | +12 | 200 | 四臂 × 3 crossed seeds |

核心停止路径为 `18 + 36 = 54` 个 full runs；完整内部复验路径为 `18 + 60 = 78`；再加 E4/E5 独立确认为 90。另有 4 个 10-epoch smoke，不计作 full runs。

不要预先承诺具体小时数。先用四组 10-epoch smoke 记录各自 `seconds/epoch` 和峰值显存，再计算：

\[
T_{total}=\sum_{jobs} epochs_j\times time\_per\_epoch(arm_j).
\]

当前 GPU 为 Quadro RTX 5000 16 GB。best checkpoint 约 15 MB；为 configs、history、predictions、checkpoints 和日志预留至少 2 GB。

---

## 10. 可选机制与稳健性扩展

这些扩展不能改变核心四组的配置。

### 10.1 Phase-TPP sentinel

仅用于验证“phase 不需要位置池化”：

- `APS`：amplitude scale 0.25、phase scale 0.25、锁定后的 SupCon；
- 用 source-only validation 与匹配的 Y11 比较；
- 先跑 E1/E2/E3 三个 source contexts、一个新 seed；
- 只有 APS 平均 macro-F1 至少高 `0.5 pp`、C3 至少高 `2 pp` 且 phase 梯度余弦无持续冲突时，才补 `AP0` 完成交互解释。

### 10.2 Temporal-order null control

若要把收益进一步归因于“bin 的绝对顺序/位置”，增加一个保留七个局部 bins 的 control：分别在 level `[1]`、`[2]`、`[4]` 内使用共享映射和 permutation-invariant 聚合消除 bin 顺序，再按固定 level 顺序拼接，必须保留 level 身份、局部统计和匹配尺度。只有真实有序 TPP 稳定优于该 control，才能声称 position/order 本身有价值；核心 Y10-Y00 只支持“时间局部统计”解释。

### 10.3 Group-fold sensitivity

核心结论条件于 F0。先用 Y11 与正式实验中更强的单组件在 F1 上做三个 source-context、单 seed、target-disabled 比较，共 6 runs。若任一 context 的排名反转超过 1 pp，再扩到 F2–F4；否则不做完整五折。sidecar 通过后可称 subject-fold，否则只能称 group-fold。

### 10.4 原始 CE anchor

若需要回答“是否超过原始小模型”，必须用本计划的新 split、preprocessing、batch、sampler、seeds 和 macro-F1 checkpoint 重新运行原始 CE。旧 CE 结果不能混入正式 CI。

---

## 11. 独立确认

本仓库过去已经反复查看 E1/E2/E3 target 结果。新的 model seed 只能验证优化稳定性，不能把旧 target 重新变成未见测试集。

严格确认优先新增从未参与选择的 E4/E5：

- 四个 arm 均在 E1+E2+E3 上训练；
- E1/E2/E3 各自沿用显式 F0：每环境 8 groups train、2 groups val，总计 train 24 groups/7200 samples、val 6 groups/1800 samples；checkpoint 仍按 val macro-F1、平局取更早 epoch；
- 按第 3 节同一规则仅用 E1/E2/E3 source-train 重新生成 confirmation calibration artifact：每个 source 环境取 256 条、seed 9001，并在读取 E4/E5 前锁定；
- 在读取 E4/E5 前，将内部三个 folds 选出的 lambda 按多数票锁成 `lambda_confirm`；平票取较小值；
- 三个 source domains 仍保持每域 8 个样本，因此 confirmation batch 固定为 24=`[8,8,8]`，恰好 300 steps/epoch、60,000 total steps，并保证每域六类都出现；SupCon warm-up 固定为 6000 steps；
- seeds 401/402/403；
- 共 12 次训练；
- 每个 checkpoint 一次性测试 E4、E5；
- 配置、代码 commit 和分析脚本在采集/读取 E4/E5 前锁定；E4/E5 不得参与 basis calibration、checkpoint 或任何统计量。

在读取 E4/E5 前先仅用 E1–E3 做 batch-24 source-only smoke。若显存不足，只允许在四个 arms 同时启用已验证的 activation checkpointing/AMP；不得静默改成破坏三域六类覆盖或 60,000-step 预算的 batch。仍无法运行则独立确认路径记为资源阻断，不临时改协议。

同一个 seed checkpoint 同时预测 E4/E5，因此 seed 是跨 target 的 crossed repeat，六个 target-seed cells 不能当作完全独立训练。独立确认按下列预注册方向判定：

- E4、E5 各自的 seed 均值中，\(D_S\ge0\) 且 \(D_T\ge0\)；
- E4/E5 等权均值中，\(D_S\) 与 \(D_T\) 都至少 `+1 pp`；
- 若要复现 synergy，E4、E5 各自 \(I\ge0\)，等权 \(\overline I\ge+1 pp\)；
- C3 safety guardrail 仍然成立。

CI 以 target 内 subject/group 为顶层配对 cluster，同时保留四个 arms 和三个 seeds；另以 `seed` 为 crossed block 报告优化稳定性。只有两个新环境，环境总体推断仍以方向一致性和效应量为主，不把六个 cells 当六个独立环境做显著性包装。

若只能采集 E1–E3 的新受试者，并且 sidecar 能证实身份，可确认 unseen-subject，但不能确认新环境。若完全没有新数据，最终结论必须写成“同一三环境测试集上的内部复现”。

---

## 12. 明确不再运行的组合

已有结果足以淘汰：

- 非 residual 的 standard TPP；
- TPP `[1,2]`；
- residual scale 0.5；
- mixed max/average 0.5；
- phase-only TPP；
- batch 8 + random sampler；
- all antenna views SupCon；
- 当前 GCI mask；
- feature-axis phase unwrap；
- 同时扫描 alpha、lambda、view mode、batch 和 mask 的笛卡尔积。

旧实验的作用是缩小候选，不是为新协议提供可合并的正式样本。

---

## 13. 执行顺序与交付物

1. 生成并校验 subject sidecar；若无法生成，显式锁定 `inferred_group_block` 降级口径。
2. 实现并测试 group split、global-only SupCon、amplitude-only TPP、global-basis control 和 target-disabled mode。
3. 先生成四个 arm templates；运行 template alignment、phase audit 和 exact parameter-count check。
4. 四组各做 10-epoch smoke，生成资源预算。
5. 对三个 outer folds 分别跑 6-run nested lambda screen，共 18 runs；任何选择不得跨 fold。
6. 锁定每个 fold 的 lambda，生成 `3 targets × 4 arms = 12` 个 resolved configs，并锁定 selection records、代码 commit、split manifests 和分析脚本。
7. 跑 36-run 正式矩阵；只执行预注册 futility 检查。
8. 未停止则完整补到 60 runs。
9. 运行 factorial analysis，输出主差值、交互项、target 分层结果、per-class 结果和配对 cluster bootstrap。
10. 根据结论决定是否做 phase sentinel、order null、fold sensitivity 和新环境确认。

最终必须产出：

- `protocol_lock.json`
- 四个 arm templates、12 个 target-arm resolved configs 和三个 fold selection records
- 每个 run 的 `split_manifest.json`
- `run_registry.csv`
- `factorial_results.csv`
- `interaction_report.md`
- 每样本 predictions/probabilities
- 完整可复现命令和代码 commit hash
