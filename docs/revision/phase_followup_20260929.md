# 相位补充实验：历史与 native 网格内的容量控制

日期：2026-09-29。实现入口：`experiments/revision_validation/phase_followup_study.py`。

本研究补齐 27 次训练，使用新目录 `D:/pa-csi-dg-revision-runtime/studies/r_phase_controls_v1_20260929`。由总协调器在时间预处理的 18 次 global 对照结束后串行启动；本模块的 QA 与冻结步骤不会启动正式训练。

2026-09-29 已完成全部 QA 和 27 次运行的冻结；冻结时 `runs` 目录尚未创建，尚无新增识别结果。冻结 manifest SHA256 为 `c98c640ac361a7888bb9a8c3a636781e8c7a9d6b117c34afeb5091a45a9a277f`，冻结驱动文件 SHA256 为 `38172e9493394e5a1ff63be137667260c0e02fd36a04126aa4c1bec156676958`。已从冻结 `code` 目录验证入口及三个 control 各 9 次的完整矩阵。

独立报告 QA 记录为 `D:/pa-csi-dg-revision-runtime/qa/phase_followup_report_20260929.json`，其开始和结束的模块哈希均与上述冻结文件相同。九项检查通过，覆盖 63 次模拟运行、七臂、八种对比各九个配对、负性能差保留、缺失或额外运行、重复引用、身份/指标/哈希篡改及输出幂等性。该报告测试没有构建模型、使用 GPU 或执行真实目标推理。

## 预定比较

| 网格 | 输入和模型 | 状态或新增数 | 可回答的问题 |
|---|---|---:|---|
| 历史 | 单路幅度，R10 | 复用 core 的 9 次 | 历史幅度基线 |
| 历史 | 双路幅度 AA，同一 twin 架构 | 新增 9 次 | 第二编码器容量的效果 |
| 历史 | 幅度＋历史相位 AP，同一 twin 架构 | 新增 9 次 | 相对于 AA 的历史相位增量 |
| native linear | 单路幅度，NATIVE_LINEAR | 复用 inputs 的 9 次 | native 幅度基线 |
| native linear | 双路幅度 AA | 复用 inputs 的 9 次 | 相同网格下的容量对照 |
| native linear | 幅度＋差分相位 AP | 复用 inputs 的 9 次 | 相同网格下的差分表示 |
| native linear | 幅度＋新历史式相位 AP | 新增 9 次 | 同网格、同 twin 容量下的相位处理链比较 |

最终汇总包含 63 次运行：27 次新增训练与 36 次冻结引用。比较限于同一时间网格；不把跨网格的分数差解释为纯相位因素。

## 数据与模型保持一致的内容

历史 AA/AP 使用 `modality_arrays_v1/historical_feature_axis_unwrap` manifests。幅度仍是原历史 float64 数组；相位严格按 `np.unwrap(original_float64_phase, axis=-1).astype(np.float32)` 构造，沿 90 列展开时跨越接收链边界。核验全部 9,000 条记录的身份及相位数组公式等价，仅用于复现历史处理，不宣称该操作构成物理校准。没有将 native 幅度与历史相位按行号拼接。

新 native 相位版本为 `native_time_wrapped_linear_feature_unwrap_v1`，由独立数据编译器冻结：DAT wrapped angle → native elapsed time → 在每条记录的 850 点等间隔网格上对 wrapped angle 作算术线性插值 → 沿最终 90 列执行 unwrap → float32。幅度与样本身份原样复用 C/native linear；不执行时间轴 unwrap、时间中心化或接收链差分。这是新的整条相位处理链诊断，不等于原历史数组，也不宣称恢复缺失区间内的运动信息。

三个新增 control 均复制已冻结 `r_inputs_v1` 的 TWIN_AA/TWIN_AP 配置，仅替换输入版本、说明、运行标识及 source/target manifest 绑定。模型和引擎原样复制旧 INPUT_CODE。每个 twin 模型总参数 **3,655,286**，CE 分类有效参数 **3,556,598**；余下 98,688 参数的投影头不参与 CE。AA 与 AP 使用相同结构及相同初始化，第二编码器独立于第一编码器。

固定三个留出环境，E1 seeds 401–403、E2 seeds 501–503、E3 seeds 601–603。每次训练 200 epoch × 300 updates，batch 16，Adam、学习率日程、源平衡采样、运行时及训练预算与旧 twin 实验一致。源训练/验证样本数为 4,800/1,200，按被试和采集记录隔离；checkpoint 选择最早达到源验证 macro-F1 最大值的 epoch。

## 检查、冻结和运行接口

```powershell
python -m experiments.revision_validation.phase_followup_study qa-history --output D:/pa-csi-dg-revision-runtime/qa/phase_followup_20260929/history_v4 --gpu-evidence D:/pa-csi-dg-revision-runtime/qa/phase_followup_20260929/history_gpu_smoke.json
python -m experiments.revision_validation.phase_followup_study qa-native --output D:/pa-csi-dg-revision-runtime/qa/phase_followup_20260929/native_v1
python -m experiments.revision_validation.phase_followup_study freeze --bundle D:/pa-csi-dg-revision-runtime/studies/r_phase_controls_v1_20260929 --history-qa D:/pa-csi-dg-revision-runtime/qa/phase_followup_20260929/history_v4/source_qa.json --native-qa D:/pa-csi-dg-revision-runtime/qa/phase_followup_20260929/native_v1/source_qa.json
```

正式运行由协调器从新 bundle 的 `code` 目录执行：

```powershell
python -m experiments.revision_validation.phase_followup_study run --bundle D:/pa-csi-dg-revision-runtime/studies/r_phase_controls_v1_20260929
```

QA 检查真实源 batch 的直接 CE、梯度有限性、分类有效参数和各 seed 初始化。CPU adapter 与直接 twin 调用要求逐位等价；历史 AA/AP 另有 GPU 有限梯度冒烟证据，未执行优化器更新。native 新输入使用 CPU 检查，避免与时间实验争用 GPU。原 twin 引擎的精确 CPU 断点恢复检查来自其冻结 QA，模型/优化器/RNG/历史及选中预测均经过核验。

全环境输入 QC 会读取三个环境的数组以核对哈希、有限性及历史公式，包括某一折作为 holdout 的信号；不会对这些 holdout 执行模型预测或依据它们选模。源模型 QA 的前向/反向仅使用 E1 折的源环境 E2/E3。报告分别记录 `target_signals_opened_for_input_QC=True`、`source_model_target_data_loaded=False` 与 `target_model_inference_performed=False`，不把两种检查混称为整个进程完全未打开目标信号。

history_v1/v2 检查产物保留，均已通过当时版本的源模型和数据检查；history_v3 在报告措辞最终修改时中断并保留，未启动训练。最终冻结使用 history_v4，绑定定稿代码。最终版本补强了数据编译定义快照、GPU/旧断点恢复证据哈希绑定、上述输入 QC 范围说明及报告的完整矩阵/引用门禁，并将正、零、负计数明确标为性能差值，避免与对比学习的正样本配对混淆。

27 次源训练全部完成、哈希验证及选模规则验证通过后，才创建目标数据 loader。随后统一评估，重建每次 macro-F1，保存七臂结果、同网格内配对差和全 seed 结果。历史引用的配置、checkpoint、源/目标预测和完整冻结 manifest 均重新核验。先前已使用 E1–E3，因此本研究是开发环境上的补充证据。

## 输出与结论边界

`comparison/report.json`、`report.md` 与 CSV 包含全部预定运行，主比较为历史 AP−AA、新 native 历史式 AP−AA、差分 AP−AA 及差分 AP−历史式 AP。AA−A 描述额外编码器容量的效果；AP−A 同时变化模态与容量，不独自证明相位有用。

`queue_status.json` 最终包含 `status=completed`、`jobs=27`、`bundle_sha256`、`evaluation_summary_sha256` 和 `comparison_report_sha256`。异常记录于 `queue_failure.json`，不静默跳过失败或负收益运行。没有按目标成绩选择种子、增加搜索或提前停止。

GPU 重复 backward 已观察到与相同初始化/RNG并存的数值差异；不承诺 CUDA 轨迹逐位复现。具体导致差异的算子未完全定位。

当前状态以新 bundle 的冻结 manifest、源训练状态和比较报告为准；方案和 QA 本身不是识别结果。
