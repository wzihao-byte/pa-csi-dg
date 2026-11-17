# 时间预处理补充实验：协议独立核查

检查日期：2026-09-29

结论：18 个 native global-MLP 补充训练可复用已冻结 R00 模型及 inputs 引擎。核查通过；本检查未启动训练、未打开源/目标信号数组、未修改历史文件。

## 已验证证据

- 核对 core、inputs 的完整冻结文件哈希，36 个相关已完成 run 的配置、结果附件和源验证最优 epoch。
- 3 个环境、9 个配对种子，四个历史分支各 200 epoch 的采样哈希均一致；独立重算全部 1,800 个 schedule，与记录一致。
- 历史与 native 的 train/val/test 样本身份、标签、环境、索引、被试和采集顺序一致；native 回指原 MAT 文件及哈希验证通过。
- 每折 4,800 个源训练样本、1,200 个源验证样本、3,000 个目标样本。
- 9 个种子的非池化投影公共张量完全一致；新 global 配置构造结果与 R00 完全一致。
- R00/TPP 总参数均为 1,878,142；CE 有梯度分类参数均为 1,779,454；98,688 个训练用 projection-head 参数在 CE 中未使用。
- CPU 合成输入检查：extension objective 与直接 CE 完全相同，梯度有限，优化器步数为 0。

## 实施边界

1. Create a new study; never add runs or mutate protocols in r_core_v1 or r_inputs_v1.
2. Copy the input-study frozen scientific code. Use extended_engine.train_experiment and the VerifiedInputs adapter.
3. Set BOTH arm and experiment.base_arm to R00; pool remains unchanged, objective ce, model_kind single. No head_kind field is consumed.
4. Copy each B/C source and deferred-target manifest without modifying contents. Bind their SHA256 in each new configuration.
5. Keep total_parameters 1878142. Verify active CE/inference classification count 1779454; the 98688 projection-head parameters are unused by CE.
6. Keep all nine matched fold/seed combinations, 200 epochs x 300 updates, batch 16, fixed Adam/LR, and earliest source-validation macro-F1 checkpoint.
7. Sampler is fixed at 16 samples: eight per source environment, every class per source, epoch-seeded independently of model initialization.
8. Do not replace cached native arrays or cast/resample them again. VerifiedInputs hashes the actual files and casts batches to float32.
9. Resource profiling uses a separate model and no optimizer updates. Training RNG resets to seed+17 after constructing the training model.
10. Use a separate new run directory per configuration. A resume requires exact config/runtime/model/optimizer/RNG hashes; recorded failures must be diagnosed.
11. Require every one of the 18 new source jobs to finish and verify before new target evaluation. Existing E1-E3 scores have been observed, so this is development extension evidence.
12. The stock extended evaluator pairs only same-study or core-reference results; it does not directly resolve historical input-study TPP references. Add explicit cross-bundle keyed pairing after verifying all reference hashes.
13. Historical A uses MAT provenance; native B/C use DAT provenance. A-to-B includes representation/origin differences and is not a pure isolated timestamp manipulation.
14. B-to-C shares native clock and record identities but changes merge/interpolation processing; primary inference compares TPP-minus-global within each input version.
15. CUDA adaptive max-pool backward remains nondeterministic. Matched seeds and common initialization do not imply bitwise replay on GPU.

## 新增矩阵

|输入版本|模型|种子|新增训练|
|---|---|---|---:|
|native_time_legacy_merge|R00 / global-MLP + CE|E1:401–403; E2:501–503; E3:601–603|9|
|native_time_linear|R00 / global-MLP + CE|同上|9|

这份审计证明方案可配对，不预判训练结果；全部环境和全部种子均应保留。
