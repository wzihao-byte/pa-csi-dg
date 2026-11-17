**WiMANS 真实源 QA 与执行冻结（2026-10-23，日本时间）**

本轮完成 PLAN 阶段 3 的真实源训练 QA 与阶段 4 的配置／作业清单冻结。没有启动正式训练，没有读取任何目标房间信号。独占 GPU 测速留到 H-S unwrap 验证队列（`studies/h_phase_link_unwrap_20261023_v2`，9 次）结束后进行。

| 项目 | 结果 | 证据 |
|---|---|---|
| 实现单元测试 | 12／12 通过；代码 receipt 与 10-14 候选配置、三折 shape QA 完全相同 | `python -m unittest experiments.wimans_20261014.tests.test_implementation` |
| 真实源 W11 QA | 三折各以 W11 seed 42 候选执行：batch32 hybrid（完整 32＋masked 96）2 次真实更新、396 条源验证、checkpoint strict reload logits 逐位相同；目标信号读取 0 行；全部 loss 有限 | `D:/pa-csi-dg-revision-runtime/data/wimans_20261014/source_qa_v1/<room>/source_qa.json` |
| QA 资源 | CPU，每折 80–85 秒，峰值 RSS 约 32 GiB（与 GPU 训练并行，未出现分页失败） | `source_qa_v1/<room>.resource.json` |
| 冻结配置 | 36 个四臂正式配置＋9 个 CORAL 源搜索配置，各绑定本折 QA receipt、代码 receipt、输入审计与划分 receipt；`validate_config(formal=True)` 全部通过 | `D:/pa-csi-dg-revision-runtime/studies/wimans_20261014_v1/frozen_configs` |
| 作业清单 | `registry.json`（SHA256 `55fc8d02…a0f1`）：四臂 36 次按 房间×种子×臂 排序；CORAL 搜索 9 次单列；执行代码 `execution.py` 哈希一并冻结 | `studies/wimans_20261014_v1/registry.json` |

QA 只说明适配与完整训练契约在真实源数据上可执行；两步更新后的源验证 accuracy（约 11%）不是模型结果。输入仍为 storage_order_v2 软件存储顺序，物理 Tx/Rx 导出来源未确认；三个辅助视图沿用 pseudo-Tx 软件特征组语义。

**执行规则（已冻结）**

- `run --stage four_arm_source`：串行在 cuda:0 执行 36 次源训练；中断后在同一 run 目录以 `--resume` 从上个完整 epoch 恢复，每次调用另存 attempt receipt。启动前若有其他训练／评价进程则拒绝。
- `run --stage four_arm_target`：只有 36 个源 checkpoint 全部冻结并核验后才能运行，统一产生目标成绩。
- `run --stage coral_search_source`：9 次 source-only 搜索，seed 20261014 禁止目标评价；随后 `prepare_configs select-coral` 按源验证 accuracy 选 gamma（平局取小）。WC 正式 9 个配置在选择后另行构建、冻结，不在本清单内。
- 冻结后修改任何受控代码、配置或 QA receipt，`verify` 会失败；变更须另建新版本 study。

**待完成**

1. 已于 2026-10-23 13:39 JST 以 WMI 脱离会话启动链式脚本 [chain_after_unwrap.py](../../../experiments/wimans_20261014/chain_after_unwrap.py)：每 60 秒检查 unwrap 队列；仅当 9／9 完成、results 完整且 runner 已退出时，先执行独占 GPU 测速（`execution benchmark` → `studies/wimans_20261014_v1/gpu_timing.json`），成功后启动 `run --stage four_arm_source`（36 次）。unwrap 队列失败、停止或 runner 消失，或测速失败，链条记录原因并退出，不启动 WiMANS 训练。状态见 `studies/wimans_20261014_v1/launcher/chain_status.json`。工时据实测写，不按 MultiEnv 速度推算。
2. 36 次源训练完成后另行运行 `four_arm_target`（链条不包含目标评价）。
3. CORAL 搜索与 gamma 选择、WC 正式配置；MC／MA／ME 的 MultiEnv 适配；可选 AA、differential AP 与未见主体扩展仍按 PLAN 另行冻结。
