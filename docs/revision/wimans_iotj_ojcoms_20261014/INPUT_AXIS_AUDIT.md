**WiMANS MAT 轴、置换与存储顺序协议核验，2026-10-14 JST**

原始 MAT 再下载核验和候选预处理已经实际执行。作者 MAT 导出来源尚未确认，因此原计划中的“已确认物理 Tx 索引 0、三个物理 Rx”输入不能标为通过。按 PLAN 已列明的替代路径，在任何 WiMANS 目标模型评价前登记 [存储顺序 v2](axis_storage_protocol.json)。它固定软件操作，保留物理映射的不确定性。

| 核查项 | 实际证据 | 结论 |
|---|---|---|
| 作者数据来源 | 作者固定 commit `c36f94f20a5834fed22c6c208bb893ae4ae3f4e4` README 指向 [Kaggle WiMANS](https://www.kaggle.com/datasets/shuokanghuang/wimans)；公开 API 元数据版本 1，初次发布 2024-02-16 | 下载仅选定 1,782 个 MAT；不下载视频或幅度缓存 |
| 三条抽查文件 | `act_19_1`、`act_55_1`、`act_91_1` 保存字节的 SHA256 与 PLAN 预登记值全部相等 | 原始文件身份通过；不是完整子集验收 |
| MAT schema | `trace[packet,0][0,0]` 含 `timestamp_low,Nrx,Ntx,perm,csi`；三条文件 `Ntx=Nrx=3`，CSI shape `[3,3,30]`，complex128 | 实际存储 shape 和字段已核查 |
| 非恒等 perm | 三条记录第一包分别 `3,1,2`、`2,3,1`、`2,3,1`；第一条内部还存在 perm 变化 | 非恒等字段本身不能证明存储 CSI 尚未置换 |
| 作者幅度代码 | [固定 preprocess.py](https://github.com/huangshk/WiMANS/blob/c36f94f20a5834fed22c6c208bb893ae4ae3f4e4/benchmark/wifi_csi/preprocess.py) 直接取已保存 CSI 的绝对值 | 支持从 MAT 直接提取幅度，不证明 MAT 的导出处理 |
| 作者导出代码 | 固定仓库 recursive tree 共 38 个 blob，无 DAT→MAT、`read_bfee` 或 `read_bf_file` 导出代码 | 导出时是否调整 Rx、是否换轴仍 unresolved；不推断 |
| 标准 CSI Tool | 固定参考 commit `08ea7cc082c2eb903c0a6fecb0af92d545734673` 的 [read_bfee.c](https://github.com/dhalperi/linux-80211n-csitool-supplementary/blob/08ea7cc082c2eb903c0a6fecb0af92d545734673/matlab/read_bfee.c) 创建 `[Ntx,Nrx,30]`；[read_bf_file.m](https://github.com/dhalperi/linux-80211n-csitool-supplementary/blob/08ea7cc082c2eb903c0a6fecb0af92d545734673/matlab/read_bf_file.m) 将 perm 应用于第二轴但保留 perm 字段 | 说明一种标准软件惯例；不能单独证明 WiMANS 使用了该导出过程 |

v2 选择每包存储 `csi[0,:,:]`，按 C-order 展为 90 列。三个视图继续同步遮蔽列 `[0,30)`、`[30,60)`、`[60,90)`。不应用 `perm`，不声称物理 Tx/Rx 映射已校验、不声称相位校准。原始元数据中的 `Ntx=Nrx=3` 与本协议的软件选择分别描述。历史 `AntennaViewBuilder(mode=tx,num_rx=1,num_tx=3,num_subcarriers=30)` 仍仅表示三个 pseudo-Tx 特征组。

时间戳以 signed/unsigned uint32 位模式解释，按模 `2^32` 的向前差求累计相对微秒；差值达到 `2^31` 及以上时拒绝，重复时间保留并算术平均。幅度为 abs；相位为 float64 wrapped angle，重复时间算术平均，同一个含端点 850 点网格线性插值，再 feature-axis unwrap 后 cast float32。不再次 unwrap、不循环平均、不按目标成绩改选处理链。

三条文件候选结果已保存在 `D:/pa-csi-dg-revision-runtime/data/wimans_20261014/preview_storage_order_v2`。原始字节 receipts、逐样本统计、amp/phase 数组及其哈希均保留。第一条实测 2,825 包、2.997995 秒、最大间隙 13.999 ms，选定 90 列含 77 个零幅度值；这不是排除条件。相位跨特征轴 unwrap 后可超出 `[-pi,pi]`，是预声明软件表示的结果。

完整 1,782 条取得与存储 v2 数值输入验收现已通过，见 [原始独立验收](raw_acquisition_verification.json) 与 [处理数组独立验收](prepared_input_verification.json)；runtime 为 `acquisition/status.json` 与 `processed_storage_order_v2/input_audit.json`。完整输入验收通过仅表示该存储顺序协议的成员、数值、哈希与处理语义通过；不解决物理导出来源问题。任何正式训练还需真实源完整 batch 短训 QA 和冻结运行包，目标模型评分未执行。

全子集共 4,317,606,012 个 MAT 字节，没有完全相同字节文件；实测每记录 2,726–2,993 包、2.895086–3.031014 秒，最大包间隙 42,707 微秒，重复时间与回绕均为 0。选定 90 列累计 23,997 个零幅度值均按预声明 angle(0)=0 约定保留，没有因此排除记录。全量矩阵逐行哈希复核通过，按 metadata 房间×类别首记录预定的 27 条独立重建与当前输出一致。

元数据中的数据许可为 CC BY-NC-SA 4.0，API 原始响应保存在 runtime `acquisition/kaggle_dataset_metadata.json`。论文应引用作者论文和数据来源，补充包发布脚本与身份哈希；是否再分发原始或处理后信号需按该许可处理。仓库 tree、作者代码与标准工具参考字节保存在 `work/wimans_20261014/author_source`；机器核验入口为 [input_axis_audit.json](input_axis_audit.json)。
