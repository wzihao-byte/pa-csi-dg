# 原始子集独立字节验收

2026-10-14，日本时间。**1782 个原始 MAT 及1782份 receipt 全部通过独立核查，合计4,317,606,012字节**。报告见 [raw_acquisition_verification.json](raw_acquisition_verification.json)，可重核脚本见 [verify_raw_acquisition.py](verify_raw_acquisition.py)。单线程核验仅读取 receipt、raw 和已登记元数据，不调用scipy、不运行模型、不使用GPU、不更改acquisition或预处理状态。

逐文件重算SHA256、检查长度及MATLAB v5头／字节序，并与独立receipt及最终acquisition索引双重比对。1782个身份、路径和版本1均精确匹配固定manifest，raw目录没有额外或缺失MAT；1782个原始文件的字节哈希均不同，**无完全相同字节文件**。字节唯一不等于解码CSI数值没有重复。

从固定作者commit `c36f94f20a5834fed22c6c208bb893ae4ae3f4e4` 的原annotation重新筛选5GHz、单人条目，并逐项核对标签、唯一非空用户活动确定的主体、位置、房间、采集组和instance。原annotation与manifest哈希均匹配已登记元数据。

| 原标注分布 | 本次独立复核 |
|---|---|
| 每房间 | classroom／meeting_room／empty_room各594 |
| 每房间每类 | 九类各66；固定类别序列与编号0–8一致 |
| 每房间每主体 | 主体1–6各99 |
| acquisition groups | classroom 19–24，meeting_room 55–60，empty_room 91–96；18组各99 |

初次下载pass的**1584成功＋198 query端点404**已保留在 `D:/pa-csi-dg-revision-runtime/data/wimans_20261014/acquisition/attempts/2026-10-14T07-09-20.062530+00-00.json`。其1584份成功receipt与最终索引完全一致；198失败身份全部以**同一数据版本1、同一manifest**恢复成功，没有按解码或模型结果排除、替换或补样本。最终来源为3个重核saved probes、1581个query式单文件下载和198个path式单文件下载。完整子集验收没有抹去初次失败记录。

源码归属需单列：保存的恢复版 `0d1c4e961a1358545fc631fdd41d1db8726871ce2ea51d7198d425cba810292d_acquisition.py` 在写status时用 `sha(Path(__file__))` 读取磁盘源码。当前源码及final status记录的hash为 `1c56724a7e4a0460c8c5edf67255432c6fee4dfb4817246c9e018cc50dfb741c`；后续磁盘修改可能改变该记录，却不改变进程已加载的代码。因此，**不能将该final hash声称为已独立确认的inflight执行版本**。报告分别保存saved／current／reported hash，并将执行版本确认标为false。本次原始字节、身份与receipt验收独立于这一归属不确定性。

本项通过的是原始取得与身份／字节完整性。全量MAT解码、CSI及时间戳数值质量、处理数组哈希、storage_order_v2映射、adapter及正式配置冻结属于后续独立验收；这里不称完整输入QA已通过，也不确认物理Tx／Rx对应或相位校准。目标记录访问仅限原始身份和字节质量核查，没有读取目标模型分数。
