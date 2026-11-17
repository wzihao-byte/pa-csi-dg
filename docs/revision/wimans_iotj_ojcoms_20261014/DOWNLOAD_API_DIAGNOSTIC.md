# WiMANS 单文件 API 有界诊断

2026-10-14 16:17 日本时间。读取现有 acquisition 状态后，执行一次由三个串行请求组成的无认证诊断。没有重试循环、下载视频或完整大包，没有更改、停止原 `prepare_workflow` 进程 PID 21400，没有写入 raw 文件或 acquisition receipt。响应摘要与实际字节哈希见 [download_api_probe.json](download_api_probe.json)，诊断代码见 [probe_download_api.py](probe_download_api.py)。

| 请求 | 结果 |
|---|---|
| 原 query 式，`fileName=wifi_csi/mat/act_21_44.mat`，v1 | 404，134字节HTML，无重定向，标题404 |
| 同一 query 式，先前已成功收取的相邻 `act_21_43.mat`，v1 | 相同404／134字节／HTML SHA256 |
| 文件名放入路径 `/download/shuokanghuang/wimans/wifi_csi%2Fmat%2Fact_21_44.mat?datasetVersionNumber=1` | 一次302→Google Storage 200，`application/zip`，恰一个成员 `act_21_44.mat`，CRC可解包且MATLAB v5头通过 |

路径式返回 MAT 为2,349,356字节，SHA256为 `0162c8161116962ae777850103dda2b9db8e0174e8326945a5d77982250667b7`。解包仅在内存中进行，payload未保存。`act_21_43`原有receipt记录07:10:22 UTC已通过query式收取；后来相同URL404不能解释为该样本从数据集不存在。当前证据支持**路径式单文件API是一个已实测可用的恢复路径**，不能据此确定query式404的服务端原因、所有失败文件状态或后续吞吐。

建议：保留原流程的所有成功receipt和失败记录；原运行结束后，从固定1782成员中精确列出无receipt项，再用文件名路径式API、v1及低并发进行独立恢复。仍逐项检查ZIP唯一成员、CRC、MAT头、长度、SHA256及已声明探针哈希；不排除失败样本、不替换版本，也不重下已验收原始文件。恢复日志独立保留并重新汇总完整子集验收，不能只把 `failed_rows` 改成零。

Kaggle官方仓库的 [nested-file接口问题记录](https://github.com/Kaggle/kaggle-cli/issues/739)亦记录路径形式与子目录兼容问题；该外部记录仅提供接口背景，恢复是否可用以本次三请求的直接响应为依据。
