# 三种 masked-view 规则：类别变化补充

从已有27次运行的逐运行类别指标恢复，无新增训练或目标前向推理。下表为类别F1的配对变化（百分点），先按每个环境内三个种子平均，再对三个环境等权平均；完整162条类别配对差和各环境结果见CSV，负差未删除。

| 活动 | All−Hybrid | Remove−Hybrid | All−Remove |
|---|---:|---:|---:|
| No movement | +1.82 | +2.84 | -1.02 |
| Falling | +5.72 | +2.83 | +2.89 |
| Walking | +2.95 | +3.43 | -0.48 |
| Sitting / standing | +8.70 | +4.62 | +4.09 |
| Turning | +0.31 | +0.59 | -0.28 |
| Picking up a pen | +2.14 | +4.80 | -2.66 |

相对Hybrid，All-view在坐下/站起（+8.70）和摔倒（+5.72）的平均改善最大；Remove-unselected在捡笔（+4.80）和坐下/站起（+4.62）的改善最大。All-view与Remove-unselected之间没有逐类别一致的排序：前者在坐下/站起和摔倒更高，而后者在捡笔、不活动、行走和转身更高。六类平均差全正不表示每个环境或每个种子的差都为正；逐对正差计数单独保存在汇总CSV。

这些结果描述匹配规则下的识别变化，不把“从分母排除”或“增加正样本”直接解释为已被证明的目标域机制。不使用H/Y分数来排序规则，也不从总体差推断某个具体样本流向。

可用于英文结果节的文字：

Under the matched four-view protocol, all-view same-class positives improved classwise F1 over the hybrid rule most for sitting/standing (+8.70 percentage points) and falling (+5.72). Excluding unselected same-class pairs from the denominator produced its largest improvements for picking up a pen (+4.80) and sitting/standing (+4.62). The two alternatives did not show a uniform classwise ordering: all-view positives were better for sitting/standing and falling, whereas the exclusion rule was better for the other four classes. These are descriptive paired changes averaged over the predefined seeds and development environments; they do not establish the mechanism of target-domain improvement.
