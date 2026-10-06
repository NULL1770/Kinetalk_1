# Phase26 stable 固定两轮／十二轮对照

恢复时先读 `CURRENT_OPTIMIZATION_STATE.md` 和 `PHASE26_RUNBOOK.md`。本页结果是完整两轮基线；十二轮仍在训练，不把中间分数当最终结果。

全部三训练seed47/48/49、全部1367验证段、三固定draw42/123/2026，clip_all等权均值：

| 指标 | centered对照 | 标准化候选 |
|---|---:|---:|
| MBE | 0.873147 | 0.855815 |
| LBE | 0.419423 | 0.416628 |
| 原128生成动作macro-F1 | 0.272033 | 0.778163 |
| 原64生成动作macro-F1 | 0.286513 | 0.665453 |
| 辅助stable128 F1 | 0.414928 | 0.792092 |
| 辅助stable64 F1 | 0.337609 | 0.748873 |
| 嘴部位移MSE | 0.002202157 | 0.002201024 |

三个seed的中性jaw相关性分别下降0.026895、0.028825、0.013651，因此联合门槛全未通过，默认timing000不替换。原64 F1、MBE也未达用户目标。这里的F1均来自最终生成动作；音频分类头及oracle不纳入这张表。

| 情感 | 原128生成F1 | 原64生成F1 |
|---|---:|---:|
| neutral | 0.461 | 0.389 |
| angry | 0.774 | 0.662 |
| contempt | 0.806 | 0.777 |
| disgust | 0.828 | 0.692 |
| fear | 0.779 | 0.581 |
| happy | 0.895 | 0.892 |
| sad | 0.880 | 0.638 |
| surprise | 0.801 | 0.693 |

happy已能被判定器识别，整体F1较低主要与neutral及原64的fear、sad混淆有关。原64的fear约32%被判surprise，sad约23%判disgust／20%判angry。眉部误差约95%来自平均姿态，嘴部约50%；这比“所有嘴都太小”更符合当前证据。GT本身两判定器macro-F1仅约0.66／0.64，生成F1超过GT不等于生成更逼真，仍需几何和时序检验。

可视化均固定seed47、draw42、M025句005；非neutral用L3、neutral用L1，最长有效连续原生区间。无gain／lag拟合或系数修改。全部音频哈希与原native来源一致，offset0，按真实native开始时间裁剪。三列为GT／匹配对照／标准化；原blend未变、每帧shape-key误差0、首中末帧非空检查通过。

- [八种情感中间帧汇总](../final_experiment/evaluation/diagnostics/phase26_channel_coordinates_stable_budget_20261006/descriptive/all_emotions_midframe.png)
- 视频保存在 `final_experiment/evaluation/rendered/phase26_stable2_{emotion}_s47_verified_camera_20261006/comparison.mp4`，全八类索引 `descriptive/all_video_manifest.json`。
- t-SNE／混淆矩阵：`descriptive/pilot_emotions_validated/`。共同t-SNE使用原128冻结TRAIN probe的hidden特征，GT／对照／标准化全部1367验证段一次无标签联合投影，固定seed47/perplexity30/1000iter，未重训probe或选图；对应三个F1从savedfeature重放精确闭合。它是描述性图，不能以“分簇”替代生成质量指标。

十二轮从相同warm起点重新训练，三个centered对照在1568更新处全部权重和输入流已逐位等对应新两轮基线；候选也必须通过相同门槛。固定final12后报告全指标、分组、近恒定通道、配对区间及2→12变化。不按epoch／seed／draw选结果，loss和架构均保持不变。

若十二轮仍无法共同改善F1、几何和时序，先判断两轮至十二轮的TRAIN收敛和验证偏差：收敛后平均姿态仍差，应定位连续global与身份条件；jaw时序仍差，应拆分B0与残差贡献。每次只修改有证据支持的一项，不能预先继续加loss。论文比较还需要相同split、参考、音频缓存、rig、指标和预算的对比方法，当前不能声称SOTA。
