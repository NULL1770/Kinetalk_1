# 02a / 02b 精选定性案例（2026-10-09）

输出目录：`final_experiment/evaluation/paper_reference_20261009/`。

这些图是论文中的 **selected qualitative examples**，不是平均性能表。每种情感从同一个已生成的 Phase53 shared-rig development clip 中选择一个共同原生帧；只在有效帧且 GT 嘴部活动位于该片段第30百分位以上的候选中，选 KineTalk Phase53 joint 嘴部 MAE 最小的帧。happy 额外在四个时间区间各选一帧，间隔至少3帧。所有方法、GT和KineTalk使用完全相同的 clip、frame、25 fps、rig、相机和渲染设置。

选择没有改变 checkpoint、训练、全量指标或推理结果，也没有为不同方法分别挑时间。原始固定样例和完整视频保留；图注必须写 selected qualitative examples，不能把这些帧当作整体 SOTA 证据。逐帧记录、输入 SHA、方法、时间和选择误差在 [stills_manifest_selected_20261009.json](../final_experiment/evaluation/paper_reference_20261009/stills_manifest_selected_20261009.json) 和 [selected_examples_receipt_20261009.json](../final_experiment/evaluation/paper_reference_20261009/selected_examples_receipt_20261009.json)。

## 文件

- [02a_emotion_comparison.png](../final_experiment/evaluation/paper_reference_20261009/figures/02a_emotion_comparison.png)：八种情感，各一帧，GT / Neutral B0 / VOCA-core adapted / EmoTalk-core adapted / FaceFormer adapted / KineTalk Phase53 joint / FaceDiffuser adapted seed42。
- [02b_native_time_sequence.png](../final_experiment/evaluation/paper_reference_20261009/figures/02b_native_time_sequence.png)：happy 同一句话的四个固定原生时刻，所有方法共用时刻。
- `stills_selected_transparent/`：84 张透明底 1024×1024 PNG。
- `stills_selected_white/`：84 张白底 1024×1024 PNG，可直接拼入 PPT。

方法名称中的 adapted 表示已经绑定 SHA 的 shared-rig 改编基线，不是官方论文端到端复现；完整指标和方法限制仍以 development tables 为准。

## 风格候选结果边界

Phase63 native-affine 候选完成续训、1367 development 评估和2026条匹配有向风格审计，但**未推广为正式模型**：clip_all 的 prior mean 为 MBE .7984、LBE .3896、lip LVE 3.2890 mm、jawOpen range .09384、jaw correlation .4464、closure F1 .2274，未超过 Phase53 style-only 的联合几何/闭口结果。

其风格诊断为：同人 A/B mouth MAE .00840、centered correlation .9883、velocity correlation .9839、closure disagreement .0395；跨人 reference swap mouth MAE .02785、centered correlation .9545、velocity correlation .9391、closure disagreement .1356。目标统计向 donor 移动比例为 mouth mean 78.13%、range 72.31%、displacement 69.35%。这些是候选 receiver 的补充诊断，不能与 Phase53 主模型指标拼接。

当前可以写的主张仍是：Phase53 证明了共享 rig 上的部分参考响应和同人稳定性；Phase63 说明更强的条件接收器仍可能损害发音/闭口，需要后续单独处理。不能只因为精选图片好看或风格距离变小就宣布整体模型完成。
