# 粗事件日程教师与评估（2026-09-19）

本轮采用 `scripts/event_schedule_teacher.py` 的 `event_excursion_teacher_v2`。最初的“相对 pooled anchor 活动”草稿把恒定高表情当作动态，已在训练前弃用；不保留其活动定义作为当前方法。

教师只读取既有 `motion9` 中 up=(2,3,4)、down=(0,1)、squint=(5,7)、wide=(6,8)，每个组取通道均值。在 native valid 与该组各 motion channel mask 的连续有效区间内，使用固定五点三角滤波，再按 `scipy.signal.find_peaks` 找正向突出峰。每组 prominence 阈值只由 fit 的 raw−smooth 噪声 MAD 拟合，固定为 max(0.02,6×MAD)。不使用 query 均值、幅度分位数、身份 anchor 或按样例调阈值。正负常量偏置不改变事件标签。

完整事件需要峰两侧都找到两相邻低值支撑，低值为峰值减去 0.9×prominence。原始信号的半突出度连续宽度至少四帧，用于拒绝滤波把单帧尖峰扩成动作。没有完整回落的边界事件、短峰、复合重叠和每个连续区间前后两帧标为 unknown。缺失段不补零后当作回落，也不跨 gap 连成事件。恒定高值不生成 onset。

`fit_teacher(train_clips)` 返回可 JSON 序列化配置；`extract_schedule(motion9, observed_bool, teacher, motion_mask=None)` 返回：

- `schedule[T,12]`：四类 active、四类线性 phase（事件内 0→1）、四类 duration（秒）；无幅度或基线。
- `known[T,4]`：训练和评分的已知支持；unknown 的 condition 清零，但必须同时保留 known。
- `onset[T,4]` 与 `events`：完整事件的 group/start/end/duration。

known 只控制监督与评分，不能代替生成的 native support，也不能在推理读取目标 motion。GT 日程只能用于训练或显式命名的 oracle 控制验收。音频推理必须使用音频预测/采样的日程。

`event_schedule_metrics.py` 使用同一个冻结教师对多采样输出评分，报告 activity Brier、Bernoulli NLL、活动总时长误差（秒）、事件数。共同评分支持来自真实 motion 的已知支持；生成输出不能用自己的 unknown 缩小分母逃避错误。无已知帧明确输出 pending/null。缺少 prediction 直接报错，不回退 target。fit 与 evaluation clip ID 必须不相交；所有 audio/static/reverse arms 的成员、targets、valid、motion mask 和 score mask 必须完全一致。

这些是粗运动日程弱标签，不能声称心理状态或人工语义 GT。时长误差是活动总时长误差，不是配对单事件持续时间误差；反序可能保持总时长，因此必须同时看逐帧 Brier/NLL。它们用于单独验收音频时序控制，不能替代 ARKit 主表、感知自然度和最终音频生成验证。

测试覆盖常量偏置/恒定高表情、单帧尖峰、gap/删失、组级 motion mask、fit generator iterable、缺预测拒绝、空评分 pending、arms 集合与 fit/eval 防泄漏，以及正确日程优于错误时间的对照。
