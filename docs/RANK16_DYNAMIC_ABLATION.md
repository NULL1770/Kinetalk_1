# Rank-16 动态场小对照

当前实现的低率情感控制点由 `model.affect_rank` 配置，默认主线仍为 8。`configs/neutral_affect_pilot_rank16.yaml` 是一个独立的 16D 实验配置：motion teacher、audio student 和共享 `local_projection` 同时使用 16D 控制点，时间步长仍为 4 帧，renderer 的输入/输出契约不变。

rank16 必须从相同 teacher 输入重新训练。旧 rank8 checkpoint 的控制头和 `local_projection` 形状不同，严格加载会失败；不能通过补零、截断或部分 `strict=False` 加载来制造“公平”的 rank16 结果。这样可以把“维度是否是瓶颈”与随机初始化和数据/批次差异分开记录。

建议对照固定数据切分、seed、步数和音频特征，只改变 `affect_rank`，并同时报告 full/mean/reverse、眉眼/嘴部时序相关、全局情感准确率和口型指标。rank16 训练改善但 heldout 动态不改善时，说明瓶颈主要在目标或监督，而不是控制点容量。
