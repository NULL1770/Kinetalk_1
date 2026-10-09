# Phase63：直接内容路径与条件表达响应

## 问题与可证伪假设

Phase53全开发集张口范围.124295，GT.175279；但固定happy/angry片段并非全部低幅度，angry会过开。Phase60单调neutral校准只改善约1.8%范围并损害闭口，不能继续统一增加幅度。现有decoder对B0做多层时序卷积并输出任意残差，有能力重新改写发音轨迹。新试验检验：明确的直接内容路径与低容量条件响应，能否同时降低几何误差、幅度误差和时序偏差。

## 唯一网络改动

替换响应decoder，旧decoder不保留为额外支路。最终52D动作：`positive_gain(g, reference_response) * B0 + expression(g, u, reference_response) * TRAIN_scales + frozen_reference_posture`。

- 内容轨迹B0保持原始25fps索引；gain是整段全局正值（预定范围1/4–4），不移位、不重采样B0，也不从B0时序卷积重新生成发音。
- 表达分支只看已有g32/u16和参考response32；Linear→128、3个masked TCN、52D输出。没有B0/HuBERT/标签/query GT输入表达分支。
- 全部观察通道都可受情感/风格影响，没有硬口型遮蔽。加性表达仍可能改变闭口和局部峰值，所以不能宣称严格音素不变；保留原生闭口/相关性/位移/参考交换验证。
- prior/student、posterior、语义头、统计参考encoder及已有posture bias全部冻结。只训练新gain与表达decoder，p使用均值且无重建梯度。
- neutral B0与数据尺度不改；不使用Phase61映射作为新网络起点，避免两个改动混淆。原posterior未随新decoder校准，oracle只作诊断不作选模。

## 训练、验证与预算

同一TRAIN-fit10903，内部身份743、句子890；相同620条独立neutral支持池，排除query录音/句子，2条不重复参考。batch16、lr1e-4、seed47、固定8epoch/5456update。保留原位置+.5原生相邻位移目标，不加probe/F1/范围等新loss。

必须先验证：旧模型默认行为兼容；正gain、内容逐帧路径、表达输出对B0不敏感、padding/缺失帧隔离、HuBERT-NaN隔离、所有冻结参数与梯度、支持排除、checkpoint恢复。预先保存Git。新目录GPU平衡smoke通过后才正式训练；训练与开发评估不得使用sealed/test。

完整1367 raw/clip、四冻结probe、jaw range/corr/closure/位移、几何与各情感表格。比较同一检查点整套指标；不能拼接历史最佳值或把t-SNE分离当作成功。八个固定情感视频及单因素身份曲线/目标统计审计。固定训练轮数，不根据外部开发集选最好epoch。

成功需要几何与幅度误差实质改善，闭口/时序不恶化，并保留参考的方向性与同人稳定性；否则否决。正gain结构只限制B0路径，不是完整内容解耦证明，不保证一定达到SOTA。

## 已执行状态

预修改Git6895d4c；实现15356f93fdf293284502596dfc639952d6f8296f精确推送核验。61本地/61远端测试通过，120步平衡GPU smoke的末10步/首10步loss=.574757，冻结条件/B0/HuBERT-NaN检查通过。worker12910在freshroot运行，禁止重新训练已完成smoke或重复dispatch。当前epoch4/8；正式生成指标未知。已有句子留出位置loss下降，身份留出未单调改善，不能把训练loss当成果。

collector51592负责本轮完整原件收集/重放、完整指标表、八情感及风格16视频和本轮原生身份曲线。队列有限2小时，不自动推广候选。主视频同时展示GT/neutral B0/Phase53/Phase63/adapted FaceDiffuser固定seed42；后者是已归档改编版本，不是官方end-to-end模型，保留其checkpoint哈希绑定缺失与预算差异限制。
