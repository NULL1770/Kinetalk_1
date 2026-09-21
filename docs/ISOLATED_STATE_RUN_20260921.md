# 独立均值/状态学习与冻结状态残差训练

2026-09-21。当前设计已经实现并正式启动，实际运行信息见末节。不能把启动、单测或 smoke 完成称为动态成功。

## 改动与依据

前一轮 calibrated run12 的上脸 raw MSE 主要来自片段平均姿态误差：眉部91.44%、眼部86.03%。原 state/local 共用可训练骨干，flow 更新可能破坏显式状态，残差还可抵消状态。该证据支持先拆分学习；尚不能证明这是所有失败的唯一原因。

新 `IsolatedAudioState` 用独立参数预测上脸9维平均姿态与4维中心化慢状态。均值使用池化音频、冻结全局情感、身份码、独立参考；状态使用逐帧音频的独立 TCN。GT只用于训练目标，不作为推理输入。状态尺度只由 TRAIN 拟合。

确定性阶段通过验收后，冻结整个状态预测器，只训练 `DCProtectedTemporalFlow` 与独立局部条件适配层。残差目标固定为 `DC((GT_upper - deterministic_upper)/TRAIN_scale)`。残差仅去除片内常量项，每个生成步保持零均值，保留低频时变动作；不再使用删除整个慢子空间的 spline-Q 投影。

这保证随机分支不能改写上脸片均值，并阻止它反向训练状态预测器；不保证随机残差无法抵消某个随时间变化的状态，因而最终仍必须进行音频/静态/反序比较。系数没有硬幅值裁剪，domain penalty 只是软约束；越界率、速度、视觉自然度均需检查。

## 数据与保留模块

- 固定 MEAD 四类情感协议：4098 train query、454566有效训练帧、446 validation query，22/3训练/验证身份。完整原生25fps序列与真实mask；这是所声明四类训练划分的全部，不是原始MEAD全部八类。
- 来源：`/root/autodl-tmp/kinetalk_calibrated_20260920/run12/protected/audio/final.pt`。身份、motion emotion→audio emotion教导及全局情感已在前一轮训练。本轮继承并冻结它们，不再声称从零重训所有模块。
- 原已修复口型和其余43个非upper通道保持同seed Stage4输出逐值一致。此保护只证明没有新增回归，不能替代口型同步、身份或情感的独立感知评估。
- 数据：`/root/autodl-tmp/kinetalk_repair_20260920/prepared`。统计量仅来自训练；sealed test不读取。

## 固定训练顺序与验收

1. audio状态分支100轮、独立static分支100轮，batch16，seed47。分别优化均值与状态，互不共享梯度。static保留音频预测均值，但无逐帧状态。
2. 固定最终epoch，在全446验证上测试。要求均值/MBE不差于Stage4、口型不变、内部情感读数下降不超过3个百分点，TRAIN状态拟合R²和相关为正，真实状态MSE优于同模型static、reverse及独立训练static（句子簇配对95%区间上界<0）。内部情感评价器并非独立认证。
3. 只有上述验收通过才训练audio residual40轮、static residual40轮，两臂共享同一个冻结且已验收的确定性预测器。状态验收失败会保存结果并自动停止残差，不改门槛强行继续。
4. 全446验证×3采样种子42/123/2026，输出full/base/deterministic/static/reverse。真实audio的centered ES和variogram均须优于独立static、同模型static及reverse，同时检查口型、非upper43逐值相同、上脸均值不变、MBE不差于Stage4、内部情感保护。MBE相对确定性输出作为诊断，随机多样性和点误差有权衡，不要求随机输出优于其确定性均值。
5. 即使数值验收通过，独立情感/AV和视频自然度仍待审查，不自动替换默认模型。

所有状态/残差曲线以原生帧无损保存到持久磁盘，带模型、代码、数据、mask及随机种子来源校验。自动恢复必须核对协议、最终checkpoint和曲线哈希；不能仅凭 status=complete 跳过检查。

## FaceDiffuser 对照

适配固定官方commit的FaceDiffBeat GRU/x0与cosine1000 DDPM，512维、两层、Adam1e-4，整脸随机初始化，不复制KineTalk口型。使用相同manifest/mask/缓存输入和可推理冻结参考/全局条件。正式计划100轮、全446验证×3种子；1000步完整采样的计算量另行测量。

这是 `FaceDiffuser-ARKit (cached matched audio + frozen conditions)`，不是官方BEAT结果复现。官方微调HuBERT且拼接为1536维；本轮提供与KineTalk相同的可用逐帧信息：原始content768与TRAIN归一化的audio_features1540（HuBERT768、emotion2vec768、prosody4），共2308维/25fps。HuBERT信息同时保留原始/归一化表示，均非新增GT信息。两者训练架构及既有编码器成本不相等；该试验是统一数据/计分的第一条适配基线，尚不能充当完全等训练预算的最终论文主表。差异见 `FACEDIFFUSER_ARKIT_ADAPTATION_20260921.md`。

自动输出ARKit-MBE/LBE/FDD、multimodality及动态诊断；AV与FD/WInD缺少合规评价器时保留pending，不制造数值。旧FaceDiffuser论文BEAT表和本项目MEAD开发表不能直接排名。

## 已验证

- 新模型、状态runner、残差流/runner、队列、FaceDiffuser适配和runner、套件编排共73项本地测试通过（60+6+7）；远程37项状态/队列/残差测试以及21项FaceDiffuser/套件测试通过。
- RTX4090真实state smoke约50.7秒，32训练样本1轮；residual smoke约44.0秒，16样本1轮，均exit0。smoke只检查功能，未通过正式效果验收，不能替代全数据实验。
- FaceDiffuser matched-all-audio真实GPU smoke55.3秒，4train/2val，正式512维模型、1000步DDPM×3种子，各采样batch约3.2秒，完成exit0；无效果通过宣称。
- 核对两个冗余HuBERT缓存与独立模型全部211个张量相同，删除可再生缓存，释放约721MiB；数据盘目前约1.6GiB可用。数据、唯一模型和旧实验结果保留。证据在 `artifacts/isolated_state_20260921/cache_*.json`。

## 运行记录

2026-09-21 00:31:56 CST启动；suite PID9880，KineTalk queue PID9968，初始state_audio PID10033。

持久输出 `/root/autodl-tmp/kinetalk_isolated_20260920/suite100`：

- `status.json`：整套状态；`kinetalk/status.json`：动态分支阶段。
- `kinetalk/state_audio`、`state_static`：100轮确定性分支、完整验证和native曲线。
- `kinetalk/state_acceptance.json`：第一道效果验收，失败自动跳过残差。
- `kinetalk/residual_audio`、`residual_static`：各40轮，条件性执行；`acceptance.json`保存最终配对结果。
- `facediffuser`：100轮同数据匹配音频信息适配版，随后独立执行；状态门槛失败不取消基线，执行异常则停止整套。

启动与前台会话无关，使用独立进程组。训练不会自动替换模型、不会读取sealed test，亦未创建定时通知。实际训练状态以远端status/log为准。

启动后实测前4轮完整4098clip分别8.15/10.03/5.95/7.00秒，第4轮均值目标loss0.1722、状态loss0.7875；这只是训练拟合信号。预计两条状态分支及验证约25–40分钟；如状态通过，残差两臂约1–2小时，FaceDiffuser训练与1000步评估另约30–60分钟。整套暂估2–4小时；后两者由小规模GPU测时外推，实际以日志为准。状态未过则跳过残差，总时间显著缩短。

## 2026-09-21 01:20 CST 实时复查

音频状态与独立static两臂各100轮、每轮4098条TRAIN，以及全446开发集验收均已完成，KineTalk队列耗时1162.55秒。状态验收未通过，自动跳过后续随机残差，不替换默认模型。

- 固定64条TRAIN诊断：状态相关0.98970、R²0.97722；这是训练探针，不能当作独立测试。相比此前训练也拟合不出的现象，本轮已能充分拟合训练样本。
- 全446开发：状态相关0.07766、R²−0.31700；真实音频状态MSE0.029865，static0.022677，reverse0.032593。真实音频优于反序，但劣于static；独立static配对差+0.006901，句簇95%CI[+0.005792,+0.007905]。目前支持明显训练/开发差距，不能只靠增加轮数解决，也不能据此推断音频普遍无法预测眉眼。
- 平均姿态MSE0.022710→0.018814；ARKit-MBE0.761587→0.735192，LBE0.314745保持。口部保护通过。
- 内部非独立情感读数（seed42）65.25%→47.98%，未通过保护门槛。教师对GT本身准确率有限，仍需独立评价；不能声称只有动态有问题、其他部分都已合格。
- FaceDiffuser完成100轮，但实例重启中断了第3采样种子的第12/28批。复查时原训练/队列进程均不存在，GPU空闲；原status仍写running，是过期状态。

01:19:40 CST已从epoch100的last.pt恢复FaceDiffuser评估，监督进程PID1420；继续使用原--resume路径，核对全部源码、源权重、数据索引和训练协议。无需重训100轮，评估会从seed42重跑全部446×3样本，因为中断前完整曲线尚未持久化。保持原1000步采样、不改变评分协议。按先前每批约4–6秒估计恢复评估约8–12分钟，完成以实际status为准。

状态快照及精简结果：`artifacts/isolated_state_20260921/endpoint_status_summary_20260921.json`；恢复记录`evaluation_resume_launch.json`。本次未生成视频或声称视觉成功。
