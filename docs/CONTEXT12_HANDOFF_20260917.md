# Context12：真实前缀接收与整窗机制对照

更新：三臂已于20:44:33全部完成，exit0，共835.29秒。真实前缀训练组在generated历史部署下，眉/眼拼接位移降至参考1.51/1.21倍；时序相关仍仅0.070/0.059，整体未达成目标。完整来源/评分审计通过，未替换默认或追加训练。报告见[RESULTS.md](../artifacts/context_mechanism_20260917/context12/RESULTS.md)。

2026-09-17 20:30:35（Asia/Shanghai）已在服务器启动，训练PID20650、监督进程PID20649。输出目录：`/root/kinetalk_full_staged_20260917/context12`。20:32核验第一组`chunk_empty`完成4/12轮，每轮约20秒。初步预计20:45左右完成三组训练和自动评价，实际以完成文件与退出码为准。

关闭对话或SSH不会终止后台进程。日志为同层`context12.log`；监督状态为`context12_process.json`，逐轮状态为`context12/status.json`。

## 本轮改动

三个独立臂均从原始history12/no_history共享backbone与local出发，seed83，各12轮/1740次更新。没有加载上一轮prefix12更新后的权重。

- `chunk_empty`：过去8槽无效，分6块，每块生成16帧。
- `chunk_teacher`：训练始终使用严格过去8帧真实动作；部署时只能用自己此前生成的动作。真实历史与逆序真实历史另列oracle诊断。
- `whole`：8槽无效padding加96当前帧，共104token，一次生成完整96帧；相同16帧网格处的测量不称解码接缝。

三组共享每clip的96帧初始噪声、flow time、数据顺序和有效帧权重；标准FM，每batch一次更新。没有新损失、额外数据成员、文本标签、音频移位、动作增幅或后处理平滑。整窗对照同时改变注意力范围、后续位置编码和分块求解机制，不能单独归因于某一项。

全局情感、身份、B0和local声学条件冻结并校验hash；全局情感继续作为条件，没有新增在线motion→audio蒸馏。正式输出的43非上脸通道及无效帧复制同seed已有完整基座。权重/系数保持不能替代独立感知评价。

## 已完成验证

新增58项测试本地与远端全部通过。真实小跑三臂各2次更新、32dev三seed评价及fit诊断完成，总44.15秒；三臂初值与随机流一致，43通道和无效帧逐位保护检查通过。

仅基于clip_id、speaker_id、emotion_id分层固定128条fit诊断片段，清单在推理前保存并绑定hash。训练前保存step0诊断，包含chunk_empty同权重/同噪声的position0与position8比较。终点分别保存这128条fit的一seed接收诊断，以及405条内部开发片段的三seed完整结果。fit文件其他43维是零占位、未评分，不可用作完整人脸展示。

## 完成后核验

1. `context12_process.json`为complete、exit_code0；`status.json`为complete，三组各12轮；`matched_audit.json`为equal。
2. 每臂`complete.json`与final/curves/fit_curves文件hash一致，全部epoch的clip/noise/time流匹配。
3. 对比`fit_evaluation.json`与`step0_fit.json`的真实前缀接收，并与dev oracle分开。逆序GT使用实际反序末token计算连接，不把原时间的GT末帧冒充输入端点。
4. 正式主表仅三seed `full`，比较时序相关、动态幅度、raw/centered误差、越界、位移/块首误差与分布评分。oracle不能混入主表或主视频。
5. whole在16帧网格改善不代表自动成功，还须检查是否静态化、偏置或整体时序退步。对比旧state_aligned/state_white与本轮配对基线，保留完整失败结果。

固定12轮后不临时续训挑epoch，不自动替换默认。尚无正式终点效果结论。协议见[CONTEXT_MECHANISM_PROTOCOL_20260917.md](CONTEXT_MECHANISM_PROTOCOL_20260917.md)。
