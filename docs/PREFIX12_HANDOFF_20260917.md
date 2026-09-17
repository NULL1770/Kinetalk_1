# 显式动作前缀正式训练交付

更新：训练已于19:17:20完成，exit0，共1980.26秒。完整审计通过，但正式效果未成功：眉时序相关0.0344、幅度2.00倍、接缝8.91倍；眼相关0.0596、幅度1.57倍、接缝5.94倍。口等43通道保持。结果见[正式报告](../artifacts/prefix_context_20260917/prefix12/RESULTS.md)，未继续训练或替换默认。

2026-09-17 18:44:17（Asia/Shanghai）已启动后台实验。输出：`/root/kinetalk_full_staged_20260917/prefix12`；训练 PID13889，监督进程 PID13888。两个进程均独立于 SSH 会话运行，关闭对话不影响训练。

18:46核验首轮完成145次更新，实测77.70秒/轮；两臂训练加完整评价预估共35–40分钟，约19:20–19:25完成。首轮正常不代表最终效果已通过，实际完成以process/status与两个complete文件为准。root磁盘尚余3.4GB。

本轮解决的问题是原历史压缩码没有学好跨块续接。新模块把严格过去的8帧动作作为干净已知token，与当前16帧待生成动作一起进入attention，并在每一步Euler中固定已知前缀；只在当前有效帧计算标准flow matching。它仍需用正式结果验证连续性，不能由结构直接保证效果。

## 已完成

- 模型与pilot相关40项、本轮正式训练/评价40项本地检查通过。正式训练/评价新增40项也在远端通过。
- 真实数据小跑：两臂各2次optimizer更新、32条内部开发片段、三seed评估全部完成，耗时35.05秒。初始权重和随机流匹配，43非上脸通道及无效帧逐位保持基座输出。
- 固定8条已暴露fit的接收pilot已完成，累计30小集轮/180更新，真实前缀接收诊断在指标对象更正、一次15轮延长后通过。三seed生成前缀接缝RMS相对无前缀下降50.05%（眉）/45.53%（眼），但眼raw误差增加19.17%，眉越界26.17%。这是训练重建证据，不是泛化成功。
- 小集页面：`artifacts/prefix_context_20260917/index.html`，包含全部8片九通道原始曲线、三seed均值和来源hash。未把pilot其余43零占位当完整人脸展示。

## 后台固定计划

两臂`no_prefix`/`scheduled_prefix`各12轮、每轮2315条fit、batch16，预计1740次更新/臂；从原始`history12/no_history`相同模型重新开始，不加载pilot权重。seed79、12步Euler，末5轮teacher概率为0，完全使用自己的生成历史。

全局情感、身份、B0及局部声学网络保持已训练的冻结权重，全局情感仍是条件。本轮不另做motion→audio在线蒸馏。输出只改五个眉和四个squint/wide通道；包括嘴部在内的其余43通道使用既有完整基座。系数保护不能证明新上脸表达的整体情感/身份感知已合格。

每臂epoch12后自动评价完整405条内部开发片段，固定seed42/123/2026，单列GT前缀oracle，保存`evaluation.json`、`curves.pt`、`final.pt`及`complete.json`。每轮保存可核验的optimizer/RNG状态，但此入口当前没有自动续训CLI；失败不能直接在原目录重跑覆盖。

## 回来后先检查

1. `prefix12_process.json`必须是complete且exit_code0，`prefix12/status.json`必须两臂完成；只看进程退出不够。
2. `prefix12/matched_audit.json`必须equal，核对两臂complete中的权重/曲线hash和source/代码绑定。
3. 比较全部三seed部署结果：眉眼raw/centered、相关、幅度、越界、接缝/块内速度、多seed分布和各chunk偏差；独立列出oracle，不混入主表或主视频。
4. 与既定state_aligned/state_white/history12候选在同metadata下比较，再按已固定样例导出正式完整脸视频。小集视频/系数不能代替正式评价。
5. 当前口等43通道保护不等于独立唇音同步、身份自然度评价，motion teacher情感读出也不是独立指标。默认模型尚未替换。

训练日志：`/root/kinetalk_full_staged_20260917/prefix12.log`。协议：`docs/PREFIX_FORMAL_PROTOCOL_20260917.md`。本轮不读取封存test、不改2315/405成员、不临时挑epoch或扩大正式预算。
