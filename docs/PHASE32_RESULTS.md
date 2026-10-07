# Phase32 中性B0＋772D匹配迁移结果

固定三seed47/48/49、每臂2轮1568update、全部1367 validation、draw42/123/2026。全部新输出已SHA验证；无sealed/default推广/Git上传。

| 指标（clip_all三seed/三draw平均） | legacy1540 | affect772 |
|---|---:|---:|
| MBE | 0.886137 | 0.915546 |
| LBE | 0.432223 | 0.439017 |
| 生成F1 原128 | 0.200109 | 0.173692 |
| 生成F1 原64 | 0.248267 | 0.236896 |
| Lip mean mm | 3.711738 | 3.929126 |
| Lip max mm | 6.913026 | 7.403034 |
| Expression mean mm | 0.781409 | 0.794924 |
| Expression max mm | 2.722860 | 2.792208 |
| 绝对mesh FDD mm² | 151.429368 | 153.170995 |
| 嘴部位移MSE | 0.002588 | 0.002989 |
| jaw centered相关 | 0.297508 | 0.259364 |
| jaw q90-q10 | 0.196554 | 0.212473 |

raw两原F1与MBE：
- legacy: F1 0.143750/0.125162，MBE 0.902422。
- affect772: F1 0.199904/0.186450，MBE 0.935955。

三seed pilot门槛全部通过：False。各项gate和配对speakerCI见本地report/readout；只有3位validation speaker，不能宣称SOTA。

输入隔离/冻结/实际sample-noise stream已验证。该结果只比较完整中性坐标上的输入迁移；不能证明teacher/u_a纯情感，也不能将native Phase26旧成绩算作新772性能。

## 结果解释与下一步

772音频情感头的validation准确率三seed约90.08%，legacy约87.15%；这是分类头准确率，不是生成F1。seed47生成动作由自身motion teacher评判的准确率约89.9%，独立生成probe却很低；因此应检查生成动作的物理/统计分布与条件接收，不能通过继续加强同一teacher分类loss来当成改好了。

当前嘴幅度并非普遍过小：全部情感jaw范围B0 .11709、GT .17528、legacy最终 .19655、772最终 .21247。中性单独：GT .11188，legacy .13831，772 .14802；772为GT的132.3%。中性jaw相关 .27163→.25881，位移误差 .003008→.003466。情感残差补上幅度后又超出，并继续扰乱时序；不统一放大嘴，也不恢复硬嘴mask。

眉部95.44%的MSE仍来自整段mean姿态偏差。独立float64全量核查的四个近恒定cheek/nose通道，GT时间std约3.6e-7–1.2e-6，772 raw约.0033–.0038，clip后仍约.0017–.0021。这是明显的生成精度/分布不匹配；它可能影响probe，不能据此单独宣称解释了全部F1低。完整报告本地near_constant_precision.json，SHA0a431ef3e14fb53e984d1ba6f2fd2884e033fc3dce9378da930688abd155b5c2。

1. 先保持772输入和中性B0，做冻结TRAIN通道统计/连续global目标审计。检查中性坐标的精度问题，以及global目标是否主要编码表达而混入B0发音误差；两轮warm截列的适配不足仍是混杂因素，不能直接宣布泄漏已证明。
2. 历史native中通道坐标标准化曾改善生成F1，但不能搬其权重/统计来修中性模型。若当前匹配TRAIN统计和冻结定位支持，提出单变量的中性772坐标修正对照，全部原loss/支持/条件保持；先审批具体实验，再训练。
3. 用新的772模型重验固定global/identity/content/noise的u_a干预，再决定D2：只选择一种明确表达监督/梯度职责修正，替换有污染风险的学习通路。D2/D3/D4未在本轮实现；不扫loss/seed/epochs来掩盖D1失败。

本轮419个新文件全部本地SHA验证；原评分/模型/GT/probe定义没有改。固定neutral/happy四列视频均已完成，显示GT/B0/legacy/772，seed47/draw42/预登记M025/utterance005，无幅度校准或时间平移。fixed_video_state.json为complete；原blend未改变、系数重放maxerror0、无填帧；Happy103帧，Neutral101帧。输出为`final_experiment/evaluation/rendered/phase32_neutral772_{happy,neutral}_s47_20261007/comparison.mp4`。Happy SHA e56fa4a26725d48680f70de038344d2eeb9de37b86bb92b32f4963f76635dba5；Neutral SHA b65d129c5b823c565fb00b623dcc20d936b275f26840c8ab503f53b47390d3ed。初次报告UTF8读取失败已保留；修复报告读取后复用neutral、完成happy，不重跑已验证视频。

恢复入口：CURRENT_OPTIMIZATION_STATE.md；完整协议：PHASE32_NEUTRAL772_MIGRATION.md；阶段数据/结构：CURRENT_MODEL_AND_TRAINING.md。
