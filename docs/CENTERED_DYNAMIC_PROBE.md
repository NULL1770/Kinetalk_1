# 去时间均值动态目标：固定对照

## 预先固定的实验（2026-09-16，结果产生前）

问题：原12步生成损失的下降主要来自平均表情修正，中心化时序误差反而增加。检验仅改变损失能否纠正这一竞争，不保证成功。

对照为 `teacher_schedule_v1/rollout_constant_teacher`。新组 `centered_rollout_constant_teacher`，其他条件完全相同：19人2315条fit、3人405条内部dev，冻结B0/identity/global/renderer/audio head及rank8基；只训练原64×8投影512参数。持续motion teacher概率0.5，推理仍为audio；seed46、18epoch、2610步、batch16、Adam学习率0.0002、相同batch/noise/time/teacher draws。

唯一变化：12步原始生成误差 `e = prediction - target`，按每条片段/每个通道的实际观测帧取时间均值，使用单一 `MSE((e - observed_mean_time(e)) / residual_scale)`。均值不detach；不新增均值锚定loss，不修改生成输出，不增加网络或区域生成头。这保证损失不奖励恒定偏置纠正，**不保证非线性模型的输出均值不漂移**，也不等于证明身份与情感语义解耦。

固定epoch18为主结果、epoch2仅辅助，不挑best。相同3个生成noise、12步、full/zero/reverse/oracle。核验来源hash、冻结参数和全部18epoch随机数摘要一致。原280/new439及封存test512/15目标不读；该405仅是新动态模块留人开发，旧B0/global可能见过这些人，不能称整套模型未见身份测试。

评估原始输出：中心化动态R²/相关/幅度、逐人眉/眼、时间反转对照、速度误差；同时报告原动作MSE、均值误差、neutral口型与上脸保护，以及冻结motion teacher情感读出（非独立感知评估）。沿用已有保护门槛，不因结果调整。比退化对照好不充分，必须比自身zero有用且不伤保护指标。单seed成功也不能直接替换默认或保证发表。

## 实际执行与预算调整

SSH服务重启后确认进程已停止，last.pt保存完整epoch10/1450步；epoch11仅有1500步日志、没有保存。用户明确后续小对照8epoch足够，故**不续训到18轮**。原18epoch终点未完成：改用现存epoch8报告做同预算分析，epoch10最后完整权重作补充只读重评。这是用户中途调整，不称预注册8epoch，也未按best选择。

两组epoch1–10的batch/noise/time/teacher draws、实际teacher比例完全一致，冻结模型/head及数据来源核验通过。8epoch报告保存每句SSE等充分统计，先跨3noise平均误差、再配对5000次句簇bootstrap。8epoch没有保存权重/全曲线，不能补造其reverse/oracle或逐人曲线。

## 同预算8epoch结果

405内部dev中的331非中性片段、41句。下表为audio full相对自身zero-local的动态ΔR²，越大越好；不是总准确率提升百分比。

|区域|原始rollout损失|去均值rollout损失|新组相对zero的95%区间|
|---|---:|---:|---|
|上脸|−.02437|+.01595|[+.00199,+.02983]|
|眉毛|−.03740|−.00515|[−.02306,+.00970]|
|眼部|−.00735|+.04351|[+.02344,+.06498]|
|嘴部|+.10043|+.09903|动态正收益，较旧组差值区间跨0|

新组比旧组上脸+.04032，CI[+.02948,+.05181]。neutral嘴部原动作MSE相对zero约+.89%。**去均值目标有效缓解退化，眼部有用；眉毛仍未通过，不能说完整动态问题解决。** 单训练seed、仅3个留人身份；句簇CI不代表身份总体。

## epoch10补充：真实权重完整检查

只加载last完整权重，无新优化步，重建full/zero/reverse/oracle×3noise。full/zero与保存的dev10指标严格匹配，zero曲线与冻结对照逐值相同。

- 上脸full-zero ΔR² **+.01556**，CI[+.00141,+.02931]；full-reverse **+.08003**。眼部+.04409；眉毛−.00628，CI[−.02583,+.00902]。
- 逐人上脸均小正，但M024/M030眉毛分别 **−.10098/−.03430**，M023眉毛+.03694；不能让上脸合并值盖过眉毛退化。
- neutral嘴部原动作MSE **+.91%**，单侧90%上界 **1.59%**，在原3%线内；非neutral上脸速度误差 **+4.24%**，在原5%线内但仍恶化；嘴部相关保护通过。
- 非neutral上脸raw MSE **.0265463→.0263936**，时序误差 **.00100574→.00099324**，均值误差 **.0255405→.0254004**，三项均下降。眼部raw MSE点估计却+0.186%（区间跨0），不称所有通道无退化。
- 冻结训练motion teacher类别读出full **95.47%**、zero **95.06%**，只作诊断，非独立情感/感知证据。

目前不仅是“幅度小”：上脸中心化残差RMS比从zero **.544**变full **.608**，但相关仅 **.040→.110**；motion oracle约 **.476**。眉毛幅度也变大（.582→.648）却更不准确，因此统一放大不是解法。九例固定系数图仍可见browInnerUp漏掉明显变化；图不是脸视频，尚无独立感知认证。

这些结果支持继续用单一去均值动态目标做短对照，但不替换默认、不保证发表。下一步应先在同一数据上定位眉毛的audio预测误差与renderer传递损失，再做最多8epoch的单因素改动；暂不增加维度、文本、区域输出头或冗余loss。

## 代码、证据与边界

训练：`train_projection_centered_rollout_probe.py`。8epoch同预算统计：`audit_centered_equal_budget_reports.py`。实际中断权重重评：`evaluate_interrupted_centered_probe.py`。原18轮成对审计入口未执行，不能误读为已完成18轮。

完整259项测试通过（77条Transformer警告）。新增测试覆盖真实12步梯度、observed中心化、来源/随机数一致性、句簇充分统计与常偏置退化。独立review发现既有`same_arm_checks_all_pass`不含非neutral均值保护，报告明确不把它当完整可用认证，并另列所有群体raw/mean误差。

本地证据：`artifacts/teacher_schedule_v1/centered_equal_budget_reports_audit.json`及`centered_epoch010_audit/report.json`；小权重、来源、图已保存，完整曲线远端保留。默认权重、数据、封存test均未动。论文数值与协议见 [原文指标比较](DYNAMIC_PAPER_METRICS_COMPARISON.md)，MEDTalk EIE/FRD与SubtleTalk FDD均不能和当前ARKit52中心化R²直接相减。

![固定九例系数曲线](../artifacts/teacher_schedule_v1/centered_epoch010_audit/montage.png)
