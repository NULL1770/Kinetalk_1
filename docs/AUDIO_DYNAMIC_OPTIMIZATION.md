# Audio动态瓶颈：证据与下一步

本记录接续 `NEUTRAL_AFFECT_PILOT_RESULTS.md`。现有teacher能重建动态，audio尚未稳定胜过静态条件；不重做已可用的B0或身份模块。

**B项已执行**：长上下文、83D声学／HuBERT／emotion2vec matched probes及锁定26条新句子审计已完成，见 [实测报告](AUDIO_DYNAMIC_CONTEXT_FEATURE_RESULTS.md)。更强的clip分类没有解决动态，尚未触发C项renderer适配；下一步先测教师低频动态目标的跨句可预测性。

## SubtleTalk的真实区别

原论文 https://arxiv.org/html/2608.06408v1 ，§3.2–3.4：

- 单独的VADP使用冻结emotion2vec＋时间卷积预测逐帧二维VA；不是预测逐帧动作类别/AU/ARKit标签。
- 残差生成器还直接接冻结WavLM的内容/中间层、多尺度声学特征、F0/log-energy。VA不是其唯一音频动态通路。
- 动作监督来自TEASER拟合FLAME，VA来自视觉情感教师，5个区域强度为动作窗口统计。DMP也不是本项目neutral-only口型B0。
- 总数据73.83小时（训练59.76小时）。有audio-only用户实验，但主文未详述VADP损失及主量化表oracle/predicted条件；官方代码目前仅README，不能补猜。

这些差异不说明VA不可替代，也不证明音频能确定每个真实眨眼时刻。

## 本实现的可确认缺口

1. motion teacher只用全动作残差/速度重建及clip标签训练。8维低率不保证它只携带音频可预测情感；GT特有眨眼、发音误差等仍能进入。
2. audio分支从83维缓存声学特征和随机TCN学习。frame hidden局部感受野15帧约0.6秒，后续池化/插值和全句去均值另扩大依赖范围；与预训练语音情感模型不同。DiT本来就接B0的content隐藏特征，并非没有直接内容条件；缺的是专门的丰富声学/韵律条件，不能把它说成所有audio都被截断。
3. audio阶段仅global/controls坐标蒸馏＋分类，未用audio条件生成动作的误差训练audio；renderer只在teacher条件上训练过。教师隐空间L2与最终动作质量未必一致。
4. global与local是并行输出，local不显式条件global；renderer同时读两者。global错误也影响整体动作，不能把全部失败都归给local。

teacher夹带不可预测信息、表示度量错配、音频特征不足是待区分的原因假设；已证的是teacher/audio的保留集落差，而非任何一个原因已确定。

## 按顺序做单变量实验

A. **先改变监督去向**：从同一run03 checkpoint各续训800步。对照沿用latent蒸馏；试验冻结renderer但允许梯度穿过，以audio条件计算已有flow＋动作差分，替换local逐点隐变量MSE；保留global蒸馏和clip分类。无新动作头、VA或region划分。固定非audio权重、数据、minibatch与尺度。用从纯噪声解码的full/mean/reverse判断，不用混入GT的训练误差宣布成功。

B. **再改变音频信息**：若A不足，冻结emotion2vec提取原时钟对齐的帧级特征，替换audio输入；小TCN输出原低率场，不改B0输入。增加约2–4秒上下文，并让local读取global条件，各作为独立消融。冻结预训练提取器，减少小样本拟合压力。是否需要给renderer额外直接音频特征，后测，不能同时打开高维旁路再声称低率情感场贡献。

C. **最后消除teacher-only训练偏差**：B0/identity/teacher及场投影继续冻结。audio与小学习率renderer交替更新，renderer批次使用完整teacher条件或完整audio条件；复用现有重建目标。检查renderer没有通过忽略local获得均值误差收益。

每步要求：audio full稳定优于真实mean和reverse、眉眼动态相关和幅度合理、global与jaw不退化。随机微动作评价分布与自然度；可预测表达评价时序，不能以“随机性”解释全部低相关，也不要求音频逐帧复现每次随机眨眼。当前28条已作开发诊断，正式结论需要新的锁定测试集。

## A项已实测：有局部改善，未通过

从run03同一audio checkpoint、224条训练各续训800步，seed43，3噪声解码，所有非audio权重冻结。`run04_latent_cont`继续原蒸馏；`run05_task_cont`用audio条件flow/差分替换controls逐点蒸馏。两者保留global对齐＋clip分类，使用checkpoint原尺度、相同抽样与新AdamW。

| 保留集指标 | 继续latent蒸馏 | 动作任务监督 |
| --- | ---: | ---: |
| 完整audio MSE↓ | 0.010811 | 0.010909 |
| 自身静态mean MSE↓ | 0.010884 | 0.010628 |
| 眼部时序相关↑ | 0.07395 | 0.10267 |
| 眉部时序相关↑ | 0.07275 | 0.06569 |
| 眼部动态幅度/GT | 0.5162 | 0.5573 |
| jaw相关↑ | 0.38645 | 0.40994 |
| 全局类别准确率 | 0.6786 | 0.6786 |

动作监督组12/28片段优于自身静态条件，平均逐片段改善-0.000259，描述性区间[-0.000791,0.000200]。B0 jaw相关0.46428，仍高于两组。结果支持“音频分支可从动作任务获得反馈”，但没有解决总体动态，不能断言根因已确认或单靠改变loss可成功。

下一优先项为B：升级冻结预训练音频特征，之后独立测试global条件与更长上下文；保留本轮A作为比较臂。若仍有教师到audio条件偏差，再进行C的交替训练。暂不引入逐帧VA/q伪标签，不重新开启高维自由动态向量。实验脚本 `scripts/train_neutral_affect_task_ablation.py`，梯度契约两项测试通过；原模型及原报告不覆盖。
