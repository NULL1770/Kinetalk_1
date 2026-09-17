# 真实小规模训练结果

结论：**中性身份基线初步可学，motion 教师的低率动态条件有用；audio 动态泛化、最终口型保护尚未通过。** 不能把本次测试写成完整架构成功、严格情感解耦或 CCF 录用保证。

## 实际训练

远程 RTX 4090 / PyTorch 2.5.1，产物根目录 `/root/autodl-tmp/kinetalk_neutral_affect_pilot_20260916`。本地报告在 `artifacts/neutral_affect_pilot_20260916`。

- MEAD M003/M005/M007/M009；neutral、angry、happy、sad，非neutral L1/L3。56条训练、28条保留，4454/2393有效帧；每条最多96帧，25Hz。另有每人4条neutral登记参考。
- 训练/保留/登记参考按句子与片段隔离；保留集是已登记身份的新句子。已有B0可能在预训练见过这些数据，不是全系统未见数据或未见身份测试。CREMA-D未参与本次小训练。
- `run01`：身份200步，motion teacher+单一残差DiT 1000步，audio student 800步；batch8。B0严格加载并冻结，原权重未覆盖。没有VA/q或硬生成通道划分。
- 教师损失为flow重建＋0.1类别/等级监督＋0.1动作差分；audio为标准化global/controls蒸馏＋0.1类别/等级监督。推理audio条件不读目标动作或真实情感标签。
- `run02_control` / `run03_data224`：从同一teacher checkpoint和相同audio初值重训audio 800步，分别56/224条训练。后者新增168条；保留集文件、身份参考、输入归一化、其他模块不变。蒸馏尺度均用原56条统计；因批量计算数值差异，global/controls尺度最大绝对差为0.0000885/0.00000572，非逐位相等，已记录于 `final_integrity.json`。预算按步数相同，遍历次数不同；这是数据量诊断，非完整收敛比较。

## 结果

全部动态对照使用相同身份、global、强度、content及噪声，仅改变local；3个解码噪声seed（42/123/2026），12步Euler。静态条件广播local真实有效帧时间均值，另测置零、倒序。指标在有效观测通道上计算，未clamp。

| 保留集条件 | 动作MSE↓ | 眼部时序相关↑ | 眉部时序相关↑ | jawOpen相关↑ |
| --- | ---: | ---: | ---: | ---: |
| B0口型先验 | — | — | — | 0.4643 |
| motion教师完整动态（读取真实动作） | 0.003926 | 0.4866 | 0.3314 | 0.5843 |
| motion教师静态均值 | 0.007351 | 0.0422 | 0.0443 | 0.3691 |
| motion教师倒序动态 | 0.010412 | -0.0102 | 0.0096 | 0.2770 |
| audio完整动态，run01，56条 | 0.012581 | 0.0252 | 0.0588 | 0.3851 |
| audio静态均值，run01 | 0.011499 | 0.0391 | 0.0293 | 0.3727 |
| audio完整动态，run02，56条重跑 | 0.012727 | 0.0203 | 0.0471 | 0.3804 |
| audio完整动态，run03，224条 | 0.011208 | 0.0536 | 0.0188 | 0.4179 |
| audio静态均值，run03 | 0.011359 | 0.0410 | 0.0308 | 0.3807 |

**身份**：独立neutral query检索4/4，随机为1/4。登记参考预测的neutral基线MSE为0.000453；直接该人参考均值0.000522；训练neutral全人均值0.003063。说明这4人的中性系数差异能学习，不代表几何脸型、未见身份或统计显著性。原run01的100%参考检索是训练视图拟合，不能当独立验证；新结论来自 `identity_heldout.json`。

**教师动态**：完整local比静态均值MSE低46.6%，28/28片段均改善；平均逐片段改善0.003556，描述性clip bootstrap区间[0.002665,0.004546]。训练MSE也从0.10104降到0.002376。这证明低率8维时序条件能承载、驱动动作变化，但教师读取真实motion，因此只是重建诊断上限。它可能包含发音残差、眨眼和跟踪误差，不能直接叫纯情感动态。

**audio仍弱**：run01控制点保留集MSE为0.051918，甚至差于全零预测0.038637；训练集则0.015190，说明有拟合但泛化不足。扩大至224条后为0.038480，仅比零预测好约0.4%。最终动作MSE比自身静态均值仅低约1.3%，13/28片段改善，逐片段平均改善0.000129，描述性区间[-0.000244,0.000532]跨零。眉部相关未改善；嘴部速度相关0.0665低于B0的0.1153。控制幅度也从约0.140收缩到0.075，误差下降可能部分来自趋向均值；不能解释为动态问题已解决。表格为有效帧加权MSE，区间基于片段等权改善，统计口径不同。

**瓶颈定位**：run01用audio global搭配教师local，眼部相关恢复到0.4838；教师global搭配audio local仅0.0245。只换教师强度几乎无效，故本轮没有开展“换强度头即修好”的训练。global替换会改善总误差，local预测则直接限制正确时序。教师/混合条件均不计入audio部署成绩。

所有区间仅描述本次小样：seed先在同一clip内平均，不能当新增样本；只有4人且有共享句子，另存speaker-cluster敏感性区间，不作总体显著性结论。保留集在初次诊断后又用于扩数据复测，属于开发集，正式结果需要新的锁定测试集。

## 当前决策

保留neutral身份和低率动态表示；**暂停全量训练，暂不替换原可用模型**。冻结B0不保证最终口型，残差仍可改嘴部。下一轮应针对audio可预测的表达变化构造教师目标，并检验内容/随机眨眼混入；在单一残差输出内限制对已有发音时序的破坏，优先替换现有差分约束的目标，避免叠加大量loss。旧全局情感是否不退化尚未做同协议对照，不能承诺。

## 复现与审计

代码入口：`scripts/train_neutral_affect_pilot.py`、`scripts/evaluate_neutral_affect_pilot.py`、`scripts/evaluate_neutral_identity_pilot.py`。扩样本与audio控制试验分别为 `scripts/expand_neutral_affect_student_data.py`、`scripts/train_neutral_affect_audio_ablation.py`。`train.py`仍是历史v9，不能用于本方案。

```bash
python -m scripts.train_neutral_affect_pilot --config configs/neutral_affect_pilot.yaml --data DATA --output FRESH_RUN --stage1 outputs_protocol_v1/stage1_neutral.pt
python -m scripts.evaluate_neutral_affect_pilot --config configs/neutral_affect_pilot.yaml --data DATA --checkpoint FRESH_RUN/audio.pt --output FRESH_EVAL --seeds 42 123 2026 --steps 12
python -m scripts.evaluate_neutral_identity_pilot --config configs/neutral_affect_pilot.yaml --data DATA --checkpoint FRESH_RUN/audio.pt --output identity_heldout.json
```

每次训练保存checkpoint、数据/配置/源码hash与日志；补评估修正了缺失通道减baseline后再次mask、真实均值对照与身份检索命名。原run01报告保留，原先`mean`实际为zero，最终引用 `robust/report.json`。固定展示索引0与首个非neutral索引1，未按效果选图。

本地50项测试通过；其中专门核验动态梯度、B0冻结、padding、共享蒸馏坐标、均值/倒序干预与固定噪声。单元测试只证明实现契约，真实效能结论以上述训练与保留集报告为准。

后续针对audio映射问题的同初始权重训练对照见 `AUDIO_DYNAMIC_OPTIMIZATION.md`（run04/05各续训800步）。用动作任务监督替换local蒸馏只改善部分指标，完整方案仍未通过；新增两项任务梯度测试通过。
