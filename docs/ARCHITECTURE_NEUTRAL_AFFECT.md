# 中性身份基线与时序情感场（v10）

> 2026-09-17 状态：本文前部描述v10来源架构。最新独立direct实验已使用1540D声学TCN→64D local绕过ridge8，开放完整renderer；本轮又完成输出位移/std的audio/zero各8epoch，仍未通过、不替换默认。最新事实及失败边界以`BROW_DYNAMICS_RESEARCH_AND_ACTION_20260917.md`和`OUTPUT_MOTION_DYNAMICS_RESULTS_20260917.md`为准，不能继续将这些新实验描述为只训练512接口。

本次真实试验入口：`scripts/train_neutral_affect_pilot.py`，配置 `configs/neutral_affect_pilot.yaml`，模型 `models/neutral_affect.py`。旧 v9 的 train.py/semantic.py 仍含 VA，不能作为新方案验证入口。

## 定义与推理

- B0：冻结的音频内容→口型先验；原检查点严格加载，不训练或覆盖。
- S_id：同人多句 neutral 的 `(motion-B0)` 均值/波动统计，经一个共享编码器等权聚合。表示中性执行特征，不是几何脸型。
- P_id：S_id 预测的全52维恒定中性偏置，有限幅、零初始化。不分配硬眉眼/口型生成通道。
- E_g：整段情感表示，类别和已知等级只提供监督。
- D_k：8维动态控制点，每4帧一个。时序卷积提取，去掉时间均值，保留幅度；共享线性投影和固定时钟插值成 E_local(t)。不是每帧独立64维自由向量。
- Motion teacher：从 `motion-B0-P_id` 提取 E_g、D_k。通过情感标签和动作生成重建共同训练，不能只靠分类得到动态。
- Audio student：从目标audio预测相同 E_g、D_k；教师冻结后蒸馏。

```text
训练: motion-B0-P_id → motion teacher → E_g,D_k → residual DiT → 动作
                         ↓冻结后监督
                     audio student ← 目标audio

推理: 目标audio → B0 + audio student(E_g,D_k)
      多neutral参考 → S_id,P_id
      DiT(B0隐藏特征,S_id,E_g,E_local,noise) → ΔM
      M_hat = B0 + P_id + ΔM
```

情感场是DiT条件，不能把8/64维情感特征再直接加到52维动作。只有一个残差动作输出；没有单独的 A(t)+F(t) 双动作解码器。参考neutral本身说话，计算其B0需要参考content；可提前缓存身份。

## 阶段和最少监督

1. 核验已有B0及native时钟。小试验冻结，不重训它。
2. neutral参考分成独立两组，以另一组neutral统计重建P_id，加一个同人对比目标；冻结身份模块。
3. motion teacher与DiT联合：flow重建＋clip类别/已知等级＋低权动作差分。20%流训练在纯噪声端点，防止只依赖混入GT的中间状态。
4. 冻结teacher/DiT，audio学教师global与低率controls（仅训练集统计归一化）及clip标签。推理不读query motion或GT情感标签。

删除主方案中的VA/q条件、情感参考充当身份、GRL/多套cycle/强度范数伪标签、donor逐帧GT。neutral不足应失败，不能悄悄回退情感参考。

## 小试验解释边界

固定噪声、身份、global，比较完整local、置零local、真实有效帧均值广播、反转local；同时报告训练与heldout的MSE、动作/速度相关、幅度比、嘴部时序及teacher-audio差距。眼部和眉部在评估中单独报告，评估分组不是生成通道划分。teacher和混合条件读取真实query动作，仅是诊断重建上限。

非零变化、单元测试通过或训练loss下降不能证明动态情感成功。需在heldout中改善正确动态、mouth/global不恶化。新句子小试验不代表未见身份泛化；既有B0可能曾见这些句子。CCF-C录用不能由架构保证。

## 已执行的真实验证

run01已完成：4人、56训练/28保留片段，另16个neutral参考；200身份/1000教师+DiT/800audio steps。3个固定噪声下，teacher完整动态MSE 0.003926，静态均值0.007351；audio完整动态0.012581，静态均值0.011499，且jaw相关0.385低于B0的0.464。独立neutral检索4/4、身份基线MSE 0.000453，仅说明已登记4人的小样本可学。

因此只通过身份初步验证和motion动态表示验证，audio动态泛化及最终口型保护未通过。冻结B0不等于最终口型不变，残差仍可修改嘴部；低率字段也不保证情感与内容严格解耦。全局情感及新架构相对旧模型的非退化还需正式同协议对照。

run06–09进一步检验上下文和预训练特征，结论见 [音频优化实测](AUDIO_DYNAMIC_CONTEXT_FEATURE_RESULTS.md)。保留默认83D声学输入；768D emotion2vec仅为可复现消融，虽提升clip类别识别，动态及口型未达验收。现有构件和阶段定义不因单项分类准确率上升而改写。

## 情感方向与单标量时序的诊断（run27，未替换主模型）

训练句中按人计算每类 emotional-minus-neutral 的平均残差方向 `r_e`，排除 blink/gaze；motion 提供有符号强度 `a*(t)=<center(motion-B0-P_id),r_e>`。音频小头预测去均值标量 `a_hat(t)`，由冻结 audio global 概率调制，输出诊断动作场 `a_hat(t)·sum_e p_audio(e)r_e`。概率混合不单位化，保留中性抑制；真实类别只用于构造监督和 oracle 对照。该52维投影仅检验目标可预测性，并未直接加到生成动作或替换DiT接口。

此假设把时间变化限于情感方向，避免自由逐帧隐向量和新增loss，但没有证据说明一个强度能统领眉、眼、嘴。run27四组固定600步仍未超过保留片段的零动态基线，真实标签条件也失败；既有global在28条开发片段准确率96.4%。因此瓶颈不能只归咎于global误判，单标量方案暂不作为最终架构。后续方向需先满足眉眼动态与时序干预评估，再接入冻结主干；详见实验记录第11节。

run28的训练句内部选正则ridge，content→同一强度目标外部proxy R²=.1457，反转为−.2100，证实部分时序可预测；但真实眉眼R²=−.1001，收益主要在嘴部。GT诊断也发现上脸/嘴部强度不同步，所以本轮不把“全脸单强度”定为最终时序场，不以扩大rank或堆loss补救。Motion-derived监督仍在训练端保留；推理仍只用音频及neutral身份参考。

## 无文本共享动态基试验（run30/31；默认模型尚未替换）

新增真实emotion2vec中间层＋韵律，和已有content合并；在train句配对拟合共享rank8基U。训练目标为真实中心化motion残差乘U，音频预测相同坐标，U在后续阶段冻结。PCA和direct ridge作匹配基线。RRR是既有统计工具，不能仅凭采用它称创新；原神经motion teacher仍保留用于原模型对照。

1200训练/280开发/12人结果：upper R²=.02354且句簇CI正，眉/眼均有小收益；RMS却只有真实动态13.75%，neutral嘴部残差预测变差。未证明强动态、严格解耦或发表充分性。

run32已将标准化低率坐标经原local_projection送入单一ResidualDiT；不直接加到52维动作、不加文本/VA/区域头。共同训练audio头与renderer导致眉眼和neutral口型退化，未采用。

run33改为冻结已验证audio头、U和整个renderer，只用原flow训练512参数的共享local_projection。推理仍是audio→低率8维场→64维local条件→原单DiT；motion在训练/oracle端提供真实投影目标，无推理motion输入。B0/identity/global保持原权重。相对自身zero-local upper动态ΔR²=.01262且CI正，oracle有效，neutral mouth比原模型改善；但upper绝对R²仍负、眉毛独立收益不稳、neutral比zero-local差。因此仅保留独立实验适配器，默认模型不替换，后续先检查global对neutral误激活的条件化。详细结果见实验报告第12–14节。


## 当前候选收敛（run34/35）

完整推理：`audio → content(B0) + frozen global E_g + centered(content,middle2/4/6,prosody) → frozen linear8 → g·D → local_projection64 → frozen shared DiT`；neutral多参考给固定S_id/P_id。`g=1-p_audio(neutral)`只抑制误激活，不提供新时序标签。训练端真实`center(motion-B0-P_id)U/trainRMS`仍作motion动态教师，前段混合teacher/audio，后段纯audio，只有既有flow损失训练512参数接口。没有文本、VA、逐帧自由向量、区域输出头。

两训练seed通过既定工程准入，但upper绝对R²仍负、眉毛独立CI跨零；可进入有保护选择/断点恢复的正式epoch训练，不能宣称完整效果已解决。正式入口`scripts/train_formal_predictable_projection.py`，数据/准入/选择规则见`FORMAL_TRAINING_READINESS.md`。默认权重仍不替换；前文各阶段的小样本状态按对应run解释。

后续正式4组各18epoch均早期最好，新身份眉部失败，见`FORMAL_TRAINING_RESULTS.md`。训练内19/3身份三日程对照进一步显示：持续teacher缓解但不能修复退化；同样本单步flow改善而12步生成变坏。`TEACHER_SCHEDULE_PROBE.md`记录用同一512接口/单一损失检验实际12步生成监督的隔离实验，尚未升级默认。不能再将“teacher撤除后纯audio长训”描述为已验证最终阶段。


最新去时间均值单一rollout目标对照见`CENTERED_DYNAMIC_PROBE.md`：同预算8epoch上脸动态有正增益、眼部改善，眉毛仍退化。训练端motion教师仍保留、推理仍audio+neutral参考；不加loss或头、不替换默认。用户指定后续短对照8epoch，不能再默认追加18轮。

`SCALED_DYNAMIC_PROBE.md`的两组8epoch进一步检验单一loss的训练尺度定标，眉毛仍不及zero、嘴部相对uniform退化，未采用。`SCALED_MOTION_BASIS_PROBE.md`已完成共享rank8基量纲的训练内OOF：眉motion投影.363→.557，但音频.02065→.02004，嘴/jaw退步；基定标没有提高音频眉时序相关，未接入生成器。后续固定native目标改善音频时序映射，尚无新生成效果证据，不改变以上推理架构。

共享小型pointwise/temporal音频修正现已按三折各8epoch测试，见`TEMPORAL_AUDIO_REFINER_RESULTS.md`。约10.8万参数的零输出初始化修正提高fit能力但破坏跨句预测，未接入生成。仍使用原冻结ridge；未来需在fit内部验证受约束的预测改动，而非把网络增大或幅度增加当成功。默认推理架构保持。

随后完成的 `CONSTRAINED_AUDIO_REFINER_PROBE.md` 将修正限制为 rank8 场内 64/192 个线性参数，并固定偏离 ridge 的单一软约束。temporal 眉 ΔR² 为 +.000442（95% CI 全正、11/19 人改善），但 neutral 嘴部残差预测 MSE 增加1.4197%（上界1.6020%），保护未通过；眼部和非neutral嘴部保护通过，仍不接入。只读多 seed 分解显示现有 DiT 的眉 seed 方差约占中心化单样本误差 13.9%，而 ensemble 均值相关仅 .142；这证明有随机差异，不能单凭方差判断一对多样本是否合理。

`AUDIO_CONDITIONED_FLOW_RESULTS.md` 已完成最后层 cross-attention 输出矩阵的 audio-local/zero-local 两臂各8epoch。单flow、36,864参数、8采样seed，不改推理输入和其余结构。眉 fair ES 相对原full仅改善0.037%，R²/相邻帧variogram变差，仍不胜关闭local；汇总保护通过但逐人退化仍在，默认不替换。只读条件路径诊断用于判断后续应改音频信号还是DiT的时间读取，不能把本轮输出重组失败说成DiT一对多能力失败。
