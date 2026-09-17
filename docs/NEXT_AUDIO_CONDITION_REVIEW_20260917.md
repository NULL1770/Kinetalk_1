# context12 后的音频条件通路审查

2026-09-17。只读检查本地源码、已完成的内部开发评估和已有论文核验记录；未改生产源码、读取封存 test、执行远端训练或重新渲染。本文区分代码事实、结果证据和待检验假设。正式两臂续训的参数与执行契约以 [AUDIO_PREFIX_ADAPTATION_PROTOCOL_20260917.md](AUDIO_PREFIX_ADAPTATION_PROTOCOL_20260917.md) 为准；本审查不改变已经启动的配置。

**建议：本轮只比较同一个已训 prefix receiver 下，专用 local 冻结与随 FM 联合适配。均值保持合成作为独立离线部署输出，不回灌训练。不要把原点条件、逐帧状态入口和 ridge 校准同时塞入本轮。** 这是定位声学表征与新接收器是否失配的最小对照，不是新架构或创新保证。

## 1. 当前真正进入模型的音频信息

| 路径 | 实际张量和来源 | context12 的使用方式 |
|---|---|---|
| 原生音频输入 | `audio_features [B,96,1540]`：content768 + emotion2vec blocks2/4/6逐帧LayerNorm后等权均值768 + prosody4 | 只用2315 fit有效帧的 mean/std 标准化；96帧25Hz，保留原mask |
| 韵律四维 | log-F0（unvoiced置0）、log-RMS、periodicity、voiced | 已包含在1540维中，并非缺少全部韵律 |
| 全局情感 | 原 `audio` 的 `input:1540→128`、四层 dilation1/2/4/8 TCN，masked pooling 后 `global_head:128→64`；另有级别期望 `[B,1]` | 缓存并冻结；进入flow的全局调制 |
| 内容 | 冻结B0/backbone从content得到 `h0 [B,96,128]` | 当前槽及允许的过去槽逐帧进入flow；不是只用全局音频 |
| 上脸专用局部特征 | `local_features [B,96,64]`，来自独立的 local 副本；history12/no_history 时已有input/blocks/local_head的联合训练 | context12缓存为 `prefix_local` 并全部冻结，只训练upper接收器 |
| 四维慢状态 | 原冻结 `audio` 的 `state_head:128→4`，经stride16 spline；四维是raise/down/squint/wide运动代理，不是VA | 只取有效帧均值 `[B,1,4]`，lift后与独立neutral anchor形成 `static_upper [B,9]`；其时变部分不直接送prefix flow |
| 过去动作 | GT训练、generated部署的归一化上脸残差 `[B,8,9]` | 与当前16帧同attention，干净known，只有unknown受FM监督 |

代码依据：`scripts/prepare_label_guided_audio_cache.py` 的 `FEATURE_SLICES/combine_frame_features/fit_feature_statistics`，`scripts/full_staged_data.py:load_training_inputs`，`SlowStateAffect.forward`，`train_history_upper.cache_audio`，`train_prefix_upper.load_context/window_batch`，`train_context_mechanism.whole_conditions/context_batch`。

**纠正旧阶段描述：当前1540维是原生逐帧、未做clip中心化的缓存。** 早期“中心化stride4低率音频缓存”的问题不能直接用于解释context12。模型内部的TCN和state spline是另一层处理；不能混为缓存已丢失原始时间变化。当前也不是WavLM第3–11层可学习加权融合，不能写成完整复现SubtleTalk多层音频。

全局音频在full_staged的audio阶段接受过 motion-global MSE蒸馏、语义分类、全脸FM及四状态监督；后续冻结不代表它从未被教导。相反，context12并未再做在线motion→audio蒸馏。`local`复制后仅input/blocks/local_head用于上脸，它的旧global/state头没有随新trunk共同训练，**不能再读取这个local副本的global/state当可靠预测**。原独立audio缓存仍是全局和静态状态的唯一来源。

## 2. 为什么旧 state_white 仍比新 prefix 准

旧state_white不是“同一个prefix只换噪声”。其动态来自whole96的 `CenteredUpperFlow`：目标、noise、velocity和输出均按有效帧投影为零时间均值，使用fit动态RMS（floor .005），local与upper共同训练。随后不训练地合成 `frozen_audio_static + centered_dynamics`。它因此不能任意改变整段静态均值。

context12的局部特征来自另一条history12/no_history分支；目标是 `(GT9-static_upper)/source_residual_scale`，不减GT均值，scale含绝对残差且floor .02。upper须同时学整体残差偏置与变化，context12又只训练upper，local冻结在旧no-history接收器所需的坐标中。teacher训练改善了续接，但不保证这条声学表示适合新receiver，也不保证first chunk起点正确。

同405内部dev、三seed均值，依据旧 `history12/audit.json` 的 `deployment.state_white` 与新 `context12/chunk_teacher/evaluation.json`：

| 指标 | 旧state_white眉 | 新teacher部署眉 | 旧state_white眼 | 新teacher部署眼 |
|---|---:|---:|---:|---:|
| raw MSE | .01785991 | .02254560 | .00695785 | .00987195 |
| centered MSE | .00112264 | .00192220 | .00138538 | .00201807 |
| 均值误差分量 `raw−centered` | .01673727 | .02062339 | .00557248 | .00785389 |
| 均值误差占raw | 93.71% | 91.47% | 80.09% | 79.56% |
| centered correlation | .0998 | .0704 | .1184 | .0587 |
| 动态RMS/参考 | .8289 | 1.3395 | .9511 | 1.2628 |

该分解沿现有每clip/每channel有效帧中心化及对应权重；不是简单把全数据平均脸相减。新旧raw差中，均值误差分量约占眉82.94%、眼78.29%。**大部分raw退化来自整段位置，但centered动态也确实退步，不能只修均值后宣布音频动态已好。** DC常量平移不会提高centered相关、改变动作幅度、修正事件时机或改变相邻有效帧位移。

context12生成历史的眉/眼接缝已经降到GT的1.51/1.21倍，而相关仅.070/.059。GT过去可把raw状态重置得更近；GT过去逆序、声学和位置不变时相关下降，支持接收器确实使用动作值。然而这同时揭示：模型可能平滑延续了错误起点，不能把连续性当成正确的音频响应，也不能把GT oracle当成跨模态可预测性证据。

以上是已发生的多个差异，尚不能单独证明“local冻结就是失败原因”。旧state_white还有目标投影、尺度、whole上下文与训练历史等混杂因素。

## 3. 当前最小12轮对照：固定receiver下解冻专用local

两臂都从同一context12/chunk_teacher upper和其原history12/no_history local恢复，新增固定12轮。`frozen_local`只训upper；`adapt_local`同时训练local的input/blocks/local_head。global audio、四状态静态原点、system/identity/B0与43通道基座固定。保持真实严格过去训练、生成过去部署，不添加loss、标签、新数据或调度。

具体接口已经由独立runner实现：

```python
native = local_features(local, audio_features, valid)  # [B,96,64]
loss, randoms = context_batch(
    upper, batch, identity, native, source_scales,
    shared_generator, arm='chunk_teacher')
```

`native`在adapt臂必须保留到local参数的计算图，不能继续读取detached `batch['prefix_local']`；后者只供step0一致性检查。local头外的模块不执行，以防旧state/global头被误用。每个batch共同 `[B,96,9]` noise与 `[B]` time，六块按有效unknown加权，只更新一次。解冻增加可训练参数是此次干预本身，不能声称参数量或算力严格公平。

这与旧direct/history联合训练有重复的基本技术，但**不是重复同一因果问题**：旧实验同时建立生成器和local，尚无现在已训练的prefix接收行为；本轮固定共同已训来源后比较是否开放local梯度。若适配优于source但未优于同预算frozen续训，不能把改善归因于声学共同适配。

均值保持输出另列部署模式：

```python
dc_upper = static_upper[:, None] + center_valid(raw_upper, valid)
```

这里只对预测整段做常量平移；不读query GT，不回灌历史，不重新训练，不称在线因果。raw、DC分别报告，并验证centered轨迹/相邻位移不变。GT mean只能放入独立诊断，不能进入主模型或同DC模型的GT-prefix端点评分。

暂不加入clipmean ridge：它只能检验或校准整段位置，不能解决逐帧动作方向/时机；与local适配同时添加会让结果无法定位。若后续单独做，必须fit内按sentence分组OOF确定是否有净收益，fit-only统计，锁定后再看dev，不能直接拿405开发目标拟合。

## 4. 一个更明确的结构候选：让flow看见自己的音频坐标原点

这是**下一项候选，不加入当前已启动两臂**。当前FM目标减去 `static_upper`，但 `window_batch/whole_conditions` 只传h0、identity code、global/intensity、native local，未显式传构造原点所用的音频均值状态。global可能包含相关信息，但不能保证接收器能从压缩后的global/local再次恢复准确的原点。GT prefix则已经携带“相对这个原点”的动作偏差，可能减轻了这一步。

推荐的最小结构信息入口是缓存并传入**现有冻结audio的整段四维均值状态**，而不是另预测几乎等于完整眉眼轨迹的9D条件：

```python
audio_result = frozen_audio(audio_features, valid)
mean_state = masked_mean(audio_result['state'], valid)  # [B,4]
# 必须就是构造 static_upper 的同一张量，hash/数值校验。
affect = {'global': audio_global, 'intensity_value': audio_intensity,
          'origin_state': mean_state}
origin_code = origin_projection(mean_state)            # Linear(4,192), bias=False
global_condition = original_global_condition + origin_code
```

`origin_projection`零初始化以保持step0原函数；只有这条投影与upper学习，是否开放local在候选两臂中保持同一政策。输入是音频预测的clip级四状态，不是标签、GT均值、VA或窗口std，不含新增逐帧目标条件。训练推理同源；只给upper，不改变mouth基座。最小对照为origin入口启用/禁用，同初始化、同source、同12轮预算；不能同时更换local来源或做ridge。

此入口主要瞄准残差坐标和绝对状态，不承诺事件时机提升。常量均值状态与已有global可能冗余；必须证明首段raw状态、generated rollout和跨clip配对指标有净收益。若只有raw均值改善而时序不变，就把结论限定在原点条件化。

另一个可选入口是冻结原audio的 `state(t)-mean_state [B,96,4]` 投影到local。但此前repair direct/soft已经比较过时变4D state且未见独立收益，所以不能称其尚未尝试或必然有效；更不能取已经失配的local副本state头。本轮不优先重加该分支。

## 5. 如何证明跨模态学习，而非只让动作变多

本轮不需要新增监督目标来做验证。至少保留以下相同权重/噪声的对照与边界：

1. **local单独的static/reverse干预**：只修改 `[B,96,64]` 时序，保持h0、global、音频静态原点、identity与noise不变。与旧“同时修改h0/local”的static/reverse分开命名。full比reverse好只说明顺序相关；full还必须比local-static有净收益，才支持时间变化条件有用。
2. **两臂与source三方比较**：本轮adapt vs 本轮frozen隔离local更新许可；两者各自vs source说明新增12轮整体变化。比较raw与centered、首段/各段均值、动作幅度、接缝和块内位移、越界、三seed分布评分，不能只挑一个相关提升。
3. **训练接收与开发预测分开**：128固定fit的GT/逆序GT是接收诊断；405dev的generated/full才是本轮部署证据。fit改善、dev无益提示过拟合或信息不足，不能靠GT过去掩盖。
4. **相同状态条件响应只能作诊断**：可以在同一个noisy motion/flow time下比较full与local-static的velocity变化、local梯度和条件使用，但响应更强不自动是信息更准；必须结合独立噪声完整12步生成质量。
5. **若需更强可学性检验，单列fit内跨句任务**：用2315fit的sentence分组OOF，在每fold只用其他句子的fit数据拟合小型线性/非线性读出，分别预测clip级四状态/均值及低频中心动态。与静态、仅global和metadata保持的音频时序破坏对照比较；不把只在fit训练上的高相关当可学证明，也不把弱线性读出当非线性上界。此诊断不得同时用来反复调405dev门槛。

不要把motion输入的GT历史成功与audio可预测成功混为一谈。已有全局情感分类也不能替代逐帧动态评价；同一句话可以对应多条合理表情，单GT误差须与随机分布、条件响应和感知证据共同解释。

## 6. 与现有论文的关系

依据已核 [LITERATURE_DYNAMICS_RECHECK_20260917.md](LITERATURE_DYNAMICS_RECHECK_20260917.md)：SubtleTalk组合WavLM多层+prosody、视觉VA/音频VA、眉眼/头窗口活动条件、过去motion、确定性prior和随机residual，并有输出动态约束。MEDTalk的逐帧强度来自选定rig通道，辅以情感指导与文本。公开材料不能支持把它们的所有自动评分当作与本任务同条件。

本轮既不复制VA+窗口活动控制体系，也不宣称“不像某论文”就足够创新。最直接的差别是：基于独立neutral身份原点和冻结全局情感，隔离专用声学条件与已学prefix receiver的共同适配；未来的origin-state入口只传已有音频预测坐标信息。能否形成论文贡献需要干净消融、数据协议、外部评估和优于合理基线的证据，本次12轮不能保证。

## 7. 对已启动runner的只读代码复核

审查 `scripts/train_audio_prefix_adaptation.py` 与其协议时未发现阻断正式运行的问题：

- 两臂重新设置seed89，从同context upper和同原local复制；独立generator统一batch/noise/time，更新预算一致。
- `local_features`与原forward的local计算逐式相同，只执行input/blocks/local_head；adapt计算图保持，protected头与归一化冻结。
- frozen system/audio/original_local 的hash与梯度检查、adapt指定模块首步非零梯度检查、local实际更新检查齐全。
- local_static/local_reverse只使用 `time_intervention` 的local返回值，仍传原h0，和原双路径干预分离。
- `compose_dc`只接收baseline、预测upper、部署static_upper与valid；GT评分与合成分离，不回灌生成历史；独立DC evaluator在过滤oracle后才检查/合成预测。
- source replay固定前32dev、seed42，不做拟合或挑选；父任务报告smoke结果max_abs=0、73项测试通过。这里只记录已提供运行证据，不声称本审查重新远端执行。

非阻断的来源断言建议已反馈：显式核对complete、checkpoint、provenance的recipe SHA一致。父任务已补充只读核验通过。本审查不更改运行中的源码。最终效果须等待完整两臂12轮、全量评分与文件绑定核验，本文不预报训练结论。
