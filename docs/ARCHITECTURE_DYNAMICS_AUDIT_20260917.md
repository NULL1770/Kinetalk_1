# 动态架构与训练链独立审计（2026-09-17）

范围：本地实际源码、既有实测文档，以及本轮主审从服务器确认的执行状态。本文不把旧 v9、默认冻结权重、rank8 适配器和最新 direct 实验当作同一个模型；既有训练源码与模型未改。审计后按主审委派另建独立输出动态实验，见第6节。行号对应本轮工作树。

**最终执行状态：direct独立48组曲线审计与新output-motion两臂各8epoch训练、72组曲线审计均已完成，均未通过效果验收，默认不替换。** 下文“正在执行/待审”描述保留审计当时的阶段；最终结论以`DIRECT_AUDIO_DYNAMICS_RESULTS_20260917.md`、`OUTPUT_MOTION_DYNAMICS_RESULTS_20260917.md`为准。下一路线见`BROW_DYNAMICS_RESEARCH_AND_ACTION_20260917.md`。

**结论：没有找到足以解释所有失败的“眉通道没接上/没有梯度/忘了加眉真值”的单点模型 bug。此前确有训练覆盖、学生条件和生成目标之间的缺口，但完整 renderer 三臂已补做且失败；最新 direct 两臂也已实际完成，必须先独立核验其结果，不能再次建议执行已经做完的步骤。** 一对多仍是合理建模方向，不过给每个随机样本同一条 GT 的 L1/L2，与学习完整条件分布不是同一件事。

## 1. 当前共有三条不同路径

| 路径 | 真实输入及监督 | 可训练范围/当前边界 |
|---|---|---|
| v10 pilot/default | motion teacher 由 `M-B0-P_id` 生成 global/低率8D；DiT flow+标签+低权差分；随后学生蒸馏 | 原始 teacher/renderer 学习来自很小 pilot；学生阶段冻结 renderer。不能把后来23k资产数当作这些模块都训练过23k |
| rank8 predictable | centered content+emotion2vec middle+prosody → ridge8 → 共享64D local；真实 motion 投影作训练/oracle | 先512投影，再末层 out-proj；**已另做完整renderer+projection、固定学生、oracle/audio/zero各8epoch**，不能再说只试了512参数 |
| direct audio（2026-09-17） | 1540D低率声学 → masked TCN → 64D local；绕过rank8 | 新encoder+完整renderer，`flow` vs `flow + 12步生成眉眼中心化L1`各8epoch1160步已完成；旧B0/identity/global冻结。独立audit正在执行，未由本文判定效果成功 |

源码定位：`scripts/train_neutral_affect_pilot.py:254,265,268,281,297`；`scripts/train_predictable_renderer.py:48,65,68,83,116`；`scripts/train_audio_conditioned_flow_probe.py:40`；`scripts/train_renderer_capacity_probe.py:51,130`；`scripts/train_direct_audio_dynamics.py:59,145,228`。

规划恢复证据：`.planning/2026-09-16-renderer-capacity/progress.md:11` 已记录容量三臂结束，oracle最终flow MSE约0.2061、audio约0.2612而眉时序退化。它否定的是本次预算/数据/目标组合有效，不是证明完整renderer理论上无用。主审本轮另已从SSH确认 direct 两臂结束；最终数值以独立重建审计报告为准。

## 2. 已排查的路径和量纲

### B0、identity与最终输出

`models/neutral_affect.py:287-317` 中目标为 `r=M-B0-P_id`；训练归一化 `r/.25`、线性插值 `x_t=(1-t)z+t·r/.25`、速度 `r/.25-z`；Euler生成最后乘回`.25`。训练/推理的方向、缩放相符，未见反号、重复缩放或残差忘加的错误。

最终是全52维 `B0+P_id+residual`，没有嘴部硬屏蔽。因此B0冻结只证明B0参数不变，**不能证明最终口型保持**。`P_id` 是恒定偏置（`models/neutral_affect.py:250`）；中心化后它严格抵消，因此rank8中心化预测收益不能证明身份对动态有帮助。`S_id` 仍进入DiT，理论上能够影响动态，但该贡献需实际参考干预证明。

### 时钟、padding与条件读取

`models/neutral_affect.py:231` 按真实末有效长度分组计算B0，避免继承的GroupNorm被批量padding改变。低率教师用真实bin计数去均值；`project_affect:255` 与 `train_direct_audio_dynamics.py:79` 都按 `(frame-1.5)/4` 固定坐标插值，未随批长重定时。direct temporal block先清padding、按帧LayerNorm，再卷积并重清padding，未发现旧时间GroupNorm污染。

DiT的local既直接加到同帧token，也进入cross-attention的K/V（`models/dit.py:131-136`），不是只被整段均值池化。DiT本身无独立frame PE，但B0的h0来自TCN+正弦位置编码+Transformer（`models/encoders.py:77-112`）。因此“完全没有时间信息”不成立；位置/声学信号能否被当前权重正确使用仍是效果问题。末层out-proj单矩阵只能重组已有注意力输出，不能据其失败推翻Q/K学习或flow的一对多能力。

现有native接口从每段取固定中央96帧，25Hz时至多3.84秒（`neutral_data.py:23,55-62`）。固定窗口不等于错位；但2720段或2315段训练应报告实际有效时长，不当作全视频训练。当前24个stride4 bin也限制了可传递的快速变化：没有必要先增加rank，先对固定GT眉事件确认96帧/stride4保留率。

### rank8与“音频不可预测”结论

rank8按原单位运动的RRR误差学习，`PredictableAudioHead`通过训练RMS定标（`train_predictable_renderer.py:48-69`）。先前基尺度实验已把眉oracle投影能力从约.363提升到.557，却没有提升audio R²；先前TCN也出现fit好、OOF差。因此再重复放大坐标、增加相同头容量、基RMS扫描，没有新因果依据。

ridge的弱R²是“这些中心化特征/基/线性估计器的条件均值预测很弱”，不是音频包含的条件分布信息上限。原DiT同时有h0和global，并非所有音频都只能过8D；新direct已进一步把较丰富声学直接送入同一DiT。

## 3. 最新 direct 实验的实际含义与缺口

代码已真正做到了训练期rich local条件和最终rollout监督：`training_loss:145` 先flow，再从同一noise运行12步`generate`，没有把目标状态塞进生成器；`generated_losses:129`按观测掩码比较真实眉眼轨迹。encoder末层零初始化保证起点等价旧zero-local（`DirectAudioEncoder:67`）；它仅让更上游encoder在第一步没有梯度，末层更新后上游就能学习，属于常规稳定初始化。

**不是“以前完全没有眉眼数值监督”。** 原flow的目标含所有观测眉眼通道，pilot还含差分；只是velocity监督发生在noised-GT状态，对只有有限步数的实际生成轨迹，优化信号与评价可能不同。新direct额外提供了最终眉眼轨迹L1，这是有意义的实现变化，但与论文是不是直接眉loss要逐篇区别。

有三个必须保留的解释边界：

1. **随机rollout逐GT L1倾向条件中位数。** 新项是 `E_(a,y,z)|center(G(a,z))-center(y)|`，z与y独立；在没有其他约束的情形，每个z的最优输出是相同的条件中位数。它可提高可预测节奏，却可能压制合理自主变化。flow项仍在，最终是否坍缩需要多seed统计；不能从公式武断断言已经坍缩，也不能称添加L1本身补上“一对多”。相同道理适用于此前rollout MSE的条件均值倾向。
2. **两臂只隔离附加轨迹项。** flow与dynamics同rich audio初始化、数据/RNG，可以比较新增L1；`full/zero/reverse`是推理干预，说明模型是否依赖条件，但zero/reverse对训练模型可能是分布外输入。缺同预算同loss的zero-local训练臂时，不能把相对旧rank8或own-zero的全部提升归因新音频信息。此前capacity的zero臂可作背景基线，但训练目标/初始local不同，不能冒充严格匹配对照。
3. **仍是低率、中心化缓存声学。** middle是emotion2vec第2/4/6层按帧LN后固定平均（`extract_predictable_audio.py:86-97`），不是学习多层WavLM融合。输入在构建bundle时已按段去均值（`prepare_predictable_motion_bundle.py:51-64`）；有损降采样和中心化的效果需要实证，不可以叫“原始声学全部保留”。

最低限度的后续决策：先看已完成direct两臂经真实checkpoint重建的眉/眼相关、R²、幅度、速度、fair ES/VS与口型。如果dynamics仅变平、单GT误差好而ES/事件统计坏，应拒绝把它当最终动态方案。若flow已经改善多seed分布而L1破坏它，则保留flow方向并检查有限步数/真实事件。任何新实验都应由这些对照确定，暂不再盲目追加长训或loss。

主审已下载的`artifacts/brow_review_20260917/quick_metrics.json`提供尚未完成独立重建的粗数：非neutral眉flow R²约−1.9258、能量比2.174；dynamics R²约−.04665、相关.1468、能量比.1664，而其own-zero R²约+.00438。它支持“flow过强而不准、逐GT L1明显减幅但full仍输zero”的当前定位，尚不能代替最终审计CI/ES/口型保护结果。

## 4. 确认并修复的审计代码问题

本轮仅按主审委派修改独立审计器，训练源码保持：

- `audit_direct_audio_dynamics.py`此前读取 `data['summary']['epoch000']`，但真实trainer只在`epoch000_evaluation.json`保存起点评价。会使已完成训练无法审完。现于`:103`读取实际独立文件，并在`:489`使用它。
- 原source-curves只校seed与单seed zero数值，没有把所有seed的source-full绑定回正确来源模型/cache。新增`load_historical_source_curves:163`，验证历史recipe、真实checkpoint各参数、head/projection、输入cache、checkpoint/curve sidecar与8seed4mode。避免拿同形状却来自别的模型的full曲线比较。
- `tests/test_direct_audio_dynamics_audit.py`覆盖真实分文件协议、改动另一seed的full、替换cache、改动来源权重。连同原direct训练测试共 **7 passed**。这只能证明被测代码行为，不代替远端真实权重重建和效果审计。

另一个部署陷阱尚未改：`NeutralAffectSystem.generate(initial_noise=None)`传入`ResidualDiT.decode`，其默认 `stochastic=False` 会从全零开始（`models/dit.py:164-180`）。旧/新正式评价都显式传`torch.randn`，所以它不解释现有八seed失败；将来正式推理必须显式规定random seed/noise，不能只调用默认generate然后声称已启用随机一对多。

## 5. 身份、口型、global与投稿判断

身份编码的是中性执行统计，不是几何脸身份；4人独立neutral检索成功不等于新身份动态保持。口型需检查最终输出而非B0权重hash。global的高准确率主要来自小样本audio标签分类或冻结训练teacher读出（`train_formal_predictable_projection.py:209-216`），不能替代独立生成情感验证。

当前还不适合以“已实现细腻眉动态且身份/口型/情绪均可靠”为主张直接投CCF-C。四类、3人共享句子的反复开发不是独立论文测试；同rig视频、眉事件真值、独立嘴同步/情绪/行为身份评价均需补齐。已有renderer显示链还曾存在隐藏对象、旧action及相机问题；新脚本已清理旧动作并驱动可见mesh（`blender_render_dynamic_rig.py:61,95`），但synthetic smoke不是模型感知成功。

有价值的完成顺序是：完成当前真实direct审计 → 固定样本GT/B0/source/direct连续视频 → 依据失败类型选择最小新对照 → 通过后才扩大训练覆盖与独立测试。不能用下一篇方法的某个loss名称，替代当前模型已经缺失或已经补做的实证定位。

## 6. 审计后新增的有限对照（未由本文宣告训练成功）

主审依据上述direct粗数要求实施`OUTPUT_MOTION_DYNAMICS_PROTOCOL.md`，已先写协议再新增独立`train_output_motion_dynamics.py`，不改前序训练源码。以原direct epoch000同初始化，flow保留，去掉centered逐帧L1，改成最终12步动作的相邻位移MSE与时间std MSE；两组权重均1。眉/眼/嘴位移三组等权、眉/眼std两组等权；scale仅fit目标计算并有固定floor。两臂audio/zero同预算训练，补上“更多renderer训练本身”这个匹配反事实。

这仍是普通工程基线而非一对多理论修复：速度逐GT项有均值化倾向，std项仅约束幅度且不保证事件时间或自然度。必须看matched-zero与ES/VS，不能以变大即为成功。新增5项有针对性测试覆盖poison mask、ddof0/常量有限梯度、相邻时序/区域等权、同初始化RNG及zero独立性；连前述direct/audit共12项通过。真实源权重/epoch0逐值核验由runner强制执行，最终训练结果需另行审计。
