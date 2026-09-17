# 显式动作前缀：完整fit的12轮配对协议

日期：2026-09-17。本文在正式两臂训练前固定实施与评价口径；接收器pilot已经读过，其探索过程不能包装成原始预注册成功。本次是完整fit的独立实验，默认模型、旧结果及现有数据锁均不替换。

## 1. 为什么进入正式对照，以及pilot的限制

`artifacts/prefix_context_20260917/pilot30/` 保存了8条固定fit、两个身份、同一句台词、四类L1的接收试验。每轮实际47个有效当前窗，batch8，每臂最初15轮/90步，随后按独立延长协议只继续15轮，累计30轮/180步。

最初接收门槛误用oracle分段预测的“整段拼接边界”：每块由GT过去续接，却把当前首预测与上一块预测求差，而非与实际供应的GT前缀末帧求差。原始结果保留，另作**看过结果后的指标语义修正**。修正后第15轮眉位移RMS为参考2.02064倍，仍未通过<=2门槛，未四舍五入放行。一次固定延长后的第30轮为：眉1.71240倍、眼1.52769倍；相对无前缀同GT端点诊断的位移误差比为0.37696/0.47145，centered MSE均不劣，满足原数值阈值。

这属于“指标修正后、一次有上限探索性延长获得接收证据”，**不是最初15轮预注册门槛通过**。真实前缀与自生成前缀另报；小集结果不证明audio动态可预测，也不证明新句/新身份泛化。正式实验不加载任一pilot15/30上脸权重或optimizer，避免将只有prefix臂经历的小集适配混入比较。

## 2. 数据、初始化与冻结

数据严格沿当前2315 fit / 405内部开发、19/3身份及独立neutral enrollment；仍用原生25Hz、固定96帧缓存与原valid/channel mask。不新增情感类别、L2、人物、句子、随机crop或长片段。不读取封存test；405是反复用于开发且允许共享台词的内部开发，不能改称最终测试，也不宣称整个系统未见其身份。

`no_prefix`与`scheduled_prefix`都从**history12/no_history同一原始backbone与local**重新初始化，严格检查source checkpoint、recipe和数据hash。共享层逐项加载，新known/unknown embedding零初始化，两臂初值hash一致。丢弃旧history压缩器的非共享参数时须显式枚举，不能静默接受任意missing/unexpected keys。

仅训练九维上脸flow及known/unknown embedding。local声学网络固定；audio-global、四状态预测网络、B0、identity、motion teacher、原local projection与旧全脸renderer都冻结，并检查权重hash及无梯度。全局情感特征继续作为条件，但本轮**不新增在线motion→audio蒸馏**。

正式生成的其余43通道以及无效帧直接复制冻结Stage3原始local完整模型的基座输出；同输入、同seed逐位检查。基座曲线必须与来源绑定hash和405条metadata完全匹配。此复制证明系数保持，不等于本轮已经通过独立唇音同步、身份感知或整体情感自然度评价；九维的新变化仍可能改变感知表情。

## 3. 模型、坐标和对照含义

每窗固定过去H=8帧+当前C=16帧，共24槽。当前槽索引始终8–23，过去0–7；首窗过去槽invalid，其他窗取严格之前8个原生时间位置，缺失帧不压缩、不跨clip，不以当前/未来目标填known。继承 `_LayerConditionedDiT` 的固定sinusoidal位置。known动作与待生成动作进入同一self-attention，known/unknown embedding显式区分状态。

运动坐标完全沿history12：`(upper9 − 独立neutral anchor − 冻结audio四状态整段均值lift) / source_fit_residual_RMS9`。尺度来自完整2315fit、下限0.02，正式直接复用已核source尺度；不减query GT均值、不逐chunk重新中心化，不额外套用旧renderer residual_scale。同一clip各chunk保持同一audio均值原点。音频/global可用离线整窗信息，所以仅运动历史来自过去，不声称整个系统是在线因果。

训练与12步Euler推理都将known作为**干净固定前缀**；每步solver强制保留known值，只有unknown从一次采样的高斯噪声积分。known值不随flow time重新加噪，不在solver中重新采样噪声。clamp不自动保证第一未知帧连续或chunk均值稳定，须用输出结果验证。

`no_prefix`将过去8个token整体标invalid，故同时去掉这些槽中的动作和声学attention上下文，当前16帧位置与条件不平移。两臂比较定位“前序token上下文”的总作用，**不能独立归因于已知motion数值**；若以后要作纯motion消融，需另建保留声学前缀、去掉motion值的条件。本轮不临时增加这一臂。

## 4. 固定训练计划

| 项目 | 固定设置 |
|---|---|
| 两臂 | no_prefix / scheduled_prefix，均完整12epoch |
| 随机seed | 79；Python、NumPy、Torch及独立batch generator可恢复 |
| batch | 16个clip，epoch遍历2315 fit一次；145次optimizer更新/轮，预计1740次/臂 |
| 优化 | AdamW lr=1e-4，weight decay=1e-5，梯度范数clip=1；非有限loss/梯度直接报错 |
| 目标 | 仅有效unknown上的标准flow-matching MSE，无额外接缝/速度/std损失 |
| 当前帧 | 96帧分6个16帧chunk，尾段及缺失依原mask；只有有效unknown参与分母 |
| solver | 训练self-rollout与部署均12步Euler |
| checkpoint | 固定epoch12；不以开发表现选择中间epoch、seed或gain |

每个batch先在当前更新前、`no_grad`下生成一次完整96帧self-rollout，过去只能来自自己此前生成的块，随后detach。再对每个clip×chunk按固定随机decision选择真实过去或该self-rollout过去，计算六段unknown FM，按有效当前帧计数汇总后执行**一次optimizer更新**。不是每块更新一次，也不是把teacher前缀rollout偷偷作为self-rollout。

使用一基epoch的teacher概率如下：

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8–12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p(teacher past) | .75 | .642857 | .535714 | .428571 | .321429 | .214286 | .107143 | 0 |

即第8轮起归零，最后5轮完全使用generated过去。首块无过去，不计teacher使用。实际日志同时记录抽中的teacher次数、存在有效历史的次数与真实被使用次数，不能把no_prefix的随机decision当成它用了teacher。

两臂匹配初始化、clip顺序、FM噪声、self-rollout初始噪声、flow time、teacher decision和更新数，逐epoch保存hash验证。no_prefix也执行同样的self-rollout/随机抽样流程后忽略历史，避免随机流漂移；两臂生成出的动作数值自然不同，不要求self-history张量相等。

## 5. 正式评价与边界指标

固定epoch12后在全部405开发clip、固定seed **42/123/2026** 上评价deployable full生成；首块空前缀，其后全程generated过去，不用GT past或query GT统计。以相同target/mask/seed比较matched no_prefix，并列此前音频状态均值+对齐动态、音频状态均值+flow及history12结果；新旧比较若改变目标/参数化/训练路径，应明确为候选比较，不作单部件因果归因。

seed42另做empty/reverse-history、static/reverse local-audio/content、明确标记的oracle-history干预；保持global、identity和基座不变。reverse-history使用实际反序后的token，不能把它的末帧当成原历史末帧。GT过去只出现在oracle诊断，其预测不进入部署主表、主视频或多seed分布分数。

边界必须区分以下三个对象，并保留原值：

1. **连续输出拼接边界**：当前块预测首帧减上一块预测末帧。full rollout可解释为真实部署接缝；oracle中只能叫“teacher重置的独立预测块拼接不一致性”。
2. **实际输入prefix连接**：当前预测首帧减实际供应的known末帧；full使用generated末帧，oracle使用该GT末帧，reverse使用实际反序后末token。empty/no_prefix无known时记不适用。
3. **同GT端点诊断**：当前预测首帧减GT前一帧，对所有臂用相同评分锚点，但GT不是no_prefix的输入。teacher下此位移误差MSE代数上等于首未知帧endpoint MSE，不应当作额外独立证据。

只在紧邻前一原生时间帧和当前首帧均有效时计边界，不跨gap拿“最后一个有效帧”补接。full的前两项应一致，oracle的后两项应一致，作为实现审计。各项报告位移RMS、GT参考RMS、相对GT位移的MSE及有效配对数；同时给块内速度、每块raw均值和前2/4帧误差，避免只修首帧却块内漂移。

眉五通道、表情眼squint/wide四通道分别报raw/centered误差、相关、RMS比、越界及多seed fair ES/VS；neutral/nonneutral分别报告。blink/gaze保留基座，不把本轮称为完整眼部动态重建。单GT误差不是随机自然度充分判据，方差接近也不等于正确动态。情感若仍用训练motion teacher读出，必须标注NONINDEPENDENT。

使用已固定样例、同一音频与rig展示真实系数、来源、两臂连续曲线/视频及接缝；不按效果挑样本，不输出smooth、gain或时间偏移。若显示需要裁剪到[0,1]，显示规则单独记录，全部数值评分保留原始值。不得用best-of-K或修改渲染增益制造成功。

## 6. 审计、结果解释与停止边界

保存协议、源码快照、source/输入/初值hash、每轮随机流hash、optimizer/RNG恢复文件、最终权重及曲线hash。验证known每步精确保留、无当前未来GT推理输入、首chunk历史干预一致、no_prefix无论传何历史不变、padding/缺失不污染梯度、冻结模块无变化以及43通道逐位复制。

pilot校正和延长仅支持开展这次完整fit比较，**不预先决定正式结果通过**。正式评价以固定终点如实报告matched部署收益、oracle接收与分布差异；不能因局部指标变好忽视异常接缝、静态化、越界或表情退化。若epoch12依然不能获得合理连续动态，本轮停止且不替换默认，不按开发结果临时改teacher概率、续训预算、gain或门槛。新上下文/八类/L2扩展另按独立数据契约实施，不能混入当前两臂实验。
