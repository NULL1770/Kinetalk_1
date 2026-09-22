# Bounded expression center：实现前审计与逐身份 OOF 方案

日期：2026-09-22。状态：设计审计，尚未实现或训练；不构成新实验结果。本文件只提出下一轮协议，不修改现有 checkpoint、生成入口或历史结果。

**实施选择更新（同日）。** 正文初稿完成后，已决定使用下文备选中的独立 anchor-relative 中心头，而不使用 c0-relative 校准。实际 API 为 `BoundedExpressionCenter(global_dim=64, identity_dim=128, hidden=64, max_logit_delta=6., anchor_eps=.01)`；公式为 \(\sigma(\operatorname{logit}(\operatorname{clamp}(a,.01,.99))+6\tanh(f(g,s,a)/6))\)。幅度限制6取代初稿讨论的2，目的是允许更大的表情中心变化；旧生成器仍作为真正外部epoch0候选。模型和组合的27项局部测试已通过，训练结果尚不由本文件认证。最终预算选择也已固定为全TRAIN上的独立三折身份选择后重训，而不是初稿的内选择epoch中位数；不会使用外层评价分数选轮。正文各节保留设计推导及审计背景，实施时以runner写出的protocol与源码为准。

## 1. 这次应解决的实际问题

现有 reference-intensity decoder 的 development 归一化 upper9 总误差为 0.547519，其中片段均值误差为 0.510292、中心化误差为 0.037227。因此中心偏差值得独立建模，但其下降不能作为时序动态恢复的证据。真实动作库已改善若干运动分布指标，同时使点对点 MSE 增大；下一步应检验中心校准是否改善同一生成器的表情基调，并完整报告其对动态幅度、共变与分布评分的影响。

当前数据仍为四情感 MEAD 子集：4098 TRAIN、446 development；TRAIN 有 22 个身份。`configs/paper_full_v1.yaml` 虽列出八种情感，但配置标签不是完整数据已准备或已训练的证据。本次只读搜索未在仓库发现可确认八情感或 CREMA-D 覆盖的本地 prepared manifest；不能据此断言远端没有这些数据。

## 2. 旧 probe 不能支持的结论

审计文件：`scripts/probe_expression_center_calibration.py`、`scripts/reference_decoder_data.py`、`scripts/train_reference_intensity_decoder.py`、`kinetalk_b0/models/empirical_upper_motion.py` 与相应 evaluation。

1. 旧 probe 使用已在完整 TRAIN 重训的 reference `final.pt`，并继承全 TRAIN 的尺度和输入统计。其 fit/cal 分割仅约束新 calibrator；base 在 cal 上的 0.054948 均值误差属于上游见过的样本，不能与新 calibrator 的 0.174123 直接比较后断言“明显过拟合”。原说明中的这一解释应修正。
2. `CenterCalibrator` 的线性输出无界，虽然后续逐帧 headroom-tanh 保证系数范围，但“中心 head”本身无范围约束。该组合依赖每一帧的 headroom，因此改变中心化运动。原 docstring 的“temporal residuals remain unchanged”不成立。
3. 没有旧模型的 epoch 0 候选；总会选一个新头，即使全部训练 epoch 都更差。选择项中 `.037` 和事后 static+0.0002 并非可迁移的统计标准。
4. 选择后未按确定的训练预算在全 TRAIN 重训；没有逐身份 OOF。随机初始化没有被固定记录；每轮重复同一排列也与常规的 epoch-specific shuffle 不同。
5. `scores(pred, zeros, ...)` 产生了不对应真实预测器的 predicted-intensity 指标。中心模块应只报告从最终运动计算的强度，或将“请求强度”明确标为不适用。
6. 旧 empirical 供体 run 在裁切前去均值；裁切后均值可以偏移，逐帧 tanh 又会产生附加偏移。因此不能把其输入 `center` 直接解释为最终样本的精确片段均值。

旧 probe 的 development 数字可以保留为探索结果，但不能作为模块级 OOF、精确中心保持、无动态干扰或论文结论的证据。

## 3. 最小可行中心头

### 3.1 输入与参数化

冻结基线输出为 \(B\in[0,1]^{T\times9}\)，有效帧集合为 \(V\)。定义其部署可计算均值

\[
c_0=|V|^{-1}\sum_{t\in V}B_t.
\]

输入仅为冻结音频全局码 \(g\in\mathbb R^{64}\)、独立中性参考的个人运动码 \(s\in\mathbb R^{128}\)、参考锚点 \(a\in[0,1]^9\) 和 \(c_0\)。均不读取 query 的真实运动、真实均值、情感标签或 teacher 输出。小网络为 210→64→9，SiLU；最后一层零初始化。所有新输入标准化统计仅由对应拟合身份计算，anchor 与 c0 保持物理量纲或使用该折拟合统计，不继承旧 reference 的新头归一化。

推荐主方案是 **reference-conditioned、base-mean-relative** 中心校准，不要将它写为纯 anchor-relative 方法：

\[
\delta=2\tanh f_\theta(g,s,a,c_0),\qquad
c=\sigma(\operatorname{logit}c_0+\delta).
\]

此式将 log-odds 位移限制在 \([-2,2]\)。常数 2 预先固定，本轮不在 development 搜索。为了处理 0/1 端点并使零初始化精确还原，实际使用代数等价式：

\[
c=c_0+\frac{c_0(1-c_0)\operatorname{expm1}(\delta)}{1+c_0\operatorname{expm1}(\delta)}.
\]

分母至少为 \(e^{-2}>0\)，输出在 \([0,1]\)，\(\delta=0\) 时修正项精确为 0。端点 0/1 保持不变，这是该保守校准的明确限制。不要通过事后 clamp 掩盖无界 head，也不要声称它能将任意饱和通道重新激活。若后续必须移动端点，可研究独立 bounded center，但应单独与本方案比较，保留原模型回退。

相比直接 anchor-relative sigmoid，这一方案有可解释的 epoch 0，避免零初始化从原预测突然跳回中性锚点。独立 anchor-relative 全中心预测可作为后续研究，当前不同时扩展两条路线。

**实施前必须明确的取舍。** 上述主案是“保守增量校准”，不是对当前偏差来源的保证：reference final 的 c0 在 TRAIN 上可能已有较小误差，而 development 偏差较大。即使 head 自身完全按身份 OOF，冻结 reference 在外层身份仍是 in-sample，这会鼓励头学习恒等并选择 epoch0，不能据此排除独立中心建模的价值。若目标是替代旧中心，另一项可预先选定的方案为 \(c=\sigma(\operatorname{logit}(\operatorname{clamp}(a,0.01,0.99))+2\tanh f_\theta(g,s,a))\)，输入201维，直接监督真实片均值，外部将旧生成器作为独立 bypass 候选。此处 clamp 只定义参考锚点的有限logit，不是输出越界后的掩盖；输出仍由sigmoid保证有界。它不使用旧 c0 作学习输入，不要求新头零初始化等价旧模型，但主张也必须改成“参考条件下的独立有界中心预测”。无论选哪一种，应在训练前固定；不根据外层/dev结果来回切换。真正解决旧c0的TRAIN/dev误差差异需要对reference decoder再做cross-fitting或重新划分上游训练，本文件的轻量模块OOF不完成这一点。

### 3.2 中心与动态的有界组合

令已有确定性或固定 seed 的完整经验生成结果为 \(Y\)，仅由部署输入和合法 donor 得到。先在有效帧上取其均值并构造

\[
R_t=Y_t-\bar Y,
\quad m^+_j=\max_{t\in V}\max(R_{tj},0),
\quad m^-_j=\max_{t\in V}\max(-R_{tj},0).
\]

对每个通道使用一个时间不变缩放，而不是逐帧 tanh：

\[
\alpha_j=\min\left(1,\frac{c_j}{m^-_j},\frac{1-c_j}{m^+_j}\right),\qquad
\widehat Y_{tj}=c_j+\alpha_jR_{tj}.
\]

分母为 0 的比值按 \(+\infty\) 处理；不要以固定 eps 错把全零残差压缩。有效帧外直接复制原结果。由 \(R\) 零均值可得 \(\overline{\widehat Y}=c\)，且每个通道输出都在 \([0,1]\)。它保持该通道的时间顺序、极性和相对波形，但可能降低幅度；不同通道的 \(\alpha\) 不同会改变协方差大小，不能宣传为完全不改变动态。每个样本报告 \(\alpha\) 分布、受限帧/通道比例和均值数值误差。

这个过程会把经验样本的随机片段均值移到预测中心；因此要有“中心化组合但不学习”的独立消融，不能把组合规则带来的变化全部归功于中心 head。经验生成器在这轮保持原 top32、距离权重、温度和 8 个 seeds，避免同时优化检索条件。

**epoch 0/回退须真正绕过整个新模块**，返回原来 \(Y\)，以逐元素一致为准。除了该部署基线，再报告 \(c=c_0\) 的新组合基线；后者对经验样本不是恒等变换。选择时不得把两者混作一个 base。确定性样本理论上 \(c=c_0\) 时恢复 \(Y\)，浮点有微差，应通过直接旁路保证回退的逐 bit 一致。

若本轮实现资源紧张，可先训练和验证确定性分支，再对预先固定的经验样本做相同中心替换评估；不可因此省略组合基线。

### 3.3 可选更保守的幅度保持版本

若最优先目标是完全保留每通道动态，则将静态位移限制在 \([-\min_tY_t,1-\max_tY_t]\)，直接令 \(\widehat Y_t=Y_t+d\)。该可行区间保证有界且中心化轨迹不变，但高幅度或接近两端的轨迹可校准空间很小。不得同时声称“任意中心都能纠正”和“动态绝不变”；两者在有限系数区间里通常不能兼得。本轮推荐主方案明确报告缩放代价，不额外开展这个分支的模型搜索。

## 4. 逐身份 OOF：严格到模块层，不能冒称整系统未见身份

冻结 Stage4 和 reference final 已见过完整 TRAIN。可以严格做到新中心头、其统计、经验 bank 在外层 held-out 身份上未拟合，但不能把外层结果写为整个系统的 unseen-speaker 泛化。应称“冻结上游条件下的中心模块逐身份交叉验证”。要做整系统未见身份结论，必须每折重训所有数据依赖模块，或改用上游从未见过的独立身份测试，后者也不能被再次用于调参。

### 4.1 固定协议

1. 根据完整 TRAIN 元数据先构造全部 22 个 leave-one-speaker-out 外折。每条 TRAIN query 恰好作为外层评价样本一次，不丢混合 speaker×sentence 单元，不先截 smoke 样本再划分。每折保存 clip IDs、speaker IDs、manifest/checkpoint/source-file SHA。
2. 外层身份的 query 运动从所有中心训练、统计、经验 bank 中排除；其独立中性 enrollment 是部署可用参考，可用于计算身份条件，但不可用其 query 均值或情感标签。ENROLL/query 交集必须为零。
3. 外层剩余 21 个身份按排序确定 3 个内折组，轮转留出；每个内折只在其训练身份拟合 head 的输入统计和监督尺度。训练预算固定 12 epoch，batch64、AdamW(lr=0.003, weight_decay=0.01)、gradient clip1，预先固定 seed53。若要降低成本，可固定 12 epoch、不选择 epoch；不能用外层挑 epoch 后仍称其独立 OOF。
4. 监督中心是有效帧 raw upper9 均值，尺度由内拟合 clips 的 reference-relative RMS 计算，floor0.02；每条 clip 等权，先各身份平均再宏平均作为内选择指标。训练损失为归一化中心 MSE；不把不存在的 predicted intensity 加进去。
5. 对 epoch1…12 和真实 bypass epoch0 在三个内验证组评估。主选择项为身份宏平均中心 MSE；若没有新头严格优于 bypass，取 epoch0。并列时取更简单的 epoch0/更早 epoch。任何动态/分布保护阈值必须在运行前写入 protocol，不能读取外层/dev后再改变；本轮可不设未经论证的数值门槛，完整报告取舍。
6. 用选定 epoch 在该外层的全部 21 个拟合身份从零重训 head，重新计算新头统计，再评价完全未用于 head 选轮的外层身份。外层经验 bank 也仅由这 21 个身份构建，并继续排除同 sentence donor。评估不得将外层 query 加回 bank。保存每个外层预测和全部 donor 元数据。
7. 汇总外层 OOF 报告；该报告评价的是上述“内选择→外重训”的算法。不要在看完 22 个外折后选一个全局最优 epoch再把同一 OOF 当成无偏最终成绩。
8. 最终训练预算用预先规定的“22 个内选择 epoch 的中位数，0 同样参与，偶数向较小整数取整”产生。重新计算全 TRAIN 新头统计，按该预算在 4098 TRAIN 从零重训；若预算0保存显式 bypass 配置。全 TRAIN 建最终经验 bank。之后仅一次 development 评估，且明确 development 历史上已反复使用。

嵌套方案只需约 88 个小 MLP 训练（每外折三个内训练加一个外重训），上游条件缓存一次，通常远小于重复训练整个网络。内层仅中心选轮，不必每epoch运行8-seed完整分布评分；外层和最终development才做完整生成指标。若改变宽度、log-odds范围、损失权重或组合算法，要算新的探索轮，不能继续称同一预注册协议。

### 4.2 统计口径

使用逐 clip 原始误差供复核，但主要报告逐身份宏平均及所有身份行，避免数据量大的身份主导。外层 head 的尺度在不同折会变化：跨折汇总以 raw center MSE 为首要通用单位，同时报告折内归一化误差；不能让更宽松尺度的折获得隐性优势。内选择仍使用各自拟合尺度且两候选共享同一尺度。

配对差按身份聚合，固定 seed 的 cluster bootstrap 或对22个身份配对差给出置信区间；不得把帧当作独立样本。句子重复问题另报 sentence-cluster 敏感性分析，不能把两个单轴bootstrap叫作双向cluster。development 只有3个身份时，不以狭窄CI强宣称泛化。检验是在既定历史系统上的模块估计，不覆盖选择旧 upstream、历史试验和全部数据处理的不确定性。

## 5. 必须比较的输出与指标

最低三组：旧生成器 bypass；新中心化组合但使用 \(c_0\)；学习 bounded center 后的新组合。每组分别报告 deterministic 和 empirical conditional；固定 unconditional donor 对照至少在最终报告保留，用于检验条件检索贡献。经验采样使用全部固定8 seeds，不挑最好样本。

- **中心**：raw及normalized mean MSE；每身份/每情感；最大身份退化、改善身份数；头请求中心与最终样本均值误差；相对独立anchor偏移。
- **重建分解**：raw upper MSE、normalized total = mean + centered；使用相同valid和尺度验证恒等式。中心化MSE不应是随机生成器唯一的门控标准。
- **运动分布**：raw/centered fair energy score、fair variogram(lag1/4/16/32)、四组 up/down/squint/wide 的幅度比例、速度/加速度、协方差距离；全部源于最终输出。眼区均值不能掩盖wide不足。
- **约束**：有效upper范围、非upper43和invalid帧逐bit保护、\(\alpha\)分布、幅度收缩比例、donor不合法/长度不足计数。复制嘴部只说明相对原模型不干扰，绝对MLE/音画同步仍须独立测量。
- **因果范围**：中心头没有局部时序输入，不会因此恢复逐帧音频眉眼对齐。局部full/static/reverse/shuffle/mismatch可保留在reference动态部分；不要对静态中心自身计算峰值延迟后包装成新的音频时序成绩。
- **论文级后续**：生成脸情感识别、身份运动特征保持、同协议口型质量、盲评自然度与情感匹配。现有coefficients误差与抽样分布指标不足以宣布MEDTalk同水平；ARKit可比指标与原论文mesh指标应分开命名。

## 6. 必要测试与复现产物

实现阶段优先测试真正可能损坏结论的性质，不写只镜像实现的冗余测试：

1. 极端logits、c0=0/1与接近边界时输出有限且有界；零初始化修正精确为0；一般内点训练梯度非零。
2. 组合对负/正/零残差、多连续run、NaN无效padding有效；最终均值与请求中心相符，单通道波形只受统一缩放，边界无post-hoc clamp。
3. bypass逐bit返回旧结果；nonupper43/invalid逐bit不变；同seed不依赖全局RNG状态，CPU/GPU差值按实际数值误差记录，不事后修改阈值掩盖失败。
4. 将 held-out query motion 替换成随机值只改变评价，绝不能改变输入、预测、拟合统计、donor、选轮；通过哨兵/拒绝读取测试确认test和其他role数据没有打开。
5. split在完整元数据上构造，outer/query与fit/bank集合不交；每个身份只作一次外评价；所有索引显式long；空fold早失败，smoke在分割后分层采样。
6. 同候选共享条件、mask、donor draws与噪声；模型评价时eval、no_grad；缺少观测通道不静默以0监督。归一化只在当前fit上计算，checkpoint记录统计来源IDs哈希。
7. 指标分解和已知玩具序列校验；没有请求强度时不生成伪predicted-intensity栏目。全部8 seeds写入评分与donor manifest。

最低产物：`protocol.json`、外/内split manifest、每fold selection/history/checkpoint/per-clip scores、OOF总表与逐身份表、全TRAIN refit checkpoint、final evaluation、固定预览manifest和四情感视频。源码/数据/依赖/随机种子哈希绑定。初稿完成后再实施本设计，不能把本文件当作实验已经完成的证明。
