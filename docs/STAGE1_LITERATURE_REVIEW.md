# Stage1 文献核查与 KineTalk 决策

## 目的

本记录回答三个问题：

1. DESTalker 的 neutral 口型是怎样得到的，哪些部分可以借鉴；
2. 跨情感输入为什么不会要求推理前先把音频转换成 neutral；
3. KineTalk 的 Stage1、Stage2、Stage3、Stage4 应该保留哪些监督，哪些监督不能继续使用。

本文只记录已读取论文或本项目实验得到的结论。论文中没有公开的配对、对齐和数据清洗细节，不推断为已解决。

## 直接相关工作

### DESTalker

来源：Xu Yang, Shiguang Liu, Qing Xu, “DESTalker: Disentangling Emotion and Style for Expressive 3D Facial Animation via Residual Generation”, *Computer Animation and Virtual Worlds*, 2026, DOI `10.1002/cav.70127`。

论文描述的 Stage1 是：冻结 HuBERT 得到音频表征；内容分支和情感分支并行；内容分支的时序卷积部分冻结；neutral decoder 只接内容特征，并用 neutral motion template 做重建。neutral motion 先生成，之后才加入 emotion/style residual。论文还使用 motion emotion 与 audio emotion 的分类和 soft gate 处理跨模态情感冲突。

这解释了为什么推理时可以输入 angry 或 happy 音频：输入仍然是原始情感音频，content branch 从其中提取发音内容，neutral decoder 被结构上限制为只看 content。因此 neutral 是输出空间的职责，而不是输入音频必须先变成 neutral。

不能直接照搬的地方：论文没有充分公开 neutral template 的跨情感配对、逐帧时间对齐、坏 pair 剔除和 target mask 细节。冻结 TCN 也不会自动证明 content feature 已经去除情感。KineTalk 必须用 independent identity、跨情感干预和 mouth timing 评估验证这一点。

### EmoTalk 与 EmoFace

EmoTalk（Peng et al., ICCV 2023）使用同内容不同情感的 pseudo-pair，并通过 emotion exchange / cross reconstruction 约束内容和情感的职责。EmoFace 的附录也采用内容分支和情感分支交换重建，并把内容重建与情感重建分开监督。

可迁移结论是：同内容跨情感样本适合做内容一致性和情感交换实验。但这不等于可以把跨情感动作在未经验证的 DTW path 上逐帧设成同一个数值目标。KineTalk 只在 safe teacher 的 masked 区域使用数值 neutral target；其余跨情感样本只能提供 self reconstruction、事件级或序列级信号。

### EMOTE、EmoFace 和动态情感

相关工作共同表明，唇同步需要短时内容特征，情感既包含 sequence-level 类别，也包含随时间变化的局部强度。只使用一个 global emotion code 会把音节级开合、局部峰值和情感强度变化压掉。KineTalk 因此把 affect 分成 `E_global` 和 `E_local[t]`，并让两者只进入 residual renderer。

### CodeTalker

Xing et al., “CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior”, CVPR 2023，指出普通回归容易出现 regression-to-the-mean 和过度平滑。codebook 或 motion prior 可以改善动作分布，但不能修复错配的文本、错误 DTW 或缺失的短时内容监督。因此 KineTalk 先修正数据契约和 content timing，再考虑离散 prior；不能用额外 loss 掩盖错误 target。

### MEDTalk 与 DisentangledBS

MEDTalk 的 motion cross reconstruction、overlap exchange 和 cycle exchange 说明：当解耦监督在时间池化或序列级进行时，不需要把 donor 和 query 强行逐帧对齐；同时，动态情感应保留 frame-wise intensity。DisentangledBS 将 speech deformation 与 expression deformation 分开，并指出 neutral 样本仍可能带有表达动作，情感动作也会影响嘴部。因此 KineTalk 采用“content 保证口型时钟，affect/style 做受限 residual”的软职责划分，不把嘴部情感变化硬归零。

## KineTalk 的最终职责划分

```text
任意情感 audio
  -> frozen HuBERT/WavLM
  -> content encoder
  -> neutral articulation prior B0

任意情感 audio
  -> affect encoder
  -> E_local[t] + E_global

reference motion
  -> motion-only style encoder
  -> S_global

B0 + bounded residual(E_local, E_global, S_global)
  -> expressive motion
```

Stage1 的 `B0` 不读取情感标签、reference motion 或 style code。它的输入可以是 neutral、angry、happy 等任意情感音频，训练目标是同一内容对应的 neutral reference motion。推理时不需要音频情感转换。

Stage2 从 motion residual 学习 `E_local`、`E_global` 和 `S_global` 的统计分工。不同情感、不同句子的 style 交换只使用 sequence-level / statistic-level 约束和 cycle consistency，不把 donor style 生成结果与 query 的原始逐帧 GT 作为严格目标。

Stage3 预测 audio affect field：分类只约束 global affect，局部 affect 使用 mouth-event、强度和 residual 的短时统计约束。neutral 分类应作为 class-balanced metric 和干预验收，不用大量重复分类 loss 代替时序监督。

Stage4 使用 query 的 `B0` 作为口型底座，style/emotion 只通过 bounded residual 注入。最终验收包括 query self reconstruction、cross-style 的内容时序保持、mouth event F1、jawOpen amplitude ratio、peak timing 和非嘴部表达质量。

## 最小损失决策

Stage1 首先只保留：

```text
masked reconstruction + lambda_v * masked velocity
```

这里的 reconstruction 应使用真实 Huber 或明确命名为 L1；当前实现中名为 `huber` 的量实际是 L1，已改名为 `reconstruction`。RMS/amplitude loss、全局 event gate、adversarial invariance 和未经 canonical-time 验证的 pair loss 不进入首轮训练。它们会把一个错误的数值目标变得更难诊断，RMS 实验也已出现幅度改善但 temporal correlation 降到约 0.094 的反例。

mouth-event、viseme、phoneme boundary 是验收和局部 mask 的辅助信息。没有可靠 timestamped aligner 时，空 boundary mask 不能被伪装成有效监督。

Stage2-4 每项只保留能对应明确行为的约束：self reconstruction、内容时序保持、global emotion classification、local affect consistency、style consistency/cycle 和 bounded residual。不要把同一个目标分别写成 amplitude、event、RMS、adversarial、triplet、supcon 多个近似重复的 loss。

## 受控实验顺序

1. 先在 identity teacher 上做 32/128 条样本过拟合；要求 amplitude、jawOpen、temporal correlation 和 velocity correlation 同时接近 1。
2. 在 safe cross-emotion teacher 上比较 neutral、angry、happy 等输入，检查同内容的 `B0` timing 是否一致，不能只看 scalar loss。
3. 做 speaker-held-out 验证；train/val 不能使用同一 manifest。若只有单一 safe manifest，报告必须明确是 train reproduction。
4. 固定 Stage1 checkpoint 后再做 Stage2 的 sequence-level factor swap 和 cycle；任何 cross-style 结果先验证口型时钟，再看情感和 style。
5. Stage1 未超过 zero/mean baseline 且未通过 amplitude、event、peak timing 三项时，停止后续阶段。

## 本项目证据

- DTW v3 的双特征 consensus 后，经 ASR 审计得到 4,111 条 safe、460 条 review、502 条 quarantine；review/quarantine 不能进入逐帧 neutral teacher。
- 4,111 条 safe teacher 上，DTW teacher 的 mouth amplitude 比中位数约 0.991，jawOpen 比约 0.983，线性插值相对最近邻约 0.995；因此 DTW 插值不是全局“幅度减半”的主因。
- corrected Stage1 20 epoch 的 amplitude P50 约 0.777、temporal correlation P50 约 0.532；32 条样本过拟合可达到 amplitude P50 约 1.014、temporal correlation P50 约 0.906。模型容量足够，当前瓶颈仍是监督噪声、时序契约和泛化。

这些数字必须和 checkpoint、manifest hash、target kind 一起报告，不能把同一 manifest 的 train/val 结果当成泛化结果。
