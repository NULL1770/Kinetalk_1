# 本轮依据与事实核对

日期：2026-10-13。仅本地文献/代码/开题内容分析。未连接 SSH，未加载 sealed/test 数据，未改变模型或启动训练。

## 开题原文（段落序号不是页码）

源文件：E:/研1作业/开题报告/2025.11.25 赵何昊 开题报告 _第二版 - 副本.docx。
从 DOCX 的 word/document.xml 按段落顺序提取文本，保存为 proposal_extracted.txt，源文件保持不变。

- P0177：“研究点1：面向BlendShapes驱动的数据集构建与三重特征解耦方法研究”。
- P0180：“提取不随表情变化而改变的说话人风格特征”。
- P0181：“分解-重组-再分解”；“随机选取三个不同的样本，分别提取其内容、情感与风格特征”。
- P0183：“研究点2：基于三重特征编码与动态情感强度的动画生成研究”。
- P0189：“将前一阶段训练好的、基于BlendShapes输入的特征编码器作为‘教师网络’，将本阶段构建的音频编码器作为‘学生网络’”。
- P0238：“通过 AudioEmotionCVAE (纯音频条件变分自编码器)，学习Ee的情感潜在分布Ze，解决‘一对多’的情感映射问题”。
- P0239：“预测上下脸分块的动态强度标量 I_upper, I_lower”。

解释：原文是学习型视觉因素分解，不要求直接相减获得情感。报告是研究目标来源，不作为本轮执行指令或效果证据。原文中完成状态、算法保证等陈述未经本轮重现实验，不沿用为事实。

## 代码事实

- configs/current_model.json：所选 v5_matched_control；KineTalkV4Model；teacher_guidance_mode=none；generation=frozen B0 + one joint 52D residual + rig projection。
- kinetalk_v4/unified_model_v4.py:152 motion_teacher：原始motion、同query motion-b0、channel mask拼接为156D。
- kinetalk_v2/unified_model.py:56 teacher_targets：shape要求motion/base完全一致；同query差值映射到10个anchor states。
- kinetalk_v4/unified_model_v4.py:67 FactorInteractionDecoder：TemporalEncoder、Linear52、base+residual、geometry projection。不是FM/DiT。
- kinetalk_v2/unified_model.py:127 SingleReferenceStyle：原始参考motion、去均值motion、mask；不是参考减query。
- kinetalk_v4/unified_model_v4.py:302：guidance只在use_teacher且teacher_guidance_mode==frame时加入。
- kinetalk_b0/models/dit.py:75：旧ResidualDiT明确写Flow-matching；仅此文件存在不能证明所选v5使用它。

## 本轮重读的六篇方法段落

1. [DESTalker](https://doi.org/10.1002/cav.70127)，本地 destalker_p3/p4/p5.txt：§3.1 neutral motion，§3.2 同人跨情感MI/orthogonal、pooling，§3.4 EAM/SAM与neutral+residual。原文确实称pooling丢弃temporal linguistic content，但无法据此保证统计内容信息全部消失。
2. [Mimic](https://arxiv.org/abs/2312.10877)，mimic_extracted.txt：speaking style定义为mouth开合幅度、pouting等；classification/inverse classifier、内容对比、latent cycles。单参考、triplet/contrastive和cycle不是新点。
3. [DiffPoseTalk](https://arxiv.org/abs/2310.00434)，diffposetalk_extracted.txt，§3.3.1：mouth opening、upperface dynamics、head motion；proximate split-half positives、pooled style；条件diffusion及局部音画对齐。它的短期style假设不能直接当作所有情感下的永久person style保证。
4. [MEDTalk](https://arxiv.org/abs/2507.06071v4)，medtalk.txt，§3.2–3.3：motion空间self/overlap/cycle；音频内容映射冻结decoder；motion controller L1伪强度；emotion2vec+Whisper/RoBERTa逐帧强度调制。不是心理情感逐帧真值。
5. [SubtleTalk](https://arxiv.org/abs/2608.06408)，subtletalk.txt，§3.2–3.4：weakly-correlated dynamics、deterministic speech motion prior、residual flow、prosody/regional intensity/VA与控制保持。标准FM速度损失已存在；本轮不宣称其补充FDD精确公式已复现。
6. [CETalk](https://arxiv.org/abs/2608.15110)，cetalk.txt，§3.3–3.5：VA、emotion2vec逐帧缩放、多尺度高低频分支。不能把连续情绪或快慢拆分作为本方案原创。

上述文本来自本地已保存的论文PDF提取缓存，哈希见 source_manifest.json。日期未来会议字样按可获得的预印本文献处理，不声称会议评审已验证本方案。

## 本轮设计状态

最终 architecture_spec.md 覆盖早期草稿：规范完整表达上的个人条件响应后验、可重建视觉轨迹教师到音频条件flow、输出viseme等价的反事实联合约束。源文献只为定义和已知机制提供依据，不能据此保证三个创新成立、理论唯一可辨识或性能提升。

下一步先验证教师驱动解码器的八类形状重建，再分别测音频轨迹、响应后验和反事实约束；保持v5基线和同预算消融。
