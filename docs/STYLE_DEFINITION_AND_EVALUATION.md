# 风格的定义、现有证据和评价协议

2026-10-09，Phase50。本文区分已实现、已验证和待实现，不把计划写成成果。

## 可用于论文的方法定义

本工作中的人物风格指统一面部rig上的人物特定动作习惯，包括稳定的中性基础姿态，以及在语音/情感条件下的幅度和运动响应。情感类别与强度由情感通路控制；内容通路提供中性发音轨迹。人物风格允许调节嘴部幅度、眉眼运动及左右不对称，但不应重排音素时序。由于所有结果使用同一rig，本工作不声称生成不同人物的脸形、骨骼或纹理身份。

|概念|含义|现有实现/可观测证据|边界|
|---|---|---|---|
|Neutral bias|中性状态下持续的眉眼、嘴角、jaw基础姿态与不对称|当前64D参考code经decoder bias影响基础姿态；Phase48可见差异|两段neutral说话参考不是静止无表情扫描；片段瞬态和拟合偏差不能都叫人物习惯|
|Speaking/expression style|人物习惯性的发音幅度、表情响应、运动速度|同音频同情感换neutral参考，幅度/均值部分朝donor统计靠近|neutral参考只能识别其中有观测支持的部分；未观测该人物的各类情感，不能声称恢复其所有情感表达风格|
|Emotion expression|类别、强度和随音频变化的表达动态|emotion2vec768+prosody4→global32/local16 prior；MEAD clip标签|MEAD三级不是逐帧强度真值；不把speaker相关情感当纯风格|
|Geometry identity|脸形、比例、纹理|本项目统一rig，不学习此项|眉眼系数左右不对称不等于几何左右不对称|
|Content|说什么、音素对应的发音轨迹和时间|冻结neutral B0；student不输入HuBERT内容|输入隔离不等于统计或因果完全解耦，仍需交换测试和独立内容读出|

当前ReferenceStyle输入独立neutral motion、对应冻结B0与观测mask；156D逐帧特征经3个TemporalBlock、masked pooling、64D投影与聚合MLP。Decoder宽192、4个TemporalBlock，global/local/style经四层调制，style另有52D bias。尚未实现独立neutral_bias/speaking_response子空间。

## 论文原文的使用边界

- [FaceDiffuser](https://arxiv.org/html/2309.11306v1)，§3、§4.1、§6.1–6.3：[原文片段](research_20261007/evidence/facediffuser_phase50_sections.txt)。V-FaceDiffuser使用训练人物one-hot、neutral template和hidden-state乘法风格调制；**B-FaceDiffuser不使用subject conditioning**。顶点实验以同音频不同人物的mean/std motion heatmap、cross-subject diversity展示可控差异；blendshape实验以多次采样控制量曲线展示随机多样性。不能说它已验证本项目的reference人物迁移。
- FaceDiffuser的MVE/MBE、LVE/LBE、FDD和人评分别支持几何/控制量误差、上脸统计变化、realism/lip-sync/appropriateness；diversity增大不自动表示人物风格正确，其动画图明确允许上脸不逐帧匹配GT。
- Personalized Speech-driven Expressive 3D Facial Animation Synthesis with Style Control（已存原PDF提取片段）：动作参考style encoder、身份分类与content GRL、gain/bias调制；报告LVE/FID/Sync、w/o style消融及identity潜空间图。t-SNE/身份分类可作为辅助证据，不能替代未见人物参考稳定性。
- MEDTalk与EmoTalk的已核验指标和rig差异见[PAPER_EVALUATION_PROTOCOL_AUDIT_20261007.md](PAPER_EVALUATION_PROTOCOL_AUDIT_20261007.md)。本项目F1是整段动作统计probe，非逐帧情感准确率；GT主probe F1=.660344。不能把生成F1高于GT解释为生成更真实。

## 固定评价协议

1. **Own reference A/B stability**：同一源音频、B0、g/u、采样种子和rig，分别用同人物两段独立neutral参考；报告code距离、raw/clip动作MAE、mean/centered误差、闭口分歧、native lag。与cross-person difference同时报告，防止以塌缩取得“稳定”。
2. **Target-directed transfer**：保持音频/情感条件，只替换donor reference；按相同sentence_id、emotion、intensity匹配目标人物真实片段的native统计，报告mean/range/centeredRMS/displacement向目标移动的比例与方向。跨人物真实音频和表演不同，不能作逐帧GT正确率或因果transfer准确率。
3. **Bias versus response**：分别报告均值/静态左右差与去均值幅度/动态。neutral说话参考须按native观测、有效通道和B0活动程度分层；不能直接把全段GT-B0均值当纯人物bias，也不能用平均jaw低推定真实静止嘴形。
4. **Generalization**：TRAIN-fit、既定internal-held身份/句子、external dev分开；不使用dev视频/统计拟合gain或选择漂亮样例；sealed保持未读。人物级bootstrap或逐人物表，明确held只有2人、dev只有3人的小样本限制。
5. **Content/emotion preservation**：保留原MBE/LBE/lip mean与lip max、四个raw/clip probe、jaw range/centered correlation、native速度/位移误差、FDD；换风格前后的jaw lag和闭口事件仅辅助，独立phoneme/SyncNet内容评价仍缺失。强度与emotion保持不要求逐帧抄GT不可预测眨眼。
6. **Visualization/human assessment**：固定八情感样例；同音频同情感多参考、OwnA/B、GT并排；统一相机/rig/原生25fps，不做效果选择或时间对齐。论文可用静态图补neutral左右偏置、mean/std heatmap、jaw/brow native曲线、预先固定词时刻的清晰单帧。盲化A/B人评区分lip sync、表达适配和人物风格相似性；没有受试者结果就不编人评。
7. **Ablation**：原模型、仅bias、仅response、完整分离结构在相同TRAIN/内部划分及预算下对比；一次更换聚合+decoder不能单独归因于某模块。必须同时记录参数/更新/解析拟合预算和失败候选。

## 已有数据与尚未支持的结论

Phase48：2026有向配对，整体/嘴部统计改善比例80.36%/72.31%；部分眉部与嘴速度方向失败。同人A/B输出差约cross-person差异76.4%，闭口分歧22.3%。支持部分动作风格迁移，尚不支持稳定纯身份解耦。

Phase49：64D code同人/跨人RMS比TRAIN-fit .010、internal-held .367、dev .465；参考GT-B0混有B0误差/内容分布/姿态。neutral参考可能具有真实姿态差，不能靠一致性loss强行抹平。当前最佳Phase47-latent F1 .719825 / MBE .825160 / LBE .383860 / lip mean3.276868mm，未因诊断改善，未达到联合SOTA。

## 本轮优化顺序

先审计原approved TRAIN pairs及neutral enrollment的native质量边界：teacher_mask AND event_local_mask AND source observation（mouth）；上脸另报原支持边界。对每个合法观测分解GT-B0=(GT-aligned neutral)+(aligned neutral-B0)，报告两项的均值/中心化/相邻变化以及交叉项，不能把非正交两项能量简单加成比例。

随后用TRAIN-only可观测的稳健bias与内容条件下响应统计对照原code；优先低容量、可解释的双参考共同分量，再决定最小结构变化。neutral静态bias与动态response必须各有作用检验。保留冻结neutral B0及772D情感student，模型修改前Git推送，smoke通过再训练；不放大gain、不增加一组无法识别的loss。
