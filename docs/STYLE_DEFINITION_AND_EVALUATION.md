# 风格的定义、现有证据和评价协议

2026-10-09，Phase50。本文区分已实现、已验证和待实现，不把计划写成成果。

## 用户澄清后的主定义与执行准则（2026-10-09，优先于后面的历史计划）

用户要求的是适合当前数据和学术领域、实际可训练并能通过图表证明有效的风格；没有要求将基础偏置与动态硬性分离。Phase51的统计双分支仅是实验候选，不是风格定义、用户要求或已验证贡献。原时序编码器对照同样可能成为最终方案。先确定可验证的研究对象，再依据实验选最简单有效结构，不为已有结构重定义研究问题。

**建议论文定义：参考驱动的人物面部动作风格（reference-conditioned speaker-specific facial motion style），即在控制语音内容和情感条件后，从独立动作参考中提取、能够跨语句复用的人物特定面部运动倾向。** 张口与唇部运动习惯、眉眼联动、可重复的左右动作不对称、基础姿态都可能是其表现；这些是待验证的可观察特征，不是必须逐一设置的网络子空间。当前统一rig不包含人物几何身份。情感通路决定当前表达条件；人物风格允许影响其表现方式，但不能替换源语音的内容时间线。

论文英文定义草案（定义研究对象，并非宣称已成功解耦）：“We define speaker-specific facial motion style as subject-dependent motion tendencies inferred from independent references and reused across utterances under controlled speech and expression conditions.” 当前协议再明确：只提供两段neutral说话参考，声称的是该信息预算下可观察的动作风格；不能推定未见人物各种情感表演的全部习惯，也不能把整个neutral说话平均值称为无表情静止脸。

### 原文依据及差异

|论文/定位|风格含义与学习方式|与本实验有关的验证|采用边界|
|---|---|---|---|
|[Mimic](https://arxiv.org/abs/2312.10877)，方法p3–5、实验p5–7|从motion提取跨时间的人物speaking style，内容另建空间；身份分类、内容GRL、音频内容对比及latent cycle|LDD、独立motion分类器特征SCS、未见人物、style插值的嘴唇距离曲线和同帧图|支持人物动作风格的定义；不照搬整套loss；独立识别器必须先在真实独立数据上有效|
|[DiffPoseTalk](https://arxiv.org/abs/2310.00434)，§3.3.1/p4–5|明确包括张口大小、上脸动态与头动；以临近片段相似为假设，用对比学习和交叉片段条件重建|动态/多样性、可视化及人评，与几何shape分开|其短期表演风格不等于长期人物身份；本项目没有头动则不宣称该维度|
|[FaceDiffuser](https://arxiv.org/html/2309.11306v1)，§4、§6|顶点版训练人物one-hot及风格调制；blendshape版无人物条件|同音频换人物mean/std motion热图、diversity；blendshape曲线展示随机生成|热图适合作为输出证据；随机差异、diversity大不能单独证明参考风格迁移正确|
|[Personalized Speech-driven Expressive…](https://arxiv.org/abs/2310.17011)，§3.1/§4.2|参考blendweight的身份相关style；身份监督、content GRL和style调制|w/o style消融、LVE/FID/Sync、身份latent图|2023 arXiv稿PDF含占位DOI，不能当已核验TOG正式出版；不能将其含emotion的广义style完全等同本实验独立情感支|

已复核本地原文`research_20261007/fulltext/mimic_export.txt`、`diffposetalk_export.txt`及evidence中的FaceDiffuser/Personalized摘录，不是新增检索得到新的领域共识。相关论文没有要求一律拆分static/dynamic style。

### 如何在当前数据上训练出来

1. 当前实施：同人物、不同句、不同录音的support→query预测。620段TRAIN-fit neutral来自20人，每次采样两段，排除query本身及query句子；换support后仍由query自己的B0与情感先验驱动，监督query真实动作。由这种跨片段任务学习可复用信息，避免只记住固定enrollment。训练支持采样不需要将不同句的动作逐帧对齐。
2. 当前实施：仅风格encoder与表达decoder更新；B0及772D情感/韵律先验保持冻结。仍用原位置与相邻变化目标，没有增加一套身份/一致性/循环loss。嘴部开放，时序保持需通过输出检验，输入隔离本身不构成证明。
3. 当前受控比较：Phase51 temporal保留原时序style表示；statistics是压缩参考信息并分开接入decoder的候选。两者采用相同support采样和更新预算。不可把temporal/statistics比较写成单一分离模块的纯消融（还混有编码和初始化差异）。
4. 决策：先测原始参考的真实同人可重复性，再判断输出是否稳定且朝目标人物的独立动作统计靠近。若temporal已有效，优先保留简单结构；若statistics更好，它是实现选择而非“风格必然可分离”的证明。若两者都不稳定或只改变偏置，不能宣布风格完成。
5. 失败后有针对性处理：若GT参考本身不可重复，改善独立参考覆盖和观测质量；若GT稳定但code不稳定，才评估一项跨录音人物表征监督；若code有信息但decoder忽略它，检查正确/错误参考的query重建收益与接入方式。任何新增监督须单独设对照、先存Git，不能把更多loss当默认优化。

### 论文证据及交付，不以“看起来不同”为成功

|要证明什么|实际测试/图表|与主指标的关系|
|---|---|---|
|输出使用了参考|同音频同g/u/B0/rig，own参考与wrong-person参考，完整分情感/人物统计和固定八情感视频|对齐嘴部、情感和总体几何指标同时报告|
|不是参考句的偶然表演|同人物独立A/B差异，与跨人物差异一起展示；多独立query，不只一张图|防止靠风格塌缩获取稳定性；同时报告绝对差异和比值|
|差异具有目标方向|相同句子/情感/强度分组中的独立人物GT统计；mean、centeredRMS、范围、相邻变化|跨人物音频/表演不同，只作分布/统计比较，不作逐帧目标真值|
|参考不覆盖源语音和情感|固定源驱动的原生嘴部曲线、闭口/lag、情感四probe、八情感结果|lag/F1不是独立内容识别；不得用DTW重定时修饰主要指标|
|可放论文的可视化|固定语音时刻的多参考清晰单帧、对应嘴唇距离/眉眼曲线；统一rig的mean/std motion热图；OwnA/B对照|参考Mimic/FaceDiffuser的证据形式；照片或t-SNE只辅助，不替代定量验证|
|模块有用|最终结构对无风格/错误参考/原结构；推理消融和重训练消融分表标注|零code可能分布外，不能将其等同公平的w/o-style重训练|

独立SCS作为可选补充：只能使用TRAIN训练且已验证真实独立query→reference检索能力的固定motion表征；不得用生成器自身style encoder给自身打分再声称成功，若GT识别不可靠则不作主要证据。当前尚未实现该新probe，不能在结果表中填数。

当前证据仍是Phase48的部分迁移，非全面成功：整体/嘴部均值误差改善对比例80.36%/72.31%，同人A/B差异仍有跨人差异的76.4%。Phase51正式训练正在运行，新指标/图尚未产生。论文主张取决于后续完整结果；其他MBE/LBE/几何/动态指标继续共同优化，不能为风格可见性牺牲口型。

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
