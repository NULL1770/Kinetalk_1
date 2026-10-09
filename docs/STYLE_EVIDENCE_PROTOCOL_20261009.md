# 人物动作风格：定义、论文依据与最小证据方案

2026-10-09。当前讨论结论；不代表新实验已完成。本文件优先于旧风格文档中的 Phase48–51 状态和计划。当前已核验的风格结果来自 Phase53；Phase63 最后核验到 epoch4/step2147，SSH 中断后状态未知。本轮不修改模型、不增加 loss、不重启训练。

## 1. 定义与可以写进论文的边界

**风格是：控制语音内容与情感条件后，从独立动作参考中提取、能够跨语句复用的人物特定面部运动倾向。** 推荐名称：reference-conditioned speaker-specific facial motion style。

英文定义：We define speaker-specific facial motion style as subject-dependent motion tendencies inferred from independent references and reused across utterances under controlled speech and expression conditions.

- 可观察表现包括习惯性张口幅度、唇部与眉眼的联动、运动强弱及可重复的左右动作不对称。它们是待验证的现象，不要求新建“静态/动态”两个定义或网络分支。
- 同人稳定指在相同驱动条件下换该人的独立参考仍给出相近结果，以及跨语句的条件运动倾向可复用；不指不同句子/情感的原始曲线完全相同。
- 风格可以影响情感的表现方式；解耦要求保留音素顺序/时间和整体情感类别、合理的相对强度，不要求嘴部系数、表情幅度或整段运动逐帧不变。
- 统一 rig 上学习的是动作风格，不是几何脸形、纹理身份。MEAD 拟合偏差也可能具有人物可识别性，所以 latent 聚类或偏置差异不能独立证明学到了真实习惯。
- 当前只输入独立 **neutral 说话参考**。可验证“中性参考提取的风格能用于不同 query 句子与情感”；不能宣称“任意情感参考均提取同一个纯身份码”，也不能宣称恢复了该人物全部情感特有的表演习惯。neutral 参考的情感标签恒定，情感泄露分类接近机会水平不构成跨情感不变性的证据。

## 2. 其他论文实际如何论证

|原始来源及定位|核实到的证据|可借鉴及不能推定的结论|
|---|---|---|
|[FaceDiffuser](https://arxiv.org/html/2309.11306v1)，§3、§4.1、§6.1–6.3、Fig.5/6|顶点版以训练人物 one-hot 控制；Fig.5 用同音频不同人物的 mean/std motion 热图及跨人物 diversity。blendshape 版没有人物条件；Fig.6 是同音频多次随机采样的控制量曲线。人评为 realism/lip-sync/appropriateness。|借鉴固定音频换人物的热图/截图。随机曲线与 diversity 只证明差异，不证明独立参考稳定、目标人物风格正确或情感解耦。|
|[Mimic](https://arxiv.org/abs/2312.10877)，Quantitative Evaluation/Metrics、Latent Space Visualization、Style Manipulation、Fig.4/5、Table3/4|参考 motion 编码人物 speaking style；独立 motion 分类器特征的 style cosine similarity (SCS)，lip dynamics deviation (LDD)，未见人物结果；Fig.4 style/content latent 图，Fig.5 两参考风格插值的嘴唇距离曲线与同帧图；未见人物人评包含 speaking style。|比训练人物 one-hot 更贴近我们的参考设定。采用曲线+目标风格相似度+消融的证据组合；t-SNE 不单独证明解耦。LDD/SCS 不是任意两人的逐帧 GT 比较。|
|[DiffPoseTalk](https://arxiv.org/abs/2310.00434)，§3.3.1–3.3.2、§4.3–4.5|风格包括张口大小、尤其上脸的表达动态和头动；相邻片段作对比正对，交换两个窗口的参考条件进行重建。报告 FDD、mouth opening difference (MOD)、风格相似度人评及 w/o style encoder 对照。|支持动作幅度与联动的定义；短时表演风格不自动等于长期人物身份，本项目未建模头动则不声称该项。|
|[DESTalker](https://doi.org/10.1002/cav.70127)，仅 DOI/Crossref 元数据与摘要|摘要描述 content/emotion/style 解耦、neutral 基座和 residual 注入。全文仍不可得，本轮出版社 PDF 返回403，arXiv 标题检索无条目，GitHub 仓库检索无结果。|**其具体风格图、分类指标、t-SNE、嘴部遮蔽和消融尚未核实；不得填入事实对照表或称昨天已验证。**|

缓存原文：`docs/research_20261007/fulltext/mimic_export.txt`、`diffposetalk_export.txt`；FaceDiffuser 对应 `evidence/facediffuser_phase50_sections.txt`。本轮检索与文件 SHA 收据见 `evidence/destalker_style_recheck_20261009/receipt.json` 和 `evidence/style_claims_20261009.json`。这是定向原文复核，不是新增系统综述或独立多人评审。

## 3. 论文最小证据链和应该画的曲线

|要证明的命题|图或量化对照|关键控制|
|---|---|---|
|同人独立参考稳定，而且没有风格塌缩|同一音频/情感下，人物 P 的独立参考 A/B 两条原生嘴唇开合曲线应接近；人物 P/Q/R 的曲线应有可重复差异。补同人/跨人输出距离分布与绝对 MAE。|颜色表示人物、实/虚线表示独立参考。固定 B0/g/u/seed/rig；参考必须不同录音、不同句子，不能把同片段的相邻裁剪当强稳定性证据。|
|差异符合参考人物，而非任意扰动|匹配 sentence/emotion/intensity 的独立真实人物统计，与换风格输出的 mean、中心化 std/范围、原生相邻位移并列。可补共享 rig 的 mean/std 位移热图。|按条件和人物分组，不比较不同句子的原始曲线是否重合。源说话人 GT 不是换人后的逐帧真值；不同人物的真实音频/表演也是混杂因素。|
|可以跨句、跨 query 情感复用|对多个 query 句子及八情感重复 A/B/跨人测试，报告逐人物与逐情感结果；先看真实参考的同人可重复性。|不能只画 happy。可画风格码检索/距离图；t-SNE 仅辅助，neutral-only reference 不支持跨情感参考不变性。|
|换风格保留说话内容与整体情感|固定源驱动仅换参考：检查嘴唇峰/闭口事件/lag，并用可靠的独立内容、情感读出或盲化人评验证；同时展示固定时刻截图。|g/u 不变只是输入控制，不是输出证明。原生闭口/lag 是代理诊断；无可靠音素识别不能宣称严格内容不变。风格引起幅度变化时，对源 GT 的逐帧 LVE 不能独立当交换后的内容准确率。|
|风格模块有实用收益|正确参考 vs 错人参考的 own-query 预测收益；最终模型 vs 同预算重新训练的 w/o-style。|错误参考与置零仅是推理干预；置零可能分布外，不能冒充重训练消融。|

**建议主文只用两组风格图，剩余八情感放补充：**

1. **可控性与稳定性图**：一段固定语音，人物颜色 × 独立 A/B 线型；上排真实 rig 嘴唇开合距离（mm），下排一个明确的眉部运动量。标注可信语音事件时刻，配同帧截图。不能把不同句子强行 DTW 后叠画。现有 jawOpen 系数可保留为辅助，不能未经 rig 计算就标成嘴唇距离。
2. **目标方向与复用图**：按人物/情感分组的真实与生成幅度/上脸变化统计，配同人/跨人距离分布。style 插值可作为 Mimic 式可控性补图；平滑插值本身不证明身份正确，不预先保证不同 code 的输出单调变化。

曲线不平滑、不按方法各自挑峰、不重定时；掩码和缺帧保持原生25fps。统计量尺度只能 TRAIN 拟合。每个 query 等权，再报告人物宏平均；不能把数千帧或2026有向配对当数千独立人物来做显著性推断。开发只有3人，结果限于这3人的证据，sealed 不用于选择阈值或片段。

## 4. 当前做到哪里

当前风格训练是同人跨录音 support→query：620个 TRAIN neutral 支持/20人，排除 query 录音和句子，每次两段独立参考；由 query 音频/B0/情感条件驱动，监督 query 动作。Phase53 保持 neutral B0、772D emotion2vec+prosody student 冻结，重建梯度不进入 student；不需要把不同句子的参考动作逐帧对齐。

Phase53 现有统计编码器为姿态与响应两个描述分支，各32D，合并64D；这是既有实现选择，不是要求风格理论必须分两部分。仅风格更新版本的 posture 学习来自 neutral query，表达响应可由各情感 query 监督。Phase63 只替换表达 receiver，冻结 style 编码器，故不是新的“风格学习改进”；是否保留风格作用还需评估。

Phase53 单因素审计：完整1367开发 query、2026匹配有向配对；同人 A/B 嘴部 MAE=.005627，跨人=.026865，比值20.95%；mouth mean/range/displacement 朝目标统计移动的配对比例为79.32%/75.32%/74.93%。支持参考作用和部分稳定性，**不等于达到20.95%身份误差或79.32%风格准确率**。

同人闭口分歧9.60%，跨人32.45%；仅3个开发人物。说明内容保持仍有疑点，不能写“已完全解耦”。这也不能仅凭闭口阈值变化判定音素必然错误，必须结合幅度和独立内容验证。

已有 `05_identity_style_curves` 和八张 `05b_style_native_<emotion>` 是原始 Phase53 诊断图，可以保留作补充。当前 MBE=.763407/LBE=.369546/lip mean=3.262565mm/main F1=.676805；main F1 是整段动作统计 probe，GT=.660344，本身不足以证明表达更真实。不得拼接不同 checkpoint 的最佳指标。

## 5. 继续优化的顺序与失败后分支

1. 先恢复并结束唯一既有 Phase63 任务：查进程/checkpoint/source SHA；活跃则观察，被中断才按原 optimizer 状态续训。只收集原定指标与八情感/风格视频，不新开 loss 或网络变体。SSH 不通时不声称进度或 ETA。
2. 先核对真实独立参考的可重复性，再用上述 A/B/跨人/目标方向/内容情感保留对照判断当前风格能支持的论文主张。复用已有输出、掩码和计算，不重跑已完成 Phase53 训练。
3. 真实参考不稳定：处理参考质量与覆盖；真实稳定但 code 不稳定：才考虑一项跨录音人物表征监督；code 有信息但输出忽略：才考虑参考接入或 support→query 训练。若换风格破坏内容/情感：先定位 receiver 的作用，不能用增加 diversity 掩盖。任何新模型改动先 Git 存档、明确假设和同预算对照。
4. 如需要 Mimic 式 SCS，先 TRAIN-only 训练并冻结独立真实 motion evaluator，验证跨录音、留出人物真实 query→reference 检索能力。未知人物用 embedding 检索，不能要求 TRAIN 身份分类头直接识别未见类别。生成器自身 encoder 不给自身评分；若 GT 上不可靠就不作为主要指标。此 evaluator **尚未实现**。

## 6. 多方法面部截图的选择与交付

可以选真实有优势的成功案例，但图注写明 selected qualitative examples。所有方法与 GT 必须使用同一 clip、同一个原生时间戳、同一 rig/相机/光照/裁剪和已声明的推理条件；不得为各方法分别挑最差/最好帧或随机种子。选择实例不参与拟合、阈值调节、checkpoint 选择或全量指标。

- 先固定候选范围和各随机方法的 seed/检查点，再从完整对比的共同有效帧查找 ours 接近 GT 的嘴形/表情片段，可兼顾其它方法在该共同时间点的不足。它只能展示该案例的优势，不能据此证明平均 SOTA。
- 展示张口、闭口/双唇、圆唇等多个真实现象，并保留固定或随机选取的完整视频、全部指标与失败案例，避免只挑静止或幅度很小的容易帧。
- 输出每方法独立高清白底/透明 PNG，并保留选择清单：clip ID、frame/time、词/音素标注来源、checkpoint/seed、各方法误差与原输出 SHA、rig/camera 和 selected 标记。未经音频核实的字/音素不标注。
- 现有对比方法是 shared-rig 改编版本，必须保留版本/预算差异说明；不能把它们标成已经复现官方论文指标。不得为白底/透明截图修改动作。

本轮只完成定义、原文核对与证据计划，没有挑选新截图、生成新风格结果或修改网络。
