# 内容、情感动态与身份：下一轮方案（2026-10-06）

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent plan
- Origin Date: 2026-10-06
- Verification Status: UNVERIFIED（拟执行计划），本文实验尚未执行；已核实事实另列
- Version Label: disentanglement_identity_plan_v1
- 用户停止点：先上传已有成果、写明方案；得到明确同意之后才新增训练/诊断代码或启动实验。

## 恢复入口与已上传成果

压缩后先读本文、`EMOTION_STUDENT_INPUT_CORRECTION.md`、`PHASE26_FINAL12_RESULTS.md`、`PHASE29_RESULTS_AND_NEXT_STEPS.md`，再按需读 `CURRENT_MODEL_AND_TRAINING.md`。`CURRENT_OPTIMIZATION_STATE.md` 的旧状态仅为历史，不重启已完成训练/下载，不继续旧 Phase30 混合输入方案。

已推送用户指定仓库/分支：`https://github.com/NULL1770/Kinetalk_1/tree/codex/experiment-snapshot-20260918`。

- `7a24423`：当前模型/训练/评估源代码、测试、阶段记录、SHA绑定的 Phase29 readout、固定评估 rig。
- `8e154cd`：测试所需的三个既有实验协议 helper。
- 本次上传前完整本地测试：363 passed / 1 skipped；Transformer 的 nested tensor 提示不影响通过结论。没有新增训练实现。
- 不上传私有连接 helper、临时媒体、node_modules、GB级权重/归档或未经单独确认再分发许可的 VOCA 上游源码；本地原文件保留。GitHub 是源代码/实验记录快照，不是含 MEAD 数据与全部权重的自包含训练包。
- Phase29 492 文件已下载并核 SHA；默认发布权重未被新候选替换；没有新772D训练成绩，当前没有新远端任务。

## 1. 先纠正问题定义

### 已核实

1. 历史学生输入1540D＝HuBERT768＋emotion2vec768＋prosody4；global与逐帧 `u_a` 共用4块TCN，均可直接读取内容。后续入口已改为772D，真实CPU/GPU及梯度检查通过，尚未重训。输入切除只去掉显式入口，不能证明emotion2vec本身无音素信息。
2. teacher读真实残差及速度，输出global/intensity；当前teacher没有给student逐帧情感目标。teacher阶段联合训练teacher、student、renderer；audio阶段冻结teacher。`u_a` 主要通过全残差flow及生成语义项间接学习。端到端监督可以学到有用条件，但不能保证这64维只表达情感动态。
3. 最新B0用各片段自身情感GT训练，口型改善有真实证据；它不再是严格中性基座。HuBERT/h0也不能只凭名字称为纯内容。要检查情感是否已由B0带入，不能只审计student。
4. 身份encoder使用独立中性参考减B0的mean/std，预测128D code和52D常量bias；renderer另用code调制。22对中性参考训练能证明参考重建，但尚未证明生成的动态风格随参考正确变化；mean/std也可能带B0误差。只延长身份阶段已有失败记录，不重做。
5. 残差全51个观察通道开放，情感/身份可影响jaw、lip、smile；保留这一点。

### 当前指标究竟到哪里

全1367 validation、固定训练seed47/48/49、固定draw42/123/2026、12 Euler、四冻结TRAIN probes，历史Phase26 standardized12：

| 指标 | Phase26保留候选 | Phase29 dropout（未接纳） |
|---|---:|---:|
| 最终动作原probe F1，clip128 / clip64 | .796875 / .721358 | .797647 / .715031 |
| 同probe F1，raw128 / raw64 | .619331 / .591701 | .610685 / .583535 |
| MBE / LBE，clip | .870982 / .421170 | .864415 / .420868 |
| Lip mean / max，mm | 3.562193 / 6.699290 | 3.558562 / 6.696472 |
| Expression mean，mm | .762788 | .751221 |
| Vertex absolute FDD，mm² | 138.879971 | 138.434368 |
| Mouth displacement MSE | .001816626 | .001879576 |

F1>.7已在**旧混合输入候选的clip协议**达成；raw仍低，GT原probe本身约.66/.64。不能把F1单项当逼真度，不能写成新772D成果，也不能靠修改probe或t-SNE取代失败指标。Phase29三个联合gate均失败，dropout不继续扫描。

静态和动态两类误差都有直接证据：Phase26眉部MSE的96.59%来自clip平均姿态偏差；连续global TRAIN→validation MSE增约15–16倍。中性jaw，B0相关约.471，最终约.253；残差使位移误差增加约51.8%。三draw间速度方差解释约26%误差，条件均值仍干扰时序。范围接近GT并不等于发音时序正确。

## 2. 因素分工与主张边界

| 因素 | 可以影响 | 应保持/排除 | 需要的证据 |
|---|---|---|---|
| 内容：HuBERT→B0/h0 | 音素形状、开闭顺序、原生说话时钟 | 情感标签/参考替换不能另造音素时间轴 | 固定内容的情感/身份干预；事件时序及读唇/内容诊断 |
| 情感：emotion2vec＋prosody | 类别、MEAD强度、眉眼/嘴角姿态、表达动态及jaw/lip幅度 | 不作为B0发音误差的无约束补偿通路 | 独立动作指标、连续表情目标、`u_a`干预与条件化泄露探针 |
| 身份参考 | 稳定系数偏置、个人运动幅度/习惯 | 不携带参考句子、参考瞬时情感或query内容时序 | 同音频同情感换参考；同人异句参考一致性；code/bias分开干预 |

prosody天然与重音、停顿和内容时间有关，“情感动态与音频时钟完全统计独立”不是合理目标。目标是情感分支不决定音素身份及重新安排音素顺序，而能按该时钟表达韵律。探针可解码内容不单独构成泄露定论，需控制韵律、标签与身份后比较，并看生成干预。

当前身份是**固定rig上的运动风格/系数身份**，不是脸形或纹理身份。同一mesh渲染最能隔离运动差异；只换头像或mesh不能验证身份模块。若论文要宣称恢复外貌身份，需要另一个明确的几何输入/任务，本文不悄悄扩大范围。

## 3. D0：先做冻结诊断，不新增loss（首批拟实现）

所有模型状态/源代码/manifest/probe/rig/统计文件SHA绑定；先正常输出重放，GT、mask、B0、h0、global、intensity、identity、noise和native times核一致，再干预。未通过重放就停下修复诊断；不重训掩盖差异。

### D0a：身份干预

固定同一query音频、global/intensity/`u_a`、B0/h0、噪声，分别使用：自身参考A、自身独立参考B、其他身份参考、零bias、零code，以及单独换bias/单独换code。零值是诊断，不作为正常部署策略。

- 统计实际code/bias差异、生成静态偏置和中心化速度/幅度差异；验证renderer是否利用code，而不是仅加了一次常量。
- 同人A/B参考差异应小于异人差异；按身份分组报告，不把22位训练参考上的检索当未见身份泛化。训练身份及3位validation身份分别报。
- 参考只用独立enrollment，不从query GT提取身份。错误身份输出不能与原query GT算MBE后就断言身份“失败”；只有真正存在、已对齐的目标身份同句同情感GT才允许定量目标比较。
- 预登记固定视频片段/参考顺序，显示自身A/B、两位其他身份、code-only/bias-only和GT；相同mesh、视角、时钟及draw。不能用挑出的最大差异片段证明功能。

### D0b：`u_a`与内容/情感分工

旧模型仅作为**污染诊断**，替换 `u_a` 为原序列、zero、static、reverse、一次固定shuffle；保持其他条件和padding不变。旧 `content_static` 可用于定位直接HuBERT时间信息的贡献，但必须新登记为legacy-only诊断，不启动原Phase30整套方案。772D模型训练后执行前五项，HuBERT变化应无直接影响已由单元/真实验证覆盖。

分别记录B0、生成残差和最终动作：jaw/lip/smile及眉眼的meanbias、centered相关、范围、速度误差、输出对条件变化的响应；记录phonetic错误与表情控制误差的区别。reverse/shuffle属于分布外扰动，只有依赖性证据；zero无影响也可能来自冗余路径，不能直接判定无用。

在TRAIN speaker-held-out折中，用冻结表示做小型诊断探针：预测表达控制/强度，以及内容或B0发音误差；与仅prosody、情感/强度/身份条件基线比较，按说话人和句子划分，防重复句记忆。不把探针加进训练，不用validation拟合校准。特征相关只能定位，不能单独证明因果泄露。

### D0c：配对适用性审计

当前safe manifest是source→neutral teacher，代码检查`teacher_eligible`、`mouth_event_gate`、native teacher mask及局部event mask。**这些证据不自动授权任意情感↔情感、跨身份或强度交换配对**。

只在TRAIN建立新配对索引；锁定同一句文本、说话人、情感/强度、两个独立clip ID、对齐映射、有效帧/通道与artifact SHA。sentence_id相同必须核实确实同文本，不能凭编号拼配。非高质量配对不做cross reconstruction；原生单clip自监督仍可用自身同步GT。

保留目标clip的原生时钟：只把供体条件按已经批准的映射投到目标时钟；不扭曲生成结果再算native指标。逐区域mask只限定**监督可信度**，不关闭前向嘴部通道。不通过A→neutral和B→neutral间接推定A↔B也是优质对齐。先报告可用配对数、每类/强度/身份覆盖及拒绝原因，再决定交换训练是否有足够数据。

## 4. D1：772D学生的匹配迁移基线（首批拟实现）

先保留现有模块/全嘴残差/全部损失，只改变student输入，避免把输入纠偏与新teacher、新loss一起混杂。

| 项目 | 固定方案 |
|---|---|
| 起点 | 每seed同一Phase26 standardized12最终权重；历史1540D继续训练对照与772D迁移候选，从该起点重新建立相同optimizer/预算 |
| 训练输入 | 全12536 TRAIN原生query；候选只emotion2vec768＋prosody4；独立身份enrollment与HuBERT→B0/h0保留 |
| 迁移 | 显式`--allow-audio-input-migration`；选同一TRAIN mean/std，删除input.weight前768列；其余可匹配权重逐位相同；不能宣称截列后与正常legacy输出等价 |
| 冻结 | B0、identity、motion teacher、既有TRAIN flow坐标/源统计；仅student/renderer更新 |
| 目标 | 原flow、情感/强度、global蒸馏、生成语义项不变；无新增loss，dropout=0 |
| smoke | 真实TRAIN两update；检查772输入隔离、正确非零梯度、冻结state、有效时钟/mask、损失finite、私有随机流与对照样本/noise一致；不因不同参数宽度改变RNG轨迹 |
| pilot | 固定seed47/48/49，2轮/1568更新，batch16，固定final；三seed都跑，不挑最优seed |
| 扩大预算 | pilot后依联合规则决定；若批准固定12轮/9408更新，必须从共同初始状态训练到预定final12，不择epoch，不能把pilot追加12轮写成同预算12轮 |

这是旧teacher坐标下的输入基线，**不是“干净情感teacher”最终方案**。旧teacher和native B0可能带内容/发音误差，D1不能消除全部污染。若772指标暂时退步，保留这一事实，不回用1540成绩冒充新架构正确；不要以新loss掩盖输入单变量结果。

## 5. D2：明确逐帧情感目标，优先于加大网络（条件候选，尚未授权实现）

只有D0证实目标覆盖/可信度后，选**一种**机制，不同时做下述所有方案：

**首选：复用motion teacher的时序主干，增加小型逐帧表达读出。** 目标是可解释的情感姿态/动态和强度变化，不是52维全残差，也不是jaw高频发音误差。从TRAIN的眉眼、嘴角/脸颊等表达控制及高质量同句配对提取可信目标；嘴部张口幅度目标必须剥离内容/速度变化后验证，无法可靠分离就不作为伪标签。控制维数和读出具体结构在目标审计后固定，不能先宣称任意8/64维残差是“纯情感”。

motion teacher阶段先学该表达空间并通过重建/交换干预验证，再冻结它让audio学生学逐帧表达目标。使用MEAD的1/2/3作为clip级强度，不复制成“每帧真实强度”；帧级伪目标独立记录来源和可信mask。中性level0是参照，不强制中性没有韵律、眨眼或个人习惯。

student的global/`u_a`训练信号应显式针对表达目标。可比较阻断**全残差flow→情感学生**的梯度、让该学生由表达蒸馏训练，renderer仍用全残差flow；这改变梯度职责，应独立于交换训练作单变量候选。新增逐帧蒸馏优先替换含污染风险的监督通路，而不是叠加多套critic/对比/统计/ordinal权重。

这一方案不在推理时硬遮蔽嘴部；renderer仍可用情感、强度、身份调节jaw/lip幅度。teacher质量靠真实表达重建、受控交换和speaker-held-out泛化验证，不能只看teacher分类F1。

## 6. D3：高质量交换重建，用原目标增加辨识性（条件候选）

借鉴交换思想，但不把“同情感”误当整段动态相同。

1. **同句、同人、不同情感/强度的批准配对**：内容供体条件映射到目标原生时钟，目标情感/global/动态条件来自对应目标情感片段，用其已观察动作作cross reconstruction。交换后应跟内容供体的音素顺序、跟情感条件的表情与幅度。与匹配self路径用同一flow/reconstruction目标；总预算/有效frame数匹配，先不新增cycle/GRL/MI损失。
2. **同情感、不同句子**：可检验global的语义交换/内容鲁棒性；`u_a`包含合法韵律变化，不能整段交换后要求复原任意一个原GT。没有对应content＋expression动态＋identity目标时只做干预/分布评价，不能伪造逐帧交叉GT。
3. **不同身份**：只有真实同内容同情感且时间映射可靠的跨人目标才做GT交叉重建。否则做参考干预与同人异句一致性，不能将原说话人GT当成供体身份的正确GT。

同句交换可以减少内容/情感共适应，但单独不能证明逐帧情感码无内容；需同时看D0探针和跨句干预。循环重建可以存在全部信息藏入某一路径的退化解；仅在交换已有证据而仍有具体失败时，另行提出cycle实验。

## 7. D4：仅在证据支持时改嘴部接收机制（条件候选）

若D1/D2后中性jaw仍被随机残差破坏，可对比“内容驱动的发音载体＋情感/身份驱动的幅度与表达偏置”通路：内容状态只进renderer，情感学生仍不读HuBERT；所有嘴通道保留可调幅度/姿态，不恢复固定mouth mask。

这是一个待验证的结构候选，不先宣称能完全保持时序。即使没有显式time warp，时变gain和offset也会改变检测到的峰值/闭口事件，必须做事件时序、嘴范围和几何检查。B0自身相关仅约.47，不能把其所有错误强制保留。与自由残差的单变量对照不再同时加内容保持/ordinal/probe等多套loss；没有明确残差归因时不实现。

强度不要求“所有情感、所有嘴通道随level单调变大”：强angry可能压唇，强sad也不一定大张口。先在同人同句已对齐GT上核对类别特定幅度/姿态响应；若存在可靠单调关系才定义对应目标，不能用全嘴RMS强制替代真实表情。

## 8. 指标、渲染与接纳规则

- 全1367 validation、三训练seed和三固定draw、native时钟、相同mask/rig/噪声政策、12Euler；raw/clip两套同时报。不得平均draw部署、挑epoch/seed/draw或修改probe追分。
- 主指标：MBE/LBE、Lip mean/max、Expression mean、固定rig的FDD；动态：mouth displacement、jaw centered correlation、开闭事件时间、逐类/强度范围；情感：两原probe宏F1与每类F1，辅助probe单列。加表达控制/强度误差须给定义，不悄悄替代已有指标。
- 因果干预：identity code/bias和global/intensity/`u_a`分别固定或替换。各seed单列；配对差及区间按speaker报告。validation只有3位speaker，区间和泛化结论有限；TRAIN-LOSO补充定位，不能冒充sealed泛化。
- D1输入基线的功能验收是隔离/梯度/冻结正确，性能不佳也要保留。扩大训练预算前，固定三seed应在MBE或mouth动态至少一项取得配对改善，两套原clip F1及LBE/几何无明确退步，同时neutral/fear等弱类没有新崩溃。若门槛失败，不扫描权重/预算，先回到明确的监督诊断。
- 最终推广要求内容/身份干预符合设计、嘴范围和时序均可接受、两套原clip F1≥.7且raw如实报告，几何/表达/动态联合改善，所有固定seed通过既有联合gate；原gate定义不因新结果修改。MBE约≤.7是工程目标，SOTA要求同协议击败实际对比方法，不能由绝对阈值认证。
- 预登记8类固定渲染（neutral为0，其余展示level覆盖）以及同内容同情感换身份的网格视频；同时显示GT/B0/final和有效帧。t-SNE联合展示真实/生成嵌入、按情感与身份分别着色，固定算法/seed，不挑图；聚类美观不证明解耦或物理逼真。

## 9. 论文依据与公平比较

**已直接核对的一手来源：**

- MEDTalk，arXiv v4，`https://arxiv.org/html/2507.06071v4#S3.SS2`：动作内容/情感编码、self/overlap/cycle交换重建；其配对动作来自EmoFace合成，同内容不同情感可精确对齐，不能把它的数据条件直接等同于MEAD真实DTW配对。Sec.3.3使用动作controller伪强度和emotion2vec＋文本预测动态强度，再映射到冻结空间。
- MEDTalk Sec.4.2/4.3：MLE、MEE、EIE、FRD及t-SNE/交换可视化、用户评估；不是以本项目的独立macro-F1为其唯一情感标准。174D MetaHuman rig与我们52D ARKit/固定632vertex rig不能直接比较论文数值。其模型并非表格每项都最优，不能将“全指标第一”当作论文发表唯一标准。
- EmoTalk，CVF ICCV2023官方摘要/来源：`https://openaccess.thecvf.com/content/ICCV2023/html/Peng_EmoTalk_Speech-Driven_Emotional_Disentanglement_for_3D_Face_Animation_ICCV_2023_paper.html`：cross-reconstruction分离语音情感/内容，decoder结合身份/情感/内容；摘要支持因素交换方向。本文没有将尚未逐条核对的其全部loss和评价细节写成事实。
- DESTalker 已找到题名“Disentangling Emotion and Style for Expressive 3D Facial Animation via Residual Generation”的索引，但原文入口此次不可读取；**尚未核实其嘴部mask、具体loss或F1/t-SNE协议**。不把此前AI关于它的说法作为设计依据；后续原文可用时再补方法核验，不需要为此停止已有方案。

当前VOCA-core、EmoTalk-core、FaceFormer、FaceDiffuser是明确标识的适配版；后者部分条件由冻结KineTalk提供，不能冒充官方DeSTalker/MedTalk复现。新学生772D后对比要分别说明各方法真实音频输入/预训练器与参考信息，不强行改动基线后仍称官方。

同协议适配版EmoTalk-core已有MBE .7450、LBE .3315、Lip mean/max 2.973/5.563mm；FaceDiffuser适配版LBE .3231、Lip mean/max 3.072/5.864mm、absolute FDD137.962mm²。这些是具体差距参考，不是新模型成绩。最终需同split/数据预算/身份参考/输出空间/指标公式复现方法，锁模型后一次sealed最终评估；本轮不读sealed目标或成绩。

## 10. 待批准修改范围和执行顺序

**建议首批只批准D0＋D1：**

1. 新增冻结干预/配对审计及对应真实smoke helper、报告契约和固定渲染；不改模型层数或训练loss。
2. 复用现有772D迁移入口，补匹配实验配置/私有随机流核验，真实两update smoke，然后三seed两轮pilot与全validation联合评价。
3. 把identity和`u_a`诊断证据、772D结果写入本页/状态页，明确下一项是teacher表达监督、质量配对交换还是嘴部接收机制。

D2/D3/D4是完整待办和决策条件，**不因批准首批就自动全部实现**。届时先固定唯一结构、数据目标、梯度路径、预算与验证，再交用户确认。未经当前用户明确同意，本轮仅修改文档，不新增这些训练实现、不启动训练、不替换默认权重。
