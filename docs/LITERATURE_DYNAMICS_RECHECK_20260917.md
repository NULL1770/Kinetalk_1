# 眉眼动态：MEDTalk、DEITalk、SubtleTalk 原文与源码复核

日期：2026-09-17。本次重新取得 MEDTalk v2、SubtleTalk v1 的 arXiv 正文、SubtleTalk LaTeX 源包，使用 GitHub API 重新读取官方仓库树和 MEDTalk 源码；没有直接采用昨天文档作为文献事实。本文不评价本轮新训练的数值结果，也没有运行或改写模型。

同日执行状态：本文第6节提出的最终输出velocity/std对照已由主任务实施，两臂各8epoch并完成独立审计，未通过动态及输出保护验收。第6节保留为实验提出时的依据；最新结果见`OUTPUT_MOTION_DYNAMICS_RESULTS_20260917.md`，后续路线见`BROW_DYNAMICS_RESEARCH_AND_ACTION_20260917.md`。

**直接回答：MEDTalk 确有全脸 rig 数值重建和最终输出强度监督；SubtleTalk 确有最终眉、眼区域运动的速度及标准差监督。它们的动态不只是“把音频 latent 学好以后自然出现”。但两者监督、输入协议和生成方式不同，不能把“再加一个眉毛 loss”当成已证明足够的修复。** 当前值得补的是部署条件下的完整生成训练与输出动态约束，同时检查监督和显示链；这是可验证的效果基线，不是直接成立的论文创新。

## 1. 可核实来源与边界

| 方法 | 本轮获取的第一手来源 | 可复核边界 |
|---|---|---|
| MEDTalk | [论文 v2](https://arxiv.org/html/2507.06071v2)、[官方代码](https://github.com/SJTU-Lucy/MEDTalk/tree/97df2c75408c9b82cbdfac1bb76a022d7ac05a43) | Git tree commit `97df2c75408c9b82cbdfac1bb76a022d7ac05a43`；有模型、训练代码。论文没有完整公开测试清单与本任务可直接对齐的数据划分 |
| DEITalk | [ACM DOI](https://doi.org/10.1145/3664647.3681359)、[作者仓库](https://github.com/KangShen-seu/DEITalk/tree/c17dbf6336dffac18397742951808ca6fc221709)、Crossref / Semantic Scholar 元数据与作者摘要 | 正确题名为 **DEITalk: Speech-Driven 3D Facial Animation with Dynamic Emotional Intensity Modeling**，ACM MM 2024；全文/PDF 本轮仍为 403，仓库 `code`、`dataset` 各 1 字节占位。不能声称核实其四项 loss 细则 |
| SubtleTalk | [论文 v1](https://arxiv.org/html/2608.06408v1)、[LaTeX 源包](https://arxiv.org/src/2608.06408v1)、[官方仓库](https://github.com/molly-ding/SubtleTalk/tree/5ea4de0f28797ef3754885d1516304eedf3c84ae)、[项目页](https://molly-ding.github.io/SubtleTalk/) | commit `5ea4de0f28797ef3754885d1516304eedf3c84ae` 仅有 README，注明 Code and Dataset Coming soon；源包也未包含正文提及的补充材料。具体 λ、mask、std reduction、训练重建方式仍不可核 |

本轮获取的正文/仓库 API/源码证据缓存于 `artifacts/literature_recheck_20260917/`。网页原文中的章节和公式编号是本报告定位依据，未用推测的 PDF 页码。

## 2. MEDTalk：直接 rig 重建 + 眉眼嘴角强度，不是随机残差生成

来源：[Methods / Dynamic Emotional Facial Animation](https://arxiv.org/html/2507.06071v2#Sx3.SSx3)，Eq.5–12；[Implementation Details](https://arxiv.org/html/2507.06071v2#Sx4.SSx1.SSS0.Px2)。

### 实际生成路径

1. 用 motion 的 content/emotion encoder 与 decoder 进行自重建、交换重建、循环重建；监督是 **174 维 MetaHuman rig 的 MSE**。这一步的 decoder 在后续阶段冻结。
2. wav2vec2 音频内容映射到 motion content embedding；训练不只有 embedding similarity，也通过冻结 decoder 重建最终 motion。
3. 使用 **外部情感 label** 的 embedding，再由 emotion2vec 和自动转录文本的 RoBERTa 特征预测逐帧强度。强度改变情感 embedding 的范数，经可训练 fusion encoder 映射后进入 decoder。
4. FIM 阶段依然通过最终输出 motion 监督，既不是只蒸馏 teacher latent，也不是在固定 ARKit52 表情 ray 上乘标量。

Eq.8：对选定控制器取逐帧 L1 范数作为强度。Eq.9：`f_t = I_hat_t * f_label / ||f_label||`；此处调制的是**特征空间**，后面还有学习映射和 decoder。

Eq.11–12：

```text
L_int = || Int(R_hat) - Int(R) ||_2
L_FIM = L_recon + λ_sim L_sim + λ_int L_int
λ_sim = λ_int = 0.1
```

论文 Eq.11 记作 L2，官方代码具体使用 `F.mse_loss`，不能把公式排版和代码 reduction 混为一谈。

### 官方代码能确证什么

来源：[audio_semantic.py L22–28](https://github.com/SJTU-Lucy/MEDTalk/blob/97df2c75408c9b82cbdfac1bb76a022d7ac05a43/models/audio_semantic.py#L22)、[L130–147](https://github.com/SJTU-Lucy/MEDTalk/blob/97df2c75408c9b82cbdfac1bb76a022d7ac05a43/models/audio_semantic.py#L130)、[rig 名单](https://github.com/SJTU-Lucy/MEDTalk/blob/97df2c75408c9b82cbdfac1bb76a022d7ac05a43/visualization/metahuman_attr_names.txt)。

`get_intensity` 使用 14 个控制器取绝对值求和，再乘 0.1：

| 区域 | 控制器（左右两侧） | 维数 |
|---|---|---:|
| 眉 | brow_down、brow_lateral、brow_raiseIn、brow_raiseOut | 8 |
| 眼 | eye_blink、eye_squintInner | 4 |
| 嘴角 | mouth_cornerPull | 2 |

L141–147 为 `MSE(Int(x_recon), Int(label)) + MSE(x_recon,label) + embedding cosine`。**强度 loss 落在最终生成结果，不是 `pred_intensity` 与标量标签直接做 MSE。** 因而强度相同但方向错误的眉动作，仍会受到全 rig MSE 约束。强度一项本身不能识别抬眉/压眉方向，更不能单独保证眼眉事件时刻正确。

公开推理实现没有随机 noise latent；给定音频、情感 label 和文本处理后是确定性输出。`validate` 另使用 window=5、polyorder=2 的 Savitzky–Golay 后处理；60fps 输出。它不是证明随机一对多模型一定必要的例子。

### 不能直接迁移的部分

- 174 rig 索引不可拷贝到 ARKit52；本项目必须按实际 channel 名与 rig 映射重新定义可解释 proxy。
- label/image/text guidance 与本项目纯 audio 自动情感协议不同；自动转录的 speech text 与用户输入的情感描述也不同。
- MEDTalk 的 decoder 虽然冻结，但它经过对应 motion 的完整重建训练，FIM 的可训练映射较完整，并受最终输出监督；不能据此为长期仅更新 512 参数接口辩护，也不能据此断言冻结 decoder 总是错误。
- 论文未给出足以核验总训练小时和独立句子/身份划分的完整材料。本项目不能宣称仅补文本就等价复现。

## 3. SubtleTalk：眉眼区域速度与幅度监督，兼有显式控制和随机 flow

来源：[§3.2](https://arxiv.org/html/2608.06408v1#S3.SS2)、[§3.3 Eq.2–10](https://arxiv.org/html/2608.06408v1#S3.SS3)、[§3.4](https://arxiv.org/html/2608.06408v1#S3.SS4)。

### 条件和监督表示

- FLAME：300 维 identity shape、50 维 expression、pose；运动映射到 5023 个三维顶点。
- 音频：冻结 WavLM 末层 content，加第 3–11 层可学习融合与多尺度时域卷积，再加 F0/log-energy。这些特征直接进入 residual flow 的可训练生成路径。
- 每帧 VA 训练条件来自视觉 EmotiEffLib 估计；VADP 在推理时从 emotion2vec+TCN 预测 VA，可选用户锚点。视觉 VA 是 pseudo label，不等于真心理状态。
- 5 个窗口强度：eye、brow、headX/Y/Z。眉眼来自 **FLAME 区域顶点轨迹的时间标准差**，头来自各 pose 轴的标准差，按数据统计归一化后作为 global condition。
- motion context：前 10 帧加本次 100 帧预测窗；推理滑窗使用以前生成的 motion 作为后续 context。生成器使用 RoPE 与局部因果 cross-attention mask。

### 两阶段的 loss 不是同一回事

DMP（确定性基础动作）用 face/lip 顶点 L1 重建和一阶速度平方 L2，Eq.2–4。其输出包含 expression 与 jaw-opening，头 pose 补零；因此它不是一个严格仅输出嘴部通道的头。

RFM（随机残差）先标准化真实 `motion - prior`，从高斯 noise 经 flow 学习残差分布，Eq.5–6：

```text
x_t = (1-t) x_0 + t x_1,  x_0 ~ N(0,I)
L_FM = E || Vθ(x_t,t,c) - (x_1-x_0) ||²
```

然后明确对重建的 `M_hat = prior + residual_hat` 加 **实际运动轨迹** 的辅助约束，Eq.7–10：

```text
L_vel    = Σ[r∈face,lips,brows,eyes,head] λ_r^vel ||ΔS_hat_r - ΔS_r||²
L_smooth = Σ[r∈face,head] λ_r^smooth ||ΔS_hat_r(t) - ΔS_hat_r(t-1)||²
L_std    = Σ[r∈brows,eyes,head] λ_r^std ||std_t(S_hat_r) - std_t(S_r)||²
L_RFM    = L_FM + L_vel + L_smooth + L_std
```

这里 Δ 是相邻**视频帧**的一阶差分，与生成 flow time 的向量场速度不是同一个概念；所有三项写为平方 L2。眉眼的 S 是几何区域轨迹，头的 S 是 pose 轴轨迹。std 按时间维计算；公式形式指向各轨迹坐标时间 std 后比较，而不是明确规定先把整个区域压成一个 scalar 再比较。

**论文的明确策略是避免对随机输出加确定性 endpoint regression，保留 flow 并约束动态。** 这对当前 rollout 中心化 L1 实验有参考价值：把每个 noise seed 都拉向单条真实轨迹，可能把不可预测细节拉向条件中心。不过它的 GT velocity loss 仍会惩罚时刻不同的有效随机动作，论文措辞并不能证明这些辅助约束绝对不会压低随机性。

### 无法从公开材料复原的细节

已查正文、源包、项目页、官方仓库，以下均无可核实现或明确公式：

1. `residual_hat` 是从含 GT 插值的 `x_t` 做一步 endpoint 估计，还是从独立噪声完整 ODE rollout。正文只写“reconstructed motion”，**不能替作者选一种并声称复现**。
2. 各 λ 的数值、warmup/IDC 切换步数、ODE solver/推理步数。
3. std 的 `unbiased/ddof`、顶点坐标/区域 reduction、mask、标准化前后应用位置及 vertex region masks。
4. Table 3 测试时 VA 是 GT 还是 VADP，区域强度来自什么默认策略/预测器。论文明确支持 audio-only（Figure 1 和 §4.5 用户研究），但不能把 Table 3 每一条自动数值都标成“已核无 GT 辅助条件”。

IDC 是另一层控制训练：固定 noise/其他状态，随机缩放一个强度维度，以目标强度 loss 控制该因素，并用 stop-gradient raw branch 约束其他因素不变，Eq.11–16。本项目若只需要 audio 自动动态，尚不必把 IDC 五个可控因子全部加进来。

### 规模与数据质量不是背景细节

§4.1/Table 1：train **29,578 clips、2,456 identities、59.76h**；test **3,301 clips、725 identities、6.67h**。总量 36,733 clips、3,905 identities、73.83h；各来源内身份隔离。源视频包括 MEAD 及四类更自然数据来源。

监督是 TEASER 提取的 FLAME motion，另有音画同步、yaw 极值/遮挡筛查与去抖。弱眉眼信号既靠模型，也依赖能保留眉眼的提取器。当前 tracker/native 数值自洽，并不能替代原视频中眉峰与拟合 motion 的视觉核验。

## 4. DEITalk：已核摘要，不编造 loss

作者摘要由 [Semantic Scholar 元数据接口](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3664647.3681359?fields=title,abstract,openAccessPdf,url) 返回，Crossref 核对题名/作者/DOI。摘要明确：

- 五类情感、MetaHuman coefficients；输入 **speech + emotional style labels**。
- DEI 模块从 speech features 提取隐式强度局部表示；DPE 针对长序列泛化。
- emotion-guided feature fusion decoder 与 four-way loss。

但本轮没有取得正文，仓库没有代码，OpenAlex / Semantic Scholar 均未提供开放 PDF。**四项 loss 究竟是哪四项、有没有眉眼系数/速度 loss、数据小时、随机 latent、各指标/消融数值，均不能确认。** 这不足以作为实施特定 loss 的出处；应以可核的 MEDTalk/SubtleTalk 作为当前补齐依据。

## 5. 论文效果证明了什么，没证明什么

MEDTalk [Table 1 / Eq.15–18](https://arxiv.org/html/2507.06071v2#Sx4.SSx2)：EIE 是逐帧强度 L1 误差，可看强度时序；FRD 比上脸 rig 的时间标准差，反转帧序不改变 std。MEDTalk 的 EIE=0.79055，为表内第二；FRD=0.00289，为第三，不能称动态全部指标最优。Table 3 完整版 EIE/FRD=0.79055/0.00289，去 intensity 为0.86488/0.00753，去 text 为0.83899/0.00757；这些是该模型内模块贡献，不是“所有方法必须文本”的证明。

SubtleTalk [Table 3 / Table 5](https://arxiv.org/html/2608.06408v1#S4.SS3)：FDD/HDD 比运动变化统计，LVE 为唇部误差。其主要消融如下，保留原表报告尺度（FDD/HDD ×10^-3，LVE ×10^-4）：

| 模型 | FDD↓ | HDD↓ | LVE↓ |
|---|---:|---:|---:|
| DMP only | 15.25 | N/A | **11.35** |
| FM only | 12.47 | 27.34 | 14.60 |
| FM+DMP | 11.44 | 25.57 | 12.46 |
| FM+MultiCond | 8.02 | 12.58 | 15.05 |
| FM+MultiCond+IDC | 5.78 | 7.93 | 14.51 |
| FM+DMP+MultiCond | 5.21 | 6.96 | 12.59 |
| Full | **4.46** | **6.61** | 11.96 |

MultiCond 同时改变 prosody、VA、regional intensity；**没有单独 velocity/std loss 消融**。因此不能根据该表断言其中某一个 loss 或 VA 单独解释全部收益。完整模型相对 DMP 唇误差略大，展示了动态与口型需要联合验收。§4.5 的 27 人、10 条域外音频用户研究明确只用 audio，给出了自然度证据，但不是单个眉峰的逐帧预测证明。

本项目 ARKit52 中心化残差 R²、FLAME FDD 与174-rig EIE不在同一表示/数据/输入协议，不能直接报差距百分比。随机模型单 GT 下负 R²不能单独否定自然运动；相反，std 接近 GT 也不能证明与音频相关或没有乱动。

## 6. 对现有实现的具体差距与最低限度补齐

阅读了 `scripts/train_audio_conditioned_flow_probe.py`、`scripts/train_renderer_capacity_probe.py`、`scripts/train_direct_audio_dynamics.py`、`kinetalk_b0/models/neutral_affect.py`、`kinetalk_b0/losses.py`；本节是本地代码事实，远端新产物由主审查另核。

### 不能再混淆三种“velocity/监督”

1. 最新 flow probe 与 capacity probe：优化 `prediction - velocity_target` 的 flow MSE，本身没有最终眉眼 std 或完整 rollout 动态约束。
2. 旧 `stage2_loss`：对 flow 向量场 `prediction` 与 `velocity_target` 取视频帧差再 L1。它不是几何眉区 loss，但也不能说与动作速度毫无关系：共享 scalar flow time 时，一步 endpoint 的时间差误差等于 `(1-t) * residual_scale * Δ(v_pred-v_target)`。所以它可解释为一步 endpoint 时序误差的时间/尺度重加权形式，**仍不同于噪声完整 rollout 的最终输出约束**。
3. 新 direct_audio 的 dynamics 臂：有独立噪声12步 rollout 后的眉眼中心化 L1；输入已是共享直接声学条件，不应再用“只有 ridge8 条件”描述该臂。但这是**单GT轨迹回归**，不是 SubtleTalk 的 velocity+std 方案，也不是其已公开的完整复现。

### 下一臂应回答一个明确问题

在新 direct_audio 相同 encoder、完整 renderer、初始化、batch/noise/预算下，比较现有 flow、flow+centered-L1，再增加 **flow+最终输出 velocity+std** 的独立臂：检验动态幅度约束能否在不过度拉向单GT轨迹的情况下保留可见、合理的眉眼变化。

实施边界应写清楚：

- 保持一个52维 residual generator，区域 mask 仅用于训练/评价，不必新增眉/眼/嘴独立 decoder。
- velocity 对 `generated_motion - target_motion` 的相邻帧差用平方误差；std 对每段、每个 observed 通道的时间 std 用平方误差；使用相同训练集尺度归一化。眉眼所用通道要逐名确认，特别区分眼球朝向、blink、squint、wide。
- 缺失帧/通道按观察 mask 处理；相邻 velocity 仅两帧均有效时参与；std 明确定义 `ddof=0`、最少有效帧数和 reduction。这些是本项目定义，不冒充论文原实现。
- 明确采用部署 noise rollout 还是一步重建。若沿用现有12步 rollout，就清楚命名为本项目的 rollout 动态约束。省算力的一步估计可另作对照，但带 GT 的 `x_t` loss 降低不能替代部署结果。
- 每个 λ 单独记录与标定；用训练数据统计设尺度，并查看实际梯度量级。不要为了“看起来会动”从 dev/test 中挑最好权重。
- std loss 只匹配幅度，无法防止时间反转、相位错位或高频抖动。需要速度/频谱/事件统计和固定noise音频反转/平移对照；平滑过强又会压掉短促眉峰。
- 同时检查多seed naturalness与 diversity。对单GT速度精确配对也存在随机性压缩风险，应测 energy score/variogram 与 seed 方差，不预设该臂一定胜出。

此实验是最小可归因的基线修复。更充分的后续方案还包括未被动作均值投影丢弃的声学事件特征、音频可预测的强度/VA条件、训练身份/句子/自然数据覆盖、上脸视觉监督质量；不能把这些互相替代。

## 7. 投稿与可宣称结论

“CCF-C 能不能发”没有仅按一段 demo 判断的统一门槛。已有身份/口型/global 模块可复用，但不能预先免检：52维 residual 仍可修改嘴，身份编码的静态检索不等于行为风格泛化，训练情感 teacher 打分也不等于独立感知评价。

如果论文主张是细粒度眉眼动态，目前至少应形成：GT 在实际rig上看得见、audio-only 多seed最终生成有自然动态、正确音频条件比时间破坏/匹配无local对照更有效、嘴和身份/global不劣化、独立句子/身份测试与公平baseline、明确的新机制而非仅补完整训练。借鉴上述直接监督是合理工作，但只有实测增益与新颖性证据齐备才能形成论文贡献。
