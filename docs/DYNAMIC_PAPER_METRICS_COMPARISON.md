# 动态指标与输入协议：原文核验

核验日期：2026-09-16。重新下载 MEDTalk v2、SubtleTalk v1 官方 HTML，与本地缓存逐字节一致；另检查官方源码。**两篇都评价动态，但评价对象不同；目前不能给出“我们的数值比它们差多少”。** 我们的 ARKit52 中心化残差 R²，与 MetaHuman rig 强度误差、FLAME 顶点动态偏差既不同单位，也不同测试集和条件。

## 1. MEDTalk：确有逐帧强度误差，但不是全部动态指标最优

来源：[原文 v2](https://arxiv.org/html/2507.06071v2)，Methods / Fusion Intensity Modeling，Eq.8–12；Experiments / Quantitative Evaluation，Eq.15–18、Table 1–3。

- **输入**：audio + 外部 emotion label，或用参考图/情感描述替代 label（Eq.1）。另外，音频通过 Whisper 自动转录，RoBERTa 提取说话文本特征，与 emotion2vec 融合预测逐帧强度。情感描述提示与自动转录的说话文本是两种不同输入。不能把其 label 引导结果当作我们的纯音频情感预测结果。
- **表示/数据**：174维 MetaHuman controller rig；动态训练使用 EmoFace 数据，七类情感。RAVDESS 用于多模态引导。基线重训到174维，并把 style code 改为情感标签 embedding。正文未充分给出 Table 1 的测试片段清单、句子/身份隔离规模，不能推定与我们的划分相同。
- **强度**：Eq.8 为选定 rig 子集的逐帧 L1 范数 `Int(R_t)`；Eq.17 EIE 是生成/真实强度序列的 L1 误差，**对强度时序对齐敏感**，但不等于每个眉眼动作都还原正确，也没有扣除 B0 或 neutral identity。
- **FRD，不是 FDD**：Eq.18 计算上脸各 rig 时间标准差之差再平均。印刷公式是有符号 `mean_j(std_t(R_j)-std_t(pred_j))`，没有绝对值；不能自行补绝对值后冒充原指标。它测幅度统计，反转帧序不改变标准差，不能证明事件时刻准确。
- **MEE** 是情感区域 rig 数值的平均 L1 误差，不是分类准确率。Table 2 的 “Emotion Acc” 则是42人对10条测试音频的五级主观评分，MEDTalk **4.13±0.09**、GT **4.39±0.06**（95% CI），不是百分比。

Table 1 原表数值；均越低越好，无额外 `10^-k` 表头倍率，单位为论文 rig/强度标尺：

| 方法 | MLE | MEE | EIE | FRD |
|---|---:|---:|---:|---:|
| FaceFormer | 0.00662 | 0.00952 | **0.69221** | 0.00618 |
| EmoTalk | 0.00756 | 0.02303 | 0.92046 | 0.00116 |
| EmoFace | 0.00651 | 0.02072 | 0.88574 | 0.00507 |
| DiffPoseTalk | 0.01525 | 0.01664 | 0.89151 | **0.00075** |
| MEDTalk | **0.00596** | **0.00906** | 0.79055 | 0.00289 |

因此 MEDTalk EIE 第二、FRD 第三，不能写成“动态指标全面最好”。Table 3 同协议消融：去文本 EIE/FRD **0.83899/0.00757**，去动态强度 **0.86488/0.00753**，完整 **0.79055/0.00289**。这些支持其模型中相应组件有用，不能证明文本是所有方法的必要条件，也不能据此算我们缺文本造成的差距。

官方实现已发布：[SJTU-Lucy/MEDTalk](https://github.com/SJTU-Lucy/MEDTalk)，核验 commit `97df2c75408c9b82cbdfac1bb76a022d7ac05a43`。[audio_semantic.py L22–28](https://github.com/SJTU-Lucy/MEDTalk/blob/97df2c75408c9b82cbdfac1bb76a022d7ac05a43/models/audio_semantic.py#L22) 实际取14个174-rig通道的 L1 和乘 `0.1`；L141–143 约束**最终输出**的强度与GT强度 MSE。[demo_label.py](https://github.com/SJTU-Lucy/MEDTalk/blob/97df2c75408c9b82cbdfac1bb76a022d7ac05a43/demo_label.py) 显式接收 `--emotion`；实现转录指定 Chinese，输出60fps。当前公开树未见 Table 1 评价脚本及完整测试划分；不能把训练 helper 的缩放直接当作已复现的 EIE evaluator。

## 2. SubtleTalk：主要量化运动幅度统计，并有纯音频用户研究

来源：[原文 v1](https://arxiv.org/html/2608.06408v1)，§3.1–3.4、§4.1–4.5、Table 1/3/4/5。

- **输入/监督**：WavLM末层与3–11层特征、F0/log-energy；训练读取视觉估计的逐帧VA，以及GT窗口眼/眉/头XYZ的五个强度统计。shape `β` 和历史motion context也是条件。VADP可在推理时从audio预测VA，可选用户锚点；并非“音频直接预测每一帧的全部动作标签”。
- **协议边界**：正文支持自动音频推理，§4.5 明确用户研究只输入audio（27人，10条域外音频）。但 **Table 3 没有明确交代VA使用GT还是VADP，五个强度的推理来源/默认值也未充分说明**。因此不把Table 3数字标成已核实的“纯音频、不含GT辅助条件”成绩，也不反过来断言它使用了GT。
- **数据/表示**：FLAME 5023顶点、300维身份shape、50维expression及pose；25fps。Table 1：训练 **29,578片段/2,456身份/59.76小时**，测试 **3,301/725/6.67小时**，在各源数据集内身份隔离。TEASER拟合、同步/姿态筛选、时序去抖，EmotiEffLib提供VA伪标签。
- **定量并非VA预测准确率**：Table 3 为FDD、HDD、LVE，未报告VA CCC/R²或情感分类准确率。不能把低FDD当作已证明音频VA准确或眉眼事件逐帧匹配。

Table 3 原表；FDD/HDD表中数值乘 **10^-3**，LVE乘 **10^-4**；这些是报告倍率，正文没有充分说明几何单位换算，不能擅自改写成mm：

| 方法 | FDD ↓（上脸） | HDD ↓（头部） | LVE ↓（唇部） |
|---|---:|---:|---:|
| FaceFormer | 15.39 | 不支持 | 13.87 |
| FaceDiffuser | 14.54 | 不支持 | 14.30 |
| DiffPoseTalk | 9.74 | 21.71 | 13.72 |
| ARTalk | 11.37 | 26.70 | 11.98 |
| SubtleTalk | **4.46** | **6.61** | **11.96** |

§4.3把精确公式转交补充材料，当前官方仓库只有README，标注“Code and Dataset: Coming soon”。[arXiv源包](https://arxiv.org/src/2608.06408v1) 的 `sections/subsections/4_3.tex` **被注释的草稿公式**为：FDD=`mean_v ||std_t(pred V_v)-std_t(GT V_v)||_1`，HDD为头轴标准差差的L1均值，LVE为每帧最大唇顶点平方距离再时间平均。这能佐证其统计意图，**不能冒充已发布附录或可运行官方评价**；顶点mask、std约定、窗口汇总仍需正式脚本核实。此类std指标对时间反转不敏感。

Table 5：DMP-only 的 FDD/LVE **15.25/11.35**，完整 **4.46/11.96**：完整模型明显改善上脸幅度统计，却略增唇误差。MultiCond把VA、区域强度、prosody一起消融，不能仅凭该表把收益全归因于VA或某一个loss。[官方仓库](https://github.com/molly-ding/SubtleTalk)，核验 commit `5ea4de0f28797ef3754885d1516304eedf3c84ae`。

## 3. DEITalk：输入可核，定量表尚不可核

[ACM DOI](https://doi.org/10.1145/3664647.3681359)，2024 ACM Multimedia。可得作者摘要明确：MetaHuman五种情感风格，**speech + emotional style labels**，动态强度模块从speech提取隐式局部表示。它不等于无情感标签的纯audio任务。

本轮ACM正文/PDF返回403；公开元数据无开放PDF。[作者仓库](https://github.com/KangShen-seu/DEITalk) commit `c17dbf6336dffac18397742951808ca6fc221709` 只有demo压缩包、README，`code`/`dataset`为1字节占位。**未核实其原文指标公式、定量表或划分，不填数、不从MEDTalk的二手描述推断其监督细节。**

## 4. 对本项目可以怎样比较

当前代码 `audit_projection_readiness.py` 使用
`r=center(motion-B0-neutral_identity)`，同样处理预测；`R²=1-SSE/sum(r²)`，并比较audio full、同模型zero-local、reverse和motion-oracle。R²无量纲，误差在ARKit52系数空间；full-vs-zero的ΔR²还不同于绝对R²。上脸集合、数据、noise、是否扣除B0均影响结果，不能与上述任何一列表值直接相减或宣称落后百分之几。

| 能检验的内容 | 当前中心化残差R²/时间干预 | 论文强度/标准差指标 |
|---|---|---|
| 是否比无local条件更准确、时序是否有用 | 同noise zero/reverse可诊断 | 单独std差不够，反转不变 |
| 是否有足够动作幅度 | R²不能单独判断合理随机性 | std差可诊断幅度，但不保证事件对齐 |
| 情感是否被人感知、口型是否好 | 系数曲线/训练teacher不足 | MEDTalk主观评分、SubtleTalk用户研究另提供证据 |

可新增**只读评价proxy**：固定当前observed上脸通道的 `mean_c |std_t(pred_c)-std_t(GT_c)|`，以及预注册通道/标尺后的强度MAE，清楚命名为 **ARKit52 coefficient std deviation / intensity proxy**，同时保留R²、时序干预与视频检查。不能称复现FLAME FDD或174-rig EIE；没有头姿不能计算HDD。正式同名对比需相同表示/区域映射、时间采样、输入条件、数据划分及官方实现，并重跑基线。

原文说明我们应补的证据是：可靠上脸监督、幅度分布与时序同时评价、纯audio和GT控制协议分开、最终生成动作检验、独立视频感知检查。**它们不证明扩大动态维度必然有效，不证明必须加入文本，也不保证当前冻结512参数接口能达到同等结果。**
