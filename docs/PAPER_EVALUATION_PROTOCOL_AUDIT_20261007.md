# 论文评价协议核查（2026-10-07）

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent validate（inline）
- Verification Status: ANALYZED；MEDTalk/EmoTalk主文原文已获取，DESTalker全文未验证。
- 原始HTML/PDF、提取文本、URL和SHA：`final_experiment/evaluation/diagnostics/paper_protocol_audit_20261007/`。
- 范围：核评价协议，不复现论文结果；未读到全文的内容不由搜索摘要推定。

## 原文证据

| 论文 | 主文实际评价 | 对本项目的意义 |
|---|---|---|
| [MEDTalk v4](https://arxiv.org/html/2507.06071v4)，§4.2–4.4 | MLE/MEE为lip/emotion控制器平均L1；EIE为逐帧强度L1；FRD比较上脸控制器时间标准差；t-SNE、可视化、用户研究 | 没有把本项目这种generated-motion F1作为成功标准。其174D控制rig与本52D系数/632vertex rig不同，表格数值不能直接对标 |
| [EmoTalk ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Peng_EmoTalk_Speech-Driven_Emotional_Disentanglement_for_3D_Face_Animation_ICCV_2023_paper.pdf)，PDF6–7页，§4.2–4.4 | LVE：每帧唇顶点最大欧氏误差再平均；EVE：眼/额区域同类误差；另报lip平均L2、视频与用户研究 | 主文没有报告本项目这种F1。不能将本项目lip mean与论文主表lip max混称同一LVE；其5023vertex FLAME也不是本项目632vertex rig |
| [DESTalker DOI10.1002/cav.70127](https://doi.org/10.1002/cav.70127) | Crossref与DOI-bound OpenAlex均确认题名；Wiley full/full-xml各40s请求超时，OpenAlex未给开放全文 | 暂不确认其F1、t-SNE或mouth mask细节；元数据不是方法证据 |

MEDTalk原文：“To evaluate the accuracy of generated animation, we propose metrics Mean Lip Error (MLE) and Mean Emotion Error (MEE) ...”以及“EIE computes the ... error ... per frame”。EmoTalk原文：“For a single frame, LVE is defined as the maximum ... error among all lip vertices.” 因而不能把“这两篇未用F1”误说成“这两篇不用逐帧动态评价”。

主文检索与评价章节人工核查均未发现两篇使用本项目的F1；该结论限定这些已读主文，不泛化为所有论文/全部补充材料。MEDTalk强调t-SNE是嵌入可视化，不能由分群漂亮推导生成质量或解耦已证。

## 本项目如何评价

现有probe是整段动作的mean/std/q10/q90/相邻速度统计（6C）→冻结MLP→一个clip类别；它没有逐帧情感真值，不是逐帧分类器。它在真实validation GT上的原F1也仅.660344/.641175；生成比GT分数更高不代表更真实，也不能根据这点忽略生成低分。

继续保留原F1、raw/clip政策与GT分数，不删失败指标、不换判别器来获得美观分数。主要判断应联合：同数据同rig的口型/表情几何误差、原生时钟下开闭/速度及中心化相关、按情感/强度/身份的动态范围、换情感/参考时内容保持、固定视频。音频分类头或自身teacher对生成的评分与独立生成F1明确分开。

若加入MEDTalk式EIE，需要先定义本rig强度函数及TRAIN-only拟合来源；MEAD的三级是clip标签，不是每帧强度GT，不能复制成逐帧真值。用现有同rig对比方法重训/统一评估才能支持相对SOTA；不能跨rig、数据划分或误差公式比论文裸数字。本轮只核查协议，没有新增训练loss/评价器拟合。

## 统计解释边界

Phase32三seed/三draw来自同一数据划分，不能把9组当9个独立数据集；validation仅3speaker，speaker bootstrap估计范围有限。warm截列两轮不足、GT拟合/评估域差异、近恒定通道数值精度都可能影响读数，不能凭一项相关性定论因果泄漏。没有sealed读取、validation校准、挑seed/draw/视频或改变公式。
