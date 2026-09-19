# 论文指标来源、实现口径与当前可评价范围

核验日期：2026-09-19。本文用于论文实验表的指标选取和命名，不更改现有训练、选模或验收协议。**今天可以形成 ARKit52 系数空间的可复现主表与内部消融表；不能把这些表与 MetaHuman / FLAME 原论文数值拼接排名，也不能把冻结口型、身份和情感网络等同于三项质量已经通过。**

本轮重新请求 MEDTalk v2、SubtleTalk v1 官方 HTML，均返回 200，逐字节与 9 月 16 日缓存一致；重新查询三个作者仓库，commit 也未变化。ACM DEITalk 页面返回 403。下载时间、URL、状态、SHA256 和仓库文件清单存于 `artifacts/paper_metrics_20260919/sources/provenance.json`；正文摘录和缓存一致性存于同目录。以下公式以可访问原文为依据，没有用其他论文的描述补全 DEITalk。

## 1. 三篇论文实际测量什么

| 论文 | 生成任务与表示 | 输入条件 | 数据/划分 | 原生客观指标及来源 | 评价实现可用性 |
|---|---|---|---|---|---|
| MEDTalk | 174 维 MetaHuman controller rig；七种情感 | speech + 外部 emotion label，或以图像/情感描述代替 label；另有 Whisper 转写的语句文本 | EmoFace 数据用于音频表达训练；RAVDESS 用于多模态引导；正文未给足以复刻 Table 1 的片段清单、身份/句子隔离细节 | MLE、MEE、EIE、FRD；[Quantitative Evaluation, Eq.15–18, Table 1](https://arxiv.org/html/2507.06071v2#Sx4.SSx2) | [作者仓库](https://github.com/SJTU-Lucy/MEDTalk/tree/97df2c75408c9b82cbdfac1bb76a022d7ac05a43) 有训练/推理，公开树中未见 Table 1 评价脚本或完整测试 manifest |
| SubtleTalk | FLAME 5023 顶点、300 维身份 shape、50 维 expression 及 pose；25 fps | 多层 WavLM、F0/energy；VA、区域活动强度、shape、历史 motion；有 audio→VA 预测器 | [§4.1 / Table 1](https://arxiv.org/html/2608.06408v1#S4.SS1)：train 29,578 段 / 2,456 人 / 59.76 h，val 3,854 / 724 / 7.40 h，test 3,301 / 725 / 6.67 h；各源数据集内身份隔离；所有基线在同一划分重训 | LVE、FDD、HDD；[§4.3 / Table 3](https://arxiv.org/html/2608.06408v1#S4.SS3) | [作者仓库](https://github.com/molly-ding/SubtleTalk/tree/5ea4de0f28797ef3754885d1516304eedf3c84ae) 目前只有 README，代码/数据 Coming soon；正文把精确公式交给 supplement，本轮未取得可运行正式 evaluator |
| DEITalk | MetaHuman；动态情感强度建模 | speech + emotional style labels；不能当作未知情感的纯 audio 协议 | 全文及精确划分未核实 | [ACM MM 2024, pp.10506–10514, DOI](https://doi.org/10.1145/3664647.3681359)；**不填未核实指标公式、单位或表值** | [作者仓库](https://github.com/KangShen-seu/DEITalk/tree/c17dbf6336dffac18397742951808ca6fc221709) 只有 demo、README、`code`/`dataset` 占位 |

SubtleTalk 的 [§4.5](https://arxiv.org/html/2608.06408v1#S4.SS5) 明确用户研究只输入 audio。但 Table 3 的 VA 使用 GT 还是预测值、区域强度的推理来源/默认值没有充分披露；不可自行把该表标注成已核验的“无 GT 辅助条件纯音频”成绩，也不可断言它一定使用了 GT。

### MEDTalk 的精确可核内容

令真实 rig 为 \(R\)，预测为 \(\hat R\)，lip/emotion/upper-face 预定 rig 集分别为 \(S_{lip},S_{emo},S_{up}\)。

| 名称 | 原文公式或定义 | 单位/归约边界 | 回答的问题；不能推出的结论 |
|---|---|---|---|
| Mean Lip Error (MLE) | Eq.15：\(\|\hat R^{S_{lip}}-R^{S_{lip}}\|_1\)；文字说明 average L1 error | MetaHuman rig 系数；公式用范数，正式代码未提供，具体时间/通道/clip 归约还需官方核验 | 唇部参数重建误差；不是 SyncNet 分数或人类 lip-sync 评分 |
| Mean Emotion Error (MEE) | Eq.16：\(\|\hat R^{S_{emo}}-R^{S_{emo}}\|_1\) | 情感区域 rig 系数；同样存在归约细节未公开 | 情感区域重建；**不是情感分类 accuracy** |
| Emotion Intensity Error (EIE) | Eq.17：\(\|Int(\hat R)-Int(R)\|_1\)；[Eq.8](https://arxiv.org/html/2507.06071v2#Sx3.SSx3.SSS0.Px2)：\(Int(R)_t=\|R_t^{S_{int}}\|_1\) | 选定 rig 的 L1 强度标尺，非 VA；时间归约需核对 | 对逐帧强度时序敏感，但同一强度可对应抬眉/压眉/其他组合，不足以证明方向、所有动作或自然度正确 |
| Upper-Face Rig Deviation (FRD) | Eq.18：\(\frac1{|S_{up}|}\sum_{j\in S_{up}}[std_t(R_j)-std_t(\hat R_j)]\) | **原文印刷式有符号、无绝对值**；rig 系数，std 的 ddof 未说明 | 幅度统计偏差；时间反转不改变 std。负值可表示过动，各通道也可相互抵消，不能把任意更小值解读为更自然 |

官方 [audio_semantic.py L22–28](https://github.com/SJTU-Lucy/MEDTalk/blob/97df2c75408c9b82cbdfac1bb76a022d7ac05a43/models/audio_semantic.py#L22) 实际选择 14 个 rig 通道、L1 和乘 0.1；这只是训练 helper。未确认 Table 1 evaluator 采用何种归约/缩放前，不把该 helper 冒充官方 EIE 复现。原文 Table 1 未给额外 \(10^{-k}\) 表头倍率；不要擅自换算成 mm。

### SubtleTalk 的精确可核内容与公式缺口

| 名称 | 正式正文可确认的定义 | 公式/单位边界 | 对我们是否直接可算 |
|---|---|---|---|
| Lip Vertex Error (LVE) | 唇顶点几何误差，Table 3 以 \(10^{-4}\) 报告 | 源包中**被注释草稿**为逐帧最大唇顶点平方 Euclidean 距离再时间平均；不能当正式 supplement；正文未充分给出几何单位和 mask 细节 | 当前只有 ARKit52 系数，不可直接生成 FLAME LVE |
| Upper Face Dynamics Deviation (FDD) | 上脸运动幅度偏差，Table 3 以 \(10^{-3}\) 报告 | 注释草稿为上脸各顶点 XYZ 时间 std 向量差的 L1，再对顶点平均；std/region/window 细节未取得正式 evaluator | 可做 ARKit 系数 std proxy；不能叫 FLAME FDD |
| Head Dynamics Deviation (HDD) | 头部姿态运动偏差，Table 3 以 \(10^{-3}\) 报告 | 注释草稿为头部各轴时间 std 的绝对差均值；单位随姿态表示而异 | 当前 52 维不含 head pose，填 N/A；不能填 0 |

草稿证据来自 [arXiv source v1](https://arxiv.org/src/2608.06408v1) 的 `sections/subsections/4_3.tex`，本地保留 `sources/subtletalk_commented_metric_draft.tex` 并记录 SHA256。以上用于解释统计意图，不宣称复现该论文指标。若以后在固定 Blender avatar 上导出顶点，可报告 **avatar-space vertex error / upper vertex std gap**，明确 mesh、单位、区域和 shape-key 转换；仍不能与 FLAME 原表直接排名。

### 主观评价不能写成已完成的客观指标

- MEDTalk [Table 2](https://arxiv.org/html/2507.06071v2#Sx4.SSx4)：42 人、10 条测试音频、六组含 GT、共 60 个视频，随机顺序，1–5 分 Likert 的 Lip-sync / Emotion Acc / Vividness。其 Emotion Acc `4.13±0.09` 是主观均分和 95% CI，**不是 413% 或 4.13% 的分类准确率**。
- SubtleTalk [Table 4](https://arxiv.org/html/2608.06408v1#S4.SS5)：27 人、10 条域外音频、5–30 s，audio-only 推理、左右位置随机、两两偏好，评价 Face Nat. / Head Nat. / Lip Sync；结果为偏好比例。没有报告客观 emotion accuracy 或 VA CCC。
- 本项目今天可整理盲评视频包和问卷，但没有真实受试者评分前，MOS、偏好率一律填“待收集”。同一个人反复看/打分不替代受试者研究；生成多 seed 也不增加受试者人数。

## 2. ARKit52 今日主表的建议指标定义

本节是可复现实验表的建议规范，不追溯更改已锁定的 bounded adapter 成功门槛。若现有 evaluator 已规定区域/归约，沿用并明确披露，不能为获得较好数字更换统计方式。

令 \(X_{ktc}\) 是第 \(k\) 个生成样本的原始系数、\(Y_{tc}\) 是追踪器产生的对应 motion 参考；它们不等于人工逐帧动作真值。所有主指标仅用真实有效 frame/channel mask，剔除 padding，不把缺失的监督通道当 0。主表使用未 clamp 的原始输出，另报 `[0,1]` 越界率与显示 clamp 的影响。

固定的 zero-based ARKit 顺序见 `scripts/05_eval_stage1_fidelity.py:27`。建议公开精确列表：

| 区域 | zero-based 索引 | 范围说明 |
|---|---|---|
| Lip23 | 18–40（Python `18:41`） | mouthClose 至 mouthUpperUpRight；**不含 jaw** |
| JawMouth27 | 14–40（`14:41`） | jawForward/Left/Right/Open + Lip23 |
| Brow5 | 41,42,43,44,45 | browDownL/R、browInnerUp、browOuterUpL/R |
| EyeExpression4 | 5,6,12,13 | squintL、wideL、squintR、wideR；**不含 blink/gaze** |
| Upper9 | 41,42,43,44,45,5,6,12,13 | 与当前动态生成器相同的九通道；不是全部上脸 |

报告“全脸”时仍要列出实际被观测的通道数及名称，尤其跨源数据 `channel_mask` 可能不同。当前 upper9 方案的保护43通道包括嘴、眨眼、视线和其他表达控制，但 protected43 不意味着原模型在这些通道上已经准确。

| 建议表列名 | 确切运算 | 单位 | 与论文关系和解释 |
|---|---|---|---|
| Lip coefficient MAE ↓ | 对有效 Lip23 元素计算 \(\langle |X-Y|\rangle\)，先每个 clip/seed，再平均 seed，再 clip-equal 汇总 | raw ARKit coefficient | 与 MLE 相同类型的区域 L1 评价，**不是同一 rig/测试集上的 MLE** |
| Jaw+mouth coefficient MAE ↓ | 同上，改 JawMouth27 | raw coefficient | 完整发音器官参数误差；与 Lip23 分开避免读者误解 jaw 是否在内 |
| Upper9 / Brow5 / Eye4 MAE ↓ | 同上，指定区域 | raw coefficient | 区域重建，不命名 MEE emotion accuracy |
| Absolute temporal-std gap ↓ | 每个 clip/channel/seed 用相同有效帧、population std (`ddof=0`)；\(\langle |\sigma_t(X_c)-\sigma_t(Y_c)|\rangle\) | raw coefficient | 受 MEDTalk FRD / SubtleTalk FDD 的幅度统计启发；**加绝对值是我们的清晰定义，不冒充 MEDTalk Eq.18** |
| Signed temporal-std gap | \(\langle \sigma_t(Y_c)-\sigma_t(X_c)\rangle\)；同时报 absolute gap | raw coefficient；0 为无偏，正为较静、负为较动 | 与 MEDTalk 印刷式方向一致的 proxy；不标无限制“越低越好”，不与 FDD 同名 |
| Upper9 intensity MAE ↓ | 定义 \(I_t(Z)=\frac1{9}\sum_{c\in Upper9}|Z_{tc}|\)；再 \(\langle |I_t(X)-I_t(Y)|\rangle\) | raw mean coefficient；与 sum 版相差9倍 | 与 EIE 同类但不等同；必须说明这是**动作系数强度 proxy，不是 VA/心理情绪强度**；方向已被绝对值抹去 |
| Dynamic velocity MAE ↓ | 在连续有效相邻帧上计算 \(\langle |(X_{t+1}-X_t)/\Delta t-(Y_{t+1}-Y_t)/\Delta t|\rangle\) | coefficient/s；若未除时间则必须写 coefficient/frame | 时序重建诊断；随机合理动作未必与唯一参考逐帧一致，不单独验收条件生成 |
| Raw out-of-range rate ↓ | 原始有效系数小于0或大于1的比例 | % | 保留幅度质量证据，不能靠显示 clamp 隐藏异常 |

std 对所有有效帧计算时会同时包含不同有效 run 的均值差；如果改为每个连续 run 独立中心化，必须另命名为 **within-run dynamic std gap**。当前 `joint_motion_metrics.py` 对每个连续 run 做动态中心化，不能默默把它与全 clip std 写成同一个数。

另保留已实现的 **raw / centered fair energy score、variogram、协方差差、多 seed spread、速度/reference 与四组 RMS/reference**。这些补充生成分布和时序结构；它们使用 fit-only 归一化尺度，单位/归一化与原始 ARKit MAE 不同。对至少两个样本，轨迹级 fair energy score 为

\[
ES_f=\frac1K\sum_k d(X_k,Y)-\frac1{K(K-1)}\sum_{k<l}d(X_k,X_l),
\]

其中 \(d\) 是全部有效时间×通道上归一化差的 RMS。第二项是有限 ensemble 的公平 spread 校正，不能改成 best-of-K 误差，也不能用平均轨迹替代全部生成样本。当前实现为 `scripts/joint_motion_metrics.py:_fair_es`；`speed` 的现有值是 normalized coefficient/frame，论文引用时不要误写成 coefficient/s。

## 3. 情感、身份和口型还需要什么证据

| 项目 | 今天可以实际评价/整理 | 可支持的表述 | 尚不能支持的表述 |
|---|---|---|---|
| 全局音频情感 | 冻结 audio emotion head 的 accuracy、balanced accuracy / macro recall、macro F1、混淆矩阵、各类支持数；四类与八类分开 | 音频分支的分类质量 | 生成后人脸一定呈现同样情感 |
| 生成动作情感 | 用只在真实训练 motion 上拟合、训练/验证分离、从未用生成输出拟合的独立 probe，冻结后同时评真实参考和生成动作 | 在该独立 coefficient probe 下的标签可识别性 | 人类感知情感准确率；与 MEDTalk Likert Emotion Acc 相等 |
| 训练 motion teacher 读出 | 可作 frozen teacher consistency 的内部诊断 | 与训练教师的一致性 | 独立情感评价；冻结不等于独立，曾给生成器梯度/蒸馏信号即存在循环 |
| 身份/个人动作风格 | 不同句子的 enrollment/query；独立检索 evaluator 的 Top-1 / macro accuracy、支持数、chance baseline；同人参考组稳定性和跨人参考交换 | 该动作风格表征下的可识别性/参考控制 | 几何脸型身份相似度；ARKit动作相同或同一avatar脸不证明本人身份保持 |
| 口型 | Lip23/JawMouth27 MAE、jawOpen相关、速度误差；固定同rig视频盲评 | 与参考动作的发音系数接近程度 | 音画同步已认证；MAE较小的静态平均嘴也可能不同步 |
| 渲染唇同步 | 未来跑独立 SyncNet 时，先核验真实rig GT视频/打乱或错位音频的区分力，完整记录检测成功率和同avatar合成域偏差 | 已通过域校验的 evaluator 对当前视频的同步诊断 | 直接把真人基准阈值当动画视频合格线；挑检测成功样本报高分 |

已有 `scripts/10_eval_independent_emotion.py` 提供真实 motion probe 的训练/冻结协议；但旧脚本绑定旧 Stage4 checkpoint 和旧 manifest，**存在脚本不代表已对当前 bounded/global-prior 输出得出结果**。`scripts/evaluate_neutral_identity_pilot.py` 的四人 pilot 检索也不能直接继承为当前完整模型的新身份泛化成绩。任何 evaluator 的预训练/选模数据是否与被评 clip 重叠都要记录。

对当前连续先验/有界音频模型，保护43通道与冻结基线逐值相同，因此相同 raw-mask 下的嘴部系数指标应相同；这是一项可验证的回归保护结果。情感相关的眉眼九通道已改变，所以 generated-motion emotion score 仍需重新计算。恒定均值保持也不保证动态 style/人类身份感知保持。

## 4. 今天可写的表格和消融边界

**主结果表：同一组数据、原生时钟、mask、身份参考、生成预算及 seed。** 行建议为 Frozen baseline、Global motion prior、Bounded audio adapter、Matched static-audio adapter。列至少包含口型系数 MAE、Upper9/Brow5/Eye4 MAE、absolute std gap、intensity proxy MAE、raw/centered fair ES、速度比、越界率；所有实际数值从归档重算，不从截图抄写。

**控制输入诊断表：同一已训音频模型**的 full / own-static / reverse / mismatch，用同 seed 成对比较。这是推理干预，不能代替 matched-static 的重新训练消融。Oracle 真值条件如果加入，必须单列“target-informed diagnostic”，不能当部署输入或主方法成绩。

**机制消融表：需要真实匹配训练**。当前可严谨解释的是冻结先验与新增有界修正、按同初值/样本顺序/noise/time/budget训练的 audio vs matched-static。1000/3000 steps 是预算轨迹，不是两次独立训练 seed。历史模型更换了数据规模、归一化和主体结构，只能列实验演进/诊断，不能伪装成只关一个模块的因果消融。身份、情感模块的 `w/o identity`、`w/o global` 如果只是推理置零，标 inference intervention；要写训练消融需重新训练匹配对照。

当前64段 outer 被此前反复用于研发，是 **historically exposed held-sentence diagnostic**；不能改称 sealed final test。新的613/206内层划分支持本轮选模比较，不会抹掉上游预训练/全局模型的历史暴露。误差条至少按 sentence 聚类、配对 bootstrap；只有7个历史 outer 句子时明确说明区间不稳定。多个生成 seed 不是独立句子、身份或独立训练运行。另报逐情感/逐身份支持数和失败情况，避免宏平均掩盖退化。

建议论文中的英文口径：

> We evaluate native ARKit coefficient trajectories using region-wise MAE, absolute temporal-standard-deviation gap, and a predefined upper-face coefficient-intensity error. These coefficient-space metrics are inspired by prior rig- and mesh-space evaluations but are not numerically comparable to their published MetaHuman or FLAME results. We additionally report proper ensemble energy scores and paired audio-condition interventions to distinguish motion magnitude from temporal conditioning. The present development results are reported separately from a future sealed test evaluation.

所有标准名、区域、归约、尺度、evaluator与划分都随表导出。写作今天可以开始；未收集的主观分数、未完成的独立分类/身份指标和未复跑的外部基线填“pending / N/A”，不填估计值或“基本没问题”。

## 5. 来源与复现记录

- [MEDTalk v2](https://arxiv.org/html/2507.06071v2)：Eq.8（强度）、Eq.15–18 / Table 1（客观）、Table 2（主观）、Table 3（消融）。下载 SHA256 `9241c6f10511cecbf6a979c3ec78a0b9512e976d538d9bf3548585a35a858015`。
- [SubtleTalk v1](https://arxiv.org/html/2608.06408v1)：§4.1 / Table 1（划分）、§4.3 / Table 3（客观）、§4.5 / Table 4（主观）、§4.6 / Table 5（消融）。下载 SHA256 `74b2fa05cfcda909a067b2e22765ee33d91b7cca07a13ffe1ee2b6f019ccb64d`。
- [DEITalk DOI](https://doi.org/10.1145/3664647.3681359)：题名、会议信息和页码由本地 Crossref 元数据核验；全文403，不补写指标。
- 本地前序核验：`docs/DYNAMIC_PAPER_METRICS_COMPARISON.md`、`docs/BROW_DYNAMICS_RESEARCH_REVIEW_20260916.md`。
- 本轮源缓存：`artifacts/paper_metrics_20260919/sources/`；含 HTML、GitHub tree、provenance、逐段公式摘录、commented draft、cache equality；不包含任何新模型评价数值。
