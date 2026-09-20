# KineTalk 接手核查：动态失败的证据、论文差异与全量实验

2026-09-20。本轮读取指定三份交接文档、训练/生成/评价源码、原论文正文与官方代码，并通过用户提供的 SSH 重新核验服务器。没有训练、替换模型或读取 sealed-test 动作目标。本轮读取了 test 的划分元数据；这与读取测试目标或计算测试成绩不同。

**目前最有依据的判断：V2 的主要待解问题已更偏向跨句泛化，而非“音频路径完全失效”。** 本轮额外的冻结权重诊断发现，真实音频在固定训练样本上同时改善 flow 误差与最终生成分布，在开发句子上却输给独立静态训练对照。该发现更新了交接文档中“下一步优先怀疑 flow/rollout 不对应”的优先级，但还不能断言唯一原因，更不能声称动态已成功。

## 1. 核验了哪一个系统

用户目标包含四项：content 决定发音/口型时机；global emotion 决定整体情感状态；dynamic 根据局部音频改变眉眼事件/活动分布，同时容纳随机细节；identity 保留跨句稳定的个人动作习惯。这四项不能以单一整体重建分数互相替代。

旧系统为 `M = B0(audio content) + P_id(neutral references) + residual DiT`。motion teacher 学情感/局部控制，audio student 学部署时可用的条件。身份来自独立多句 neutral residual 统计，是行为/动作身份，不是几何脸型；恒定 `P_id` 不能本身解释中心化动态，identity code 对动态的贡献仍需参考交换实验。

最新 V2 已是另一条 upper9 路线：

```text
冻结 run12/audio 基线 -> baseline52
                      -> global(65), identity(128), b9(9维均值)
训练 motion9-b9 的连续 AE（5帧有序块 -> 16维 latent）
global/identity/b9 + noise -> 冻结 latent prior
1540维 native audio -> fast/slow adapter -> 有界 flow 速度修正
24步 Euler -> AE decoder -> b9 + residual
以生成的 upper9 替换 baseline52 的对应九通道
```

65维 global 是64维情感编码与1维强度；上下文共202维。1540维音频是768维内容、768维 emotion2vec 中层及4维韵律。V2保留原生帧并每5帧有序打包，不能继续描述为旧的 stride4 去均值 rank8 输入。upper9 为五个眉通道和 squint/wide 四个眼部通道，**不含眨眼、视线、头部**。

原生基线明确绑定 `run12/audio/final.pt`，不是 `run12/dynamics/final.pt`。源 prior 没有旧 DiT 的逐帧 B0/h0 条件，只接收全局上下文；V2 时变条件来自音频 adapter。这不是已证实的 bug，但论文/消融必须准确区分两条生成路径。AE重建成功不等于条件生成器能受音频控制。

V2 的训练 objective 是插值 GT latent 上的 FM 加错配排序，未对最终24步自由生成轨迹加训练损失；旧 DiT 的 rollout/L1/velocity/std 实验已经做过并失败，不能称为“全项目从未做”。FM 在理想条件下能学习条件分布；当前实现使用 FM 本身不是理论错误，不能凭看到 noised GT 就断言它必然忽略条件。

## 2. 来源、权重与数据 hash

本地源码、远端执行源码、V2 保存快照的13个关键文件逐项相符；AE、prior、统计量、两条 adapter checkpoint 与协议绑定相符。54个本地备份文件的尺寸/SHA256全部通过。

| 对象 | 本轮重算 SHA256 |
|---|---|
| continuous_latent_dataset.pt，654,187,344 bytes | `6f876fd7596eba204185d4a646c0886a15c7d1984a8550c19d4f6e20ddceb32c` |
| AE | `e2291d52c0c4cccb64fb7e1d3dba3d4b41a40527042ab1431a2e9796eaaf7865` |
| source prior | `66aa7e8028be64b5486f7dc91cd453b798a0e6d80c5a3366bc9f69313cd63019` |
| V2 audio step1000 | `d426e159724d943d2d8059a9db646912a8123a3a8be383bbf4fa7bdbc7ab1088` |
| V2 matched-static step1000 | `4de4f82d9ec8a7d9180da1b9b337e0c183ea70a38599faeefd98b8e179fec6cd` |
| run12/audio/final.pt | `73a8f17137772f882bf6d8c65dbf2f6c36b6c42f5724091f7a2e7a42870df42c` |
| run12/dynamics/final.pt | `9f7c40d201ef782dce46ed337e32eda9f819cd27686d7de3de058b38db152501` |

重新执行原独立审计并拦截写报告函数，保留旧报告：重建613片/66,272监督帧，重放24,000次 donor/batch 随机链；两样例×两训练臂的24步生成与保存数组最大差均为0。它验证实现/复现，不认证自然度，也不是对所有历史实验、所有原始资产逐个重新校验。

证据目录：`artifacts/research_takeover_20260920/`。主要文件为 `local_hash_verification.json`、`remote_hash_verification.json`、`upstream_verification.json`、`audit_rerun_result.json`。

## 3. 本轮新诊断：训练样本能改善，开发句子失败

固定每类8片，共32 fit +32 exposed inner-dev；按预设字符串与 clip_id 的 SHA256 排序选样，不按结果挑选。使用原checkpoint、原4个种子、24步Euler；无梯度、无更新、无参数搜索。原生生成完全不读取目标，FM评价另在监督轨迹上计算。该诊断耗时143秒。

| 同一端点，越低越好 | 训练32片 Centered ES | 开发32片 Centered ES | 训练 Variogram | 开发 Variogram |
|---|---:|---:|---:|---:|
| source prior | 0.208888 | 0.265787 | 0.051184 | 0.060228 |
| real audio | **0.167733** | 0.269909 | **0.045969** | 0.059882 |
| 独立静态训练臂 | 0.204894 | 0.264712 | 0.050430 | 0.059861 |
| audio模型自身静态输入 | 0.209313 | **0.265067** | 0.050801 | **0.059632** |

注意开发列最小 Centered ES 是独立静态的0.264712；表中自身静态仅用于同模型比较。

real−独立静态：训练为−0.037161，32/32片改善；开发为+0.005197，仅11/32片改善。固定五个 flow time（.05/.25/.5/.75/.95）下，训练集 real FM误差均低于独立静态，开发集均高于独立静态。例：t=.5时训练0.31225 vs0.35559，开发0.68180 vs0.64915。

该结果支持：**现有受限 adapter 在已见片段上具备产生时序收益的能力，跨句迁移没有建立。** 这不是只有FM下降而生成完全无收益的情况。不足以证明模型学到可泛化的韵律因果关系：音频内容、位置与身份也可能充当片段/句子记忆索引。训练片段已经参与拟合，32片不是总体显著性实验；未做新的同接口 motion-oracle 训练，也未由此证明所有生成能力问题已排除。

开发子样本结论与已有全206片一致：real 0.254140，独立static 0.250434，差+0.003706。static仍保留 audio-derived global、b9及clip音频均值；它测试的是**新增时变局部音频的价值**，不是“整个系统无音频”。

复现入口：`artifacts/research_takeover_20260920/frozen_diagnostic.py`；完整逐片指标、固定选样和五个时间点见 `frozen_diagnostic_result.json`，紧凑汇总见 `frozen_diagnostic_summary.json`。

## 4. 四篇论文究竟证明了什么

不能把“视频有自然动态”“情感强度逐帧贴近参考”“局部音频在控制其他因素后确实提高动态分布”混为一个命题。论文确实有动态证据，但证据层级不同；没有本项目的全部反事实对照，也不意味着论文没有成功。

| 论文与一手来源 | 实际条件/训练机制 | 已报告证据与边界 |
|---|---|---|
| [DESTalker，§3–4](https://doi.org/10.1002/cav.70127) | 音频+参考动作；参考emotion/style经时间平均池化成全局向量，neutral base上做FiLM与style gated residual；最终L1及视频帧速度监督。正文没有噪声采样生成路径 | 3DMEAD 35/6/6人、8类；MVE/LVE/FDD、换参考风格及20人MOS。确实展示表达/动态，但未找到隔离局部audio时间信息的独立static训练对照，不能把全局emotion/style效果等同精细眉峰时刻预测 |
| [MEDTalk v2，Eq.8–12、Table1/3](https://arxiv.org/html/2507.06071v2) | 音频content；emotion label embedding；emotion2vec+转录文本预测逐帧强度，调制情感特征，经fusion/decoder输出174维rig；最终rig MSE+输出强度MSE+embedding loss。公开推理确定性 | EIE逐帧强度误差是比单纯std更直接的时序证据；去intensity/text消融退化。FRD主要反映变化统计。支持动态表达有效，但强度混合了眉眼嘴角，不能单独证明每个眉事件都来自音频；协议包含额外情感指导，与纯音频自动global不完全相同 |
| [SubtleTalk v1，§3–4](https://arxiv.org/html/2608.06408v1) | WavLM多层可学习融合+韵律、逐帧VA、窗口眉眼/头强度、前10帧motion context；DMP+随机residual flow；最终运动velocity/std/smooth约束。训练强度由GT motion窗口统计，VA来自视觉teacher；audio-only推理的VA由预测器提供 | train59.76h/2,456人；FDD/HDD、消融、27人对10段域外纯audio的用户研究，支持自然弱相关动态。自动主表的窗口强度来源、辅助loss重建方式/权重等细节仍不充分；公开仓库本轮仍仅README。不能擅自指控其测试用了GT，也不能声称完整audio-only主表协议已核实 |
| [DEITalk，ACM MM2024](https://doi.org/10.1145/3664647.3681359) | 可核摘要明确speech+emotion style label、局部动态强度DEI、长序列DPE、emotion-guided decoder | 本轮ACM正文/PDF均403；OpenAlex无开放全文；作者repo的code/dataset仍各1byte占位。已阅读摘要，但**没有读到可核正文**，因此不编造其四项loss、消融数值或严格动态验证结论 |

DESTalker使用已有完整正文转录 `artifacts/stage1_literature_20260911/destalker_normalized.txt`；MEDTalk/SubtleTalk正文及GitHub状态本轮重新下载。MEDTalk官方 `models/audio_semantic.py` 明确在最终 `x_recon` 上计算强度误差，不能误写成只有标量预测头监督。SubtleTalk正文避免确定性endpoint回归，但逐GT速度损失同样可能压缩随机性，仍须实测多样性；不能因作者称随机就忽略该权衡。

**一对多不是不能学动态。** 合理目标是让音频改变条件分布中的活动概率、幅度、时间范围和动作组合；自主眨眼等细节由noise承载。确定性方法也能学到稳定可预测部分。给每个独立noise都配同一GT的L1/L2，会偏向条件中位数/均值；反之只拟合std可产生时间错位但“幅度正确”的动作。需要同时评价条件信息、分布与感知质量。

## 5. 为什么多次尝试没有得到可靠音频动态

已有证据支持多种失败机制同时存在，不能归纳为漏接一个眉通道或少抄一个loss。

1. **很少的独立训练句子下反复优化高维条件。** V2实际15人、21句、44.18分钟监督；1000步×24样本约24,000次采样，相当于每片平均约39次抽取，并非“只看了很少batch”。本轮训练收益/开发退化符合过拟合或句子分布偏移。增加相同小集合更新不等于增加数据多样性。
2. **修改过的环节多，但没有稳定建立可部署条件的跨句收益。** VA、event、prosody、直接audio、activity条件都已有实验；事件真实条件有响应也不能证明audio能预测它。前轮coarse-condition probe及三seed事件NLL也未胜static。当前缺的是这项实证，不是条件名称本身。
3. **历史训练目标可能通过均值补偿、幅度或错误对照改善。** teacher schedule有raw均值变好但dynamic变坏记录；rollout L1压幅；velocity/std把眉RMS拉近却相关很低；swap ranking也可以只使错配更坏。这些已解释具体试验现象，但本轮V2的训练样本实际生成收益说明不能把FM/solver错配当唯一主因。
4. **数据/表示/输出仍有边界。** 旧event曾丢失77.5%监督帧，已修复；AE保留主要动作，但无约束解码仍越界，显示clamp会遮掉负眉动作。source prior可动不等于自然，upper9也不是全部上脸。原视频→拟合系数→同rig显示的峰值保留需持续抽检；没有证据说所有失败都由tracker导致。
5. **有些验收比论文原主表更直接测试audio增益。** 这能防止用随机摆动冒充音频响应，但不应把论文数值/视觉成功贬成“都没验证”。也不应把“每项指标必须胜所有对照”的开发门槛当作领域唯一成功定义。最终论文应预设主假设和次要指标，保留失败项。

## 6. 全量数据与四模块训练范围

以下库存来自已核hash的native manifest，仅统计metadata，没有打开test动作。

| 数据 | train：clips/人/有效小时 | val：clips/人/有效小时 | test：clips/人/有效小时 |
|---|---|---|---|
| 当前MEAD处理库存 | 12,959 /22 /15.7626h | 1,435 /3 /1.6124h | 1,659 /3 /2.0313h |
| 当前CREMA-D处理库存 | 5,797 /72 /4.0259h | 723 /9 /0.5047h | 806 /10 /0.5654h |

这里“库存全量”不是MEAD官方完整数据集。四类neutral/angry/happy/sad、非neutral仅L1/L3筛选复算为4,265/470/546；这也不是八类库存全量。CREMA-D有unknown intensity=-1，必须mask，不能当低强度或零。

MEAD训练/验证/测试的sentence ID数164/141/141，train与val/test各重合141；CREMA三个split各12句且全部重合。身份确实隔离，句子没有隔离。库存数也尚未扣除neutral enrollment、历史曝光名单及所选正式泛化协议排除项，**不能直接宣称它们就是最终paper query counts**。

| 模块 | 已完成训练事实 | 论文全量阶段要求 |
|---|---|---|
| content/B0口型 | run12 articulation在2315 fit、19人、四类上12epoch；继承预训练来源 | 所有符合正式train协议的有效发音片段；完整时间覆盖/随机连续窗口；若继续主张neutral-only B0，明示neutral子集与中性化规则；冻结预训练声学编码器不影响“完整训练分区”定义 |
| identity | run12 identity在fit identities的独立neutral多参考上12epoch；dev参考仅enrollment | 所有训练身份的独立多句neutral参考；测试参考只作enrollment，不参与梯度，排除query同clip/同句；记录每人参考数及不足处理 |
| global emotion | run12 teacher/audio各12epoch、2315fit/405 internal-dev；近期冻结 | 在完整train的支持情感类上拟合teacher/student与统计；生成后用独立评价器验证，不靠训练teacher准确率认证 |
| dynamics | run12旧动态12epoch；另有2720片18epoch仅512参数projection；当前AE4000/prior14000/V2两臂1000步仅613片 | 正式train上重拟AE、统计、prior和音频路径；记录clip/句子/人/有效帧/抽样覆盖；不能只训adapter就称四模块全量重训 |

论文划分应同时保留两个清楚命名的任务：A，新身份、允许共享文本；B，新句子，并报告是否同时新身份。要称B，必须从所有可学习阶段和参考/统计拟合中移除该句子的样本；不能只在最后adapter剔除。新的正式任务专属权重需要从无test泄漏的来源重训，历史反复调试模型仅作开发初始化/诊断记录。测试人物只有3人，不能用几百clip掩盖独立身份数量有限。

正式manifest在训练前锁定selection规则、参考与query、排除原因、各split/资产hash、音画时钟、通道mask和历史曝光账本；调参只用dev。方案锁定后训练完整train partition，保留validation选预算规则；sealed test最后用于锁定版本的最终评价，不把test分数回流改方案。本轮完成了库存与风险核查，尚未生成/批准一个新的最终split，也未启动全量训练。

## 7. FaceDiffuser指标与公平对比

已重新取得 [FaceDiffuser官方BEAT评价器](https://github.com/uuembodiedsocialai/FaceDiffuser/blob/e15f3500fdae0eda962f5d018488dfa0a1a9d552/evaluation/compute_objective_metrics_blendshape.py)，文件SHA256为 `ea909e1346f78e5b1c930b794020654945cb224c08fb1f4c18efcb4b5eaeb10f`，与现有实现引用一致；并阅读[原论文§6.1](https://arxiv.org/html/2309.11306v1#S6.SS1)。

- ARKit/BEAT系数采用 **MBE、LBE**；顶点空间才用MVE/LVE。官方blendshape实现是每帧区域系数差的L2范数再时间平均，不能把它替换成逐元素RMSE而仍称同一指标。
- 官方BEAT FDD：`std_t(sum_upper GT²) - std_t(sum_upper pred²)`；同时有ABS版本。signed以接近0为合理解释，不能无限越负越好。它对任意帧排列不敏感；官方16通道映射到本项目为eyes/nose，**不含眉毛**。保留原定义用于比较，另报Brow5/Upper9 centered ES、速度、窗口活动、事件/节奏诊断。
- FaceDiffuser原文Diversity主要定义为不同subject条件间差异；B-FaceDiffuser不使用subject条件，原文没有同样的定量Diversity表。固定audio/identity的多seed Multimodality须另定义，不能把当前spread冒称其原指标。FD/WInD、AV offset/confidence是扩展指标，需独立评价器和合成域校准，并非FaceDiffuser原始必备整套指标。
- 原论文BEAT表中diffusion的MBE/LBE高于deterministic baseline（0.5152/0.1358 vs0.4170/0.1077），FDD略优（0.1471 vs0.1482）；作者明确讨论随机性与逐帧误差的权衡。因此所有逐帧误差都最优不是一对多模型的必要条件，但条件信息和自然度要有独立证据。
- FaceDiffuser、FaceFormer、CodeTalker等要在相同ARKit52、split、帧率/时钟、mask、合法条件与训练预算下重训；输出维度适配必须披露。各自原论文的不同数据/rig数字不能直接拼主表。audio-only与emotion-label/参考指导任务分表；本方法额外neutral reference条件也要公开，最好提供去identity的audio-only消融。
- 主表固定多seed逐draw评分，禁止best-of-K或先平均随机轨迹；报告真实参考质量、越界、完整带音频同rig视频与盲评。独立emotion/identity评价器先展示在真实留出数据上的能力，AV评价器先能区分GT与人为错位音频。冻结43通道只证明未改写，不认证其嘴同步或身份情感效果。

## 8. 下一步执行顺序与投稿主张

目前不应再按“完全没有训练收益”开始换生成器，也不应承诺只要全量就一定成功。优先固定当前结构，先把本轮fit/dev诊断扩成一次有完整反事实记录的开发诊断，保留real/独立static/自身static/reverse/合法mismatch；如仍怀疑接收端，新增同接口的motion-oracle正对照，不能用AE重建替代它。不要把已经做过的旧DiT容量训练重新包装为新实验。

随后在排除所有预留身份/句子和参考后，做**覆盖更多独立句子的训练规模对照**，各规模AE/统计/prior仅用自身fit，audio与static共享初始化/预算规则。每个训练规模同时报告训练自由生成与句子留出表现；若扩大独立句子后差距收窄，才有证据支持全量改善泛化。若仍训练强、验证弱，查语义/情感与句子混杂、音频归一化与跟踪域差异；新增文本/条件/正则必须由这些证据驱动。强度的可预测部分和随机细节保持分开解释。

结构/数据协议在dev锁定后，执行四模块完整train训练与同协议基线，最后一次最终test评价；当前613/206和64外层的数字只放开发诊断/失败分析。论文需要的是可验证的新机制和完整证据链，CCF-C不对应某个自动合格分数。现在最值得推进的贡献候选是“在口型与身份/全局情感约束下，让局部audio对弱相关动态分布产生可泛化增益”；该主张尚未成立。本轮明确缩小了排查范围，没有把诊断完成当作研究成功。


## 9. 本轮续做：split 审查、paired mismatch 与训练前 gate

第一版 `joint_disjoint` 候选的 query 句子完全不重叠，但稳定 hash 没有按三个 source role 的支持度分层，test 只有 45 个 query clips。它保留为可复现实验记录，不进入训练。新增 `joint_balanced_disjoint` 候选只从三个 role 都有至少 3 个非中性片段的共同句子中选 12 个 val 句、8 个 test 句；当前 query 为 train/val/test = 2516/115/105，manifest SHA256 为 `074043d12c88c8fba0c529e45fe421bfed1eafb73800197583c83cac710ee437`。三份候选的 metadata-only preflight 均通过，结果见 `artifacts/research_takeover_20260920/paper_manifest_preflight_20260920.json`，但仍是 candidate，尚未批准训练。

V2 现有 206 片结果的 mismatch 只覆盖 161 个有合法 donor 的 query。新证据包将 audio、matched-static、mismatch 限制到这 161 个共同 clip 后重算逐片均值：audio−matched-static centered ES 为 `+0.004247`，audio−mismatch centered ES 为 `−0.006232`；对应 variogram 差为 `+0.000360` 与 `−0.000190`。这说明 mismatch 不是干净的同分布反事实，不能用它的单项优势认证时序因果。证据包为 `artifacts/research_takeover_20260920/v2_causal_diagnostic_pack_20260920.json`，pack SHA256 为 `8b117d312e6b451550ce150235772e66db43bd0e7ef046ce09bb4cb5c00f8a7f`。

因此当前可写进论文的结论仍是：V2 在已见 fit 片段上存在 audio adapter 的训练收益，但在句子留出的开发诊断上没有超过 matched-static；“音频对应动态已经成功”尚未成立。下一步应先锁定论文轨道和 exposure ledger，再按同一正式 train partition 做四模块全量训练与公平 baseline，最后才读取 sealed test。
