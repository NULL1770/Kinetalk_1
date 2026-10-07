# Phase39表达接收器结果

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent validate（inline）
- Verification Status: ANALYZED；全部原SHA核验、三seed聚合逐值重算一致；未重新训练/生成。
- Date: 2026-10-07；固定seed47/48/49、每seed1568updates、1367validation×draw42/123/2026。
- 新方案尚待用户批准；sealed未读，Git暂停，发布默认未替换。

## 结论

三seed联合gate全部false，拒绝推广本次1920参数调制；不扩大该失败候选预算。相对Phase37，F1/MBE基本持平，LBE小幅下降；仍未超过Phase34整体质量门槛。固定pilot不能证明所有训练预算下该结构都无效。

| clip_all三seed/三draw平均 | Phase34参考 | Phase37表达/韵律 | Phase39新调制 |
|---|---:|---:|---:|
| 原128生成F1 ↑ | 0.683872 | 0.617731 | 0.616752 |
| 原64生成F1 ↑ | 0.581147 | 0.549179 | 0.548809 |
| 辅助两个生成F1 ↑ | 0.700589/0.656377 | 0.631700/0.600021 | 0.630281/0.599428 |
| MBE ↓ | 0.904603 | 0.906694 | 0.906927 |
| LBE ↓ | 0.446154 | 0.451459 | 0.450506 |
| Lip mean mm ↓ | 4.008668 | 4.021237 | 4.014739 |
| Lip max mm ↓ | 7.598816 | 7.640006 | 7.628092 |
| Expression mean mm ↓ | 0.769161 | 0.762216 | 0.763266 |
| Expression max mm ↓ | 2.669011 | 2.635881 | 2.637344 |
| 绝对mesh FDD mm² ↓ | 153.447688 | 162.015833 | 161.654398 |
| 嘴部相邻位移MSE ↓ | 0.003709 | 0.006340 | 0.006324 |
| jaw中心化相关 ↑ | 0.255015 | 0.226260 | 0.226293 |
| jaw q90−q10 | 0.229451 | 0.229957 | 0.228965 |

GT jaw q90−q10=.175279；新模型=.228965。整体幅度已不小，动态相关仍仅.226293；不能据合并统计断言每种情感/每个通道幅度正确。

## 每seed与保留的配对区间

| seed | 新原128/64 F1 | 新MBE/LBE | ΔMBE 95%区间，39−37 | Δ原128 F1 95%区间 | gate |
|---|---:|---:|---:|---:|---|
| 47 | 0.635258/0.572974 | 0.893344/0.440135 | [-0.000293, 0.001308] | [-0.008565, 0.000167] | false |
| 48 | 0.573553/0.518637 | 0.919972/0.453549 | [0.000774, 0.003125] | [-0.003973, 0.006041] | false |
| 49 | 0.641446/0.554815 | 0.907466/0.457833 | [-0.002795, 0.000753] | [-0.005936, 0.000592] | false |

区间沿用原报告的2000次speaker-cluster bootstrap；仅3位validation speaker，区间可靠性有限，且多指标未作整体显著性校正。不将零散区间改进等同论文成功。

## raw与类别

| 分支 | raw原128/64 F1 | raw MBE |
|---|---:|---:|
| full | 0.487035/0.431355 | 0.936852 |
| named | 0.486281/0.430323 | 0.936860 |

clip_all沿用现有[0,1]系数裁剪；raw完整保留。未通过改裁剪、重训练probe、挑seed或重定时提高分数。

| 情感 | Phase39原128类F1 |
|---|---:|
| neutral | 0.214286 |
| angry | 0.601792 |
| contempt | 0.793322 |
| disgust | 0.428124 |
| fear | 0.551080 |
| happy | 0.831304 |
| sad | 0.843533 |
| surprise | 0.670577 |

原128 probe在GT的F1为.660344，原64为.641175；均为整段motion统计分类，非逐帧情感判别。它们不是质量上限，也不是唯一验收标准。

## 正确性、视频与报告恢复

399passed/1skip及两真实2update已通过。最终三seed新调制权重RMS约.0185–.0190，bias RMS约.0242–.0246，确实发生更新；不能把无性能收益解释为层完全没有训练。
八类固定视频已生成：GT / Neutral B0 / Phase37 / Phase39，seed47/draw42/M025固定句005，不做gain、retiming或最佳样本选择。25fps与全部帧数核验，1/4与3/4固定帧人工检查：37与39非常接近；这些静态检查不能证明完整口型时序正确。视频入口见PHASE39_VIDEO_GALLERY.md。
收尾finish_phase39最后一行残留completed未定义，导致driver/collector报告状态失败；训练和三个评分此前全部成功。保留原helper/failed状态/日志；新report_recovery_v2/state.json确认153正式原成员、226smoke/source成员与12收尾原文件全SHA闭合，summary逐值等于原报告平均。v1恢复helper因字符串replace优先级失败，也保留。没有重训或重评分。

## 统计谬误检查（11/11）

| 项目 | 本次检查与限制 |
|---|---|
| Simpson | 各seed方向不完全一致，保留每seed和分组，不凭总平均推广。 |
| Ecological | 群体幅度/宏F1不推出每个clip或通道正确。 |
| Berkson | 固定MEAD分割与批准配对是筛选样本，不外推任意音频群体。 |
| Collider | 沿用固定条件与mask，不增加结果相关筛选。 |
| Base rate | 保留八类宏F1与类F1，不以Happy或总体accuracy代替。 |
| Regression to mean | 同seed对照，不以相对失败基线的恢复当超过Phase34。 |
| Survivorship | 三seed全保留，报告失败证据也保留。 |
| Look elsewhere | 四probe、raw/clip与全部指标保留，不挑有利项。 |
| Forking paths | 固定预算/seed/draw；新方案明确探索性质。 |
| Correlation vs causation | oracle与probe读出只定位条件利用，不证明纯表达或身份风格正确。 |
| Reverse causality | 只比较固定架构干预，不把观察相关解释为表达造成正确口型。 |

## 论文与下一步边界

MEDTalk/EmoTalk已核主文采用几何/动态指标与视频，没有本项目这种F1；DESTalker全文仍未取得，不推断其具体方法。已保存对比是共享音频/同rig改编、训练预算不同，不能称官方SOTA复现；其历史几何评估子集与本全1367结果也不得混排。
具体新设计与审批范围见PHASE40_ARCHITECTURE_PROPOSAL.md；当前模型训练代码没有新增修改。
