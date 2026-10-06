# 固定final12连续情感条件定位

Phase27已完成，三个模型×完整1367验证段×三固定draw。所有正常audio输出先逐位等于现有final12曲线，GT、native时间、mask、B0、identity、u_a、强度和噪声保持不变；之后只替换global为真实query动作teacher条件。模型state前后逐位不变，未拟合、未读sealed test。全部22个诊断文件已下载并核SHA。

| clip_all指标 | 正常audio | 仅teacher global |
|---|---:|---:|
| MBE | .870982 | .632573 |
| LBE | .421170 | .335949 |
| 原128/64生成F1 | .796875/.721358 | .795215/.699641 |
| 眉平均姿态MSE | .052693 | .013014 |
| 嘴平均姿态MSE | .007964 | .003528 |
| 嘴位移MSE | .001816626 | .001874296 |
| jaw时序corr | .315339 | .312886 |

这表明连续global条件是主要几何瓶颈：眉平均姿态误差下降75.3%、嘴下降55.7%。三个seed的配对speaker MBE CI都严格小于0。但这不是部署成绩：它用了query GT，而且mouth位移误差略增、原64 F1略降且配对区间跨0。还不能说联合目标已解决。

global本身validation MSE分别2.305/2.323/2.354，cosine .645/.626/.628，其中整体均值偏差占29%–33%。音频分类头F1=.844/.863/.841，只说明类别信息已存在，不能代替连续几何的质量。当前眉41–45通道GT平均约.254/.225/.234/.157/.103，B0约0，身份bias约.025–.033；生成残差均值普遍不足。这里的加法项描述不等于因果身份归因，也不支持统一嘴部gain。

十二轮最后TRAIN记录global distill约.159/.157/.158，与validation MSE差距很大。为了区分网络没拟合好与未见speaker泛化问题，下一步Phase28读取全部12536TRAIN/22speaker的当前audio/teacher code、真实TRAIN enrollment和固定B0。只做误差分组与条件去均值speaker方差；一个预先固定的常量偏移假设只在TRAIN留一speaker检查，验证集不拟合、不选择系数。若TRAIN已经接近teacher而未见speaker偏差很大，不能盲目加同样的MSE或直接拟合validation校准。

证据目录：`final_experiment/evaluation/diagnostics/phase27_global_20261006/`。阶段实际状态、结果和下一步见CURRENT入口；default未推广，总目标未完成，论文同协议对照和sealed最终评估仍待开展。
