# FaceDiffuser指标核对与下一版修改依据

2026-09-20，用户要求下一版同时改善动态与其他指标，并询问0.5/0.6量级。

后续更新：此设计已于2026-09-21实现并启动，见 `ISOLATED_STATE_RUN_20260921.md`。实际残差采用DC保护与软domain penalty，没有硬幅值界限；下述早期“有界”设想不能作为实现保证。

## 原文与本地公式

FaceDiffuser原文 https://arxiv.org/abs/2309.11306 ，PDF第6页Table3（BEAT）：B-FaceDiffuser MBE0.5152、LBE0.1358、FDD0.1471；无diffusion消融0.4170/0.1077/0.1482。此表没有10的负幂倍率。Table2为BIWI/Multiface mesh评价，不能与ARKit系数表混用。PDF与表格渲染经独立代理核对，原文缓存位于artifacts/arkit_benchmark_20260919/sources/facediffuser_2309.11306.pdf。

本轮MEAD开发集KineTalk MBE0.74082470、LBE0.31474539、ABS-FDD0.10998095，signed-FDD−0.03561010。官方代码同时打印signed和ABS，但论文Table3标签FDD不能仅凭标签断定是哪种；因此不声称本轮FDD优于原论文。

当前实现沿用固定官方commit的BEAT区域L2公式，无乘1000或通道平均，按语义名称映射。MBE每帧sqrt(sum_channel(error^2))，当前全部446条都恰好51个有效通道；LBE12通道。0.74082470/sqrt(51)=0.10373621，是同样片段/采样权重下的“逐帧通道RMSE的均值”，不是全帧汇总RMSE，也不能替换主表数字。LBE对应归一化值0.09085917。0.74不是74%系数误差。

原论文BEAT、30fps与本轮MEAD四类、25fps、身份划分、系数来源、mask、训练预算不同。公式追溯不等于官方benchmark复现；不能按原论文数字直接排名。FaceDiffuser同数据适配重训仍未完成。Table3的消融也说明MBE/LBE更低不自动表示随机动态更好。

## 本轮结构审计（结论与假设分开）

没有发现四状态readout/lift的符号或尺度反转；共用通道映射与TRAIN尺度，已有roundtrip检查。

确定事实：

- seed42眉rawMSE0.02395041、centeredMSE0.00205092；均值偏差分量0.02189949，占91.44%。眼raw0.01162689、centered0.00162434；均值偏差占86.03%。这是对raw平方误差的分解，不是对MBE百分比的分解。
- SlowStateAffect的state/local共享input与TCN；Stage5 flow通过两者回传。residual target使用当前state.detach()，输出为anchor+lift(state)+unrestricted residual，因此残差可以抵消状态，独立状态语义没有结构保证。
- Stage4 slowstate训练Huber终点0.122705；Stage5前两轮0.137360/0.143206，最终0.125206，static0.089157。固定64训练探针state centered correlation−0.285，validation−0.025，不能只归因于身份泛化。
- Stage5绝对替换upper9，Stage4上脸的平均表情没有硬保护。全局码冻结并不保证最终情感保留。内部生成情感读数下降符合这一路允许覆盖的事实，但评价器本身有限，尚非独立感知因果证据。
- oracle同时替换显式mean与flow condition，是训练分布之外干预；oracle恶化不能单独证明状态监督本身无用。

联合优化的梯度竞争和可抵消分解是有依据的待检验原因，不宣称已经找到唯一根因。

## 下一版具体顺序（本条尚未实现或启动新训练）

1. 保留已修复口型；分离身份中性偏移、情感平均姿态与随时间变化项。先让确定性基座拟合原始系数和音频可预测变化，单独训练/验证状态通路，flow不能更新其共享骨干。用完整fit覆盖及固定train探针区分拟合与泛化；静态、反序对照仍保留。若独立音频状态都无法优于静态，不直接进入昂贵联合训练。
2. 固定已验证的均值/状态预测，再拟合条件运动分布，使用稳定残差目标。保护口型和上脸平均姿态，动态只加入有界、片内零均值修正，不重新放开上脸绝对姿态任意覆盖。零均值不等于此前失败的固定spline-Q高频投影，仍允许低频时变动作；但平均表情保护也不能代替独立情感验收。
3. 在相同MEAD/ARKit划分重训FaceDiffuser适配版，使用同通道、时钟、条件、mask与计分器建立可比较基线。新候选同时考核原始系数误差、口型/身份/情感、速度/幅度与真实音频对静态时序收益；不能仅通过其中一项便宣称成功。基础训练按预先锁定的开发集规则与收敛证据设预算，不再默认12轮即充分训练。

不修改本轮终点、原有门槛、默认模型和sealed test；不承诺下一次必然研究成功。
