# Phase42：固定audio prior方差的单项消融

2026-10-08。属于已批准proposal_v2 R2表达空间匹配/概率对照，不新增架构主张。先读本文与PHASE41_RESULTS.md，避免上下文恢复重复分析。

## 证据与假设

Phase41纯音频MBE .810014、LBE .372413、lip 3.193374mm、原128生成F1 .636091；jaw范围比GT低28.6%。完整1367冻结替换发现音频g＋教师u MBE .307617，而教师g＋音频u .758158；主要缺口在u预测/接收的联合关系。混合路径用了GT，不是成绩，不是因果独立证明。

u prior raw logvar 83.465%超过硬上限2，均值差平方15.5705，prior variance7.1013、posterior .3011，q均值95%覆盖82.39%。学生可用大方差减弱均值匹配，硬clamp以外方差参数没有梯度。先检查是否应移除这一自由度，不能简单增KL权重或加幅度/分类critic。

## 唯一模型改动

`ResponseConfig.prior_variance='unit'`：p只输出g/u均值，返回logvar0，协方差为I。删除p global/local head的方差行6,192个参数。q仍输出均值和方差、训练仍q采样；D/S和两neutral独立support不变。

匹配依旧`KL(q||N(p_mean,I))`：每维0.5×(−q_logvar+exp(q_logvar)+(q_mean−p_mean)^2−1)，按g/u、有效token/维归一化。没有新loss，没有额外调权。不等于确定性teacher、不等于纯point模型；不声称fixed variance是已校准人类不确定性。部署主输出仍p mean，随机部署收益另验收。

默认`learned`保持历史检查点/旧代码行为。初始化先用原高斯头原seed，再仅移除p方差行，不多耗RNG；q、D、S、语义头与p均值权重逐位等于同seed控制，mean初始值相等。active新参数2,170,756。正式训练从头，不从已学会发音补错的q蒸馏。

## 固定协议与上线前检查

- 同fit10903，TRAIN内speaker743/sentence890留出；同fit-only normalization、原50独立neutral参考。
- seed47/24epochs/16368updates，AdamW2e-4，batch16query×两个单独support；同4轮KL anneal、epoch5 detached-p适配。
- 同中性B0/checkpoint、772D选择先于NaN检查、不传h0、全部观察嘴通道开放；不把full-motion重建反传p。
- 本地单测：默认兼容、unit闭式KL/有效token/梯度隔离、相同初始化、HuBERT NaN独立、config与sample恢复。远端：原Gaussian checkpoint重放，真实GPU梯度与optimizer/RNG精确恢复，fit8条120update可学习性，16条完整eval接口。
- 独立远端目录与source binding；权重/原Phase41证据不覆盖，核实GPU/容量后启动。所有原final保留，不删除唯一数据。

## 收尾与判断

训练后自动原1367validation raw/clip_all、四冻结probe、同rig几何/动态/参考干预、固定八情感显示导出；本地备份与渲染。sealed不读取，默认不推广。

与Phase41同seed/数据/更新/初始化对照，同时报告MBE/LBE/lip、jaw幅度/相关/闭合、眉动态、F1及local匹配差距。GT/oracle仅诊断。首轮若无真实音频联合改善，不以挑probe、加幅度、加预算救它；再检查音频可预测性和q包含的B0误差。只有改进可信才做三seed、公平基线、可靠pair交换和独立音素读出；本轮不是全部论文实验完成，不保证F1≥.7或SOTA。

## 当前状态

本地全套回归为 418 passed / 1 skipped；Phase42 预检四阶段、旧 checkpoint 兼容性、真实 GPU 梯度与精确恢复、120 update 小样本可学习性、16 条评估接口均通过。正式远端训练已启动，目录 `/root/kinetalk_phase42_fixed_prior_20261008`，seed47、24轮、16,368 updates；训练和后处理未完成前不读取 sealed、不替换默认、不宣称提升。状态以远端 `pipeline_state.json` 与本文为准。
