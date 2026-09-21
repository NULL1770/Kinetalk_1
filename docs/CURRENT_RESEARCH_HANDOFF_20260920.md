> **2026-09-21 清理与架构指针**：本仓库按“v9 历史 → v10 基础 → 冻结 v10 Stage4 的 upper9 动态实验族”理解；不同分支结果不能拼成统一已验收系统。当前主边界、推理条件和指标限制见 [CURRENT_SYSTEM_20260921.md](CURRENT_SYSTEM_20260921.md)。本地脚本清理为可恢复迁移（74 scripts、84 tests 保留，1571 个历史/散落文件进入 `archive/cleanup_20260921/`）；SSH 代码副本清理不在本页宣称完成。最新 relative 路径直接组合冻结 Stage4 上脸均值、中心化状态和 zero-DC 残差，`mean_preserving_upper.py` 主要属于较早 centered/prefix 实验。训练可使用动作/标签/教师监督，部署目标仍是音频加独立 neutral reference；upper9 不含 blink/gaze。动态读出有弱 state 信号，但完整时序门控仍未通过，不能写成“已成功”。
# KineTalk 当前研究交接：音频到 ARKit52 眉眼动态

更新时间：2026-09-21

## 最新启动：独立状态100轮，条件残差40轮，FaceDiffuser适配100轮

2026-09-21 00:31:56 CST正式启动，实例11473，suite PID9880，路径 `/root/autodl-tmp/kinetalk_isolated_20260920/suite100`。当前先训audio state100，再独立static100；只有完整验证通过才继续两臂冻结状态残差各40轮。状态质量门槛失败时保存结果并跳过残差，随后仍跑FaceDiffuser同数据适配100轮；进程异常则停止全套。

数据为四情感协议全部4098train/446validation，25fps原生帧。继承并冻结已修复Stage4的口型、身份、全局情感及teacher→audio权重；本轮不是从零重新训练这些模块。新mean/state分离参数和优化器，residual不能回传状态且生成过程保持片内零均值。没有硬幅值限制，必须检查越界/速度/自然度。

实现与断点续训/来源验证已测试；state、residual、FaceDiffuser真实GPU smoke均完成，后者完整1000步×3种子。启动不代表动态成功。FaceDiffuser是缓存全部音频信息与冻结参考/全局条件匹配的适配版，非官方BEAT复现；全预算也尚非最终主表等总计算量协议。详细设计与实际运行记录见 `ISOLATED_STATE_RUN_20260921.md`。sealed test未读、默认未替换，正式结果待产生。以下均为历史进展。

## 最新终点：校准时序队列完成，口型保住，音频动态仍未通过

run12全部exit0、总41.43分钟，GPU空闲。完整4098训练query，teacher/audio/两臂dynamic各12轮；完整446validation×3噪声，口型保护30组合全部通过。Audio MBE0.740825、LBE0.314745、centered ES0.115918；独立static ES0.116007，两者差区间跨零，audio variogram0.033334较static0.031794更差。时间先验降低本轮Stage4过快运动，但不能声称已学会有用音频时序。慢状态训练64探针相关−0.285、validation−0.025；生成情感teacher诊断Stage4 65.25%→最终50.67%，独立情感/AV仍待验收。详细完整表、CI和备份见`CALIBRATED_TEMPORAL_RUN_20260920.md`终点部分。下面启动描述已是历史；没有新训练排队。

## 最新：参考口型校准通过，时序眉眼自动队列启动

恢复B0后加入TRAIN拟合、独立enrollment驱动的固定口部偏移，完整446 validation overall raw MSE 0.01499158→0.01078375、neutral 0.01470305→0.00652539，时序轨迹保持，原保护门槛通过。尚有越界与视觉/AV验收限制。已实现native时钟+相邻状态+相关噪声的upper9时序分支、完整条件static/reverse干预、独立静态训练及全446三采样验收。2026-09-20 16:35启动`/root/autodl-tmp/kinetalk_calibrated_20260920/run12`，队列PID3559；当前先进行真实CUDA smoke，再自动identity200参考对轮次→teacher12→audio12→两臂dynamics各12。实际状态必须查看queue_status.json，不能把启动当成成功。详见`CALIBRATED_TEMPORAL_RUN_20260920.md`。下文是历史阶段记录。

## 最新终点：口型30轮完成，neutral均值误差门控未过

完整4098训练query的30轮已完成，耗时24.68分钟。446 validation口型相关0.304501→0.445307、raw MSE−37.32%、centered MSE−14.87%、邻帧位移MSE−5.91%，真实输入优于static/reverse的句簇区间全负。唯一失败是80条neutral raw MSE+5.96%超3%上限；独立曲线分解确认neutral时序改善但均值项+14.24%。因此identity/teacher/audio/Stage5均未继续，GPU空闲，默认未换。详细结果见 `MOUTH_REPAIR_RUN_20260920.md`。这不是眉眼动态验收通过。

## 最新进度：口型修复训练已启动

本轮已修 fixed train-only motion/residual support，隔离未监督通道和嘴部随机残差，加入identity/audio阶段口型保护。来源已知的full_v1 B0正在全部4098训练query上继续30轮，完整446 validation固定终点验收，通过才自动训练identity/teacher/audio；未启动Stage5眉眼长训。实际路径、PID、边界和测试见 `MOUTH_REPAIR_RUN_20260920.md`。下文“未启动新训练”只描述前一轮排查当时状态。

## 口型异常排查补充（本轮）

用户反馈最新六格口型不对后，已确认三条固定视频的 native/prepared 输入、音轨 SHA 和 25 fps 时钟一致。最新 full_v1 的 B0 是重新随机初始化、715 条 neutral×12 epoch，而非旧已训练基座；全446 validation 的嘴部相关仅0.3045。Stage4 进一步把嘴部速度误差增到 B0 的2.91倍；Stage5与Stage4嘴部逐值相同。修复了 renderer 忽略 channel_mask 而显示未监督 tongueOut 的错误，以及 identity 阶段评估误用未训练renderer的错误。三条显示修正版已经重渲染，模型权重未修改，口型尚未修好。详见 `MOUTH_REGRESSION_AUDIT_20260920.md`；不能继续假设身份/情感/口型已合格而只优化眉眼。

## 当日后续覆盖说明（13:03更新）

以下V2历史结论保留。最新已执行完整MEAD4098train/446validation、完整native帧、全新KineTalk五阶段与独立static动态对照，总38.6分钟，全部结束；不是仍待全量训练。详细协议见`PAPER_FULL_TRAINING_20260920.md`，结果见`PAPER_FULL_RESULTS_20260920.md`。修复历史test15封存冲突后重新锁manifest，不能使用旧4118候选。最新audio raw ES优于static，但centered ES差+0.005276，句簇95%区间[+0.003063,+0.007669]，动态目标仍失败。独立全部曲线复算最大差2.22e−16；sealed test未读、默认未替换、无进一步训练排队。先定位train/val和solver/条件差异，不能重复声称扩大数据即能解决。

## 1. 当前结论

目前还不能声称已经实现了可靠的“音频到眉眼动态时序”。AE 可以重建 upper9，source prior 可以产生一定运动，但最新多尺度音频残差 V2 没有通过严格的真实音频优于静态对照验收。

V2 冻结了 AE、source prior、global emotion、identity、audio b9 和其余 43 个通道，只训练 fast/slow 音频残差。使用 613 个 fit clips、66,272 个有效 joint supervision frames；206 个内部开发 clips、4 个固定噪声种子进行评估。两条训练臂各训练 1000 步，未达到预设门槛，因此没有继续到 6000 步，也没有替换默认模型。

V2 结果如下（越低越好）：

| 条件 | Centered ES | Variogram | ARKit-MBE | ARKit-LBE |
|---|---:|---:|---:|---:|
| Frozen source prior | 0.252139 | 0.058985 | 0.643935 | 0.358831 |
| Real audio | 0.254140 | 0.058685 | 0.638804 | 0.358831 |
| Independent static training | 0.250434 | 0.058316 | 0.638910 | 0.358831 |
| Same-model static | 0.252303 | 0.058233 | 0.639703 | 0.358831 |
| Reverse audio | 0.261950 | 0.059071 | 0.640050 | 0.358831 |

真实音频相对 reverse/mismatch 有响应，但相对 independent static 退化：Centered ES 差为 +0.003706，句簇区间为 [+0.001496, +0.007617]。因此不能把音频优于反向输入、非零通路响应、loss 下降或动态幅度保持当作时序成功。

## 2. 已经做过的路线

以下路线已有实际代码、训练或审计记录，不能再次当作“尚未尝试”：

- direct audio；
- VA/语义 student；
- prosody residual；
- event supervision/event schedule；
- prefix/history/context；
- bounded audio adapter；
- centered/rollout supervision；
- temporal refiner；
- renderer adapter；
- motion/source prior；
- continuous latent/projection；
- multiscale fast/slow audio residual V2。

旧 event 分支曾把完整支持从 613 clips、66,272 frames 错裁成 348 clips、14,898 frames，原因是 `known4 ∩ joint / min10`。该数据支持问题已经修复，但修复后仍没有得到可靠音频时序，因此不能把数据修复当作最终解决方案。

## 3. 目前真正未解决的问题

必须区分三种可能性：

1. 音频到真实眉眼动作的跨句/跨人可预测性弱；
2. flow matching 训练目标与最终有限步自由生成轨迹不一致；
3. 生成器虽然接收条件，但动态控制能力、通道量纲、mask 或训练/推理分布仍有问题。

V2 的结果不能判断是哪一种，因为真实音频没有胜过 independent static。下一步不应继续盲目增加条件、loss 或模块，而应在固定结构上做 train-vs-validation 因果定位：同时记录 flow matching loss、最终多步自由生成轨迹、real/static/reverse/mismatch 以及 motion-oracle 上限。

判定规则：

- train 也不胜 static：优先查生成器控制能力、目标/solver 对齐、mask、量纲和有限步 rollout；
- train 胜而 validation 败：优先查跨句/跨身份泛化、音频特征泄漏或数据分布差异；
- flow loss 下降但 rollout 不下降：训练目标与最终推理轨迹错配；
- oracle 成功而 audio 失败：生成器通路可用，瓶颈在音频条件预测；
- audio 与 oracle 都失败：先修生成器/接口，不能继续声称是“音频不可预测”。

## 4. 数据与训练覆盖边界

当前 V2 的 613/206 是内部开发协议，不是论文最终主表，也不是完整原始数据集。早期 run12 五阶段是在 2,315 fit / 405 internal dev、19/3 identities、4 类情感上完成各 12 epochs；这只能称为该实验 fit split 的完整训练。另有 2,720 segments、22 speakers、67 sentences、18 epochs 的 formal local projection，但只训练了 512 参数 local projection，其余模块冻结，不能称为完整模型全量重训。

因此目前不能独立证明 identity、global emotion、mouth 都已通过论文级感知验收。近期动态实验冻结这些模块；“43 个通道未变化”只说明新 upper9 分支没有改写它们，不等于身份、口型和情感质量已经合格。

正式数据库存量审计显示：MEAD 原始 train/val/test 为 12,959/1,435/1,659 clips，身份分别为 22/3/3；当前正式候选四类筛选后为 4,265/470/546。论文实验仍需重新锁定 train/dev/test manifest、speaker/sentence 隔离和 sealed test；sealed test 尚未读取。

## 5. 为什么现在的指标是局部数据

这些指标来自快速开发和安全门控：在尚未证明音频分支有效前，使用 613 fit clips 和 206 内部开发 clips 可以固定结构、对照静态/反向输入并避免消耗 sealed test。它们适合回答“这一版是否有局部因果信号”，不适合回答“论文方法在完整数据上泛化如何”。

论文主表必须另建固定协议：训练集只用于 AE、统计量、identity/global/mouth/dynamic 模块拟合；validation 只用于选配置和 epoch；方法锁定后在完整 train split 重训；sealed test 最后只读取一次。主表统一 ARKit52、帧率、mask、身份/句子划分、条件和训练预算，并报告 ARKit-MBE、ARKit-LBE、ARKit-FDD、AV offset/confidence、Multimodality、FD/WInD，以及 upper9、身份、口型、全局情感和消融结果。当前开发数不得冒充最终论文结果。

## 6. 交接给下一个 AI 的任务说明

> 你接手的是 KineTalk 项目中“音频驱动 ARKit52/upper9 眉眼动态”的研究。先阅读 `docs/CURRENT_RESEARCH_HANDOFF_20260920.md`、`docs/MULTISCALE_PRIOR_AUDIO_V2_20260919.md`、`docs/ARCHITECTURE_DYNAMICS_AUDIT_20260917.md`，并核对源码、checkpoint、dataset hash。
>
> 当前已知事实：AE 可以重建眉眼，source prior 有一定动态幅度；旧 event supervision 曾把 66,272 帧裁成 14,898 帧，现已恢复完整 joint/min5。direct audio、VA/语义 student、prosody residual、bounded adapter、rollout/去均值、renderer adapter 和 V2 multi-scale residual 都没有通过“真实音频优于静态对照”的验收。V2 的 real audio Centered ES 为 0.254140，independent static 为 0.250434，real audio 仍然更差。不要重复这些实验，也不要用 oracle motion、训练 loss 下降、动态幅度保持、通路响应或音频优于 reverse/mismatch 宣称动态成功。
>
> 第一项工作是建立最终论文数据协议，明确完整 train/validation/test 的 speaker、sentence、clip 数、有效帧和 sealed test 是否读取，并分别列出 identity、global emotion、mouth、dynamic 四个模块的训练范围。当前局部开发指标只能用于诊断，不能直接写成论文主表。
>
> 第二项工作是做一次固定结构的 train-vs-validation 因果定位：同时记录 flow matching loss、最终多步自由生成轨迹、real/static/reverse/mismatch 和 motion-oracle 上限。若 train 也不胜 static，查生成器控制能力或训练目标；若 train 胜而 validation 败，查跨句/跨人泛化；若 flow 下降而 rollout 不下降，修改训练目标与最终 solver 的对应关系。不要在没有定位证据时继续换模块或堆 loss。
>
> 后续任何方案必须同时满足：真实 audio 相对 independent static、same-model static、reverse 和合法 mismatch 的 paired centered ES 改善；variogram/速度/眉眼分组动态改善；口型、身份和全局情感不退化；并在完整论文 train split 上重训。最终只读取一次 sealed test，输出 ARKit52 主表、消融、基线复现、固定 seed 视频和失败项。若真实 audio 仍不能胜过 static，应明确报告当前设计未解决音频眉眼时序，停止继续堆模块。

