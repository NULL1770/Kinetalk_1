# KineTalk 当前研究交接：音频到 ARKit52 眉眼动态

更新时间：2026-09-20

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

