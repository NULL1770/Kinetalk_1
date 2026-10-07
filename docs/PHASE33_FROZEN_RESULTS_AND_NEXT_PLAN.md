# Phase33：中性B0＋772D冻结定位与下一步（2026-10-07）

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent validate（inline）。
- Verification Status: ANALYZED；全12536TRAIN/1367validation冻结提取；三模型66cell因素干预精确重放通过。
- 原始提取报告在accuracy比较失败；仅报告修复完成，未重跑GPU提取。runtime guard边界下文明确，不伪称所有guard均过。
- 全部31个远端产物67,864,368字节已本地逐SHA验证；`all_download_verified.json.passed=true`。
- 无optimizer、参数更新、新loss、新teacher、新默认、sealed读取或Git上传。

压缩后先读本文与`CURRENT_OPTIMIZATION_STATE.md`；原PID1308和因素driver1531均已结束，所有localdownload/render sessions已结束，禁止重跑。

## 1. 连续global的真实情况

三seed均值；MSE针对同一个冻结中性motion teacher的64D目标，非动作生成误差。

| 读数 | legacy1540 TRAIN | legacy validation | affect772 TRAIN | affect772 validation |
|---|---:|---:|---:|---:|
| global MSE | .246633 | 3.383114 | 1.170863 | 3.104422 |
| global mean-bias MSE | .001185 | 1.004648 | .017280 | .574494 |
| cosine | .965504 | .715997 | .880978 | .760746 |
| 情感×强度内中心化相关 | .924069 | .309837 | .610771 | .181765 |
| 音频头准确率（不是生成F1） | .995506 | .871495 | .973516 | .900756 |

772的validation总MSE较低，但内部连续细节相关更低、TRAIN拟合也不充分，不能由较小TRAIN→validation倍数（约2.65 vs13.72）宣布泛化已解决。这是两轮warm截列迁移，预算/适配不足仍是混杂因素。teacher在情感×强度内的speaker mean解释比例为TRAIN12.97%、validation22.01%；这种关联也可能是个人表达差异，**不是因果身份或内容泄漏证据**。

输入职责已纠正：HuBERT只直达B0/h0，学生只772。监督职责仍未完全纠正：`u_a`还由全残差flow间接学习，teacher的global目标也来自完整残差；因此不能称“完全解耦架构已证明正确”。

## 2. 物理残差尺度

只使用TRAIN真实`GT−中性B0−独立身份bias`的1,372,104有效帧/51观察通道；validation没有参与拟合。

| 通道 | TRAIN residual mean | population std |
|---|---:|---:|
| jawOpen | .050778 | .146300 |
| mouthSmileLeft / Right | .054268 / .049363 | .199121 / .195875 |
| browInnerUp | .234623 | .320494 |
| cheekPuff | .000106 | .000232 |
| cheekSquintRight | .000025 | .00000850 |
| noseSneerLeft / Right | −.0000169 / −.0000211 | .00000526 / .00000621 |

最大/最小std约60,918倍。Phase32使用共同scalar .25单位及标准高斯source；在12Euler步下拟合近恒定通道与大幅度眉部的数值任务差别很大。它支持测试匹配中性TRAIN的坐标/误差单位，但不能单独证明是全部F1或时序问题的原因。近恒定通道仍开放，不通过删除通道/改probe/滤输出来提高成绩。

统计文件SHA：`eed104a6ec78ecdb77a5a834890895e3436c6a6718216faf6e37427ed70b7ca5`。起点仍完整中性checkpoint SHA`e17659536a6fcaaaec2d9c22f99403fe4d9f1c8690c3cd182583894545f5ab45`；没有搬入native模型的统计。

## 3. 新772的u_a确实起作用，但动态质量还不对

固定content/B0/h0/global/intensity/identity/noise，只换`u_a`。每个三seed模型先复现原66个evaluation batches的全部3draw，误差严格0，再干预draw42。表为预登记66cell（三speaker×情感×强度）平均、clip_all；**不是全1367性能**。

| 条件 | jaw相关 | jaw速度MSE | jaw范围 | smile范围 | brows范围 |
|---|---:|---:|---:|---:|---:|
| GT参照 | — | — | .176906 | .124451 | .049491 |
| 原u_a | .328305 | .004418 | .204434 | .225110 | .064374 |
| static u_a | .408874 | .003818 | .151182 | .121275 | .023020 |
| zero u_a | .414707 | .003610 | .149571 | .132597 | .024003 |
| reverse u_a | .331200 | .004513 | .192193 | .218896 | .048257 |
| shuffle u_a | .289038 | .013961 | .215759 | .215910 | .060140 |

static/zero改善jaw时序，三个seed方向一致；同时压掉眉部动态，说明不能把u_a清零当修复。原眉部中心化相关约−.0019，smile约.1604，生成有运动不等于该运动对应GT时序。shuffle产生大量跳变，属于分布外敏感性，不是“ua编码音素泄漏”的证明。neutral子集仅3clip，另存readout，不用它冒充全80中性验证。

## 4. 身份通路有动态响应，正确身份仍待配对证据

validation自身两参考code RMS差.00559，异人平均.01794；同人参考相对稳定但并非完全一致。固定其他因素，raw换他人code的动态响应RMS：jaw .01497、smile .06594、brows .01289；只换bias的动态响应约1e−9，为静态位移。说明动态风格确实通过code进入renderer，不能说身份仅加常量。

无可靠异人同句同情感GT时，不对换人结果用原身份GT打重建分，也不能仅凭响应大就称身份正确或发音内容已保持。

## 5. 报告修复与边界

原accuracy使用torch整数相除后float32；既有evaluation使用Python `correct/n` double，两者约1e−8舍入差导致`<1e−9`assert失败。完整TRAIN/validation arrays和TRAIN stats均在assert之前持久化；报告修复只从已有arrays计算并以整数correct/n严格等于原accuracy，没有重复GPU提取或改任何评分/模型。

- v1原源码、失败state/log全部保留，不能把它覆盖成success。
- 完整arrays clip顺序、labels/speaker/强度、finite、数据metadata/manifest/模型源码/全部checkpoint SHA复核通过。
- 原提取最后runtime-state digest guard位于失败assert后，**未执行**。新报告明确写NOT_REACHED；实际input stream digest/完整TRAIN target moments当时未持久化，也不编造补齐。
- 三个独立因素诊断各自最终runtime state/无grad/输入checkpoint guard全部通过；它们是该诊断的验证，不冒充原提取guard。
- 首次SFTP train_codes零字节长时间未进展，停止唯一writer；bounded16prefetch顺序重试完成，原因未证实。原失败收据保留。12项已有干预/固定因素回归通过；git diff whitespace检查通过。

主报告SHA`792ac291276c769303a15390aa0d250ac587202a5e400a80f7d7fbea44e0ba59`；TRAIN arrays SHA`58afb4a3799dd40bd083dd1d94a704cf0923cf5e43de40729ba9c10a6855a86b`；validation arrays SHA`e3a0de780a86d93ff10e5ca3f352016cf8fd2d6e9a7ed43f5049fc5925eadd1b`。

## 6. 下一项可审核优化：N1中性772坐标对照（未启动）

目的：测试当前真实TRAIN残差单位是否制约生成精度和条件接收。**这是数值优化，不能据通过它宣称ua监督已经纯情感。** 不改变B0/输入/身份/teacher/条件支持，不新增loss或结构。下一新结构D2仍单独决策。

| 项目 | 固定方案 |
|---|---|
| 两臂起点 | 同一个完整中性checkpoint；学生按已验证规则截列到772；其余初始state共享，不从Phase26/29或挑选的Phase32 seed续训 |
| 共同设置 | 全12536TRAIN原生数据；B0/id/teacher冻结；audio+renderer更新；使用本次中性TRAIN moments；同一diagonal source、TRAIN mean、共享清零output head，保证两臂起始物理速度/样本/噪声相同 |
| 对照 | `flow-coordinate-system=train-centered-scalar`，坐标std=1，flow误差保持scalar单位 |
| 候选 | `flow-coordinate-system=train-standardized`，使用本次TRAIN std；配套flow误差按相同std度量。现有nondegeneracy floor normalized1e−4，不扫floor |
| 唯一比较变量 | 坐标/flow误差单位。数学上改变通道误差的相对权重，不能称所有loss数值完全不变；没有新增损失项或调大语义loss |
| 预算 | seed47/48/49，每臂固定2轮1568updates、batch16、12Euler、final固定，无择epoch/seed；pilot失败不扫参数或自动扩大 |
| 验证 | 先真实2update smoke：物理头初值一致、TRAIN mean/std与checkpoint绑定、B0/id/teacher frozen、772梯度隔离、非零学生/renderer梯度、finite、actualsample/GT/mask/time/white-noise相同；再训练 |
| 完整评分 | 同1367validation×draw42/123/2026，raw+clip_all、四固定TRAIN probes、同rig MBE/LBE/lipmean+max/expression/FDD、native jaw/嘴动态、逐类/强度/身份、固定视频；normal输出重放通过再做因素干预 |
| 判断 | 与匹配N1对照比较，至少MBE或mouth displacement有配对speakerCI改善；LBE/同rig几何及两原F1无明确退步；中性jaw相关/速度无明确退步、弱类无新崩溃，三seed均满足。保留Phase32参照但不将其当共同清头/diagonal source的等价对照；3speaker CI不能认证SOTA |
| 容量 | 当前root约2.08GiB；启动前按现有weight/last/curve实际size核预算和持久化方案，不能仅放RAM或盲目写满。保留原失败与唯一证据；不自动清理 |

本计划未改训练代码、未启动训练。原用户要求“明确后我同意了你再改代码”；故具体新训练方案先交审核，不因旧Phase26高分或本次诊断强行推广。

## 7. 后续表达职责修正，不能与N1一起堆叠

N1后仍需解决已定位的ua动态干扰及连续表达对齐。D2应先审计明确表达目标，再选一种监督/梯度机制：例如以可信眉眼/嘴角表达和优质同句配对的幅度变化监督时序情感，而阻止全残差flow让情感学生补B0发音误差。完整flow继续训练renderer，h0仍提供内容；嘴部前向开放。现有CE/global蒸馏不能单独监督当前ua，直接detach全部flow而不提供逐帧目标会让local head失去训练信号，**不这样盲改**。

需要先锁目标通道/时钟/可靠mask、配对覆盖及teacher重建验证；MEAD1/2/3保持clip标签，不冒充逐帧真值。身份交换只用真正优质对齐；source→neutral安全对并不自动等于情感↔情感/跨身份安全对。一次只做一种职责修改，不并行加D3/D4/多个loss。

论文协议核查见`PAPER_EVALUATION_PROTOCOL_AUDIT_20261007.md`。F1留作辅助而非唯一成功标准；保留失败分数。两个正确中性基座/772输入四列视频见PHASE32_RESULTS.md，GT/B0/legacy/772、seed47/draw42，未做gain/retiming。
