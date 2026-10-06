# 当前优化恢复入口（2026-10-05，Phase16）

压缩后先读本页，再看 EMOTION_MOUTH_OPTIMIZATION_PLAN.md / progress.md 最新章节。历史完整保留，不重做旧审计、不重复启动。

## 正在进行

Phase16 TRAIN时间相关审计complete；rho .827–.972，jaw .857。ARsource实现已通过35本地/23远端检查和真实4sample2step smoke；三臂×三seed正式训练已排队启动（最多3组并发）。远端/root/kinetalk_temporal_source_20261005；先查launch/state防重复。详细门槛见主计划最新Phase16章节。

**Phase15六组训练/验证已全部complete。四probe/三seed F1全部显著提升，但口型位移+3.61%、几何混合，联合门槛失败，不推广。**

所有训练、postprocess、native复核、mouth tradeoff分解均complete，全部本地保存。目前无任务运行、无新模型推广。恢复先读phase15 README和native_replicate_summary.json。
- 远端 /root/kinetalk_source_spread_20261005
- seed47 control/source_spread wrapper9929/9930；48=9931/9932；49=9933/9934；finalizer9935。
- 先查 launch.json、seed*/{control,source_spread}_state.json、各.log、postprocess_state.json；不重启已有任务。
- 同一固定 timing000 warm、源码、12536 TRAIN、1367 validation；三seed47/48/49，每组2epochs/batch16/1568steps，final only。
- sole arg: flow_source_noise standard -> train-residual-std；两组均使用同一 train_source_stats.json。
- 保持 motion teacher、global MSE .5、generated consistency .2、B0、身份、u_a、全嘴部支持和所有loss权重。
- normalized source std = 冻结TRAIN人口std(motion-B0-independent neutral identity)/.25，固定数值非退化floor1e-4，未观测tongue支持外保留1。正值Gaussian、无零通道/输出缩放/增益/新loss。
- 在 flow 的 x_t/velocity_target 及 rollout 初态使用同一source transform；checkpoint config保存52std，历史state_dict无新key。
- 旧 generate(initial_noise=None) 是 deterministic zero state，已保留；正式随机评估始终显式传入相同white noise。
- 26初始本地检查、增加兼容测试后9项通过；9远端检查、修正源码fresh smoke_retry1通过。4samples/2steps，firstloss .238463 / grad2.419279，84renderer/26audio tensors更新，其他system完全等于warm。
- smoke_verification.json绑定训练source/statshash，launcher查验后启动。首次implicit-noise行为错误已修正并保留原smoke；verifier初次错误读取inference_only checkpoint的total_steps，已改成严格epoch_complete日志2steps，无重训。

完整report/features/provenance/state/complete/comparisons/replicate_summary已下载phase15_source_spread_20261005。均值control/source：MBE .887251/.888017，LBE .430269/.432589，位移MSE .00256787/.00266068；原F1 .202691/.253625→.255245/.327616，稳定 .460048/.434124→.527568/.505934。每seed每probe F1 CI>0，但每seed位移CI>0。native3draw复核已完成：四F1 .2003/.2480/.4581/.4332→.2460/.3171/.5159/.4960；jawcorr .29751→.30380，范围 .19655→.19856，没有幅度坍缩。mouth位移增量66.2%来自三draw随机速度方差、33.8%来自均值预测误差，不能只看边际尺度。

## 本轮已完成

Phase14 五条件冻结诊断1367validation已完成，**否决仅修scalar强度作为下一方案**。远端 /root/kinetalk_intensity_factorial_20261005，PID8093 complete，本地 phase14_intensity_factorial_20261005 report/perclip/state/launch/smoke已下载，不重跑。
- audio clip MBE/LBE .886043/.429660，位移MSE .00260116；四F1 .185877/.219977/.441821/.414801。
- teacher scalar-only .881039/.426183，位移 .00261140；真实label scalar-only .880680/.428082，位移 .00260652。两种位移均显著变差，label scalar稳定F1 .41877/.39920，均显著下降。
- teacher global-only MBE .659836/LBE .345109；teacher both .659867/.345671。几乎全部oracle几何收益来自global而非scalar。
- global-only位移 .00269251、四F1 .13518/.18036/.42153/.37645，没有语义/时序修复。所有oracle是GT-informed诊断，不能当推理成绩。
- 当前平均jaw范围q90-q10 .19637，GT.17528，相关 .31072；不声称所有口型均幅度不足。
- GT四probe F1 .66034/.64118/.66395/.64589；稳定probe不是原协议替代。

TRAIN source spread审计12536clips/1372104frames已完成，不重提取。本地 flow_source_spread_20261005/train_source_stats.json，SHA fc9645c7719b7ac130b9d187900d87db8e45b63f1b794fcf27ebb1412bbc82e0。jaw std.14630，smile .19912/.19588，brows .24142/.23091/.32049，cheekPuff .000232，cheekSquint .000009，noseSneer .000005-.000006，当前raw source统一std.25。尺度差是事实，不能称唯一主因；Phase15测试这个机制。Stage1 .05 target-scale floor与此无关。

## 固定参考和限制

当前仍无新模型获选，MBE≈.7/生成独立F1>.7未达到。
checkpoint /root/autodl-tmp/kinetalk_final_20260922/checkpoints/phase2_fullmouth_timing000_20261004/audio/final.pt
SHA e17659536a6fcaaaec2d9c22f99403fe4d9f1c8690c3cd182583894545f5ab45
manifest d4ef98bcb7f4e5bc94a15f27516bb3e67c4e61dcfff936e02cf6befefa6ceb1a
原1367validation seed42/batch16/Euler12：rawMBE/LBE .9024/.4503，原F1 .1698/.1468；clip .8860/.4297，原F1 .1859/.2197，稳定 .4424/.4134。GPU微小跨脚本numeric差异已记录，只解释同运行匹配对照。

只用TRAIN/validation；sealed test禁止调参/改报告。条件分类头与训练critic不是生成独立F1；GT oracle不部署。不得盲目叠loss、遮蔽嘴部、扫描权重/选择seed；大量历史改动禁止reset/清理。

## 已确认，不重复审计

- audio emotion条件F1 .8456，teacher head读取audio global仍.8040；global MSE train .3844/val3.2133，连续值泛化差。
- Audio intensity条件F1 train.82961/val.45154；真实valL2/L3均值预测1.375/1.898，train2.065/2.833；teacher neutral强度1.042偏高。但Phase14说明scalar不能单独解决。结果 semantic_code_subspace_20261005/intensity_readout.json。
- semantic/intensity head rowspace rank10含91.78%valglobal误差和93.71%teacher variance，禁止把多数误差称为classifier-invisible nuisance并投影删除。
- B0 neutral jawcorr715train.7019/80val.4415，残差降至约.279；没有统一lag。监督3298=715neutral+2583安全pair，配对有效嘴帧55.375%；missing native帧约1%，不改GroupNorm。
- 固定TRAIN speaker折ridge neutral-reference×audio-category映射五折全退步，不增加此identity adapter。
- 原probe对GT sigma.001极脆弱，稳定probe仍敏感；四probe顺序固定，clip与raw同时保存，不把clipping称学到情感。

## 历史失败，不能重跑

Phase7 prototype、Phase8去endpoint consistency、Phase9时序adapter、Phase10冻结renderer、Phase11teacher-agreement筛选、Phase12B0dropout、Phase13去coordinate MSE均未推广。Sampler/gain/zero-noise/投影/ordinal/class balance/probe critic/线性补偿也有失败证据，详细见主计划。
Phase12中性jawcorr control/dropout .438899/.436686，jaw位移 .00180911/.00183583，mouth位移 .00117130/.00118543；不替换B0、不做下游适配。
Phase13位移误差改善3.59%，但原F1/geometry联合复现失败，默认distill .5保留，不扫描中间权重。六组wrapper6453–6458/finalizer6459 complete，全部结果已下载phase13_no_global_distill_20261005。

## 操作

数据 /root/autodl-tmp/kinetalk_data/packed_trainval_20260923；python /root/miniconda3/bin/python。SSH用 .codex-finalizer/remote_ops.py读取现有凭据，不打印密码。新实验放/root/，autodl-tmp空间紧张。上传用workspace-relative ASCII路径，避免中文绝对路径经PowerShell管道损坏。每次先查existing launch/state，不重复启动。

## 下一机制候选（未实施/未启动）

先审查TRAIN片段内、去均值的残差相邻帧相关；若支持，再比较非退化的时间相关Gaussian source与diagonal/standard controls，保持通道边际std、全嘴部支持和原loss。训练/采样变换、native缺帧规则必须相同，rho=0严格兼容。仍按三seed/全validation/四probe/口型和几何联合门槛。不能称该候选已证实；全局条件连续值泛化和均值时序误差也还存在。详见Phase15 README。

## 论文目标重新确认

目标仍是相比VOCA、FaceFormer、EmoTalk、FaceDiffuser整体指标领先，并形成公平、可复现的论文证据。下一步先补齐同协议validation差距表；既有ARKit/shared-audio适配须准确标注，历史sealed成绩不用作当前调参。新候选尚无全套LVE/EVE/FDD基线比较。随后按时间相关source、连续global条件泛化、B0发音泛化推进。Phase15目前不推广，SOTA与MBE≈.7/F1>.7均未达到；不能用辅助probe或t-SNE分离替代主成绩。详见主计划最新paper/SOTA章节。
