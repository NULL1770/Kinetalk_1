# Phase18 正在进行（2026-10-05）

Phase17预算链/两种raw-mean诊断完成，均拒绝旁路；两组失败历史实验完整归档后远端/root约6.1GB空闲。下一项仅身份参考训练预算：/root/kinetalk_identity_budget_20261005/launch.json、run.log、audit/progress.json 或 report.json；PID6886，以launch.json为准。12vs120额外参考轮次、seeds47/48/49，22TRAIN参考对，不生成视频，不改变默认；只跑这一个登记好的试验。完成后检查全冻结模块和first12精确一致、每seed联合门槛，再决定是否允许完整下游重训。原warm/data/probes未变；压缩后先读本页和主计划最新段落。

# Phase17 正在进行（2026-10-05）

预算已核清，详见 training_budget_20261005/README.md；正在登记 raw-emotion 信息保留诊断，默认模型不变。恢复后核对 /root/kinetalk_raw_audio_audit_20261005/launch.json 与 run.log，避免重复启动。存储安排待完成。下方 Phase16 历史状态保留。

# 当前优化恢复入口（2026-10-05，Phase16 complete）

压缩后先读本页，再读主计划最新章节；保留历史文件，不重复审计、不重复启动。目标仍是论文所需的公平同协议比较领先，MBE≈0.7、生成独立情感F1>0.7。**目前均未达到，没有新模型推广。**

## 当前状态

本轮所有任务已完成，无训练/验证/审计在运行。默认仍是 timing000 + standard source + 全嘴部残差支持，原loss保持。不要重跑Phase14/15/16、pooling或基线重评。

## 本轮已完成

1. Phase16 三臂×三seed47/48/49，每组2epochs/batch16、全12536TRAIN/1367validation，final only，唯一训练参数差异flow_source_noise。源码、warm、manifest、四probe、固定参数冻结均核验。
2. TRAIN原生相邻帧、片段内去均值残差相关rho .827–.972；jaw .857。固定TRAIN std不变，AR平稳初始化/缺帧重置，训练flow与rollout一致，所有嘴部开放，无输出平滑/平均推理/新增loss。
3. 三seed clip均值standard/diagonal/AR：MBE .887258/.888016/.908438；LBE .430272/.432583/.444092；mouth displacement .002567814/.002660729/.001720026。AR位移误差降低33.02%/35.36%，但每seed MBE/LBE显著变差，联合门槛全失败，不采用。
4. 原F1 128/64 standard .202964/.253380，diagonal .255627/.327671，AR .233189/.309935。稳定probe另报，不能代替原F1。
5. 原生3draw复核diagonal→AR：jawcorr .303786→.292512，jaw范围 .198559→.203536，GT .175279；mouthcorr .287769→.280753。位移误差更低不等于语音同步更准，没有幅度坍缩。眉毛静态bias仍占主误差。
6. 同协议validation差距表已完成，fixed632vertex rig、51TRAIN支持通道、四冻结probe、raw/clip、所有原生帧：EmoTalk-core MBE .7450/LBE .3315/F1 .5919/.6862，当前timing000 .8847/.4313/.1825/.2155；VOCA-core .7467/.3324/.4977/.4845。完整五方法/vertex/FDD/位移见gap README/report。适配版不是官方原论文端到端复现；FaceDiffuser使用冻结KineTalk条件且历史curves缺checkpointsha，明确记录限制。确定性基线按原CUDA/TF32 firstbatch检查，FaceFormer bit-exact；不能用CPU重放数值差异误判checkpoint失配。
7. 音频汇聚诊断complete：实际staged student为SlowStateAffect(hidden128/4TCNblocks)，不是system.audio_encoder。冻结mean128 vs mean128+populationstd128、固定ridge .001、TRAIN speaker_id%5五折。MSE .331713964→.333426501，五折全变差；allTRAINfit validation3.323732→3.319506，仍差于原audio3.213。拒绝std-pooling分支，不扫描，不用这个ridge部署。原cached audio64输出TRAIN/validation bit-exact，未重提取teacher/B0/身份。

## 本地产物

- final_experiment/evaluation/diagnostics/phase16_temporal_source_20261005/README.md、replicate_summary.json、九组report/features/provenance/complete/state、九个paired comparisons。
- 同目录native_replicate_summary.json、三组native_diag_vs_ar report/npz、24固定情感render-inputNPZ；均已下载，原noise42/123/2026，无挑seed。
- final_experiment/evaluation/diagnostics/validation_gap_20261005/README.md、report.json、per_clip_metrics.npz、complete/state；report/npz SHA已校验。
- final_experiment/evaluation/diagnostics/audio_pooling_predictability_20261005/report.json、features_predictions.npz、launch.json。
- API兼容修复：ARgenerate允许CPU初始white noise输入CUDA模型；33本地pass/1CUDA-skip、22远端pass包含CPU→CUDA bit-exact。局部源码比formal frozen snapshot多一行device conversion；formal9组显式CUDAnoise所以结果不受影响，冻结snapshot未覆盖。详见api_compat_verification.json。

## 下一步（未实现/未启动）

基线表说明真实动作重建/情感分布均有差距，不能只修判定器、追更大幅度或更平滑。下一轮先核对原模型各stage总预算、训练/验证趋势及与基线的输入信息差异，然后聚焦跨说话人的音频→动作条件泛化和B0发音泛化；以已训练EmoTalk/VOCA为诊断参照定位信息在哪里丢失，再登记一个有证据的结构变化做匹配实验。不能继续扫描ARrho/std/pooling权重，不能以oracle或辅助F1充当成绩。

服务器/root仅余约0.394GB，autodl-tmp也紧张。启动下次大训练前先规划可验证备份/存储；不得直接删历史数据或覆盖冻结结果。

## 固定参考

checkpoint /root/autodl-tmp/kinetalk_final_20260922/checkpoints/phase2_fullmouth_timing000_20261004/audio/final.pt
SHA e17659536a6fcaaaec2d9c22f99403fe4d9f1c8690c3cd182583894545f5ab45
manifest d4ef98bcb7f4e5bc94a15f27516bb3e67c4e61dcfff936e02cf6befefa6ceb1a
STD stats SHA fc9645c7719b7ac130b9d187900d87db8e45b63f1b794fcf27ebb1412bbc82e0
TRAIN lag report SHA 0b5b81c711bfb28f7e4683595c2893a2876589948851f53c6faede0902236fe6

远端phase16 /root/kinetalk_temporal_source_20261005（九组wrapper2690–2698/finalizer2699/native4843均complete）
远端gap /root/kinetalk_validation_gap_20261005/scores_retry1（PID4417 complete）
远端pool /root/kinetalk_pooling_audit_20261005/audit（PID5322 complete）
数据 /root/autodl-tmp/kinetalk_data/packed_trainval_20260923；python /root/miniconda3/bin/python；SSH用.codex-finalizer/remote_ops.py读取既有凭据，勿打印密码。

## 不重复的历史结论

Phase15 diagonal source：四probe/三seed F1收益复现，但口型位移+3.61%，不推广。Phase14仅修scalar强度失败；teacher-global oracle MBE .660是GT-informed诊断，不能部署，且不改善F1/时序。全局code MSE TRAIN .384/validation3.213；音频类别F1 .846不是生成F1。B0中性jawcorr TRAIN .702/validation .442。GT原probe F1约 .66/.64，低方差通道噪声敏感，但基线能取得明显更高生成F1。

Phase7–13 prototype、去endpoint consistency、时序adapter、freeze renderer、teacher agreement、B0dropout、去global MSE等未推广。已有gain/zero-noise/投影/ordinal/probe critic/线性与neutral-reference映射失败证据；详细读主计划对应章节。只用TRAIN/validation调参，不读历史sealed成绩决策；历史test存在，不声称从未访问。t-SNE为描述，不能替代正确情感/同步和公平比较。
