# 当前恢复入口（2026-10-05，Phase19运行中）

压缩后先读本页，再读主计划最新章节，不能重复启动或重做审计。目标仍是公平同协议论文比较领先、MBE≈0.7、生成原协议独立情感F1>0.7；目前没有新默认模型，目标未达到。

## 唯一运行任务

Phase19 /root/kinetalk_native_b0_target_20261005：pairworkers7679/7680/7681（seed47/48/49）顺序control/native，finalizer7682。查launch.json、seed*/control_state.json/native_state.json、postprocess_state.json。只训练B0，其他模块冻结。每臂同warm、batch16、现有loss、精确1568optimizerupdates；control3298片段安全中性监督，native12536片段音频自身同步GT。control8部分轮、native2完整轮；clipinputs24990/25072。没有新增loss/嘴部mask/模型层。Native B0是audio motion base，不能继续称其语义上中性。

先通过9远端/31本地测试；真实smoke2updates完成，101B0tensors改变，其他system+audio与warm完全一致。完整finalizer评1367nativevalidation几何、口型时序、四冻结probe，715TRAINneutral时序；base的F1只是B0诊断，不是完整flow生成F1。三seed预定联合门槛见主计划；即使通过，也需native分组审计和独立全teacher/audio下游重训，不能直接替换默认。

## 本轮已完成

Phase17 warm链：从头四stage各12轮 + fullmouth2audio + timing0002audio，hash匹配。B0 12轮/19788steps；identity12轮/132steps；teacher12轮/75216steps；audio-global16轮/100288steps；renderer/audio-temporal28轮/175504steps。基线80–100轮，但各loss/每step计算不同，不能直接声称欠训。详见training_budget_20261005/README.md与metadata.json。

Raw-emotion均值补充诊断：hidden128 vs hidden128+rawemotion768，固定ridge.001/五TRAINspeakerfold/fit-only标准化。teacher64 MSE .331714→.345320、真实平均动作offset MSE .00669546→.00692189；两项目的五折均变差，不加这个旁路，不扫描。只有固定线性读出结论，不能证明原音频没有信息。详见raw_audio_predictability_20261005。

Phase18身份参考预算：12vs120额外轮次/三seed，只22TRAIN独立参考对，原loss；所有非identity状态不变，前12trajectoryhash完全一致。参考TRAIN/val MSE .008240→.001556/.009589→.004217；中性query整体/valmouth改善，但TRAINmouth+30.32%、valbrows+9.75%，三seed联合门槛失败；不推广、不重训下游。报告、六identity小权重和summary已下载/hash核验。identity_budget_20261005。

## 存储与归档

/root原376MiB，完整逐文件验证后归档两组已结束rejected实验，现约6.1GB可用（启动Phase19后逐渐下降）。858文件、6.12GB全部保存在本地final_experiment/remote_archives/20261005两个完整tar，SHA/每文件/目录校验通过；远端原文件未变更复核后移除工作副本，原目录保留ARCHIVED.json恢复指针。/dev/shm有临时tar副本，重启会消失；本地是持久恢复副本。原warm、数据、基线、Phase15/16及条件缓存未迁移。详见归档README与manifest/verified/reclaimed JSON。不得把空指针目录误判为数据遗失，需先恢复。

## 默认与固定参考

仍是timing000 + standard source + 全51观察通道残差，原loss保持。
checkpoint /root/autodl-tmp/kinetalk_final_20260922/checkpoints/phase2_fullmouth_timing000_20261004/audio/final.pt
SHA e17659536a6fcaaaec2d9c22f99403fe4d9f1c8690c3cd182583894545f5ab45
manifest d4ef98bcb7f4e5bc94a15f27516bb3e67c4e61dcfff936e02cf6befefa6ceb1a
data /root/autodl-tmp/kinetalk_data/packed_trainval_20260923
python /root/miniconda3/bin/python；SSH用.codex-finalizer/remote_ops.py既有凭据，勿打印密码。

## 不再重复

Phase7–16、pooling、prototype/线性/neutral-reference映射、B0dropout、teacheragreement、去globalMSE、去endpointconsistency、冻结renderer、时序adapter、仅强度scalar与sampler均已有失败证据；详细见主计划。Phase15diagonal提高F1但位移变差未采用；Phase16AR位移降33–35%但MBE/LBE更差且jawcorr下降未采用。GT原probe约.66/.64，audio类别F1 .846不是生成F1，teacher-global MBE .660为GT-informed oracle不可部署。

同协议validation基线表已完成：EmoTalk-core MBE.7450/LBE.3315/F1.5919/.6862；当前timing000.8847/.4313/.1825/.2155。全部共享音频ARKit适配版，非官方原论文端到端复现。不要重复重评分，不读历史sealed报告来调参；原test历史存在，不声称从未访问。t-SNE仅描述，不能替代真实动作指标或公平比较。
