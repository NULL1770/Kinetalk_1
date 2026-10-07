# 最新恢复覆盖：Phase39训练/评分/原成员备份已完成，报告收尾恢复中

三seed47/48/49固定1568updates、1367val×3draw已全结束，各联合gate均false。49 F1 .641446/.554815、MBE .907466、LBE .457833、mouth速度MSE .006123984。各seed全部原成员已SHA备份并ack（49 manifest f0749e75…），没有训练/评分待跑。
唯一driver2830和collector11590因报告收尾NameError(completed未定义)已失败结束；失败发生在replicate_summary/validation_comparison写出后，保留原failed状态/日志/helper。只做报告恢复：独立原SHA备份收尾证据、核已写summary与原三seed报告平均一致，写新recovery状态，绝不重训/重评分/覆盖旧failed证据。下一步固定八类视频与具体新架构方案；未经用户确认不执行新训练/架构。

以下都是更早的进度快照，以本段为准。

# 最新恢复覆盖：Phase39 ACTIVE，seed47/48闭合，seed49参数更新已完成
最新只读状态：seed49日志已记录第二轮epoch_complete、total_steps=1568，第二轮144.405秒；参数更新已完成。尚未出现正式完整评分结果，driver仍标training（含阶段末验证/保存），不能说全实验已结束。先等原driver完成收尾与统一评分，不重复启动。此前epoch1 batch576/784为旧快照。

2026-10-07本次SSH读取：seed48训练1568updates、全部1367val×3draw评分结束，50原成员SHA备份与ack闭合。seed49为唯一活动训练（driver2830/child9014），读取日志时第一轮576/784，loss/gradient有限，collector11590等待49评分；root free647823360bytes。下文seed48仍训练的描述属于上次快照，不再适用。没有新launch或超出固定预算。

| seed48 clip_all，固定三draw | 同seed Phase37 | Phase39 |
|---|---:|---:|
| 原生成F1 128/64 | .571746/.511507 | .573553/.518637 |
| MBE/LBE | .918217/.454572 | .919972/.453549 |
| mouth velocity MSE | .006460780 | .006414645 |

seed48有小幅F1/嘴速度变化，但联合gate仍false，Phase34参考gate也false。report SHA `214928839daeacd3ac032397e168413c2541c09b31f94578d389d784e7eec15d`；manifest `e6bb994f452b79e53cd316db8005524d2330fde02e650cd5b2015f2f298fd8d9`。不能提前宣称三seed有效或使用seed间差异当作训练效果。

最新用户明确审批边界：继续完成已运行的Phase39及现有评测/备份；任何新的训练方案或架构修改，先说明依据、具体设计和正确性/效果验证，等待用户确认后执行。旧“无需确认即可后续修改”由本条覆盖。若39全部失败，优先讨论更丰富的表达表示与显式保持内容时序的生成机制；目前只列研究方向，不是已决定/实施的设计，不自动添加模块或新实验。

2026-10-07本次读取远端最新状态。唯一driver2830、唯一collector session11590（Windows PID94548）继续原固定47/48/49、每seed两轮1568updates；没有重新launch。正式binding SHA `3f270ccd16ce6f8458b15f57811713cf6e1f009f60c91fbc8cf09bf42de553a4`。seed47全部1367validation×3draw评分完成，53个原成员已SHA备份，backup_ack上传后driver仅回收已备份的本候选last/curves，进入seed48（child7533，最新正在B0缓存）；root free767897600bytes。seed49尚未启动；没有三seed最终结论。

| seed47 clip_all，固定三draw | Phase34 standardized | Phase37 named | Phase39 modulation |
|---|---:|---:|---:|
| 原生成F1 128/64 | .672117/.574858 | .638484/.572148 | .635258/.572974 |
| MBE | .900116 | .892900 | .893344 |
| LBE | .445204 | .440425 | .440135 |
| mouth velocity MSE | .003785621 | .006441897 | .006432124 |

新调制对同seed Phase37没有明显收益，联合gate=false，Phase34门槛也未通过；不要将seed47与旧三seed均值比较后声称改善。该候选不推广、不延长失败预算，但按原固定计划完成48/49。seed47 report SHA `b0fc166957a4b80b4178e9d382a00507376b9f457be48c0f1c9167f70910723f`，manifest SHA `0ec84e3f92fc325db9368e4b44b494ed383ef58c9e747c83eb3a34f04ad8b50d`。ack位于seed47/backup_ack.json，不存在seed47_download_verified.json；不要重复猜此文件名。

Phase39 seed47嘴通道合并centered RMS .07794，GT .07116；范围已不小，但centered相关约.2196。这不是jawOpen单通道统计，不能泛化为每类/每嘴通道幅度都正确。原128 probe三draw类F1 happy .8445，neutral .2757、fear .4705、disgust .5449；不能把看到happy不错等同macro全类别通过。Lip mean3.94552mm、Lip max7.49052mm、Expression mean.750688mm，均为该seed、clip_all、同rig协议。

下一步先闭合唯一顺序任务和完整汇总，再渲染固定八类新旧对照。随后优先定位学生表达均值偏差/动态预测，再针对口型残差破坏内容时序做单变量方案；维持772D/中性B0/全嘴开放/flow与generated对学生detach，不恢复内容输入或堆loss。不使用sealed、不替换默认、不上传Git。

以下为旧准备记录，仅用于历史；ACTIVE和已评分事实以上述新段为准。

# Phase39真实smoke通过，正式调度前历史记录

先读本段，不重跑下面的旧pending工作。Phase37完整392files/1782614795bytes原SHA已闭合，manifest e94da8617ea3d98b51e790e2b2784a6eb235653baaba86745f4ce1c41456e04d；recovery17983 exit0。因素11files、Phase38冻结表达预测/receiver诊断8files也全SHA闭合；65667/58248全部结束。八视频43101 exit0、正式PHASE37_RESULTS写成，原报告误写36路径已修。统一冻结probe t-SNE97003 exit0、原分数闭合，描述性输出phase37_descriptive_20261007，非SOTA证据。

Phase38报告bd38f110…：学生4state动态相关raise/down/squint/wide −.019/.239/−.003/.063；平均偏差主导stateMSE。GT state oracle把brow MSE .05392→.03913，但动态相关仍约−.02，dynamic与static oracle几乎相同；接收器动态使用不足，同时学生预测也有误差。参见PHASE38_STATE_RECEIVER_RESULTS_AND_NEXT_PLAN.md，不将oracle算生成成果。

Phase39实现仅新增renderer已命名state4→逐帧shift/scale的Linear4→384，共1920参数；调制完整52D头隐藏tokens，不划嘴mask/不接管upper9/不增加loss。默认关闭、零初始化、RNG保留、旧keys不变；config显式renderer_expression_modulation，必须配expression-prosody。3source变化dit/neutral_affect/trainer；student37源保持不变，772D/B0中性/renderer条件detach不变。20related passed，全suite399passed/1skip，gitdiffcheck clean。

唯一远端 `/root/kinetalk_phase39_expression_receiver_20261007`；真实smoke2509/2510已结束并通过：旧named默认全state+actualstreams精确复现Phase37，开启初始loss相同，两update参数梯度非零，学生state有梯度/旧local未用/fullflow和generated→学生grad0、嘴renderer有梯度、HuBERTNaN隔离、冻结/padding/finite、student及完整system config恢复生成逐位相等。不能把两update当性能改善。prepare绑定0b74a627…为prepilot；后续只加调度/备份helpers更新binding，须保存prepilot原件。

容量：原root与数据盘都紧张。Phase39 smoke写入持久数据盘 `/root/autodl-tmp/kinetalk_phase39_expression_receiver_smoke_20261007`，root/smoke为显式symlink。按真实旧tensor size+newparam余量预算389221758bytes通过。主37全SHA后回收7成员389650608bytes（3named last＋2smoke last/curves），root free850210816；全final、Phase34及Phase37正式3baselinecurves保留，receipt见verified_intermediate_reclaim.json。本次未用RAM-only产物。

下一步唯一正式pilot：3seed47/48/49，各2轮1568updates，基线复用Phase37named；另须通过Phase34Full质量gate，不能拿从37退步中恢复当超过34。由于磁盘，逐seed训练→完整评分→本地原SHA备份→仅curve/last中间文件再远端SHA复核回收→下一seed，保留全部final/报告/数组。单worker最坏实际文件数预算+128MiB原子last安全余量+过去seed持久final/report余量；无1.8GiB并发假门槛，不改变预算/抽样/公式。调度/collector待准备，尚未正式launch，没有39新成绩。

禁止重复运行prepare_phase39/build_phase37/update_phase37_recovery。phase39_status.py只读。旧failed collection/localfinish证据仍保留；当前完成由recovery_collection_state/download_verified/fixed_video_state/readout/report证实。用户目标仍正确架构下口型/表达/身份及各指标SOTA，尚未达成；Git暂停/sealed未读/发布默认未替换，不新建子agent。

以下为Phase37接续历史，用本段最新状态覆盖。

# 当前恢复入口：2026-10-07 Phase37已拒绝，备份恢复

## 最高优先级事实

用户授权继续正确架构优化、修改与实验。中性B0；学生772D=emotion2vec768+prosody4，HuBERT只直接进B0/h0；全部嘴通道前向开放。不堆critic/交换/ordinal，不用Phase26 native-GT/1540成绩。Git暂停，sealed test未读，默认未替换，SOTA未达成，不创建子agent。

Phase34–36已完整闭合，不再wait或重复分析旧代码。Phase37训练/全部评分/独立因素诊断也已经完成，三seed联合gate全false，拒绝且不延长失败候选。

## Phase37正式结果

远端 `/root/kinetalk_phase37_named_expression_prosody_20261007`；本地 `final_experiment/evaluation/diagnostics/phase37_named_expression_prosody_20261007`。
driver22623/workers22624–22626/closure22627全部完成。三seed47/48/49，各2轮1568updates，对照复用Phase34 standardized/full；全部1367validation×3draw、raw/clip、4冻结probe、same-rig、配对CI及neutral gate已完成。binding589ba6010e374f9f503cebbcf0d1323e199557476a008b82e1787da48875b367。

| clip三seed/三draw平均 | Phase34 Full | Phase37 Named |
|---|---:|---:|
| 原生成F1 128/64 | .683872/.581147 | .617731/.549179 |
| MBE/LBE | .904603/.446154 | .906694/.451459 |
| Lip mean mm | 4.008668 | 4.021237 |
| 嘴速度MSE | .003709471 | .006339550 |
| jaw centered corr | .255015 | .226260 |

嘴速度误差+70.9%，没有实现大幅提升。报告gate解释字符串遗留Physical-unit pilot，实际flow units/formulas未变，只比较named学生职责；不得改绑定证据以修文案。

## 实现与已完成检查

仅trainer/slow_state_affect两source及新测试：expression-prosody模式、ua64=4有符号上脸state+4标准化实测prosody+56zeros、TRAIN情感×强度原型global目标、renderer条件detach。state用原生眉眼GT/独立neutral anchor/TRAIN scales/stride16 spline；不读取嘴GT。原teacher仍看全残差，不宣称完全解耦。旧local_head保留但候选不用。
全suite393passed/1skip；元数据后17相关pass。两真实2update smoke通过：default精确复现Phase34，named flow/generated→学生梯度0、state_head有梯度、local_head不用、嘴renderer有梯度、HuBERT NaN隔离、padding/frozen/finite及config恢复精确。不要重复。

## SSH恢复与唯一备份writer

SSH重启后读取成功，所有训练/评分/因素证据存活；root free约0.436GiB，不能启动新实验。
旧collector50441、factor35609、localfinish42620均已失败结束（当前工具session不再存在，Windows也没有对应writer）。旧collection_state/local_finish_state/.part保存。main原manifest392files/1782614795bytes，未闭合完整备份；小报告/24渲染输入必须核原manifest SHA。
唯一恢复helper `.codex-finalizer/recover_phase37_backups.py`，session17983；先factor再main，读取原manifest、按原member SHA续传gzip/offset，保留失败状态，写新recovery_collection_state，不重训/重评分。原manifest和失败状态另存interrupted_backup_evidence。不能再并发启动原collector/finish。先前说改分块传输但未实施；断线不等于旧协议缺陷。
本地summarize_phase37_local原误写PHASE36_RESULTS路径已修为PHASE37_RESULTS，尚未执行。此修复不改远端评分证据。

## 独立因素诊断

root `/root/kinetalk_phase37_factors_20261007`，driver23079 complete；seed47 named/66cells，三draw重放严格0，模型状态不变。whole ua及分别expression4/prosody4 static/reverse/shuffle、独立身份reference/code/bias干预；异人只看response，不用原人GT评分。
helper5ea0dce56775d1334c2c164b4316973738d7659930dacf1e459eb9e8668bc9ad；report59010c9855be102f4d28ea166fb43e72460fdcb27858e61c1b3725ebb2e87556；exportsbfd7fbad31c66381c32d875751a6dbb69483136c6036f5e16055348c77d5a323；replay718377c4d0f2a234eb436f8d87d34c1face682a81ba54a3edc3b12f2696c49fd。

| clip66cells平均 | jaw corr | jaw velocityMSE | brow range | smile range |
|---|---:|---:|---:|---:|
| original | .286264 | .0105201 | .089961 | .225793 |
| expression static | .288839 | .0104916 | .090042 | .224719 |
| prosody static | .295681 | .0099801 | .083767 | .218241 |
| whole ua static | .297803 | .0099677 | .083835 | .217126 |

纠正旧口头解释：主要是static韵律/whole ua降低动态，expression static影响很小。4表达状态对renderer影响弱；尚非内容泄漏的因果证明。需读详细report的表达幅度/预测误差与身份response再决定单一Phase38改动。

## 下一步

1. 完成唯一恢复writer全部原SHA备份；不更改旧failed状态/manifest。
2. 用summarize_phase37_local写正式结果；render_phase37_fixed输出固定八情感GT/B0/Latent772/ExpressionProsody772，无gain/retiming，建立新的恢复finish状态。
3. describe_phase37_emotions：原128冻结probe、Phase34 canonicalGT与Phase37 saved per_clip，无refit/重新提取；先核源SHA。
4. 详细因素诊断分清学生表达预测/renderer条件利用/身份响应，写单一下一步方案与必要检查，然后实施固定budget实验；不盲扫权重。
5. 新训练前容量闭合，只有完整本地SHA以后才能回收远端重复last/curves；保留final、Phase34baselinecurves和receipt。

## 数据、架构与历史边界

详见CURRENT_MODEL_AND_TRAINING.md。12536TRAIN/22人、1367val/3人；每人2独立neutral身份参考。B0715native neutral+2583approvedemotion→neutral；嘴监督55.375%，eventmask0/1且局部gate有效。469强度三元组jaw严格递增49.89%，不统一ordinal。
B0四TCN+2encoder/3decoderTransformer；teacher三maskedTCN；student四maskedTCN；renderer四192D/6headDiT。身份code128/bias52改运动风格，固定rig不改脸形。u_a仍需实际时序验证，原型teacher不保证纯情感。
GT原128probe F1=.66034433，整段统计motion F1并非逐帧分类器，也不是唯一论文标准。MEDTalk/EmoTalk协议已核，DESTalker全文未核；t-SNE不能证明解耦或SOTA。发布默认仍phase2_fullmouth_timing000_20261004。
Phase35/36完整备份与失败结果见各RESULTS；9个重复回收1805119840bytes有原SHA及reclaim_receipt，全final及Phase34三baselinecurves保留。旧路径重放须恢复已删原member。
SSH只经remote_ops私有helper读取，不输出/提交凭据；远端Python/root/miniconda3/bin/python。WindowsUTF8、rg文件搜索、boundedSFTP16。
历史恢复页已归档，除缺失历史证据外不重读。
