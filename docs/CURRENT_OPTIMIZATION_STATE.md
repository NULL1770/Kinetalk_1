# 当前恢复入口（2026-10-06）：成果已上传，等待后续方案批准

先读 `NEXT_DISENTANGLEMENT_AND_IDENTITY_PLAN.md` 与 `EMOTION_STUDENT_INPUT_CORRECTION.md`。最新用户要求：先上传已有成果，再写身份/内容/情感动态优化方案，明确同意之后再新增代码。当前GitHub指定分支已推送7a24423和8e154cd；全本地测试363通过/1跳过。仅记录现有实现和新方案；没有新训练、新默认模型或772D新成绩。

Phase29完整492文件下载SHA核验已完成，原exec88017结束，不再等待或重下载；三联合gate均false，dropout拒绝。最新保留候选Phase26 standardized12原clip生成F1 .796875/.721358，raw .619331/.591701，MBE .870982/LBE .421170；默认发布权重仍旧timing000。旧Phase30未执行，已暂停其混合输入设计，不启动旧helper。当前无活动远端任务。

下一批拟批准D0冻结身份/code/bias/u_a干预及优质配对审计，D1匹配772D迁移pilot。D2显式逐帧表达监督、D3质量配对交换、D4嘴部幅度/时序接收结构是条件待办，不能自动一起实现。正文计划有数据/冻结/预算/指标/接纳门槛和论文来源边界。压缩后必须恢复此停止点，不能把历史ACTIVE状态误认为进行中。

# Historical: Phase29 full report COMPLETE, candidate not accepted; full SHA download active

Three candidate trainings, six mesh and three full native/raw+clip/probe/global/precision/CI audits COMPLETE. Repaired state postprocess_repaired_v2_state.json complete; report_repair_v3.log closed. Original failed launch/postprocess and all repair failures preserved. Three joint gates false; no default promotion or sealed read. Generated clip originalF1 .797647/.715031 vs control .796875/.721358; raw .610685/.583535 vs .619331/.591701. MBE .864415 vs .870982, LBE .420868 vs .421170, lipmean3.558562 vs3.562193mm, max6.696472 vs6.699290mm. Mouth displacement MSE +3.47%; continuousglobal valMSE2.309306 vs2.327279 (<1%gain), meanbias worse. Do not promote dropout or scan p. Geometry not SOTA; see PHASE29_RESULTS_AND_NEXT_STEPS.md.

Only local full downloader active: exec session88017, .codex-finalizer/download_phase29_repaired_v3.py. Wait/poll this session, never parallel write/download same targets or rerun model. After download_verified.passed, run summarize_phase29_local.py for SHA-bound compact readout, update status and report conclusion. All six mesh JSON/completion files already downloaded/report SHA verified. No active remote training/audit, no new trial. Old collector stopped failed. Architecture/data docs CURRENT_MODEL_AND_TRAINING.md retained; update stale Phase29 ACTIVE wording after closure.

# Latest: Phase29 v2 trainings/mesh COMPLETE; report-only closure actively scoring

Fresh report driver4838 launched via tools/launch_phase29_report_repair_v3.py. Canonical report state postprocess_repaired_v2_state.json, log report_repair_v3.log, launch record report_repair_v3_launch.json; currently seed47 native raw scoring. Actual statistics SHA and all remaining source fields exact for all3seeds; six local identity-contract checks pass. Original failed states/helpers retained; v1 additionally failed child frozen-code import, v2 launcher accidentally referenced v1 helper (caught before output), both retained. v3 binds child PYTHONPATH to frozen code and asserts executed helper matches registered hash/fresh state; no model/formula/RNG/optimizer change. Do not repeat launch. On complete use .codex-finalizer/download_phase29_repaired_v3.py (only one downloader) and inspect download_verified.json. Old collector is stopped failed. No active model training.

2026-10-06 recovery: seeds47/48/49 all finished12epochs/9408updates; exact actual input/noise streams and private RNG counts verified, durable archives verified. mesh_state complete. Original launch/postprocess failed ONLY at comparing flow_source_noise.statistics.path: independently copied TRAIN statistics have different paths but identical recorded SHA and values. No training/audit remains active. Collector stopped failed. Do not restart training or wait old PIDs. Preserve failed states/helpers/logs; repair report comparison by verifying both actual file SHAs then ignoring ONLY statistics.path, retain all other strict equality. Run fresh report closure against existing final/curves, reuse six mesh reports, download and SHA verify full evidence before interpreting F1/global results.

Initial geometry (pending complete report download): candidate three-seed clip MBE .864415 vs control .870982; LBE .420868 vs .421170. Improvement small; not .7/SOTA. Full candidate F1/global audit not yet generated. Root/data free about672/564MB; no blind new training. Previous ACTIVE status below is historical.

# Previous: Phase29 v2 fixed3seed12 ACTIVE; seed47 preserved, untouched48/49 dispatched

本轮结束前最新：47epoch7batch351/5055updates，48epoch4batch326/2678，49epoch4batch51/2403（需下次查新状态）。完整新F1/MBE尚无。32项本地回归通过（新增4 dropout语义检查），CPU/GPU及real2update证据已持久保存并重核SHA，gitdiffcheck clean。rootfree826MB、datafree1.06GiB，足够既定3archive+finalmeta，禁止盲目再开实验。durability watcher已无损替换为2665，处理sourceatomicreplace/archiverunlinkrace，复用已有SHAmanifest，不修改训练；三个epochcheckpoint已持久。

自动完整pipeline已补齐：固定3final→全1367×3draw×4冻结probes/raw+clip/nativegroups/precision/pairedCI→冻结TRAIN/valglobal核查→同632vertex rig的MBE/LBE/BSFDD/LVEmean+max+sq/EVE/FDD及逐speakerCI→与已有VOCA-core/FaceFormer/EmoTalk-core/FaceDiffuser adaptations validation比较表。Report-onlyclosure addendum保留初版finalizerSHA；不改训练/metric公式，不SOTA宣称。局部mesh控制47complete，48进行，49后续，candidate仅在各fixed12complete才评分。

本地只读collector PID91900 `collect_phase29_when_complete.py` / `.codex-finalizer/phase29_collection_state.json`，等待remote launch/mesh/postprocess全部complete才自动完整下载并逐fileSHA核验；当前waiting_fixed_run。下次先检查该state及download_verified，不能并行手工下载同目标。完整downloader `download_phase29_completed.py` 只用于collector或确认其已停止后补传。当前没有交互execsession需要wait。用户最新继续已落实为实际训练、独立诊断和完整自动验证/备份，不把正在运行说成完成。

Aftercapacitycheck128GiBhostlimit/~114GiBfree andGPU20.1GiBfree, dispatcheronly1273 replaced by2188; active47worker1274 was NOT restarted, untouched48/49 freshworkers2190/2191. Training source/budget/noise unchanged, launch=fixed_training; parallel_adoption.json persists schedulingaddendum. CPUmeshaudit accidentally inherited64BLASthreads consuming15cores; activeauditor affinity2CPUs/nice19 and futureenvthreads2 cap fixed onlyauditresources, notformula/rig/source model. GPU now98%, ~5.3GiB. Latest47epoch2batch651/1435updates;48/49basecache. Durability1532 saveslast root (epoch1/784updates verified). Fullmeshread-onlydriver1640 runs samefixed632rig+metricsascompletedbaselines; controlsnow/candidatesafterfinal12; no sealedread. No finalcandidate result yet.

Additionaljaw3drawvariancecomplete: neutralclip randomdrawvariability accounts25.5–26.5% ofnativevelocityerror; diagnostic3drawmeanstillcorr .317–.328 vsB0 .462–.481. Thus stochasticvariation contributes but doesnotexplainall timing damage; do NOT deploymean-of-draw or replace reported stochasticscore. jaw_draw_variance report persisted remotev2.

# Previous: v2 serial dispatch and container recovery

最新canonical `/root/kinetalk_phase29_global_dropout_v2_20261006`，driver1273/当前seed47worker1274。原16530/16531–16533消失且旧dev/shm全部清空；三个日志停在base_cache，没有stage_start/optimizer更新。原root保留interruption_audit及launch=interrupted，不能等旧PID或把旧launch当active。模型/source/budget/seed/p不变，仅同预算fresh串行调度，recovery_addendum记录；已从全SHA一致持久archives恢复三个completed standardized12 control curves，未重评估/训练对照。smoke报告及model检查持久继承，旧失败记录保留。新增只读durability watcher将每次atomic RAM last checkpoint（含optimizer及private RNG）复制root/durable_last并SHA核验，防止RAM再次清空丢全部更新；不修改模型或随机流。

中性jaw只读定位complete，本地report/per_clip SHA核验：B0corr .470949→最终clip .253362，B0位移MSE .00127525→.00193531；动态残差corr−.175299，新增速度能量大于纠偏。幅度已接近GT而时序受残差干扰，非单纯clip问题。docs/PHASE29_JAW_LOCALIZATION.md。当前不加mouth mask或并行时序改动。

# Previous: Phase29 first concurrent dispatch interrupted before updates

Canonical `/root/kinetalk_phase29_global_dropout_20261006` driver16530/workers16531/16532/16533，launch.json=fixed_training，GPU92%。CPU/GPU default/privateRNG/masks/nonzero-u_a/strictoldstate checks +28 local regressionchecks pass。真实2update smoke2已完成，闭合读已有states（smoke_closure_state complete、smoke2_verification passed）：old/new defaultstate/loss逐位等，实际sample/GT/mask/noise/ft与原全部RNG一致，private drawcount exact，B0/identity/teacher冻结。首次batch16仅1update、第二次report numpyRNG比较错误的失败记录保留；闭合只修report，不重复GPU smoke2。

固定p=.1/seeds47/48/49/12epochs/9408updates。整个训练输出RAM，完成后全文件/目录无损archive在data盘并逐SHA核验才标记complete；root另存final/meta，旧实验不删。reuse completedPhase26 standardized12 control，全actual streams结束必须相同；full1367×3draw×4probes/raw+clip/regions/groups/precision/raw+clipCI +冻结TRAIN/val连续global核查由driver自动接续。无最终候选指标、未推广timing000、sealed未读。jawbaseline只读定位16979独立进行，无时序模型修改。下次读此页→查launch/seedstates/RAMstatuses及postprocess，不等待旧session，不重复启动。

# Previous: Phase29 patch ready; smoke/launch pending

2026-10-06 resumed: SSH/GPU idle checked, no Phase29 or active job. docs/PHASE29_GLOBAL_REGULARIZATION.md preregisters one candidate p=.1 dropout on pooled128 before global_head; no loss/parameter/channel mask. Two files patched separately from bound Phase26 snapshot and local lineage, preserving previous differences. Independent CPU generator/checkpoint state; original unfreeze keeps eval, wrapper only marks audio root during explicit train call then restores; child modes unchanged. No candidate result or training yet. Default timing000 remains. Next run meaningful model/private-RNG/default checks and actual old/default/candidate2update smoke, storage plan then fixed3seed12/full audit. Do not repeat Phase26/27/28 or report oracle/heads as generated improvement.

# Previous: Phase26/27/28 complete; generated clip F1 target reached, continuous condition generalization gap established

### 2026-10-06 18:09–18:13 CST：最新恢复入口
Phase26fixed12及Phase27oracle定位全部complete、完整文件分别546/22全部SHAverified。Phase28当前canonical `/root/kinetalk_phase28_train_global_v2_20261006` launch及三个seedstate均complete，20远端files完整SHAverified；全部12536TRAIN/1372104native帧/22speaker/3模型，无拟合/无sealed读取，frozen model及真实TRAINstream三seed一致。原Phase28失败root及日志保留，不重启。当前没有远端训练/审计或本地挂起exec session。

TRAIN globalMSE .148714/.149859/.149300，validation 2.304595/2.323142/2.354100，差15.50/15.50/15.77倍；TRAIN meanbias仅.00213/.00182/.00182，而validation .667/.735/.765。TRAIN-LOSO常量offset只有约1%改善，不拟合validation，也不重复旧线性/prototype/ref映射。去各split情感×强度均值后的global相关：TRAIN .931/.925/.928、val .204/.192/.221，表明类别内部的连续细节跨未见speaker对齐不足；不能把它当因果身份泄漏结论。audio headTRAIN≈.999、val≈.85不是generatedF1。详细证据docs/PHASE28_GLOBAL_GENERALIZATION.md，局部oracle限制docs/PHASE27_GLOBAL_LOCALIZATION.md。

最后真实部署协议成绩仍standardized12 clip原generatedF1 .796875/.721358、MBE .870982/LBE .421170；rawF1 .619331/.591701，三个jointgates false，defaulttiming000未推广，不能称SOTA。GT-global MBE .632573只定位用，不能报作模型达标。下一单变量候选限audio-global泛化/正则化，不预先扩大嘴/加loss/改变身份或u_a；必须先核对旧失败、默认/RNG/实际stream/frozen smoke，再固定三seed matched预算。中性jaw时序另行拆B0/残差，不同时改。新模型候选尚未实现或launch。

最新下载6487首次碰到Windowsstate.json替换短暂WinError5（与同目标metadata下载同时），metadata已结束后增量重试完整通过20files，未改文件权限/远端数据。今后完整downloader和metadata不能并行写同一目标，可先下载小文件或等待同一downloader；旧partial不作完整证据。

### 2026-10-06 17:42 CST：六fixed12和完整审计均complete（当前恢复入口）
Canonical `/root/kinetalk_channel_coordinates_stable_budget_20261006` launch/postprocess均complete，driver已完成，不重启。六臂12epochs/9408updates，full1367validation×3draw×4冻结probe、source/frozen/stream/prefix/precision/groups/pairedCI/2vs12均通过完整性检查。原generated F1三seed均在clip_all协议>.7；三seed均MBE不达.7、jointgate=false，未推广默认，sealedtest未读。

三seed均值centered12→standardized12：MBE .887183→.870982、LBE .434720→.421170；原generated128/64 F1 .292569/.353879→.796875/.721358，辅助 .458828/.459079→.800471/.783213。standardized原raw F1 .619331/.591701，raw MBE .884162；必须同时保留raw与clip，不能隐去裁剪依赖。standardized2→12：MBE+.015167、LBE+.004543，口型位移MSE .002201024→.001816626（−17.46%），嘴部range .090651 vsGT .091991、jawrange .181152 vsGT .175279。嘴MSE .0147806、眉MSE .0545518，其中眉meanbias .0526932约96.59%；不能统一扩大口型或继续仅延长预算。

中性jawcorr三个control→standardized分别.283986→.255299、.293686→.247184、.285504→.257602，三个均退步。下一步Phase27：先冻结当前三个standardized12做global-only GT-informed定位，复用原native-padded/同noise/同identity/同u_a/intensity条件；必须先正常audio逐位重放当前curves，再oracle，仅用于定位、不冒充部署成绩。同时分解每通道/情感/强度/说话人静态误差及B0/identity/residual贡献，证据到位再设计单变量修改，不能盲目叠loss。

最终增量下载7256已complete：546files/12arms（六pilot+六formal）全部size/SHAverified；六formal final12权重也各自重核complete SHA，metadata93306complete。无本地挂起session。旧17:17下方为历史。当前无新训练任务，无模型/loss修改。固定12轮结果说明见docs/PHASE26_FINAL12_RESULTS.md。

Phase27已complete：`/root/kinetalk_phase27_global_20261006`。三seed×三draw正常audio重放bit-exact/maxdiff0，oracle全审计及frozen state通过；22files/3arms完整SHAverified下载结束（79361complete）。global-only GT条件：MBE .870982→.632573、LBE .421170→.335949；眉meanbias .052693→.013014（−75.30%），嘴meanbias .007964→.003528（−55.70%）。MBE配对speaker CI三个均严格负。原clip F1 .796875/.721358→.795215/.699641，CI均跨0；嘴位移MSE+.000057670（+3.17%）、jawcorr .315339→.312886，不能把oracle当部署成绩或宣称联合解决。原raw F1 .619331/.591701→.719158/.677737。global validationMSE2.305/2.323/2.354，整体均值偏差占约29%–33%，cosine .645/.626/.628；audioheadF1 .844/.863/.841不是generatedF1。眉channel41–45 GT全clip平均约.254/.225/.234/.157/.103，B0约0，identity约.025–.033；音频renderer残差偏小，非嘴mask。

下一步Phase28当前canonical为 **`/root/kinetalk_phase28_train_global_v2_20261006` driver13399**。旧root `/root/kinetalk_phase28_train_global_20261006`三worker完成forward后在report生成`sha(__file__)`处失败（str没有open）；旧launch/state/log已保存本地，不能当complete或重启旧root。修复helper哈希接受Path/str、代码指纹回归通过；v2额外在报告前先持久保存全部codes及frozen/stream检查，避免未来格式错误重做GPU。未改模型/训练配方/统计假设。v2实际launch一次，读取当前三final12全部12536TRAIN/22speaker的audio/teacher codes，B0和独立TRAIN enrollment重建，冻结不拟合网络；对比已有Phase27validation codes的连续误差、分情感/强度/说话人和条件去均值speaker方差。唯一offset假设仅在TRAIN留一speaker交叉检查，不对validation拟合/选参数；原encoder看过全部TRAIN，LOSO只检验offset估计，不代表整模型未见speaker交叉验证。目的是区分TRAIN拟合不足和未见speaker泛化缺口后再改；不能先盲目fit affine或增加loss。代码统计fixture checks通过，远端state须核验，无新模型训练。下载helper现指向v2，必须complete才下载逐SHA核验。

### 2026-10-06 17:17 CST 服务器实查（当前恢复入口）
Canonicalroot `/root/kinetalk_channel_coordinates_stable_budget_20261006`，driver3999仍为formal_training，无训练失败。三个centered controls全部完成12epochs/9408updates、全validation曲线和持久归档；三个standardized candidates均epoch11：47=batch76（7916updates），48=batch676（8516），49=batch726（8566），固定继续到12。六组prefix报告均passed，前1568updates的system/audio全state及实际sample/GT/mask/noise逐位等各自新stable2，全部本地报告也已核验，不能重复启动或放宽门槛。17:14 GPU96%/10451MiB，17:17 root剩2.061GiB。candidate完整epoch10 TRAIN flow分别.398343/.396007/.397282，低于epoch2约.508/.501/.512，仅为TRAIN下降，不能当作validation质量改善。

尚无正式12轮完整审计/replicate_summary，不能报告新F1或MBE。最后有效完整stable2指标仍是原generated128/64 F1 .778163/.665453、MBE .855815、LBE .416628；三个联合几何/口型时序gate都false，默认timing000保持。无新增loss/模型层/嘴mask。completed arms增量下载session96448已结束：372files/9arms（六pilot+三formal controls）全部SHAverified；metadata同步93481也结束，无本地挂起session。下一步只查canonical状态；自动driver会在六fixed12全部完成后审计全1367validation×3draw×4冻结probe、2vs12及联合/绝对目标，之后再增量下载候选和最终审计。只读TRAIN/validation，不开sealedtest，不重跑旧基线。

### 2026-10-06 最后检查：formalcontrols epoch11
最新47=8166/48=8516/49=8341updates，epoch11，三prefixguard passed、无失败，driver3999后台继续自动pipeline。localcheck：pilotrefs12个、234baselinefiles完整SHAverified、8native视频complete、6tSNE regressionchecks通过、gitdiffcheck无空白错误。所有本地exec会话均已完成；无需wait旧session。12标准化臂仍queued after每seedcontrolfinal12。下次继续先读本页及runbook，查root/launch.json/seed*/state/status，不重复启动；固定six12complete后自动fullaudit，最后download_stable_completed增量下载核SHA并判joint/absolute目标。当前完整2epoch F1 .778/.665、MBE .856，defaulttiming000不变。

### 2026-10-06 恢复快照：formalcontrols epoch10；所有基线可视化complete
latestupdates47=7207/48=7482/49=7382，均epoch10，driver3999 formal_training；rootfree2.8GiB，没有训练失败。三control严格ownpilot1568已passed，candidates等controlfinal12之后各自接续，不能重复launch。全部六2epoch+234files本地SHAverified；全部8emotionfixeds47draw42 nativevideos已complete/allinput-wavSHA/rigmaxerror0/首中末非blank，all_video_manifest及8midframe汇总保存descriptive目录，所有图已实际查看。newjointtSNE/probe closure/report完整。结果解释集中docs/PHASE26_STABLE_RESULTS.md，架构数据docs/CURRENT_MODEL_AND_TRAINING.md；默认未推广，目标未达。所有本地execdownload/render/tsne sessions均已完成，未留挂起；下一次只读canonicalstatus→等待fixedformal6finals/fullaudits→增量download_stable_completed（跳过已验证旧files）→joint结论。不要重复重放/修改snapshot/model/loss。

### 2026-10-06 formal controls epoch8；分情感/t-SNE完成
三12epochcontrol更新47=5814/48=5989/49=5914，epoch8，无失败。两轮全3seed×3draw原generated128/64每类F1：neutral .461/.389、happy .895/.892、fear .779/.581、sad .880/.638。剩余宏F1瓶颈主要弱neutral及原64fear/sad，不能继续盲目强化happy。GT本身两probe宏F1约.66/.64、各类不完美；生成F1超过GT只表示统计更易识别，几何/时序仍必须检查。
联合fixeds47draw42全1367val GT/control/standardized原128冻结probe hidden t-SNE和pooledconfusion已生成并查看：descriptive/pilot_emotions_validated/generated_emotion_tsne.png/pdf、generated_confusion.png；GT/control/candidate savedfeatureF1各1e-12重放闭合。固定jointmap，不按标签重新fit/选图；TRAIN冻结normalization，描述性可视化不能证明SOTA。全8固定emotionbaseline渲染session83451进行（happy/neutral已verified，angrycomplete）；各input/wavSHA/offset绑源，first/middle/last非blank/rigkeys maxerror0。第一次allrenderhelper默认Windowsdecode导致JSON反斜线错误，改UTF8后正常，不影响模型训练。旧空failedtSNE目录保留。

### 2026-10-06 三formalcontrols真实prefix guard已通过，epoch3
最新r/seed47/48/49 control分别1894/1919/1869updates，三control_contract.json.prefix_replay.json passed：前1568system/audio全部keys/dtype/shape/value逐位等对应新stable2，sample/GT/mask/noise也exact。原数值失败已修复，继续固定12不重启。GPU99%/9.99GiB。baseline所有234files/六finals/fullnativeaudits已本地SHA验证，downloadsession35479complete；prefixes和pilot_binding/summary也下载。formal candidates仍queued aftercontrolsfinal12；没有正式12最终指标。
渲染已完整通过：`final_experiment/evaluation/rendered/phase26_stable2_happy_s47_verified_camera_20261006/comparison.mp4`，fixedseed47/draw42，GT/control/standardized，103真实validnative帧25fps，audiooffset0/trim.04、wavSHA匹配；first/middle/last非blank检查及actualframe50人工查看都通过，rig keyvalue maxerror0，原blendSHA不变。两个旧失败目录不交付。新版tSNE helper仅本地修复，六checks通过，旧F1不受影响。用户新“继续”仍推进本任务，默认timing000无promotion，绝对目标未达。

### 2026-10-06 新stable两轮全部审计通过，formal12已实际启动
driver3999 statusformal_training；pilot_run/postprocess complete，pilotrefs/contracts/summary独立SHA绑定，完整性全部通过，无metric选择。stable2三seedclip均值control→standardized：MBE .873147→.855815、LBE .419423→.416628，原generatedF1 .272033/.286513→.778163/.665453；辅助 .414928/.337609→.792092/.748873。与历史Phase25结论一致，三jointqualitygates仍false；新2轮不是额外改进、未推广默认。现在root/seed*/state/status是fresh12，前1568必须逐位等对应新的stable2。基线state和近constant/std/region/group/CIs都已保存，绝对F1/MBE目标仍未全部达成。
局部视频新fixed happy GT/control/standardized正在本地verified_camera目录渲染；旧headworker相机参数产生blank、系统OCIO导致contrast枚举失败，这两个旧失败视频保留但不能交付。已恢复旧成功dynamic renderer的camera0/spacing.49/cols*spacing，并只在Blenderchild清除外部OCIO；新增first/middle/last blank检测。无模型/系数/数据变化。

### 2026-10-06 六stable2端点全部complete，native audit开始
driver3999当前pilot_auditing，全部control/standardized seed47/48/49 1568updates/full1367曲线/无损持久归档完成，无失败。pilot_run/postprocess_state auditing seed47；此审计完整性通过后driver自动formal12，尚未生成正式12指标，禁止重复launch。当前增量下载session99291（补已完成标准化臂），旧30936complete100files/3controlsSHAverified。以后读本页→远端canonicallaunch/state判断真实进展，不把较旧章节当当前。

### 2026-10-06 controls complete/local verified; standardized pilots epoch2
三个stable2controls已1568updates/fullvalidation完成并逐arm29files持久无损归档/全字节恢复核验。seed47完整system/audio还逐位等serial stable_a，parallel路线也闭合。新download_stable_completed.py bounded+resumable SSH已本地下载100files/3完整finals约150MB全SHA正确，localsession30936已结束；后续重复调用只补新完成arm/审计并跳过匹配现有SHA，不重复下载。标准化pilot分别epoch2batch401/676/576（47/48/49），无失败；driver3999会自动pilot审计后启动12，勿重复。
另在本地修复旧descriptive t-SNE utility：centered upper9 mean数学为零，改用原有正确mean/std/q10/q90/相邻速度54维特征，6regressionchecks通过，schema v3。旧v2报告搜索未发现已生成输出。该工具不参与当前四个F1 probes；F1实现本来正确，远端snapshot未变，不据此改称指标提升。HuBERT-base-ls960第6层与native recipeSHA精确匹配；emotion2vec实为4prenet+8shared/768/12head；source证据和模型文档已更新。

### 2026-10-06 新stable实验已实际启动（禁止重复launch）
Canonicalroot `/root/kinetalk_channel_coordinates_stable_budget_20261006`，driver3999，2epoch pairworkers4000/4001/4002，seed47/48/49 controls实际训练，GPU99%/2.7GiB。preregSHA2794a3bdbfa2ebcda6447b1c3b70479621f4326c53ee66c7be6d84c77c98f7a7；13helperSHA/独立source,input,stats,probe绑定/远端storage完整恢复与duplicate拒绝检查通过。launch.json是canonical调度状态；pilot_run/seed*/{control,standardized}_state.json为baseline，root/seed*/同名是12epoch。
顺序自动执行：全部六2epochfinal→全1367val audit/四probe/geometrytiming/precision/CIs→绑定新pilotrefs+summary→fresh六12epoch→各自1568state/stream逐位等新pilot才继续→全audit/2vs12/absolutegoals。pilot gate结果不用于挑选/取消12，只有完整性失败才停。每arm所有文件先持久tar.zst、全目录/文件字节SHA验证/原副本未变核验，再只回收已归档last及curves工作副本；SHA一致RAM曲线symlink用于审计，原始完整bytes可恢复。新正式指标仍未生成，旧失败root保留，不启动。最后有效旧Phase25原128F1 .778/原64 .666、MBE .856，口型时序联合gate失败；未推广默认。

### 2026-10-06 stable replay完整通过 / storage complete
stable_a/b真实1568updates的全部system/audio张量逐位一致，sample/GT/observations/noise也逐位等历史Phase25输入流；stable_summary/status及logs已本地下载。固定cuDNN deterministic、math SDPA、cuBLASworkspace:4096:8，不启用慢30倍的global deterministic排序。原失败Phase26不重启、不放宽旧guard。历史Phase25完整tar SHA1ea6c533e6cb75f6412b724ae84d01715014593602bdbe80cabd5e6c7982cd9b，760files/目录全本地验证后远端回收；storage_state complete，root剩4.8GiB，GPU空闲。
下一独立fresh stable预算实验尚未启动：相同source/inputs/stats/loss，共同stable运行，先每seed每臂独立2epoch baseline，再fresh12epoch，前1568全部state/stream必须等对应新baseline。三seed/两坐标/固定final12，全1367val×3draw×4冻结probe；不挑seed/epoch，不读sealedtest。大curves与last将逐arm无损持久压缩、整包及每文件验证后回收工作副本，curves审计使用SHA一致RAM恢复副本；完整原始字节保留可恢复。默认timing000不变、F1/MBE/时序目标未全部达成。

### 2026-10-06 数值不可复现已实证，stable-route重放验证中
legacy_a/b两次1568updates实际sample/GT/mask/noise都exact到Phase25，但两次之间84renderer/system+26audio tensor也不bitexact，已保存legacy_comparison.json；因此不是仅旧机器结果差异。global torch deterministic trial开销约30x（9分钟仅首轮251batch），已仅停止diagnostic driver2290/child2609，保留日志/未完成状态，不作为成绩。改做fixedcuDNN deterministic+mathSDPA+cuBLASworkspace:4096:8（不启用全局det排序），stable_a/b完整1568updates重放，orchestrator3000。需要严格state/stream闭合后才准备独立freshbudget对照；不能直接把旧失败视为通过，不降低容差。所有模型source/loss/数据不变，无validation调参。Archive98131~91%，storagewatch51803仍待全tar/every-file验证才reclaim；默认timing000未替换。

### 2026-10-06 numerical replay diagnosis underway（不放宽门槛）
Isolation orchestrator2290 runs realfullTRAIN seed47/1568update legacy_a,b and deterministic_a,b, no validation metrics or promotion. legacy_a actualsample/GT/mask/noise hashes **exactly equal Phase25**; 84system/26audio tensors differ, maxabs9.4e-5/9.8e-6. Originalruntime deterministic_algorithmsFalse/cudnnFalse/TF32True. Need legacyrepeat selfcomparison and strict deterministicrepeat to distinguish numerical nonreproducibility; diagnosis reports in newroot/replay_diagnostic, states temporarilyRAM. If deterministic repeat passes, preregister fresh matched deterministic2/12 budget validation, commonruntime botharms and own2epochbaseline; cannot silently relax oldPhase25bit-exact gate or call failedfirstroot success. Failedroot/logs retained/localfailureevidence downloading. Archivewatch51803 stillfullverify before reclaim; independentnewrootstats/inputs/pilot/source preserve diagnosis dependencies.

### 2026-10-06 Phase26 controls stopped at1568 prefix guard（不能绕过）
All3control workers reached first2epochs, but exact state gate failed at system.renderer.motion_input.weight. Allstopped; finalizer1812failed with fixedarmfailed. Candidate arms neverstarted; no12epoch result. Actualprefixstream not yet preserved because stateassert occurred first; need diagnose streams/environment/source and savefailureevidence beforeanyrerun. Do not relax tolerance or resume blindly. Originalfailedroot/logs/weights preserved. Previouslaunch/PIDs no longer running. CompletePhase25archive stilldownloading local98131; storagecompletion helper51803 waits fulltar then per-fileverify/reclaim. Must investigate before deleting dependency needed for diagnostic; Phase26independentpilot/code/inputs/stats retained.

### 2026-10-06 Phase26已launch（不能重复启动）
Canonicalroot `/root/kinetalk_channel_coordinates_budget_20261006`；pairworker1809/1810/1811（seeds47/48/49），finalizer1812。preregSHAaffcc9983c19d6541b6ac0e55bea7cadc755396d9b994ca3fb62cd49e6e05ce9；storageaddendumSHA46481e781095bca4178166bfbc5a61807bae29caeb4754f9716385c6f0e8be03。6arms固定12epochs/9408updates，先三controls；actualprefixgate1568state/stream严格等Phase25才继续。新artifact-dir仅RAMcurves，weights/log/contracts持久root；controlsfinal12等完成历史localtar/every-file/every-dir验证/reclaim再持久curveSHA闭合，之后启动对应candidate；finalizer只读complete。归档stream本地session98131进行，已约25%/1.27GB，原files未回收。不能重复launch/重跑完成Phase25或读取sealedtest。默认timing000保持，尚无Phase26finalvalidation。

### 2026-10-06 prelaunch storage pipeline addendum（算法/预算不变）
Archive760files tar3,564,523,520bytes SHA1ea6c533e6cb75f6412b724ae84d01715014593602bdbe80cabd5e6c7982cd9b。zstd无损transport1,269,462,251bytes。多连接SFTP和stream均受约0.5MiB/s实际带宽限制，完整下载仍进行，original untouched、未reclaim。为不空等近40分钟，新增**launch前**storage_addendum.json：3control训练可与read-only完整备份并行；权重/optimizer/log/contract在root持久存储，--artifact-dir仅大型curves临时RAM。control final12后必须等完整localtar/every-file验证及原rootARCHIVED指针，再copy curves至root并核对原completeSHA，才完成control/startcandidate。finalizer只读已持久闭合curves。启动空间阈值依据上述routing为root1.5GiB/RAM12GiB（已测root1.7/RAM55）。source/conditions/seed/loss/epochs/first1568exact门槛不变，所有嘴开放。旧prereg不可静默重写；独立storage_addendum+helper_binding保留storage例外记录。launch尚待实际执行/记录。

### 2026-10-06 SSH恢复：Phase26准备完成，尚未launch
原endpoint已恢复；GPU4090/24GB空闲，root剩2.0GiB，RAM60GiB。确认无Phase25/26训练任务及无新root/launch后建立Phase26root，11依赖上传并prepare成功，source/inputs/probes/stats/pilot独立复制SHA一致。固定12轮prereg已上传。正在完整归档Phase25；本地全tar/every-file/every-dir验证前不能回收。尚未启动Phase26，之后按PHASE26_RUNBOOK顺序核验/launch，禁止重复。

### 2026-10-06 本轮本地准备已核验；远端仍未启动
- 最后只读SSH复核再次拒绝连接；本轮未上传、训练、归档或回收任何远端文件。异步询问服务器状态/新端口，尚无回复。不得把请求JSON或本地登记当作训练启动。
- `docs/CURRENT_MODEL_AND_TRAINING.md`已完成：真实数据/层数/四阶段训练/身份与情感口型/部署输入；provenance确认TRAIN **22位**（此前摘要24位错误）、12536clips/1372104frames；val3位/1367clips。Stage19all-emotions B0，audio-only recipe的neutral scope没有执行，真实学生1540维而非旧audio_dim83。
- Phase26固定12轮登记及10个helper SHA、11个上传依赖核验通过；预算9408updates/150432inputs每臂。observer7项检查通过：exact前缀、完整state keys/dtype/shape/value、sample/noise漂移停止、继续结果与原observer一致、RNG不变。此前toy已收敛造成无效“权重必须继续变化”测试失败，已改为原observer的1570步完整闭合，非模型错误。真实GPU前1568步闭合仍未运行。
- `docs/PHASE26_RUNBOOK.md`与5个request JSON已准备；先mkdir再upload/prepare。finalizer额外保存2vs12差异和绝对F1/MBE目标；reclaim强制核验Phase26独立依赖再回收已持久归档的Phase25。源码/架构/loss本轮未更改。默认仍timing000，不声称全部目标/SOTA。

### 2026-10-06 recovery: remote training has NOT started
Previous preparation and the fresh connection check both failed before login: connect.nmb1.seetacloud.com:11473 (116.136.52.182) refuses connection. Phase26 root creation is unconfirmed; no Phase26 upload, archive/reclaim, or training launch occurred. Existing Phase25 complete weights/evidence are safe locally. Seven Phase26 helper scripts exist locally and were only syntax checked; review and preregistration are pending. Continue independent architecture/data documentation and script verification. Ask for server availability/new SSH endpoint, without re-requesting existing credentials. Never report Phase26 as running based on local script creation.


### 2026-10-06 Phase25 complete; strong F1 gain, joint geometry/timing gate failed
All6formal arms1568updates and3complete native audits finished, workers4449/4450/4451/finalizer4452 ended. All93finalartifact files/6completefinals (~303MB) downloaded and SHA verified after interruption; no pending local exec51569 (unknown session, filesystem evidence allcomplete). Default/initialfield/actualstream/frozen/config/source checks pass. Full1367validation/all3draws/raw+clip/fourprobes/classes/intensity/speaker/precision/CIs saved.
Matchedcenteredcontrol->standardized3seedclip means: MBE .873153->.855830; LBE .419426->.416625; mouth MSE .0145754->.0144379; mouthdisplacement .002202119->.002201256 (essentially unchanged); mouthcorr .328318->.321403. Jawcorr .321086->.307054, jawMSE .0154882->.0162794. Neutraljawcorr47 .269745->.242856/48 .276766->.247924/49 .270859->.257264, allworse. Threejointgatesfalse, no defaultpromotion.
Original final GENERATED independent F1 .272033/.286390 -> .778258/.665628; auxiliary stable .414684/.337615 -> .791925/.749272. OriginalF1gain all3seed/speakerCIs strictlypositive; original128 surpasses .7, original64 remainsbelow .7. Do not claim allF1target/SOTA. Fournearconstant motion MSE decreases from1e-4/1e-5 to3e-11–1.22e-10, predicted std now sameorder GT~1e-6. Stable gain substantial too, not only original-sensitive classifier. OriginalGT .66/.64 known; F1 alone cannot certify fidelity.
Remaining眉meanbias .0454809/.0479510MSE≈94.85%; mouthmeanbias .0072374/.0144379≈50.13%. Fullmouth range .098382 vsGT.091991, jaw .190668 vsGT.175279; avoid global amplification. Geometry near nativebaseline and jawtiming worse. Twoepoch TRAIN flow .94–.96 -> .50–.51, still strongly decreasing acrossall3seeds; head refit not yet converged. No loss/architecture addition warranted from those results.
NextPhase26 fixedbudget control: same Phase25 source and initialmeanfield/coordinate arms, train12 total audioepochs (original base stage budget), all3seeds, batch16/9408updates. Fresh matchedruns using identical initial/checkpoint/std source, no reset-after2 or invalidresume; compare first1568updates actualstream and full system/audio state to Phase25 before continuing. Only epochbudget changes; no source edits/default/loss/condition changes, no selection. Formalfixedfinal12/fullnativeallprobes/geometrytiming/precision/CIs required; 2 vs12 within eacharm and centered vsstandardized12. STORAGE must verifiedarchive/reclaim completed Phase25 after copying independentinputs/source/probes/stats AND pilotreference weights/contracts, keepfullcurves historical. Do not relaunch Phase25. No sealedtest tuning. Userasked architecture/data/branchtraining explanation: create explicit docs using true checkpoint config/code/provenance rather than stale cfg.training fields or unused old Stage5.

# 最新恢复状态：Phase25 formal running


### 2026-10-06 Phase25 formal running; Phase21 durable archive verified
All746files/directories and complete4.042GBtar locally verified, remote originals unchanged rechecked before reclaim. Local persistent archive final_experiment/remote_archives/20261006/kinetalk_native_b0_adaptation_20261005.tar SHA8bb170c8647ff55c42cf00f98d5cf621ac7343a23f46a61006e73a8803e99469; original/root path now ARCHIVED.json pointer, NOT missing data. Localverified/reclaimed/ARCHIVED records saved. RAMtar temporary only. Phase25 has independent3final inputs,4probes,legacy code/SHA binding before reclaim. /root free4.8GiB after reclaim; no pending local exec/archive session.
Formal fresh launch /root/kinetalk_channel_coordinates_v2_20261006: pairworkers4449/4450/4451 seeds47/48/49, finalizer4452. PreregSHA6bde2ce64ace585c9ed7c35cad3d07e4d54efcd105315a8ad5587b51949ff4b0. Never relaunch. Canonical launch.json, seed*/control_state.json/standardized_state.json, postprocess_state.json. Both arms2audioepochs/1568updates/batch16/12536TRAIN, full1367nativevalidation×3draw/4frozenprobes. Current3controls active epoch1batch326/351/326, finite losses/gradients, GPU5.5GiB/24GiB, no OOM. Candidate queued after each control. No complete validation results yet; smoke is not benefit.
Finalizer checks frozen B0/identity/teacher, actual sample/GT/observation/noise exact, checkpoint/config/source/curve bindings; full raw+clip/all emotion/intensity/speaker native audit, original/stable probes separate, paired speaker CIs for geometry+all4F1 (reuse Phase22 statistic function with new branch labels), nearconstant precision and fixed8renderinputs. Fixed joint gate unchanged; no selection/promotion/sealedtest. Default timing000 and user goals unmet. Continue by checking canonical state/results; do not reopen/reanalyze old phases.

# 最新恢复状态：Phase25 smoke passed / verified archive pending


### 2026-10-06 Phase25 smoke complete; archive before formal launch
Fresh v2 root /root/kinetalk_channel_coordinates_v2_20261006 preparation3794 complete (first root3717 failed AST before training and is preserved). Three intentional training-source changes only: trainer, DiT coordinate transform, neutral config call. AST default equals frozen Phase21; all other snapshot files exact. Existing local AR CPU-noise API fix is intentionally excluded from this isolated diagonal-source snapshot, remains in local file. Local39 targeted tests pass. Twenty-two smoke evidence files downloaded/SHA verified and matched recipe has exactly one coordinate-system flag difference.
Real2update four-arm smoke: old/new default system/audio/loss bit-exact; all sample/GT/observation/noise streams identical; centered control/standardized initial physical mean field and x_t hashes exact. B0/identity/teacher frozen, audio/renderer updated. Standardized flow~1.895; total~2.579, finite. Actual total output-head gradient square shares standardized: mouth54.19% then43.80%, brows21.57% then26.79%, four nearconstant .532% then .605%; no low-variance monopolization. This is actual head parameter gradient, unlike Phase24 physical-output derivatives. Smoke is not validation improvement.
Three fixed seed47/48/49 matched2audioepochs,1568updates/batch16/full12536 TRAIN and1367validation,12Euler/3draws/4frozenprobes preregistered locally and remotely; source/conditions/head-reset common, no selection. Full originalF1/geometry/jawtiming joint gate plus class/intensity/speaker/meanbias/range/constantprecision/CIs planned. No formal launch yet. Finalizer/helpers uploaded; original/stable scores separate, no sealed-test tuning/default promotion.
Disk~959MiB, so first preserve full completed Phase21: three native final inputs/fourprobes/legacy source copied and SHA-bound independently into v2 root. Complete archive made:746files/4,040,861,183original bytes, tar4,042,362,880bytes SHA8bb170c8647ff55c42cf00f98d5cf621ac7343a23f46a61006e73a8803e99469. Download to final_experiment/remote_archives/20261006 in progress; no reclaim before full local archive/every-file/every-directory verification and unchanged remote recheck. RAM copy alone is not durable. Current pending local exec session69202 is archive download, not training. Default timing000 unchanged; goals unmet.

# 最新恢复状态：Phase24 rejected / Phase25 coordinates planned


### 2026-10-06 Phase24 both loss-only variants rejected; Phase25 coordinates planned

Balanced fresh root /root/kinetalk_flow_balanced_units_20261006 preparation3246 complete. Default old/new diagonal-source scalar states/losses bit-exact, actual TRAIN sample/GT/observation/noise exact, frozen B0/identity/teacher guards pass. Eighteen evidence files downloaded and SHA verified at phase24_flow_balanced_units_20261006. No formal Phase24 run, no running jobs.

Balanced total loss is finite (~.483 average), but first-update four near-constant channels take99.5528% of flow loss and99.9949% of flow-output derivative squared; mouth derivative norm is .00019237× scalar, brows1.3387e-7×. This is analytical dL/dphysical-vector-output, NOT full parameter-gradient attribution. With unchanged model coordinates it exposes severe mouth/brow downweighting. Reject balanced formal trial as well; no mean-weight/STD-floor/lambda scan. Default remains timing000. Objective-only alternatives do not constitute channel normalization.

Phase25 next design: coherent TRAIN-centered channel coordinates inside the DiT. Input y=(x_t-t*mu)/s; physical velocity v=mu+s*network(y,t,conditions), where mu=TRAIN residual mean/.25 and s=bound normalized residual std (existing1e-4floor). Same physical diagonal source in matched arms. Control uses centering and scalar s=1, existing scalar MSE; candidate uses channel s and MSE in those same channel units. All output/support/native-clock/content/affect routes retained, no extra loss or mouth masking. Both arms reset ONLY renderer output weight/bias to zero, thus initial physical velocity is exactly the same mu. Shared trained trunk/audio/teacher/B0/identity inherited identically. Reset is necessary because a warm physical head has huge normalized outputs in tiny units; this compares refitting from a shared mean field, not continuation from unchanged timing000.

Implement opt-in flow-coordinate-system default scalar / train-centered-scalar / train-standardized. Save mu/s in config with nonpersistent buffers for strict old checkpoints; inference must integrate the same transformed physical vector field. Bound fullTRAIN moments to each Phase21 final SHA, finite/missing-support checks. New architecture branches stay off by default, old/new default real smoke bit-exact required. Smoke must inspect normalized head gradients (physical-output derivatives alone cannot judge the reparameterized model), actual stream equality and identical initial mean field. If passed, preregister all3 seeds,2audioepochs/1568updates/batch16/finalonly/full1367×3draw/fourfrozenprobes and joint geometry+originalF1+mouthtiming gate before large launch. Complete verified archives/reclaim required first; preserve dependencies and no sealed-test tuning. No performance or SOTA claim yet.

# 最新恢复状态：Phase24 balanced smoke pending


### 2026-10-06 Phase24 raw smoke complete; balanced smoke next

Remote /root/kinetalk_flow_vector_units_20261006 preparation2820 complete; no running jobs. Old diagonal-source trainer vs new scalar units produced bit-exact system/audio states and losses after two real TRAIN updates. All three smoke arms share actual sample/GT/observation/noise streams; B0/identity/teacher frozen, audio/renderer updated, no nonfinite loss/gradient. Evidence downloaded at phase24_flow_vector_units_20261006.

Direct train-residual-std error scaling is REJECTED for formal training: scalar total .6157948 / flow .1710928, direct-relative total282705.6836 / flow282705.2617 (99.9998508% of total). Smoke is not an improvement. No coefficient scan to compensate. Historical Phase21 and raw snapshot preserved; default timing000/scalar unchanged.

Local opt-in train-balanced-std normalizes inverse-variance channel weights to mean1 on the fixed TRAIN support, preserving relative weights with no new lambda. Source remains the same physical TRAIN diagonal source. Thirty targeted units/source/runner tests pass, including default loss+gradient bit-exact, masked NaN exclusion and balanced scale invariance. Balanced remote smoke/formal training NOT yet run. Next fresh /root/kinetalk_flow_balanced_units_20261006 snapshot must strip both helper AST definitions, verify source-only change, repeat matched real smoke and inspect actual loss/gradient concentration before any full trial. Mean weight1 alone does not prove useful mouth/affect gradients. Remote disk1.9GiB; full six-arm curves require verified archive/reclaim first. No sealed-test tuning, seed selection, new mouth mask or SOTA claim.

# 当前恢复入口（2026-10-05，Phase21 complete / Phase23 complete / Phase24 smoke pending）

压缩后先读本页，再读主计划最新章节，不能重复启动或重做审计。目标仍是公平同协议论文比较领先、MBE≈0.7、生成原协议独立情感F1>0.7；目前没有新默认模型，目标未达到。




### 2026-10-06 Phase23 complete / Phase24 opt-in loss units local-ready
ThreefullTRAIN12536audits2413/2414/2415+orchestrator2412success; allmodelstatesunchanged, nativeGT/observation/noisecontracts exactall3, reportsdownloaded/SHAverified. t=.5桥endpoint为GT-informed训练路径，不能当rolloutF1或推理MSE。
TRAIN GTstd cheekSquintLeft2.00e-6/Right9.24e-7/noseSneerLeft4.60e-6/Right2.47e-6；新B0/identity坐标残差std约1.2e-5–2.6e-5；scalarrawsource仍.25。t=.5原flow桥endpointRMSE这些通道 .0080–.0189，GTstd倍数约1738–20457。jawOpen GTstd.1678、bridgeRMSE.0757–.0761；browInnerUp.3208、.0385–.0389。因此近constant通道精度不足真实存在，不能靠原0.25固定单位MSE衡量每通道相对误差；并非mouthmask。没有实际调参/模型promotion。
Phase24隔离设计调整：不声称做可逆模型坐标变换。两个matched臂都固定相同TRAIN残差STD物理diagonalsource（reuse existingmechanism），唯一变化是原flowMSE scalarunits→error/stdunits；既有loss项/权重、全51通道支持、forward和输出单位不变，不新增loss。不和旧Phase15warm control混比。单变量仅能归因relativeunits effect GIVEN相同diagonalsource；相对standard的推广还需全metric独立对比。stats直接从Phase23fullTRAINmoments派生，每seed绑定自己finalSHA，eps沿用既有normalized1e-4，不扫描。
本地opt-in --flow-vector-units scalar(default)/train-residual-std 只允许既有隔离warm-audio+boundstats+diagonalsource路径。新helper默认直接调用原mse，单位1loss+gradbitexact，NaN未观察排除、invalidstd拒绝、单位变换不变量checks；28targetedrunner/source/unitschecks通过。原训练源码默认行为不变，历史Phase21snapshot不覆盖。实际smoke准备脚本做旧源码diagonal/newscalar/newrelative三个2update，绑定其他source不变、actualsample/mask/noise相等、非renderer冻结state、defaultstate/lossbitexact。尚未正式训练，不能把smoke当改进。空间约2.15GiB，6fulltrialcurves会超预算；正式trial必须先完整验证归档或规划artifact保存，不擅自删除历史结果。

## Phase22 complete / Phase23 TRAIN flow-units audit planned (2026-10-06)
Phase22 v2三个worker1791/1792/1793、orchestrator1790全部成功结束。1367validation×3draw×3模型正常audio重放全部bit-exact；GT/time/masks/B0严格相同，所有模型state前后完全不变。三report/perclip/state/replay/summary已下载并核验SHA，13localtargeted checks通过。旧first-replay失败保留，因原audio阶段requires_grad标志影响dispatch；修复仅诊断flags，容差仍2e-6，无训练或权重修改。当前无Phase22运行任务，不要重启。

仅global换成真实query motion teacher，clip_all三seed均值audio→oracle：MBE .866997→.620782，LBE .417398→.326710；嘴部MSE .0145246→.00977143；嘴部meanbias .00771852→.00326642，眉部meanbias .0517715→.0122579（−76.32%）。嘴部位移MSE .00187616→.00192276（+2.48%，每seedclusterCI跨0），jaw corr .322320→.315969。原probeF1 .182096/.235712→.122878/.175071，三个seed配对原F1 CI均<0；辅助stable .496938/.484648→.514020/.494310，CIs跨0，不宣称情感稳健提高。teacher-global是GT-informed，.621不是部署成绩/默认/SOTA。

Teacher-global也未解决情感：辅助fear .0196/.0769→.1236/.2121；happy .6159/.6403→.6691/.6715；sad/surprise反而下降。眉部动态corr仍 .0686，静态姿态改善远大于真实动态时序。说明至少有两个问题：连续全局几何预测缺口，以及生成动作分布/动态与评估敏感性的剩余差距。不能只提音频类别准确率或增大全嘴范围。

额外只读冻结head核验（GPU同batch16，teacher logit重放bit-exact）：audio自己head macroF1 .8514/.8557/.8497；teacherhead(audio_global) .8048/.8156/.7830；teacherhead(teacher_global) .6201/.5984/.6154。这些都不是最终生成F1，证实音频编码已有类别语义，不能把瓶颈简单归因类别识别或学生/teacher语义坐标错位。validation global MSE2.4331/2.3976/2.2479，cosine .6340/.6213/.6534；全局误差中约29.8%–33.1%为整体均值偏差。CPU teacherhead初始strict类别重放失败仅1/0/1样本、最大logit差 .00778/.00507/.00679；未混用CPU/GPU分数，改为原GPUbatch16后所有logits exact，失败事实保留在progress。

下一步Phase23：三个native最终audio模型冻结，在全部12536真实TRAIN逐observed native帧核验motion与residual通道populationstd，以及固定t=.5/noise seed20261006的原flow向量误差/GT-informed桥endpoint误差。保留原audio阶段计算标志但no_grad/nooptimizer/backward，所有支持通道不遮蔽。直接用原system.flow，现有objective不改；不评分sealed/validation、不按F1挑权重、不重复旧warm source-only trial。这是为判断新的residual坐标是否仍有严重单位/精度失衡，不是修复效果或部署评估。若尺度证据支持，再设计保持同物理先验、条件、初始输出的单变量归一化对照；不得提前把source-only失败当作归一化成功，也不能直接堆loss。归一化尚未实现/训练。

## Phase21 complete / Phase22 冻结全局条件诊断

三 pairworkers5030/5031/5032 和 finalizer5033 全部结束；六臂/三完整审计成功，无 OOM。18完整阶段权重已下载，SHA和132identity/1568teacher/1568audio更新数核验通过。实际训练sample/GT/mask/noise流跨臂完全相等；B0全程冻结，identity拟合后冻结，teacher在audio阶段冻结，teacher/audio/renderer确实更新。主训练/模型没有新增修改。

完整1367validation，native-padded三draw42/123/2026、raw/clip_all、四冻结probe。clip_all三seed均值control→native_B0：MBE .882626→.866997，LBE .432473→.417398；嘴部MSE .0154841→.0145246，位移MSE .00251557→.00187616（−25.42%），centered corr .292511→.331791；全嘴q90−q10 .100725→.0931905，GT .0919912。范围总体接近GT不代表所有情感/通道/中性jaw正确。原probeF1 .187394/.250608→.182096/.235712；辅助stable .494992/.472235→.496938/.484648，不可冒充原协议。联合门槛47false/48true/49false，不能挑48、不能推广；默认timing000保持。

native中性jaw corr47 .276929→.268940，48 .270571→.279711，49 .251959→.248467。native音频类别accuracy .874177/.874177/.874909（不是生成F1）；teacher accuracy .655450/.635699/.656181。固定暖适配pilot不是充分收敛或从头重训。本地compact_summary、三report/perclip、contracts及18checkpoint齐全；远端每seed有固定8emotion render_inputs未下载/渲染。Phase20完整归档468文件、2.899GB，本地tarSHA ae6d26878ec2e9a8784855f9534b9248ef895b409d15f6c94a2f051b4407dd90，可恢复；Phase21完成/root约2.3GiB。

下一项Phase22：全部三个native_B0最终audio模型冻结、相同validation/nativeclock/mask/三draw/batch16/12Euler，仅把audio global替换为该片段真实motion teacher global；u_a、intensity_value、logits、B0、identity等不变。先重放正常audio并与Phase21原生整体验证曲线核对，不能混入trimmed-batch噪声。仅validation+独立enrollment重建B0缓存，避免重提取12536TRAIN。保存四probe/几何/时序/情感强度speaker分组及配对区间。GT-informed仅定位，不能部署/SOTA/替换默认；不新增loss、不挑seed、不读sealed成绩。


### 2026-10-05 Phase22 local-ready, remote connection pending

本轮没有新增训练/模型/loss修改。Phase22三个native_B0最终模型、原训练源码、manifest、四probe及Phase21曲线/审计SHA已本地绑定，preregistration SHA69a652b9bdf47c0c817c3437cd7d7e8ad9309935aefa7fa8cc17463cd3af00b4。全1367、batch16、12Euler、native-padded三draw42/123/2026，仅换global，其他条件共享原对象。正常audio全部三draw先重放核对，GT/time/mask/B0严格一致，prediction atol2e-6/rtol0，并报告是否bit-exact；全部通过才oracle解码，不混trimmed-batch协议。无需保存新大curves，只保存perclip统计/特征/分组/paired speaker CIs。所有模型状态前后相等，fresh输出保留失败且不自动重跑。诊断GT-informed，不能部署或计为SOTA。

本地12targeted tests通过（全padded噪声/RNG独立性、single-global条件隔离、nativeGT/clock/mask/B0/clip漂移拒绝及原factorial）。另用三个实际Phase21 perclip数组+四冻结probe复算raw/clip_all，全部分数与原report一致到1e-12；zero-paired-bootstrap通过。run/launch脚本编译通过。注意这只是评分闭合与单元检查，**实际生成重放和oracle还没有跑**。

SSH connect.nmb1.seetacloud.com:11473两次paramiko/TCP核验及第三次TCP复核均连接拒绝（Windows10061），DNS仍116.136.52.182；不是训练报错。已异步询问机器是否停机或地址变更。没有启动/上传远端Phase22，不要声称有running PID。待用户恢复连接后，先确认Phase21路径及磁盘，再建独立/root/kinetalk_frozen_global_diagnostic_20261005、上传prereg/local_smoke/run/launcher；使用新目录launch，不覆盖Phase21冻结代码。入口 tools是本地.codex-finalizer/run_phase22_global_diagnostic.py和launch_phase22_global_diagnostic.py。本地README/prereg/local_scoring_smoke齐全，后续不重复分析/测试。

对现有三个最终模型的全部三draw分组描述（非新训练成绩）存Phase21/error_decomposition.json和class_f1_decomposition.png：原128happyF1 .018、64 .114；stable128happy .616、64 .640，stable总体仅.497/.485，fear .020/.077。原128预测75.2%片段为contempt，属于读出分布塌缩；历史GT低方差敏感性已证明，不能把当前视觉笑容低F1全部解释为缺失happy语义。stable对fear也很差，仍有实际情感差距。

眉部MSE .053670，其中mean_bias .051771（96.46%），占51观察通道加权平方误差31.71%；嘴部占46.33%，眼部21.94%，这是coefficient SSE贡献不是MBE贡献。眉部corr .0555，fear眉部MSE .0889/corr−.002，surprise .0878/.063。全嘴范围/GT：happy1.149、surprise .705；jaw范围/GT：happy1.376、surprise .779、sad .894。不能继续统一放大嘴部。下一项机制定位仍是global-only oracle，看表情平均姿态以及fear/surprise/全部类是否改善，再决定唯一修改点；不能直接凭meanbias加loss。Default timing000未替换，用户目标未达成。



### 2026-10-05 SSH restored; Phase22 generation replay failed safely
用户已重启服务器，旧Phase21/root/data完整、/root空闲2.232GiB。新独立/root/kinetalk_frozen_global_diagnostic_20261005，orchestrator1233/workers1234/1235/1236完成并失败，不能当running或重启原输出。所有绑定和nativeGT/time/mask/B0严格校验通过，但第一batch正常audio生成与原曲线最大差47 .000412107、48 .000385284、49 .000347137，超过预登记2e-6，全部在oracle前停止，无训练/模型修改。原state/logs/launch已下载到Phase22/failed_replay，远端原失败目录保持不覆盖，prereg未放宽。
下一项firstbatch数值路径诊断：固定seed47，不涉及效果/模型选择；比较原trainer evaluate入口、同一冻结forward重复、原audio阶段requires_grad标志、MHAfastpath和TF32路径，以查明接口/数值差异，不通过挑较好score绕过重放。check_phase22_replay_paths.py执行中，原oracle尚未跑。


### 2026-10-05 Phase22 v2 running, canonical replay bit-exact
Firstbatch路径诊断完成：frozen default/repeat maxvs原 .000412107、repeat与自身完全一致；恢复原audio阶段requires_grad标志后maxvs原0，原trainer evaluate入口max0。MHA slow/TF32 false都不复现原路径，不采用。这里只是诊断实现此前把所有标志关闭导致数值dispatch变更，未改训练或权重，未放宽2e-6。原失败root+logs保留，本地failed_replay含完整路径证据。
修复：run依然整体@no_grad、没有optimizer/backward/参数更新，恢复audio和renderer的原阶段requires_grad标志，其他模块false；结束逐tensor比较权重/状态。新增guardtest，本地13targeted checks和真实三seed评分闭合通过。dispatch_addendum绑定原prereg和路径证据SHA；原prereg69a652.../噪声预算/probes/输入/容差均不变。
新root /root/kinetalk_frozen_global_diagnostic_v2_20261005，orchestrator1790/workers1791/1792/1793，canonical launch/state在该root，勿重启/勿看旧failedroot为当前。三个模型完整1367validation×三draw重放全部bit-exact，maxdiff0；已进入global-only GT-informed解码/评分。真实oracle结果尚未完成，不声称改善或SOTA。Phase21被冻结，默认timing000保持。

## 最新 Phase20（完成，无运行任务）

`/root/kinetalk_b0_target_values_20261005` 三pairworkers3013/3014/3015及finalizer3016已结束，六臂/三个audit complete。同3298安全片段、原safe有效帧/channel masks、批次和noise流、1568updates/24990inputs；唯一差异为中性teacher监督值→自身nativeGT值。Actual contracts跨臂sample/observation/noise hashes完全相等，evaluation真实GT/time/masks完全相等；仅B0更新。

嘴部MSE .0231544→.0179866（−22.32%），位移MSE .00139003→.00123103（−11.44%），嘴部corr .267073→.359861。每seed八情感/四强度mouthMSE均下降。但neutraljaw corr .444223→.441229：seed47/48略退步，49通过。联合门槛false/false/true，**不推广、不挑seed49**。范围 .055034→.058828 vsGT.091991，未解决。B0原F1 .03441/.03148→.07716/.06842，不是完整生成F1。

本地 `phase20_b0_target_values_20261005` 已存三report/perclip、六完整final.pt/provenance/summary/complete、actualtrainingcontracts、sourcebinding/parity日志与compactsummary。六final SHA/1568steps已核验；完整curves仍远端。默认不变，sealed test未用于本轮调参。

Source AST去新CLI后与Phase19相同；跨源码smoke严格bit-equality失败只涉及单个float32B0元素9.31e-10/optimizer1.46e-11，实际loss/gradnorm/audio/batch/mask/RNG完全一致。失败记录保留；recheck显式atol2e-8/rtol1e-6，未声称bit-exact；正式六臂同源且全新control。

Phase19匹配下游适配现已启动（Phase21，见上）；Phase20证明中性目标有贡献，不能与Phase19收益百分比直接相减归因于覆盖。新B0重新拟合identity参考、motion teacher/renderer、audio student；原loss/全嘴部/standard source，保留已失败约束默认off；不能直接拼旧renderer。

Phase19完整远端工作副本已可恢复归档：479文件/3.012GB，本地tarSHA `52d172874ed04b904117ac26bf972d2733ae81d16fe84975b5e5ac1ae8217e7a`，每文件SHA/大小/目录逐项校验。原目录保留ARCHIVED.json，/dev/shm临时tar，本地持久完整tar在remote_archives/20261005。本地Phase19六完整final.pt仍可直接用。Phase20完成后/root约3.3GB可用；下一项先规划空间。

## Phase19 已完成（无运行任务）

Phase19 /root/kinetalk_native_b0_target_20261005：pairworkers7679/7680/7681及finalizer7682全部complete，三seed联合门槛全通过。查launch.json、seed*/control_state.json/native_state.json、postprocess_state.json。只训练B0，其他模块冻结。每臂同warm、batch16、现有loss、精确1568optimizerupdates；control3298片段安全中性监督，native12536片段音频自身同步GT。control8部分轮、native2完整轮；clipinputs24990/25072。没有新增loss/嘴部mask/模型层。Native B0是audio motion base，不能继续称其语义上中性。

结果：control→native三seed嘴部MSE .0233636→.0148477(-36.45%)，位移MSE .00138916→.000974281(-29.87%)，嘴部相关性 .264962→.438176，中性jaw相关性 .443933→.472812。所有情感/强度分组嘴部MSE均降。全嘴部范围仍低于GT、中性jaw范围略缩，不能称幅度全解决。B0原F1 .03556/.03441→.15224/.12072，仍低且不是完整生成F1。六组报告/最终权重/summary保存在phase19_native_b0_target_20261005。

先通过9远端/31本地测试；真实smoke2updates完成，101B0tensors改变，其他system+audio与warm完全一致。完整finalizer已评1367nativevalidation几何、口型时序、四冻结probe，715TRAINneutral时序；base的F1只是B0诊断，不是完整flow生成F1。三seed预定联合门槛见主计划；已通过且分组审计完成，但不能直接替换默认。

## Phase20 启动前历史记录（已完成，勿重做）

本地train_full_staged.py新增--articulation-target-values safe-teacher(默认)/native-query；仅在safe_target_batch之后替换target_motion，clip/mask不变。31本地检查通过，但尚未远端真实parity验证或启动Phase20；Phase19冻结源码未覆盖。先做新旧默认路径和native值切换的2step真实smoke，不许直接复用历史control。六finalcheckpoint本地hash和1568updates均已核验。

先分离目标/数据覆盖贡献：同3298安全片段、同原safe观察mask、同1568updates/batch顺序/seed/warm，仅替换中性teacher目标值为该片段自己的nativeGT。重用本轮control需要源码与初始轨迹一致性验证。该小型因果对照完成后才设计完整identity→teacher→audio适配，不直接把新B0拼进旧renderer。Phase19改动同时涉及覆盖范围，因此不能声称已单独证明中性目标是根因。不扫描loss或追加混杂条件。

## 本轮已完成

Phase17 warm链：从头四stage各12轮 + fullmouth2audio + timing0002audio，hash匹配。B0 12轮/19788steps；identity12轮/132steps；teacher12轮/75216steps；audio-global16轮/100288steps；renderer/audio-temporal28轮/175504steps。基线80–100轮，但各loss/每step计算不同，不能直接声称欠训。详见training_budget_20261005/README.md与metadata.json。

Raw-emotion均值补充诊断：hidden128 vs hidden128+rawemotion768，固定ridge.001/五TRAINspeakerfold/fit-only标准化。teacher64 MSE .331714→.345320、真实平均动作offset MSE .00669546→.00692189；两项目的五折均变差，不加这个旁路，不扫描。只有固定线性读出结论，不能证明原音频没有信息。详见raw_audio_predictability_20261005。

Phase18身份参考预算：12vs120额外轮次/三seed，只22TRAIN独立参考对，原loss；所有非identity状态不变，前12trajectoryhash完全一致。参考TRAIN/val MSE .008240→.001556/.009589→.004217；中性query整体/valmouth改善，但TRAINmouth+30.32%、valbrows+9.75%，三seed联合门槛失败；不推广、不重训下游。报告、六identity小权重和summary已下载/hash核验。identity_budget_20261005。

## 存储与归档

/root原376MiB，完整逐文件验证后归档两组已结束rejected实验，归档后6.1GB、完成Phase19后约3.2GB可用。858文件、6.12GB全部保存在本地final_experiment/remote_archives/20261005两个完整tar，SHA/每文件/目录校验通过；远端原文件未变更复核后移除工作副本，原目录保留ARCHIVED.json恢复指针。/dev/shm有临时tar副本，重启会消失；本地是持久恢复副本。原warm、数据、基线、Phase15/16及条件缓存未迁移。详见归档README与manifest/verified/reclaimed JSON。不得把空指针目录误判为数据遗失，需先恢复。

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
# Latest: user corrected emotion-student architecture; 772D input fix verified, no new training

先读EMOTION_STUDENT_INPUT_CORRECTION.md。用户明确student只应读取emotion2vec+prosody，u_a应表达情感动态，内容走HuBERT→B0/h0。查到旧1540D拼HuBERT内容，共享TCN的global/u_a都能直接用内容。原Phase30旧模型消融仅准备本地helpers，未远端launch；不要按旧“任意音频时间上下文”前提继续或启动它。

本地slow_state_affect.py及train_full_staged.py已修正：后续训练默认772D；选既有TRAIN统计最后772维，保留B0原内容；旧checkpoint兼容但只为历史重放，warm迁移显式授权参数且严格验证归一化/列选择。24本地checks通过；真实checkpoint+16val音频CPU/GPU无optimizer验证通过，旧正常输出逐位不变、新global/u_a完全无直接HuBERT依赖/梯度，state不变。验证报告与代码SHA保存在evaluation/diagnostics/affect_only_input_check_20261006；远端/root/kinetalk_affect_only_input_check_20261006/state.json complete，无活动任务。

尚未重训/无772新指标。emotion2vec或teacher global仍可能带内容；u_a尚无逐帧教师情感目标。后续必须核对监督和动态情感角色，再匹配迁移smoke/训练，不能称输入删除已证明纯情感或提升SOTA。Phase29完整492-file下载SHAverified，summarize_phase29_local.py已运行，docs/PHASE29_VERIFIED_READOUT.json已生成；旧exec88017结束，不再等或重下载。旧Phase29 jointgates全false/dropout拒绝，默认模型未推广，sealed未读。
