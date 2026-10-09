# Current: Phase62 rejected; Phase63 native-affine training active (2026-10-09)

## Latest: read this first

LatestreadPhase63正式epoch4/8,step2147/5456,~.103s/update。已有3轮history，内部身份loss并非单调改善，尚无外部生成指标。独立有限2小时本地collector PID51592已启动，锁在diagnostics/phase63_native_affine_20261009/collector.lock；状态local_queue_state.json，日志.codex-finalizer/phase63_collect*.log。训练/评估完成后自动SHA收集、重放336指标、与原Phase53 B0曲线逐字节比较、导出基线/消融/干预/八情感表、生成本轮身份曲线并渲染8主对比+8风格视频。主对比加入已SHA核验的adaptedFaceDiffuser固定seed42，同GT/clock/support。不能重复启动collector或remote worker。人工结果审阅与模型推广尚未执行。

Phase61 worker8268已收齐212原始文件/9827676bytes，源码与60组/840字段重放通过；禁止重启。Phase62 worker12206已完成完整1367评估并收齐19原始文件/115710802bytes，293源码SHA及336字段重放通过，8视频完成/全帧解码。V3结果MBE.762805/LBE.368069/Lip3.243318/F1.678671/jawrange.126920/corr.476134/closure.388260；闭口下降.012761，否决。读PHASE61_RESULTS/PHASE62_RESULTS。正式模型仍Phase53。

Phase63方案先于代码记录于PHASE63_NATIVE_AFFINE_PLAN。旧decoder由直接B0正全局gain+仅g/u/reference_response输入的表达TCN替代；prior/q/style/语义头/posture bias/B0全部冻结，原位置+.5相邻位移loss不变。8epoch/5456update，seed47，TRAIN10903+620独立neutral支持。实现15356f93fdf293284502596dfc639952d6f8296f已通过GitHub Git-data双次GET精确确认。61本地/61远端测试及120步GPUsmoke通过（lossratio.574757，冻结/NaN检查通过）。freshroot /root/kinetalk_phase63_native_affine_20261009 worker12910，正在正式训练流程；禁止重复dispatch。训练后自动full1367+2026style审计，不默认推广模型。实时状态读seed47/state.json，不把smoke loss当正式指标。

身份曲线36文件已打包为final_experiment/evaluation/identity_style_curves_20261009.zip；paper_reference_manifest_v2.json含240原始及新增项，原zip/manifest保留不覆盖。八情感05b曲线源于原Phase53正确单因素审计，不是Phase62或尚未完成Phase63。未平滑/重选片段；读IDENTITY_STYLE_CURVES.md。

Storage:为本轮安全移除远端Phase60以及Phase53joint、Phase51temporal/statistics的已完成curves.pt冗余副本；本地原件全部SHA验证并保留，权重/数据/报告未删除。恢复路径见.codex-finalizer/phase62_archive_receipt.json、phase63_curve_archive_receipt.json以及旧phase60_curve_archive_receipt.json。任何旧远端重放需先恢复对应曲线。最后free269176832bytes（Phase63smoke结束）。不触碰third_party/voca_reference。

## Recovery update: supersedes all older running / archive-pending entries below

Git local and independently verified GitHub HEAD: e0541c9a8214a1d5c5d8cdd2ab094819f37bd5b3.
Official model remains Phase53 style-only; no candidate promoted and no active training.
Phase59 worker6199 complete, all198 source files archived and SHA verified;840 summary fields replay exactly. V2 endpoints were forced after optimization, so do not adopt it.
Phase60 retry7161 complete/full1367, all26 originals/115013342bytes locally SHA verified. MBE .76248956, LBE .36758557, lip3.24028534mm, F1 .68222438, jaw range .12651879, corr .47589440, closure F1 .38763712. Tiny geometry gain but closure drops .013384 against Phase53 .40102120: reject. Its reference controls have inconsistent B0; never use them as identity evidence. No Phase60 videos rendered yet.
Phase61 worker8268 complete at /root/kinetalk_phase61_fixed_endpoints_20261009; candidate_passed=true. V3 fixes every endpoint inside solver bounds and checks deployed objective. ReportSHA701590dbb876765a38a12ca16ce75698edb2f5a1be38315227ca0db4e64341d1. Collect and summarize, never relaunch. A Phase62 integration is not yet implemented/launched.
Identity plots complete:05_identity_style_curves and eight05b_style_native_<emotion> PNG/PDF/SVG/CSV in paper_reference_20261009/figures. Only original Phase53 single-factor audit; unsmoothed native curves, fixed samples. SourceGT is not donor counterfactual GT. See IDENTITY_STYLE_CURVES.md. Figure package ZIP/manifest need updating.
Remote free space last53858304bytes. Phase53 style-only original curves.pt removed remotely only after verified local archive; restore before old remote audits. Phase60 curves locally verified and still remote, eligible for verified archival to make room.
Next: collect61 -> storage check -> consistent calibrated receiver integration with configure(47), BLASthreads2 and SHA bindings -> all-eight renders. If gain stays tiny/closures worsen, reject amplitude-only correction and train a justified receiver change preserving neutralB0/772D/no-student-motion-gradient/native-mask constraints. No new training design was approved or implemented yet. User requests real mouth improvement vs FaceDiffuser plus identity curves; keep honest paper limitations.

## Latestcontinuation2026-10-09 (supersedes staleheader)
Originaloverviewcompleted4db9567. Phase59ec58dd3 full3298neutralcandidate passed all8heldraw/clipgates,12local/remote tests;9resultoriginalsSHAverified,fullsourcearchivepending. Read PHASE59_RESULTS.md. Worker6199complete,neverrelaunch. No checkpoint changed;threshold-specific eventpreservation doesnotprove phonemepreservation.

Phase60retry7161 active at /root/kinetalk_phase60_event_eval_20261009/evaluation_retry;first7026failedbeforeinference(missingtest).13remote tests passed,full1367 scoring. Realprogressin evaluation_retry/state.json;retryheaderstayspreflight. Local01d5b57 addsTorchcalibration/evaluator. Primaryprediction is validcandidate,but reference_A/B/wrong_reference controls mix original/calibrated B0: do not use for identityclaims. Fixcontrols and endpointsolver before finaladoption. V2nonjaw/lastjaw endpoints were forced aftersolver instead of constrainedinsidebounds;heldmetrics use serializedmap but solverobjective differsdeployedobjective. Preserve oldcandidate/eval.

Rootstorage: Phase53style-only/evaluation/curves.pt remotelyremoved afterlocal+remoteSHA/openfdcheck;88,842,300bytes reclaimed. SHA f574fc3447e609df5f6af298e542dfe577e219cb6d3bc8ca97dd46e50487ac57. Finalcheckpoint retained. Restore fromlocaloriginalplus .codex-finalizer/phase60_curve_archive_receipt.json before originalstyleaudit. Rootfree178MB afterreclaim. Initial05categoricalidentitychartgenerated;actual8emotionnativecurvespending. Git01d5b57localpushunconfirmed(networkfailures);ec58dd3pushed. No student/B0training/sealedinput. third_party/voca_reference untouched.

Read PHASE58_RESULTS.md. Worker1604 COMPLETE in202.44sec; do not relaunch.
11local/11remote tests +24clip smoke +3298target fullfit complete. HeldmouthMSE
improves5.596–11.852%, rangeerror13.027–17.325%; closuregatefails3of4heldsubsets
(raw+clip),so no integration/newgenerator gain. Phase53 remains current.
212originalfiles9,816,755bytesSHAverified/198sources;840summaryfieldsreplayed.
Report12b0c4587b496e578c5cd3f3c92fbe01ed7ab5130b7c4064446d84e841021f4f.
No training active or queued. Read OVERVIEW_FIGURE_V2_DESIGN.md for user's
correction: original composition around one utterance/neutral scaffold/audio
expression/reference swap; previous EmoTalk-like multiemotion teaser superseded.
Latest pushed prechange8716dbe; unrelatedthird_party/voca_reference untouched.

Four requested reference types are complete: seven figures in PNG/PDF/SVG and90transparent+90white1024-square stills under final_experiment/evaluation/paper_reference_20261009. Read PAPER_REFERENCE_FIGURES_20261009.md for captions/provenance/limits. Data exports SHAverified; one joint motion t-SNE with all1367clips/domain, no seed search. Seven-method8emotion+fixed4time panels include adapted FaceDiffuser fixedseed42. GT-native timestamps shared per method; no word boundaries invented. Reference-style/architecture/teaser matchPhase53. Finalteaser squareproportions and final7method panels visuallychecked. Figures show actual mismatches; no new generator gain. Next optimization design pending; don't relaunchPhase54–57.

Read PHASE57_RESULTS.md first. Code c80ec82 pushed;43local/remote tests,24clip smoke/full2583 approved TRAIN/internal-held pairs complete. Remote /root/kinetalk_phase57_descriptor_predictability_20261009 worker5974 finished in92seconds;do not relaunch.214originalfiles31,536,754bytesSHAverified/195sourcebindings;report386deee1259965a00d477d56fe29c60cdfa826149905352fa061db4b5bed51fa. Pairhashes/countsexactPhase56;2016summaryfields replayed. Tables2304allgroup/96compactrows. No B0forward/neuralupdates/devqueries/sealed/newvideos.

Five-frame descriptor readouts are weak: u mouth localstd error reduction1.80%/5.38% on heldspeaker/sentence; mouthmean-1.94%/+1.99%. Audio772 localstd-13.53%/-3.61%,overfit. u beatsreverse on common support butthat alone isnot goodprediction;eyes essentiallyzero. Rejectuniversal paired-descriptor replacement;notproof nonlinearlearningimpossible. Stopthisdiagnosticfamily,do notrerun37/39namedstates,676Dinteraction,widenu/universalsmoothing. No newlyvalidated modeldesign/trainingqueued. Latest rootfree60,903,424bytes;recheck/archivebeforefuturetraining. Actualmetrics/videosremainPhase53below.

Read PHASE53_RESULTS.md for latest actual generated metrics; PHASE54_RESULTS.md and PHASE55_RESULTS.md for subsequent diagnostic evidence. CURRENT_MODEL_AND_TRAINING.md now matches actual Phase53 config; old Phase39flow description is archived. Do not rerun completed workers.

## Actual models
Phase53 two8epoch/5456update runs seed47 fully evaluated1367development+2026matchedstyle each. Style-only MBE.763407/LBE.369546/lip3.262565/F1.676805; joint.823704/.406063/3.471205/.690150. Jawrange.124295/.128094 vsGT.175279. No promotion: style consistency/semantic recovery trades against lip/timing. MainF1 is whole-clip statistics, not framewise; auxiliary/audio F1 not maingeneratedF1. Original113condition tensors exact each+36protectedreceiver style-only, B0exact.

24videos complete/full-decode/nativeclock/audio/rig checked. Actual sampled inspection all24middle+fivefixednativepoints forhappy/fear/angry/sad; no complete realtime playback. Happy underopens, fear temporal mismatch, sad/angry expression weak. Separate visual_review_receipt.json preserves original receipts. Gallery PHASE53_VIDEO_GALLERY.md;24method/20candidate tables PHASE53_EXPERIMENT_TABLES.md. Adaptedbaselines unequalbudgets; no verified MEDTalk/DESTalker trainedrows/SOTA claim or mixing bestcheckpoint metrics.

## New diagnostics COMPLETE
Phase54 root /root/kinetalk_phase54_paired_predictability_20261009 PID18700 complete.21local+remote tests/24clipsmoke/full2024fit+281speaker+278sentence. Frozen u/hidden/audio linear paired expression dynamics are weak held; B0 predicts neutral-base displacement difference30.79%/20.65%. That target includesB0 itself: association not causal leakage.17original30914551bytes,200preflight4873972bytes SHAexact. Reportf8c0d40d970d5175e95eb0087183546e089f6ff8388d5b7f12b9eee3cec5bf1e.

Phase55 root /root/kinetalk_phase55_paired_predictability_20261009 PID22650 complete.23local+remote tests/smoke/fullsame2583pairs. Conditional B0×audio-predicted emotion/intensity676D probe overfits; held expression centeredMSE/zeroMSE11.94/11.32,displacement4.98/5.92. Contempt dominates failure; do notdropclass.24B0controlarrays byte-exactPhase54.17original15500594bytes+191source2061122bytes SHAexact. Report58c0ec636ac04dd33ea0c1b04720c93390c1b74a0d7e52981bb0fd81ab3b3c66. No content entersstudent or neuralmodelchange. Gitimplementation5642ff9 pushed. No new deployment metrics or model promotion.

## Next decision
Phase54–57 completed the paired-target feasibility chain. Do not repeat these audits or infer that weak linear prediction proves no nonlinear signal. A future model change must target actual mouth timing/amplitude and held-speaker expression generalization, preserving independent neutral supervision and reference-style checks. No replacement teacher objective is validated yet; do not launch another failed family or call diagnostic fits generation improvements. Useralready authorizes optimization; do not invent approvalrequirement.

## Constraints/recovery
NeutralB0frozen;772Daffect/prosodyonly;no studentmotionreconstructiongradient;nativeobservation/event/channelmasks/rawclip/fourfrozenprobes;sealeduntouched;Gitbeforeimplementation. Style=reference-driven motiontendencies onsharedrig,not facialshapeidentity. Usernowrequestedreferencefigures; see PAPER_REFERENCE_FIGURES_20261009.md. Unrelatedthird_party/voca_reference untouched.

SSH ignored credential-bearing helper .codex-finalizer/immutable_transfer.py: import connect/command/transfer/upload/sha, neverprintwholefile. Lastrootfree107212800bytes;check/archivalbeforeanytraining.13oldlast.pt alreadySHAarchived/deleted;don'trepeatcleanup. GitSSHrelay currentlyunreliable; direct git -c http.proxy= -c http.sslBackend=openssl -c http.version=HTTP/1.1 push succeeded (TLS staysverified). Phase53original112files345162906bytes/preflight208files/finalarchive15files backed locally. Allsource/artifacts local final_experiment/evaluation/diagnostics/phaseXX_...; no relaunch of existing receipts.

## Phase56 COMPLETE (latest)
Read PHASE56_RESULTS.md and PHASE56_TARGET_SENSITIVITY_PLAN.md. Code1378447 pushed;33local/remote tests,24clipsmoke/full2583 complete. Remote /root/kinetalk_phase56_target_sensitivity_20261009 PID2361 finished;do not relaunch. CPU only, no modelupdates/B0predictions/queryaudio/developmentinference/sealed.
Report SHAac424e3017b0e69642071381508ced588538e6ffa7e04497c4117c1215195f9f;206originalfiles10764854bytes SHAexact;193sourcebindings. Pair/eventhashes exactPhase54. Counts/physicalexpressionenergy replay2583x3regions againstPhase50, maxdiff4.17e-17; original_physical_replay_v2.json. Original slowlocalNPZreplay interrupted, cachedarrayrepeatcomplete. All2583MFCCstrictalternatefiles unavailable;fixed±1neutralframe stress test is NOT actualalignmenterror/noisefraction.

TRAINmouth safe nativecoverage53.53%; strict5framewindow retains41.10%ofsafe (=23.03%fullnative),11frames11.14%ofsafe (=6.66%fullnative). Upper5/11cover94.82%/87.79%ofsafe. Mean native mouthshift displacementerror/signal .4547/.6031/.4876 fit/speaker/sentence;5frame .2227/.3935/.2334 butdifferentcoverage. Morestrict wholeclipgate not uniformlylesssensitive; do notdropclasses/masks. No universalsmoothing/newloss chosen. Currentteacher usesGT-B0, NOTdirectGT-alignedneutral: audit ofproposedpairedtargets doesnotproveexistingteachercontamination.

Derivedallgroup/compactCSV in final_experiment/paper_tables/phase56_target_audit_20261009;paperreadyasdescriptiveprotocolmaterialonly. The follow-up descriptor prediction audit is now COMPLETE asPhase57above;do not repeat. Actualgeneratedmetrics/videosremainPhase53. RetainnativeGT+independentneutralrefs andcontenttimingatreceiver. Useralreadyauthorizesoptimization,no inventedapprovalgate.
