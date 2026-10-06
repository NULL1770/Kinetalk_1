# Phase26 恢复执行顺序

最新storage addendum：实测约0.5MiB/s下载，完整归档验证仍在后台进行。为利用空闲GPU，允许3controls先运行，curves用RAM artifact-dir；权重/记录仍root持久。control完成12轮后等完整归档验证/reclaim指针，copy curves持久SHA闭合后再启动candidate。具体见新root的storage_addendum.json，算法/数据/loss/固定预算/first1568 exact gate不变。因此下文“先reclaim再launch”和3.5GiB阈值由此**launch前登记的存储安排**替代；禁止据此未验证删除历史，禁止仅RAM curves就让finalizer评估。

2026-10-06：本地已审查/登记固定12轮对照；7项observer检查通过。SSH端口11473拒绝连接，**以下步骤均未在远端执行**。本地生成request不表示远端已启动。

1. 先重新读CURRENT并核验SSH；远端检查新root与canonical launch.json，若已有任务，不重复mkdir/prepare/launch。旧Phase25已完成，不能重跑。
2. 使用 `phase26_mkdir_request.json` 建新root/tools，再用 `phase26_prepare_request.json` 上传并运行prepare；顺序必要，因为remote_ops先上传再运行command。
3. prepare独立复制Phase25 code/inputs/probes、三TRAIN stats、六pilot finals/contracts。检查source/input/probe/stat/pilot SHA，不依赖已归档Phase21。Phase26模型源码必须和Phase25完全相同，不能顺带上传当前checkout其他改动。
4. `phase25_pack_request.json` 仅完整归档已完成的 `/root/kinetalk_channel_coordinates_v2_20261006` 至RAM临时目录。下载完整tar及manifest到 `final_experiment/remote_archives/20261006`；不是只下载93个结果或六个final。
5. 执行 `python .codex-finalizer/verify_phase25_storage.py kinetalk_channel_coordinates_v2_20261006.manifest.json`，核验整tar、每文件SHA/size及每目录；成功后将其verified.json上传原RAM目录。未验证不能回收；RAMtar不是持久备份。
6. 使用 `phase25_reclaim_request.json`：远端再次检查原文件未变化、无原任务、完整备份及Phase26独立依赖，再只回收指定Phase25工作副本，保留ARCHIVED.json指针。保存reclaimed记录本地。
7. 确认GPU/磁盘和prereg，然后 `phase26_launch_request.json`。launch拒绝重复运行，要求剩余空间>3.5GiB，绑定所有输入/source/stat/pilot/probe与必需helper；启动六固定臂及finalizer。记录canonical PID、prereg SHA，不能根据旧PID报告运行。
8. 前1568update读取每臂prefix_replay.json；system/audio每个tensor的keys、dtype、shape、value及实际sample/observation/noise都必须等于相应Phase25 pilot。不同则停，保留失败，不放宽为近似相等，不绕过。
9. 固定final12（9408更新、150432输入/臂），完整1367validation、draw42/123/2026、raw/clip_all、四冻结TRAIN probe、每情感/强度/说话人、几何/口型/近恒定精度、paired speaker CI。两原probe与两辅助probe分开，比较centered12 vs standardized12及各臂2vs12。独立评分器不作为训练loss。
10. 下载并SHA核验完整结果/权重；保存CURRENT/progress/findings。联合几何时序门槛与绝对F1/MBE目标分别报告；即使过相对门槛，也不能自动称SOTA或替换默认。

本地request生成：`python .codex-finalizer/build_phase26_requests.py`；仅生成JSON，不访问网络。需传给已有remote_ops helper，沿用既有凭据而不输出密码。`phase22_statistics.py`对应上传的 `run_phase22_global_diagnostic.py`，这是finalizer依赖，不能遗漏。

已知结果与网络/数据解释见 `CURRENT_MODEL_AND_TRAINING.md`。若SSH仍不可用，明确报告训练未开始；不重做旧阶段，不尝试在本地CPU冒充同环境训练。
# 2026-10-06 最新：fresh stable fixed2vs12已启动

先读CURRENT_OPTIMIZATION_STATE.md。旧`kinetalk_channel_coordinates_budget_20261006`已prefix失败，不得继续或重启；旧runbook下面保留作历史。

新canonicalroot `/root/kinetalk_channel_coordinates_stable_budget_20261006`，driver3999，baselinepair4000/4001/4002；preregSHA2794a3bdbfa2ebcda6447b1c3b70479621f4326c53ee66c7be6d84c77c98f7a7。以launch.json和每armstate/status/postprocess为准，禁止重复launch。

1. 先确认driver/arm状态、finite loss/空间；pilot_run六2epoch/full1367audit完成后自动进入六fresh12；不按指标选择。
2. root/pilot保存自身stable两轮refs/contracts；12epoch在1568必须全部system/audio keys/dtypes/shapes/values及actualstreams逐位等对应自身pilot，严格guard。
3. 每完成arm都archives/budget{2,12}/seed{47,48,49}_{control,standardized}.tar.zst全量无损持久归档；manifest记录原每fileSHA与dirs，完整流解压验证后回收原last/curves工作副本。final.pt/provenance/complete保持canonical；curves.pt symlink指向SHA一致RAM audit副本。
4. RAM丢失时不能误报丢数据或重训：先验证archive整SHA，再仅恢复manifest所列curves.pt到独立RAM路径核SHA并重连symlink；完整last/model/optimizer均在archive可恢复，不擅自改变训练resume。
5. 同runtime训练/评估：cuDNNdeterministic/mathSDPA/workspace:4096:8，globaldetFalse，TF32True。固定12Euler/3draws/4冻结TRAINprobes，原F1和辅助分开。全部原联合gate/absoluteMBE,F1/2vs12保留；无promotion/epochselection/sealedtest。
6. 远端complete后下载report/npz/renderinputs/6finals/provenance/contracts/archive manifests/bindings，逐SHA核验，更新CURRENT/plan/findings/progress，固定emotion渲染供视觉检查。未全结束不声称SOTA。
