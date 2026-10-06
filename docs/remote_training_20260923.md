# 2026-09-23 共享读取、四路训练和指标队列

远端工作区：`/root/autodl-tmp/kinetalk_final_20260922`。
代码：`code/source_20260923`。本地架构记录中的四阶段主线已同步。

## 本次修复

- 12,536 train / 1,367 validation clips 采用共享 flat NumPy memmap；只对当前 batch padding。
- KineTalk 的 b0/h0 按原生长度缓存；评估拼接时统一 padding，保留 double 时间戳和 clip ID。
- 仅 KineTalk 读取 50 个 enrollment shard（25 个身份各两个），三个 baseline 不读取 reference tensor。
- 音频以 FP16 存盘、batch 转 FP32；motion FP32、times FP64。已抽查两 split 各三条与原始 shard 完全匹配（音频按 FP16 存储契约）。
- 线程从每个 baseline 约 131 个限制到 2 个 torch CPU 线程，避免三路争抢容器 16 核。
- Baseline 每 epoch 原子保存 `last.pt`，完成时保存 `checkpoint.pt`；有 `--resume`，仅允许相同协议。
- 修复 FaceFormer 自回归 neutral token 遗失与前缀音频对齐；加入自己的生成序列与 teacher forcing 一致性回归测试。
- EmoTalk 全局分类池化排除 padding。

## 正式训练

调度器 PID 启动记录在 `logs/supervisor_memmap_20260923.log`；状态在
`runs/training_20260923_memmap/supervisor_status.json`。不要依据旧 PID 或旧日志重启。

| 任务 | 输出目录（相对远端工作区） | 预算 |
|---|---|---|
| VOCA-style ARKit adapter | `baselines/voca_arkit_memmap_20260923` | 80 epochs, batch 16 |
| EmoTalk-style ARKit adapter | `baselines/emotalk_arkit_memmap_20260923` | 80 epochs, batch 16 |
| FaceFormer ARKit adapter | `baselines/faceformer_arkit_memmap_20260923` | 100 epochs, batch 8 |
| KineTalk refactored | `checkpoints/kinetalk_refactor_memmap_20260923` | 12 epochs/stage, batch 2 |

后三个消融依次在主 KineTalk 完成后运行，输出为
`checkpoints/kinetalk_ablation_{no_teacher,no_mouth,no_residual}_20260923`。

- no_teacher：复用完成的 identity checkpoint，保留相同两段 generator 训练预算，移除 teacher 条件和蒸馏；audio 类别监督保留。
- no_mouth：从随机初始化跑完整四阶段，允许 residual 修改 mouth；主线没有额外启用未拟合的 calibration。
- no_residual：从随机初始化跑完整四阶段；生成器预测完整运动，移除加性的 b0 和 identity offset，保留 content/reference 条件，mouth 必须开放。该项属于直接全脸生成控制，不能声称只移除了单个不相关参数。

VOCA/EmoTalk 为当前仓库的简化 cached-audio 适配，不是官方原始模型/数据集复现。
FaceDiffuser 保留已完成的 100-epoch checkpoint，并使用原始冻结条件编码器代码。

## 自动评估

队列配置：`code/evaluation_plan_20260923.json`。
队列状态：`runs/evaluation_20260923/queue_status.json`。

1. train/validation 指标由各 trainer 完成后直接生成：`arkit_full.json`、原生帧预测。
2. 独立 motion emotion probe 只在 train 拟合、validation macro-F1 选 epoch，保存在 `evaluation/independent_probe_20260923`。
3. 1,622 条八情感 sealed-test 音频特征已生成，路径 `/root/autodl-tmp/kinetalk_data/sealed_audio_inputs_20260923`；query motion 未读取。独立 neutral enrollment motion 是允许的推理输入。
4. 固定 checkpoint 推理得到 `predictions.npy`，全部写完后计算 SHA256、写 `predictions_sealed.json`；然后才打开 query motion 评分。
5. 每方法结果：`evaluation/sealed_20260923/<method>/report.json`，含 MBE/LBE/coefficient FDD、LVE/EVE/vertex FDD、独立 probe macro-F1。
6. 完整 sealed 报告齐全后，自动生成 `evaluation/sealed_20260923/paper_tables` 的 Table 0/1/2/3。

## 固定 rig 与通道

rig 源：`D:\实验室项目\模型修正\male_arkit_head.blend` 的 `body` 对象；52 relative shape keys。
采用原始静态 shape-key geometry、对象线性变换、scene metric scale；不应用 armature 或 subdivision。
原生皮肤权重中的唇部组权重和 >=0.5 得到 281 顶点，眼睑/眉组得到 351 顶点。
集合在测试评分之前锁定，不按模型结果调整；FDD 使用相同眼眉区域。
本地导出与审计：`final_experiment/rig/fixed_arkit_rig_20260923.{npz,json}`。
这是自有固定 ARKit rig，不能称为官方 FLAME/VOCASET 顶点拓扑。

所有 12,536 条 TRAIN 都缺失 TongueOut，其他 51 通道都有观测。所有方法按同一个固定
51 通道支持评分，TongueOut 对 prediction/GT 都固定中性。FDD 要求全部声明支持通道可见；
不存在用模型填补 GT 的情况。报告保存 rig/hash/mask/单位与通道支持。

## 进度与限制

21:00 前检查：四个正式训练都在推进，KineTalk 已通过 articulation 与 identity 完整训练/评估进入 teacher。
实际训练总时长以日志为准；目前未完成主 KineTalk、消融及全部 sealed-test 表格。
旧 `*_full_20260923` 路径不是这次正式运行；早期 raw-loader 与提前 mkdir 的启动失败已被替代。
旧 131-thread baseline 未保存 checkpoint，已记录后从头重训。
不要把 smoke / validation 报告放进 `--require-test` 表格。

磁盘总量 50 GB，启动评估时约剩余 5.2 GB；不要建立全数据 padded cache。
KineTalk 使用 `--compact`，只保存 audio 阶段曲线；已有数据和历史 checkpoint 不删除。


## Baseline adaptation correction (supersedes VOCA/EmoTalk sections above)

See docs/baseline_adaptation_20260923.md for the audited source and deviations.
All baselines are explicit ARKit adaptations; shared frozen encoders are not
original end-to-end reproductions. Old simplified VOCA/EmoTalk are internal only.

- Corrected train plan: code/baseline_core_training_plan.json.
- Train state/logs: runs/baseline_core_20260923; VOCA and EmoTalk concurrently.
- Fresh outputs: baselines/voca_core_arkit_20260923 and emotalk_core_arkit_20260923.
- Corrected evaluation plan: code/baseline_core_evaluation_plan.json.
- Eval state/logs: runs/evaluation_core_20260923.
- Evaluator: scripts/evaluate_core_sealed_models.py (old evaluator preserved).
- Corrected results: evaluation/sealed_20260923/voca_core and emotalk_core.
- Final tables: evaluation/sealed_20260923/paper_tables_adapted.
- Old paper_tables task may fail the intentional legacy-model guard; this is
  expected. Do not remove the guard or promote old simplified results.
- FaceFormer100-epoch checkpoint/test already complete; metrics downloaded to
  final_experiment/evaluation/sealed_20260923/faceformer, with adaptation audit.
- Local current comparison: adapted_baselines.csv and adapted_baselines.md.
- Core models:80epochs each; VOCA batch8/lr1e-4, EmoTalk batch4/lr2e-4; no test selection.
- Pair coverage5943/12536 cross-emotion content donors,12536 emotion donors.
- Both remote smoke checks and13 targeted local tests passed. Training verified
  at VOCA epoch2/2751steps, EmoTalk epoch1/1351steps. Always inspect current logs.
- Heartbeat kinetalk updated in place, every20minutes, meaningful changes only.


## 22:07 CST monitor
KineTalk teacher12/12 complete (marker verified), now audio epoch1/12. VOCA-core epoch22/80; EmoTalk-core epoch5/80. FaceDiffuser inference1224/1622. All live processes/queues healthy, no current failure markers, disk4.5GB free. No new complete test report; final tables and ablations remain pending.


## 23:27 CST monitor — connection refused
SSH port11473 refused the first two checks, then recovered on the third.
User confirmed restart. Processes had exited; see the recovery entry below. FaceFormer/FaceDiffuser completed test reports are local and
hash-verified. Heartbeat remains active. On reconnect inspect processes and
checkpoints before resuming; never start duplicates based on stale PIDs.
Local incremental CSV/Markdown is regenerated with
`python scripts/summarize_adapted_baselines.py`, which handles quoted method
names, verifies report hashes and excludes legacy simplified baselines.
FaceDiffuser training_protocol.json downloaded after reconnection.


## 23:33 CST SSH restart recovery
User requested continuation after restart. Original processes were gone.
Checkpoint/source audit passed: VOCA69 completed epochs, EmoTalk15, KineTalk
audio7. Resumed the same outputs with --resume; unfinished current epochs rerun.
Active supervisor plans are now code/training_resume_20260923_2332.json and
code/baseline_core_resume_20260923_2332.json. State/log paths are unchanged, so
the heartbeat continues to inspect the same four run directories.
Old states/plans and recovery audit: runs/restart_20260923_2332/recovery.json.
Completed baselines/evaluations were not restarted. Queued ablations and final
evaluation/table dependencies were restored unchanged. Check actual processes,
not the archived PIDs, before any later restart.


## 23:47 CST monitor
VOCA-core completed80epochs plus1622-clip sealed evaluation (exit0). Corrected report and full protocol downloaded/hash-verified; it replaces the legacy simplified result in the incremental comparison. Local adapted_baselines.csv/md now has3 completed methods (FaceFormer, FaceDiffuser, VOCA-core). EmoTalk epoch19 and KineTalk audio9 continue, three ablations queued; disk4.3GB free.

## 2026-09-24 00:07 CST heartbeat — KineTalk main complete

KineTalk main completed all four stages (12 epochs each,154860 total steps),
then all1622 sealed test clips; both jobs exited0. Downloaded report, completion,
recipe, prediction seal, training summary/provenance and four stage markers to
`final_experiment/evaluation/sealed_20260923/ours`. Local report SHA and recipe
bindings passed; remote actual four final checkpoints, predictions and training
source bytes match recorded hashes. Split/input/rig/probe/support/metric protocol
matches the three completed adapted baselines. Training seed47; test draws use
42/123/2026. Baseline training seed42 remains unchanged.

KineTalk: MBE0.919714220, LBE0.466377367, coefficient FDD(abs)0.108821048,
LVE9.570863073mm, EVE2.448973822mm, vertex FDD131.563911661mm2,
independent motion-probe macro-F1 0.127030406. The emotion score is lower than
the completed baselines; it is retained without test-driven retraining/selection.
Incremental adapted_baselines.csv/md now contains4 complete main methods and
only EmoTalk awaiting a verified report. This is not final table delivery.

Latest live check during this heartbeat: EmoTalk epoch34/80 step105851,
no_teacher generator training phase1 epoch2 (log stage name teacher; teacher
semantics disabled). PIDs1298/2093 and all four supervisors/queues alive; no
current failure markers. no_mouth/no_residual remain queued. GPU76%,2738MiB;
disk3.7GB free. No jobs restarted and no protocol changes. Heartbeat stays active.
