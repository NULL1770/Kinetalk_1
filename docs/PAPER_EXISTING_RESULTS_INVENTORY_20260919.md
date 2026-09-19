# 论文实验数据盘点：历史结果与当前候选分开

2026-09-19，只读核对已有 JSON、协议和评估源码。本文没有重新训练或推理，不能把历史数字列为当前 bounded-audio 模型成绩。当前可以开始写数据、方法和实验协议；主结果表须绑定同一模型、同一数据划分和评估版本。

## 目前能支持什么结论

全局音频分类和身份系数参考有已有证据，但**没有“身份、情感、口型都通过独立验收”的证据**。保留 43 个通道只能证明新的上脸模块未修改这些输出；不证明原口型质量合格，也不保证修改眉眼后感知情绪不变。当前 motion prior 能生成变化，有界 audio 未可靠改善时机。

| 证据层次 | 可用结果 | 论文中的正确用途 |
|---|---|---|
| 当前 20260919 bounded 实验 | 613/206 内层及历史 64 外层、多种子 prior/audio/static/reverse/mismatch | 该模块的配对诊断及失败分析；外层反复使用，不称最终 test |
| 20260917 五阶段 run12 与 subsequent 原 local 修复 | 身份参考、输入情感分类、mouth 系数误差 | 历史基座能力；完整模型仍需当前版本复测 |
| 20260915 v5 | 混合 MEAD/CREMA-D 的分类、检索、mouth、表情代理指标 | 历史基线、动机；其 architecture/checkpoint/data 与当前不同 |
| 20260910 及更早 | 旧 40/10/10/10 或旧 CSV | 归档，不与当前模型拼成一行 |

## A. 较新全脸基座：20260917 run12

数据：2315 fit/405 internal development，19/3 身份，neutral/angry/happy/sad 四类，96 帧原生 25 Hz。开发集与 fit 共享句子，继承基座有历史曝光。每阶段固定 12 epoch；本节全量指标是生成种子 42/123/2026 的指标均值，非挑最佳 seed。

来源：[summary_metrics.json](../artifacts/full_staged_review_20260917/summary_metrics.json)、[原始阶段目录](../artifacts/full_staged_review_20260917/remote/run12/)、[核查报告](../artifacts/full_staged_review_20260917/metrics_audit.md)。评估实现及快照见 `scripts/train_full_staged.py` 和该 run 的 `source/scripts/train_full_staged.py`；`provenance.json` 保存数据、参考、源模型和代码绑定。

| 项目 | 历史实测 | 限定 |
|---|---:|---|
| Audio 输入 emotion accuracy | 0.967901 | 405 内部开发，训练用分类头；不是生成情感识别 |
| 生成动作经训练 teacher 分类，audio 阶段 | 0.911111 | 非独立语义一致性 |
| 生成动作经训练 teacher 分类，dynamics 阶段 | 0.914403 | 非独立语义一致性，不能称外部 emotion accuracy |
| Motion teacher 输入 emotion accuracy | 0.891358 | 输入真实运动 oracle |
| 身份 fit 跨参考检索 | 0.526316 → 0.631579 | 19 训练身份，系数风格 |
| 身份 fit 跨参考 baseline MSE | 0.004819300 → 0.003464820 | 参考自身被用于训练 |
| 身份 dev 跨参考检索 | 1.000000 → 1.000000 | 仅 3 人，非 mesh 外观身份 |
| 身份 dev 跨参考 baseline MSE | 0.003126477 → 0.002247951 | 不足证明广泛身份泛化 |

| 全脸条件 | mouth raw MSE ↓ | mouth centered MSE ↓ | centered correlation ↑ | RMS/GT | centered R² ↑ |
|---|---:|---:|---:|---:|---:|
| B0 articulation | 0.014377139 | 0.005283276 | 0.424071 | 0.676638 | 0.116047 |
| Teacher motion oracle | 0.003576648 | 0.001787006 | 0.846279 | 0.969461 | 0.701013 |
| Audio | 0.012704796 | 0.006706189 | 0.413507 | 0.954811 | -0.122023 |
| Dynamics | 0.012704799 | 0.006706171 | 0.413508 | 0.954809 | -0.122020 |
| 后续原 local 修复 candidate | 0.009347868 | 0.005031983 | 0.479266 | 0.746856 | 0.158091 |

最后一行来自 `artifacts/temporal_repair_20260917/candidate/evaluation.json` 的 `distribution/populations/all/groups/mouth/mean_over_three_seeds`；与 run12 不是同预算训练消融。candidate 保存曲线 SHA256 `fe5877eceded6fa2c434f243ed3ef865e4fbe21161f4e485bf9132eabc1d3769`。其 mouth outside_fraction 为 0.274804；越界比例不等于越界大小，必须另报幅度。`audit.json` 的 original_local 单 seed 数字 0.009350314/0.479152 与三 seed 均值不同，不混用。

run12 `final.pt` 文件哈希：articulation `3bd6d5c3634b974b8120e780dbb240228a9b7dadcb7fae0a8452a1a572c45d2b`；identity `4166b28cfc18e7b6fb09ab3e40d93e6fe8e06cf63a129322f2d13902e1b1857e`；teacher `8c129f9c41204b9b7e762b0d4354bf4c49ffc7fdaa78d19c5e9e88c3978834bc`；audio `73a8f17137772f882bf6d8c65dbf2f6c36b6c42f5724091f7a2e7a42870df42c`；dynamics `9f7c40d201ef782dce46ed337e32eda9f819cd27686d7de3de058b38db152501`。本地 audio/dynamics 权重和曲线仍在，其余阶段需按归档确认权重可用。

## B. 较早 v5：可以复述，不可冒充当前结果

`artifacts/evaluation_v5_20260915/evaluation.json`：val 因子 1800 条（CREMA-D 610、MEAD 1190），生成 128 条（45/83）。val 已用于选模。配置窗口 96、采样步骤 16；generation 具体执行还需恢复对应源代码，不能只靠配置推断所有细节。data root 为 `data/processed/native_affect_style_v4_refmask`。run 位于历史 `/root/kinetalk_runs/v5_optimized_20260915`。

| 128 个相同 val 样本 | mouth MAE ↓ | mouth velocity cosine ↑ | jawOpen Pearson ↑ | speed MAE/s ↓ | expression MAE ↓ | 真实动作线性 probe emotion UAR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.056907 | 0.172811 | 0.532888 | 0.314768 | 0.103058 | 0.149148 |
| Stage3 motion oracle | 0.054106 | 0.125854 | 0.371551 | 0.554390 | 0.049990 | 0.523419 |
| Stage4 audio | 0.062906 | 0.074078 | 0.369057 | 0.551970 | 0.096583 | 0.524251 |
| Stage4 motion oracle | 0.053568 | 0.138748 | 0.368135 | 0.545819 | 0.051162 | 0.551668 |
| Style swap | 0.067291 | 0.068054 | 0.362100 | 0.546185 | 0.103208 | 0.493171 |
| Same speaker alternative references | 0.062679 | 0.073787 | 0.371859 | 0.549838 | 0.097898 | 0.506221 |
| Local reversed | 0.064456 | 0.065579 | 0.367277 | 0.555942 | 0.098657 | 0.515492 |
| Local zero | 0.063455 | 0.072157 | 0.414203 | 0.548071 | 0.104683 | 0.504286 |

真实 128 条 probe emotion UAR 为 0.537457，1800 条为 0.575103。代理对真实数据识别能力有限，生成分数接近它不等于情感成功。mouth 指标是 ARKit coefficient 误差，**不是毫米 LVE、SyncNet LSE、音素正确率**。

| 因子诊断 | CREMA-D | MEAD | 意义 |
|---|---:|---:|---|
| Audio emotion head UAR | 0.749117 | 0.409170 | 输入标签分类，不是独立生成情绪 |
| Motion emotion head UAR | 0.570845 | 0.530488 | teacher 语义 |
| Style cross-emotion speaker retrieval | 0.983607 | 0.900840 | 9/3 人闭集，非生成身份认证 |
| Style→emotion 独立线性 probe UAR | 0.367950 | 0.799573 | 随机 1/6、1/8，尤其 MEAD 泄漏明显 |
| Audio affect→speaker probe UAR | 0.351590 | 0.842149 | 身份信息泄漏，随机 1/9、1/3 |

v5 checkpoint（epoch/architecture/SHA256）：

- Stage1：16/401/`eb17629c43a5e3762ae5c2a348c76af91e2130d9c5cd610b44c120f8092c3a0a`。
- Stage2a：26/502/`033091c7d8c11e116cb227d93b97530b1f27d75c8debed39c9a4377f6dc0178b`。
- Stage3：29/503/`8482ec8281ad7261192a433a0092304db36440d9c6623982c817a2a204578646`。
- Stage4：4/503/`e8b90077e9f178a3b5edff68fc83073f33344e1345fa5d157862105e19fbd04b`。

`resolved_config.json` data_hash=`e4583169fe8f50f38bd600b7428de07bd669a748183766bba1c2e01e0e0a2a6f`，source_hash=`b8943d7c27dbf5fc6d77dcc6e2ea56aeb4f3cc5e4a15501b1380822edbc7d9ed`。其 stage1_import 指向另一个初始哈希，不应代替 evaluation.json 的实际模型哈希。pipeline_status 明确 quality_passed=false，训练完成不等于通过。

## C. 更早归档的危险用法

根目录 `stage4_disentanglement_evaluation.csv` 没有 checkpoint 哈希、manifest 哈希、采样信息和足够 split 来源；不能直接进入新论文主表。里面 Stage2 96.35%、Stage3 85.42%、style recovery cosine 0.980886 等都是旧诊断，并且其 style→emotion probe=0.795258 已明确显示泄漏。

`artifacts/eval_new/RESULTS_SUMMARY.md` 与同目录现存 `all_stages_val.txt` 不一致：README 说 Stage4 epoch4，文本报告是 epoch20。这是混合更新目录，必须优先逐文件哈希和模型元数据；禁止从 markdown 挑最好的数字拼表。

更早独立 real-BS probe 已有反例：`independent_emotion_val_stage4_new.json` 中配对真实 124 条 UAR=0.754227、旧 Stage4 生成 UAR=0.229779；历史 test 的 308 条配对真实 UAR=0.735103、生成 UAR=0.117223。其 JSON 明示 `generator_heldout_generalization_established=false`，旧模型见过这些说话人。probe SHA256=`5e348ed4e3d9ae53b2594d338f20c332d3858826dcc13dc703335c233d9e382a`，原文件当前本地不存在。这些数字只说明**内部分类高分不能替代独立生成语义**，不代表当前模型的分数。

## D. 01–10 评估入口适用范围

| 脚本 | 当前判断 |
|---|---|
| 01_eval_stage1 | 固定 train；quick diagnostic，不能叫 holdout |
| 02_eval_all_stages | 可选 split；旧 Stage1–4 架构；按 batch 均值再平均，最后小 batch 权重可能不同；旧指标名不要直接迁移到新模型 |
| 03_eval_semantics | 固定 train；模型自身 emotion/intensity head，只能训练集语义诊断 |
| 04_probe_disentanglement | train 内逐片随机 80/20；非句组隔离；同时构造后续模块可能改变被评估的嵌套权重；不作为新规范 |
| 05_eval_stage1_fidelity | 对 aligned neutral GT；目标均值是 oracle，原 GT-assisted alignment 与自然时钟唇同步要区分 |
| 06_eval_style_leakage | 独立 frozen Stage2、句组留出、train-only 标准化，思路可沿用，但需对新身份 encoder 重写接口 |
| 07_eval_audio_emotion | 独立加载 teacher；有混淆矩阵/UAR/多数类基线，适合输入分类诊断，旧网络接口不能直接套新模型 |
| 08_eval_stage4_fidelity | 同人异句参考、与情感 query GT 比较；需新部署输出接口与相同 clock |
| 09_eval_stage4_semantics | 源码明确是训练 Stage2 编码器自读出；非独立感知评价 |
| 10_eval_independent_emotion | 真实动作训练、val 选 epoch、冻结 probe；可借鉴独立评价原则，但旧 checkpoint/数据协议不通用 |

## E. 当前完整评估所需资源

已具备：bounded 的四 seed upper9 curves、full52 显示 NPZ、split/模型协议；数据构建器保留原始 `target52`、`baseline52`、`valid`、逐通道 `channel_mask`。可以基于原始目标训练一个冻结的独立 motion-only emotion/speaker-signature 统计探针，以 613 拟合、206 检查真实动作识别能力、64 历史外层评估。GT mask 仅用于离线评估，不能进入生成器；显示用 baseline-filled target 不可训练探针。

本地不具备完整 `data/`，v5 使用的 `scripts/v5_runtime.py` 和 `kinetalk_b0/quality.py` 当前树中不存在，旧独立 probe 权重也未找到。因此从历史结果复现必须先恢复原代码/权重/manifest，不能声称今天已经重新运行。

新增 benchmark 仍需：一致音频及原生时间轴；全脸原始目标和通道 mask；固定模型版本与生成参数；独立情感模型及真实数据识别基线；几何 LVE/MVE 的一致 mesh 模型、顶点单位/区域；SyncNet 所需真实视频渲染与适用性核查；感知身份需要个体几何或外观资源，统一 rig 无法认证。今日先输出 coefficient、独立动作语义、动态分布和受控消融表；不得给缺失项目编数字或沿用不等价名称。

当前默认模型没有被这份盘点修改。新指标运行的结果应作为单独日期、单独 provenance 的产物附上，再决定哪些可进入论文主表。


## F. Current paper package (audited)

See PAPER_RESULTS_20260919.md and artifacts/paper_metrics_20260919/index.html. Current raw-mask coefficient metrics, matched static/reverse/mismatch controls, source-condition separability and a failed coarse audio condition probe are organized separately. Motion-probe v1/v2 are superseded: v3 corrects split, stage, numeric speaker labels and raw GT masks. All 64 speakers are known. Generated v3 classification scores are diagnostic only because near-constant real channels amplify generated-domain deviations; do not use them to certify or reject emotion/appearance identity. No new generator training is running.
