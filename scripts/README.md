# 保留脚本索引

更新时间：2026-09-22。此目录只保留当前可追溯的训练、审计、评估和渲染入口；历史一次性脚本已移入 `archive/cleanup_20260921/relocated/`，原路径和哈希见归档清单。

## 当前生成候选：empirical upper motion

真实TRAIN动作库与参考强度解码器组合，恢复眉眼幅度并改善分布评分；条件检索尚无综合优势，局部音频时序与MEDTalk性能对等未建立。详见[实测报告](../docs/EMPIRICAL_UPPER_MOTION_20260922.md)。

```powershell
python scripts/evaluate_empirical_upper_motion.py --data <prepared-dir> --source <frozen-stage4.pt> --reference-run <reference-run-dir> --output <fresh-dir> --device cuda
python scripts/infer_empirical_upper_motion.py --bank <empirical-dir/bank.pt> --checkpoint <reference-run/final.pt> --input <deployment-input.npz> --output <fresh-animation.npz> --speaker mead_M025 --sentence text_a6ba33f343d77c4eab734e7e --seed 42 --device cpu
```

前者只用TRAIN校准温度，随后固定development评分；`--selection-only`可止于TRAIN。后者校验bank/参考checkpoint绑定，不读query目标，输出original_prior/smooth_prior/deterministic/empirical/unconditional及供体/seed/保护报告。speaker、sentence必须为精确元数据；省略时无法执行该项排除。长度不足CLI明确报错。输入仍是下述预计算NPZ，不是raw WAV完整系统。bank保存真实TRAIN动作片段，不随Git发布。

## 确定性中心：reference intensity decoder

**2026-09-22 正式训练已完成；真实强度 receiver 通过，音频输出仍低幅，未替换默认模型。** 选择预算为 20/30/10，实际选轮及全 TRAIN refit 为 13/28/0。Full 中心化 upper MSE 0.037227 高于 static 0.035894，不能据均值拟合改善宣布动态成功。详情见 [REFERENCE_INTENSITY_DECODER_20260922.md](../docs/REFERENCE_INTENSITY_DECODER_20260922.md)。

- `reference_decoder_data.py`：按 TRAIN 或 validation 角色读取和校验 shard；构造冻结音频 global、独立 neutral identity/anchor，不从 query motion/label 构造部署条件。
- `train_reference_intensity_decoder.py`：原生音频→两维非负强度→学习式 upper9 decoder。目标为中性参考相对 MAD，经 TRAIN 通道尺度和五帧平滑；主动释放旧 upper 均值，逐 bit 保护其他 43 通道。TRAIN 内选轮后重新估计全 TRAIN 统计并从头 refit，再评分 development。
- `infer_reference_intensity_decoder.py`：使用可信的本仓库 `final.pt` 和预计算部署 NPZ，输出 original_prior/smooth_prior/full/static 及保护报告。无 query motion、emotion 或 target 输入；upper 由学习式 sigmoid 限界，学习输出不做事后 clamp/平滑。
- `diagnose_regional_targets.py`：TRAIN-only 固定 ridge/目标裁剪诊断；它不训练正式解码器，不把替代目标的裁剪稳定性称为可预测性。

从仓库根目录运行，下列路径为占位符，输出必须为新路径：

```powershell
python scripts/train_reference_intensity_decoder.py --data <prepared-data-dir> --source <frozen-stage4.pt> --output <fresh-run-dir> --device cuda --oracle-epochs 20 --student-epochs 30 --joint-epochs 10 --batch-size 32 --seed 53
python scripts/infer_reference_intensity_decoder.py --checkpoint <run-dir/final.pt> --input <deployment-input.npz> --output <fresh-animation.npz> --device cpu
```

训练可用 `--smoke` 检查短流程（2/2/1 轮），或 `--selection-only` 只完成 TRAIN 内选择。正式产物含 `protocol.json`、`selection.json`、各阶段 history、`oracle_evaluation.json`、`selection_evaluation.json`、`final.pt`、development `evaluation.json`、`curves.pt` 和固定元数据四例 NPZ。Oracle 用真实强度，仅检验接收器；不能作为音频预测成绩。

部署 NPZ 必需字段：`prior [T,52]` float32、`audio_features [T,1540]`、冻结音频 `global_code [64]`、独立参考 `identity_code [128]`、原始中性 `anchor [52]`、`valid [T]` bool、25 fps `times [T]`；可附 `channels/channel_mask/clip_id/noise_seed`。**这不是 raw WAV 或完整参考编码入口**，推理不会重新生成、认证调用方提供的 prior/global/reference。输出采用 renderer 的 `channels/times/valid/mode_names/motions` 格式，并保存 `.report.json`。

所有模式保留 non-upper43 和无效帧；旧 upper 均值按设计可以变化。Static 是同一 student 的时序隐藏状态取片内均值，是推理干预而非独立训练消融。CPU与正式CUDA曲线约5e−5差异保留在复放报告，未称为严格逐值一致。四类视频已生成，自然度仍未完成主观验收。

## 前一轮候选：native regional envelope

**2026-09-22 真实训练已结束；可靠音频时序诊断未通过，未替换默认模型。** 载体平滑主要减少抖动，不代表音频表达时机已学成。实验来源、实际结果和限制见 [NATIVE_REGIONAL_DYNAMICS_20260922.md](../docs/NATIVE_REGIONAL_DYNAMICS_20260922.md)。

- `train_audio_regional_envelope.py`：从预计算音频特征预测原生帧率的 brow/eye 两维活动包络，校准折选定训练轮数后重新拟合完整 TRAIN。目标是片段去均值 upper9 的区域 RMS，经训练集通道尺度归一化和 5 帧 box 平滑；它是运动活动量，不是绝对情绪强度或 MEDTalk EIE。
- `infer_audio_regional_envelope.py`：读取训练的 `final.pt` 和部署 NPZ，按 checkpoint 中的载体平滑窗口、5 帧包络、尺度及 `[0.25, 2.5]` gain 限制生成对照。推理不读取 query motion、emotion 或 target。

从仓库根目录运行，下面路径是占位符；训练窗口须在训练前固定，已有权重的推理读取其保存协议：

```powershell
python scripts/train_audio_regional_envelope.py --data <prepared-data-dir> --source <frozen-stage4.pt> --output <fresh-run-dir> --device cuda --epochs 40 --batch-size 16 --carrier-window 5
python scripts/infer_audio_regional_envelope.py --checkpoint <run-dir/final.pt> --input <prior-audio.npz> --output <fresh-animation.npz> --device cpu
```

训练可用 `--smoke` 做短流程检查，或 `--oracle-only` 检查真实目标包络对冻结载体的控制；这些模式不能代替正式音频推理结果。正式训练保存 `protocol.json`、`selection.json`、`final.pt`、`evaluation.json` 和 `native_curves.pt`，具体实际运行预算以本轮报告和保存协议为准。

部署 NPZ 必须包含冻结 Stage4 的原始 `prior [T,52]`（float32、标准 ARKit52 顺序）、`audio_features [T,F]`、`valid [T]`（bool）和 `times [T]`（均匀原生时间轴），可附 `channels`、`channel_mask`、`clip_id`、`noise_seed`。此入口需要 prior 和预计算音频特征，**不是 raw WAV 到完整动画的入口**；它不重新生成或认证 Stage4 prior 的来源。

输出 NPZ 采用 `render_dynamic_rig_comparison.py` 的 `channels/times/valid/mode_names/motions` 格式，四个模式为 `original_prior`、`prior`（协议 carrier）、`full`、`static_gain`；`static_gain` 是音频 gain 的片内均值，保留载体本身的时序。另存 `.report.json`，记录输入/权重哈希、gain 范围和口部、非上脸、片段均值保护。原始输出不做范围裁剪或组合后平滑，渲染器的显示裁剪需单独解释。保护通过只证明相对给定 prior 的结构保持，不认证唇音同步、自然度或可靠音频动态。

## 数据与协议

- `lock_paper_training_v1.py`、`lock_paper_manifests_20260920.py`：锁定论文协议与清单。
- `prepare_paper_full_data.py`、`build_protocol_manifests.py`：准备和检查数据。
- `extract_native_hubert.py`、`extract_native_mfcc.py`、`extract_emotion2vec_pilot.py`：提取音频特征。

## 基础训练与修复

- `train_full_staged.py`、`run_paper_full_queue.py`：v10 五阶段基础训练/队列。
- `recover_paper_mouth.py`、`calibrate_paper_mouth.py`、`run_mouth_repair_queue.py`、`run_calibrated_temporal_queue.py`：口型与校准实验。
- `train_neutral_affect_pilot.py`、`train_formal_predictable_projection.py`：中性身份与可预测投影实验。

## 动态实验

- `train_relative_motion_prior.py`：冻结状态后的 zero-DC 时变运动先验。
- `train_relative_audio_timing.py`：音频到相对时序条件的实验。
- `train_supervised_audio_readout.py`：固定表示上的音频状态读出探针。
- `run_isolated_baseline_suite.py`、`run_isolated_state_queue.py`、`train_isolated_audio_state.py`、`train_isolated_residual.py`：独立训练的状态/残差对照。
- `train_facediffuser_arkit.py`：共同 ARKit52 协议下的 FaceDiffuser 适配实验；不能直接视为官方数据协议复现。
- `audit_prepared_audio_clock.py`、`audit_raw_av_sync_v2.py`：检查音频、原生帧和时钟契约。

## 评估与可视化

- `evaluate_arkit_literature_metrics.py`、`evaluate_arkit_archives.py`、`evaluate_staged_arkit.py`：ARKit 指标和阶段评估。
- `paper_generation_report.py`、`build_paper_results_package.py`、`export_paper_full_examples.py`：论文结果整理。
- `render_dynamic_rig_comparison.py`、`blender_render_dynamic_rig.py`：共同渲染和对比可视化。
- `09_eval_stage4_semantics.py`、`10_eval_independent_emotion.py`：情感与独立条件检查。

## 使用边界

脚本名不能代表结果已通过验收；运行前必须确认对应协议、数据清单、checkpoint 来源和 `channel_mask`。旧 v9 `train.py` 位于仓库根目录的历史快照中，不是 v10/upper9 的默认入口。

本页按用途列主要入口，不是完整删除白名单。部分旧名脚本承载共享函数（例如 `train_neutral_affect_pilot.py`），评估/渲染还有按路径动态加载的依赖；完整保留闭包见本地归档 `plan.json`，不能只按文件名或未出现在本索引中就删除。
