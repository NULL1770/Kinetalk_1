# 保留脚本索引

更新时间：2026-09-22。此目录只保留当前可追溯的训练、审计、评估和渲染入口；历史一次性脚本已移入 `archive/cleanup_20260921/relocated/`，原路径和哈希见归档清单。

## 当前候选：native regional envelope

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
