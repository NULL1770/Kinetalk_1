# 保留脚本索引

更新时间：2026-09-21。此目录只保留当前可追溯的训练、审计、评估和渲染入口；历史一次性脚本已移入 `archive/cleanup_20260921/relocated/`，原路径和哈希见归档清单。

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
