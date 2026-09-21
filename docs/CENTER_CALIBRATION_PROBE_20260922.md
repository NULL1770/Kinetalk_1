# 表达中心校准探针：开发集均值改善，但未通过组合门槛

本轮在现有 reference-intensity decoder 之外增加一个独立的轻量 `CenterCalibrator`，输入冻结的 audio global64、neutral identity128 和 upper anchor9，监督为 TRAIN clip 的 valid-frame upper9 均值。它只作为诊断，不修改 reference checkpoint、empirical motion bank 或默认推理入口。

探针使用 fit/cal speaker×sentence 分折，12 个固定小模型 epoch；完成选择后只载入一次 development。推理仍不读取 query motion、标签或均值。静态修正以当前 full upper9 的片均值为中心做有界组合，mouth/non-upper 通道没有被修改。

| development 指标 | 原 full | center probe |
|---|---:|---:|
| normalized upper MSE | 0.547519 | **0.511647** |
| normalized mean MSE | 0.510292 | **0.474476** |
| normalized centered MSE | 0.037227 | 0.037171 |
| raw upper MSE | 0.022826 | **0.020602** |

片段均值误差改善约 **7.0%**，达到预设的 5% 诊断目标；但 centered MSE 仍高于 static 允许门槛 `0.035894 + 0.0002 = 0.036094`。本次探针也没有保存 M025/M037/M039 的逐身份门控、empirical seed42 组合曲线和完整口型复验，因此不能把它并入主模型。

TRAIN calibration 上出现明显过拟合/分布偏移：base mean MSE `0.054948`，calibrated `0.174123`，而 development mean MSE 改善。这说明当前短探针的校准选择和有界修正仍不够稳定，不能直接作为最终表达中心 head。

结论：保留 probe 作为下一轮设计依据，当前部署继续使用已审计的 empirical motion 候选。若继续做中心校准，应改成显式 bounded center head，并补充逐身份 OOF 选择及固定 empirical bank 组合验收；若 centered 或最差身份门槛不通过，应停止扩展该分支。

产物：`artifacts/center_calibration_20260922_evaluation.json`；远端运行目录为 `/root/autodl-tmp/kinetalk_active_20260921/center_calibration_20260922_v3`。该探针未读取 test、未替换默认模型，也没有建立 MEDTalk parity。
