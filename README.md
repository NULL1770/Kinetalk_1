# KineTalk 当前研究入口

更新时间：2026-09-22

本仓库是一个连续研究工作区，包含历史 v9、v10 基础系统，以及从冻结 v10 Stage4 继承的 upper9 动态实验。它们是演进关系，不是三个都已验收的成品；论文和主表只能引用同一协议、同一代码路径、同一训练预算下得到的结果。

## 当前应先读什么

1. [2026-09-22 原生帧率区域活动候选](docs/NATIVE_REGIONAL_DYNAMICS_20260922.md)：本轮真实训练、推理产物和未通过的时序诊断。
2. [基础系统与边界](docs/CURRENT_SYSTEM_20260921.md)：模型层次、推理条件和通道范围。
3. [研究交接记录](docs/CURRENT_RESEARCH_HANDOFF_20260920.md)：完整实验时间线、失败门控和待办事项。
4. [脚本入口索引](scripts/README.md)：保留脚本的入口、命令和用途。
5. [清理归档说明](archive/README.md)：本地可恢复清理归档的范围和恢复规则。

## 2026-09-22 当前候选状态

原生帧率区域活动（native regional envelope）候选的真实训练已结束，**可靠音频时序诊断未通过，未替换默认模型**。该候选从音频特征预测眉部和表情眼部的两维活动包络，以有界 gain 调制冻结 Stage4 的上脸运动载体。监督目标是片段去均值后、按训练集通道尺度归一化的区域 RMS，再做 5 帧平滑；它不是逐帧情感真值，也不是 MEDTalk 的绝对情感强度。载体平滑主要减少抖动，不能据此认定音频已控制真实表达时机。

训练入口是 `scripts/train_audio_regional_envelope.py`；推理入口是 `scripts/infer_audio_regional_envelope.py`。推理需要冻结 Stage4 原始 `prior [T,52]`、预计算 `audio_features [T,F]`、原生 `times [T]` 和 `valid [T]`，**不是 raw WAV 到完整系统的入口**。输出保留原始 prior、协议规定的平滑 carrier、音频控制结果和 static-gain 对照，并保存通道/均值保护报告。完整命令见[脚本索引](scripts/README.md)，结果与限制见[本轮报告](docs/NATIVE_REGIONAL_DYNAMICS_20260922.md)。

## 当前主线

- **v9（历史）**：已归档的根入口 `train.py` 及其 VA/语义条件路径，仅作历史对照，不能代表当前主方案。
- **v10 基础系统**：`kinetalk_b0.models.neutral_affect` 注册 `architecture_version=10`；冻结 B0 内容/口型基座，使用中性参考统计个人执行特征，并由情感与音频条件驱动残差生成。
- **upper9 动态实验**：继承冻结的 v10 Stage4 上脸状态，仅覆盖眉毛、squint/wide 等 9 个上脸通道；不包含 blink 或 gaze。`train_relative_audio_timing.py` 和 `train_relative_motion_prior.py` 直接读取 Stage4 上脸均值，组合中心化状态与 zero-DC 时变残差，不依赖 `mean_preserving_upper.py` 作为最新主路径。
- **native regional 候选**：`train_audio_regional_envelope.py` 在原生时间轴预测 brow/eye 活动包络，使用冻结 carrier 和均值保持的区域 gain 组合；与上述 signed-state/zero-DC 分支分开报告，当前仅为未通过时序诊断的候选。

训练时可以使用动作、类别或教师状态作为监督；部署目标是语音加独立中性参考，不把目标动作或情感标签作为必需输入。这里的 identity 表示个人运动/表达执行特征，不能直接写成静态几何身份。

## 当前证据边界

完整开发协议为 4098 个 train clips、446 个 validation clips、25 fps；这是四类 MEAD 协议，不等同于全部 MEAD。2026-09-21 的独立读出探针在 validation 上 state MSE=0.0223675、corr=0.1169、R²=0.0136，并优于 static/reverse；但完整时序门控（包括 variogram/动态一致性）未通过。2026-09-22 regional 候选也未建立可靠音频时序收益，不能混用两条路径的指标声称动态成功。口型、均值、非上脸通道保护通过的局部检查，也不能替代身份、情感、口型、动态的统一验收。

论文主表必须固定 ARKit52 数据划分、帧率、mask、随机种子和训练预算，并在同一协议下报告 ARKit-MBE、ARKit-LBE、ARKit-FDD、AV offset/confidence、Multimodality、FD/WInD；未具备经过验证的输入和协议的指标应标为 pending。

## 代码与数据

保留的训练、审计、评估入口在 `scripts/`，核心模型在 `kinetalk_b0/`。数据、checkpoint、视频和大体积实验产物不随源代码发布；文档中的远程路径仅用于实验交接。清理产生的本地归档在 `archive/cleanup_20260921/`，已做快照和清单，可恢复但默认不参与测试或 Git 提交。

不要直接把旧 README 中的历史“当前/最新”措辞当作现状；以本页、`docs/NATIVE_REGIONAL_DYNAMICS_20260922.md`、基础系统说明和交接记录顶部指针为准。
