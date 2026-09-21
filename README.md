# KineTalk 当前研究入口

更新时间：2026-09-21

本仓库是一个连续研究工作区，包含历史 v9、v10 基础系统，以及从冻结 v10 Stage4 继承的 upper9 动态实验。它们是演进关系，不是三个都已验收的成品；论文和主表只能引用同一协议、同一代码路径、同一训练预算下得到的结果。

## 当前应先读什么

1. [当前系统与边界](docs/CURRENT_SYSTEM_20260921.md)：模型层次、推理条件、通道范围和当前验收结论。
2. [研究交接记录](docs/CURRENT_RESEARCH_HANDOFF_20260920.md)：完整实验时间线、失败门控和待办事项。
3. [脚本入口索引](scripts/README.md)：保留脚本的入口和用途。
4. [清理归档说明](archive/README.md)：本地可恢复清理归档的范围和恢复规则。

## 当前主线

- **v9（历史）**：已归档的根入口 `train.py` 及其 VA/语义条件路径，仅作历史对照，不能代表当前主方案。
- **v10 基础系统**：`kinetalk_b0.models.neutral_affect` 注册 `architecture_version=10`；冻结 B0 内容/口型基座，使用中性参考统计个人执行特征，并由情感与音频条件驱动残差生成。
- **upper9 动态实验**：继承冻结的 v10 Stage4 上脸状态，仅覆盖眉毛、squint/wide 等 9 个上脸通道；不包含 blink 或 gaze。`train_relative_audio_timing.py` 和 `train_relative_motion_prior.py` 直接读取 Stage4 上脸均值，组合中心化状态与 zero-DC 时变残差，不依赖 `mean_preserving_upper.py` 作为最新主路径。

训练时可以使用动作、类别或教师状态作为监督；部署目标是语音加独立中性参考，不把目标动作或情感标签作为必需输入。这里的 identity 表示个人运动/表达执行特征，不能直接写成静态几何身份。

## 当前证据边界

完整开发协议为 4098 个 train clips、446 个 validation clips、25 fps；这是四类 MEAD 协议，不等同于全部 MEAD。最新独立读出探针在 validation 上 state MSE=0.0223675、corr=0.1169、R²=0.0136，并优于 static/reverse；但完整时序门控（包括 variogram/动态一致性）仍未通过，不能声称音频已经可靠预测眉眼动态。口型、均值、非上脸通道保护通过的局部检查，也不能替代身份、情感、口型、动态的统一验收。

论文主表必须固定 ARKit52 数据划分、帧率、mask、随机种子和训练预算，并在同一协议下报告 ARKit-MBE、ARKit-LBE、ARKit-FDD、AV offset/confidence、Multimodality、FD/WInD；未具备经过验证的输入和协议的指标应标为 pending。

## 代码与数据

保留的训练、审计、评估入口在 `scripts/`，核心模型在 `kinetalk_b0/`。数据、checkpoint、视频和大体积实验产物不随源代码发布；文档中的远程路径仅用于实验交接。清理产生的本地归档在 `archive/cleanup_20260921/`，已做快照和清单，可恢复但默认不参与测试或 Git 提交。

不要直接把旧 README 中的历史“当前/最新”措辞当作现状；以本页、`docs/CURRENT_SYSTEM_20260921.md` 和交接记录顶部指针为准。
