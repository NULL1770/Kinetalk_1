# KineTalk 当前研究入口

更新时间：2026-09-22

本仓库是一个连续研究工作区，包含历史 v9、v10 基础系统，以及从冻结 v10 Stage4 继承的 upper9 动态实验。它们是演进关系，不是三个都已验收的成品；论文和主表只能引用同一协议、同一代码路径、同一训练预算下得到的结果。

## 当前应先读什么

1. [2026-09-22 真实动作库增动候选](docs/EMPIRICAL_UPPER_MOTION_20260922.md)：恢复眉眼幅度、改善运动分布；全局条件检索综合收益和局部音频时序仍未建立。其确定性中心见[参考强度解码器](docs/REFERENCE_INTENSITY_DECODER_20260922.md)。
2. [前一轮原生帧率区域活动候选](docs/NATIVE_REGIONAL_DYNAMICS_20260922.md)：冻结载体 gain 路径的真实结果与失败时序诊断。
3. [基础系统与边界](docs/CURRENT_SYSTEM_20260921.md)：模型层次、推理条件和通道范围。
4. [研究交接记录](docs/CURRENT_RESEARCH_HANDOFF_20260920.md)：完整实验时间线、失败门控和待办事项。
5. [脚本入口索引](scripts/README.md)：保留脚本的入口、命令和用途。
6. [清理归档说明](archive/README.md)：本地可恢复清理归档的范围和恢复规则。

## 2026-09-22 当前候选状态

参考强度解码器已完成20/30/10轮预算选择，选出13/28/0轮后全TRAIN重训；接收真实强度的oracle通过，但音频输出仍低幅。本轮在冻结确定性中心上补充TRAIN真实连续眉眼动作，完成固定三温度校准和8 seeds分布评分，未替换默认模型。

446 development上，新候选眉/眼temporal std为0.01833/0.02070，GT为0.01900/0.01997；centered fair ES由0.15909降到0.12033，fair variogram由0.05076降到0.02527。raw MSE略增，条件检索也没有综合优于无条件采样，wide通道幅度仍不足。**这是可运行的运动分布候选，不是可靠局部音频时序或MEDTalk等效性能的证明。** 独立推理入口 `scripts/infer_empirical_upper_motion.py`，需要bank、参考checkpoint及预计算prior/audio/global/reference NPZ，不是raw WAV完整入口。四类有声对照视频、运行结果和限制见[本轮报告](docs/EMPIRICAL_UPPER_MOTION_20260922.md)。

## 当前主线

- **v9（历史）**：已归档的根入口 `train.py` 及其 VA/语义条件路径，仅作历史对照，不能代表当前主方案。
- **v10 基础系统**：`kinetalk_b0.models.neutral_affect` 注册 `architecture_version=10`；冻结 B0 内容/口型基座，使用中性参考统计个人执行特征，并由情感与音频条件驱动残差生成。
- **upper9 动态实验**：继承冻结的 v10 Stage4 上脸状态，仅覆盖眉毛、squint/wide 等 9 个上脸通道；不包含 blink 或 gaze。`train_relative_audio_timing.py` 和 `train_relative_motion_prior.py` 直接读取 Stage4 上脸均值，组合中心化状态与 zero-DC 时变残差，不依赖 `mean_preserving_upper.py` 作为最新主路径。
- **native regional 候选**：`train_audio_regional_envelope.py` 在原生时间轴预测 brow/eye 活动包络，使用冻结 carrier 和均值保持的区域 gain 组合；与上述 signed-state/zero-DC 分支分开报告，当前仅为未通过时序诊断的候选。
- **reference intensity 候选**：`train_reference_intensity_decoder.py` 使用独立中性参考相对 MAD 强度和学习式 upper9 decoder；不保留旧 upper 均值，不依赖 prior 上脸轨迹生成新 upper。正式训练已完成，音频输出低幅，尚未建立可靠时序收益。
- **empirical motion 候选**：在上述中心上叠加真实TRAIN片段的连续变化，原生裁切、有界组合、nonupper43保护。它是非参数增动基线，保留历史medoid探索的定位，不作为已成立的新贡献。

训练时可以使用动作、类别或教师状态作为监督；部署目标是语音加独立中性参考，不把目标动作或情感标签作为必需输入。这里的 identity 表示个人运动/表达执行特征，不能直接写成静态几何身份。

## 当前证据边界

完整开发协议为 4098 个 train clips、446 个 validation clips、25 fps；这是四类 MEAD 协议，不等同于全部 MEAD。2026-09-21 的独立读出探针在 validation 上 state MSE=0.0223675、corr=0.1169、R²=0.0136，并优于 static/reverse；但完整时序门控（包括 variogram/动态一致性）未通过。2026-09-22 regional 和 reference-intensity 两轮均未建立可靠音频时序收益，不能混用不同目标/路径的指标声称动态成功。新路径主动释放 upper 均值，只保护其他 43 通道；这些结构检查不能替代身份、情感、口型、动态的统一验收。新模块的 TRAIN 内留出不等于全系统留出，因为冻结 Stage4 已见完整 TRAIN；development 也不是独立 test。

论文主表必须固定 ARKit52 数据划分、帧率、mask、随机种子和训练预算，并在同一协议下报告 ARKit-MBE、ARKit-LBE、ARKit-FDD、AV offset/confidence、Multimodality、FD/WInD；未具备经过验证的输入和协议的指标应标为 pending。

## 代码与数据

保留的训练、审计、评估入口在 `scripts/`，核心模型在 `kinetalk_b0/`。数据、checkpoint、视频和大体积实验产物不随源代码发布；文档中的远程路径仅用于实验交接。清理产生的本地归档在 `archive/cleanup_20260921/`，已做快照和清单，可恢复但默认不参与测试或 Git 提交。

不要直接把旧 README 中的历史“当前/最新”措辞当作现状；以本页、`docs/REFERENCE_INTENSITY_DECODER_20260922.md`、基础系统说明和交接记录顶部指针为准。
