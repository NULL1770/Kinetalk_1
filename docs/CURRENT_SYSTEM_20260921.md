# 当前系统与实验边界（2026-09-21）

## 结论先行

另一份审查关于“仓库不是一个已经统一验收的单一模型”的判断是正确的。更准确的关系是：**v9 历史路径 → v10 基础系统 → 继承冻结 v10 Stage4 的 upper9 动态实验族**。这些代码和结果不能跨分支拼成一个已经通过身份、情感、口型、动态全部验收的系统。

当前仍缺少统一协议下的全量动态验收。读出探针证明音频对低率状态存在弱的可测信息，但不等于生成时序已经成功；最新完整门控仍需以真实音频相对 static/reverse 的动态指标和保护项共同判定。

## 三层代码关系

### v9 历史路径

已归档的根入口 `train.py` 和旧 semantic/VA 条件代码属于历史实验。它们记录了视觉 VA、类别和片段强度等条件的探索，不是当前部署入口，也不应与 v10 或 upper9 的最佳结果拼接。核心模型文件暂时保留，因为当前包导入和历史 checkpoint 审计仍有依赖。

### v10 基础系统

`kinetalk_b0/models/neutral_affect.py` 注册 `architecture_version=10`。它冻结 B0 内容/口型基座，通过多条独立中性参考的 motion−B0 residual 估计个人执行特征；motion teacher 通过动作重建学习全局情感与低率 affect field，随后冻结并教导 audio student，生成器使用这些条件生成残差。这里的 identity 是个人运动与表达执行风格，不是静态网格几何身份。训练阶段可以使用动作、类别和教师状态监督；部署接口目标是语音加独立中性参考。

### upper9 动态族

upper9 只覆盖眉毛、squint/wide 等 9 个上脸通道，不含 blink 和 gaze。`train_relative_audio_timing.py`、`train_relative_motion_prior.py` 使用冻结 Stage4 的上脸均值，再组合中心化状态和 zero-DC 时变残差；最新 relative 路径直接计算并保留该均值，不以 `mean_preserving_upper.py` 作为必要主路径。后者主要服务较早的 centered/prefix 实验，不能当作最新主模型的唯一实现。

## 推理与监督边界

训练可以使用真实动作、情感类别、教师状态或派生强度作为监督，以便学习可预测的低率状态和时变残差。部署时不要求目标动作或情感标签，目标输入是音频与独立中性参考。任何需要真实目标动作才能驱动的演示，只能证明接收通路，不证明音频预测器成功。

## 当前实验事实

- 协议：四类 MEAD，4098 train clips、446 validation clips、25 fps；不等同于全部 MEAD。
- 最新 supervised readout：validation state MSE 0.0223675、corr 0.1169、R² 0.0136；state 相对 static/reverse 的差异门控通过。
- 完整动态仍未通过：variogram/时序一致性门控没有形成可靠优势，因此不能宣称已经实现稳定的音频到眉眼动态。
- 口型、均值和非上脸通道的局部保护通过，不代表身份、全局情感、口型和动态已在一个统一版本中全部验收。
- AV offset/confidence、FD/WInD、Multimodality 等需要共同头像渲染、固定评估器或重复采样协议的指标，缺输入时必须保留 pending。

上述读出数值来自本地实验产物 `artifacts/readout_probe_20260921__evaluation.json` 和 `artifacts/readout_probe_20260921__benchmark_summary.json`（不随源码发布）。TRAIN 内部身份与句子留出选择了 full1540、ridge alpha=10，之后用全部 4098 个训练片段重新拟合，再评估 446 个 validation；未读取封存 test。旧结果中的 `passed:true` 仅表示状态门控通过，不能扩展为完整动态成功。

音频时钟审计也有一次必须保留的纠正：早期简版跨无效帧边界计算差分，0.44–0.52 的速度相关不可引用。严格版使用有效连续片段和邻接 mask 后，validation 的简单声学与上脸速度相关很弱（mel 变化约 0.033–0.050）；它不能推出“音频无法预测动态”，也不能替代真实音画同步的独立验收。原始报告为 `artifacts/clock_audit_v2_summary.json`。

## 2026-09-21 本地清理记录

已完成可恢复迁移 1571 个文件：252 个历史 scripts、144 个对应 tests、1127 个 artifacts/tmp 内的历史 Python 脚本、7 个根目录旧入口、41 个根目录一次性 helper/下载页。活动目录保留 74 个 Python scripts 和 84 个 Python tests；所有核心模型源码、数据、checkpoint、视频、指标文档及第三方依赖目录保留。

归档位于本地 `archive/cleanup_20260921/`，包含源快照、散落实验脚本快照、逐文件哈希和迁移计划；其内容私有且不提交 Git。本地和 SSH 清理后的活动代码各通过 764 项测试。远端四份旧代码已完整归档并校验，新的活动目录为 `/root/autodl-tmp/kinetalk_active_20260921/code`。详细范围、恢复方式及验证见[清理完成记录](CLEANUP_COMPLETION_20260921.md)。本轮没有启动新训练。

需要恢复历史实验时，在隔离目录恢复完整源码快照，并按原 `source_inventory` 校验。旧训练协议绑定了源码哈希，不能将清理后的代码直接当作旧协议源码续训，也不能改哈希绕过验证。

## 论文使用规则

论文只能选择一个明确的代码快照和训练协议。主表、消融和可视化必须来自该快照，注明训练/验证划分、通道 mask、随机种子、预算和是否使用训练期监督。历史 v9、v10 和 upper9 的结果可作为消融或失败分析，不能合并成一套“最佳结果”。
