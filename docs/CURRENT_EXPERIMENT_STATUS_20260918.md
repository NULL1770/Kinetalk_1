# 当前实验状态与代码入口

更新于2026-09-18。本文件描述当前代码快照与已核验结果；旧文档中的“下一步”或“正在训练”需结合各文档终点更新阅读。

**主要瓶颈仍是音频对应的眉眼动作时机。** 固定时钟先验、有界修正和低容量韵律修正已真实训练完成。最新353开发集Centered ES从0.553357降至0.550722，约改善0.48%，区间跨零且未胜过倒序/错配，时序不通过。身份、情感和口型保留既有基座，仍需独立验收。默认模型未替换，没有追加训练在后台排队。详见[本轮完整结果与后续决策](CLOCKED_PRIOR_RESULTS_20260918.md)。

## 已完成实验

| 实验 | 主要结论 | 协议／结果入口 |
| --- | --- | --- |
| 固定时钟双臂30、有界静态30＋修正30、韵律修正30 | 有界先验修复九通道范围，韵律有微小开发收益；音频时序仍失败，校准睁大眼幅度也不足 | [实测结果](CLOCKED_PRIOR_RESULTS_20260918.md)、[韵律协议](CLOCKED_PROSODY_PROTOCOL_20260918.md) |
| native_context30、motion_process30、joint_prior_formal30 | 扩展原生时序、动作过程与迁移后的联合先验；仍未达到时序目标 | [原生上下文](NATIVE_CONTEXT30_HANDOFF_20260918.md)、[动作过程结果](MOTION_PROCESS30_RESULTS_20260918.md)、[迁移记录](JOINT_PRIOR_MIGRATION_20260918.md) |
| 完整五阶段 run12，各12轮 | 身份系数基线和全局情感分类有收益；音频到运动的时序差距仍大 | [完整训练协议](FULL_STAGED_SPLINE_PROTOCOL_20260917.md)、[执行记录](FULL_STAGED_RUN12_HANDOFF_20260917.md) |
| local 对齐、direct/soft、残差修复 | local 对齐有部分收益；soft-state未显示独立优势；恢复Stage3原local口部输出改善口部误差 | [修复协议](TEMPORAL_REPAIR_PROTOCOL_20260917.md) |
| centered white/AR1 | white有连贯性收益，AR1未胜出；更准均值不等于更准动作时机 | [中心动态协议](CENTERED_TEMPORAL_PRIOR_PROTOCOL_20260917.md) |
| history12、prefix12 | 历史接收和分块机制得到验证，但早期正式生成结果未通过 | [历史协议](HISTORY_CONTEXT_PROTOCOL_20260917.md)、[prefix执行记录](PREFIX12_HANDOFF_20260917.md) |
| context12，三臂各12轮 | 使用自己生成的历史，眉/眼接缝位移较无前缀降低约86%/82%；时序相关仍弱 | [机制协议](CONTEXT_MECHANISM_PROTOCOL_20260917.md)、[执行记录](CONTEXT12_HANDOFF_20260917.md) |
| audio_prefix12，双臂各12轮 | 开放local有部分相关性收益，但误差和分布指标混合，未整体胜过旧候选 | [适配协议](AUDIO_PREFIX_ADAPTATION_PROTOCOL_20260917.md)、[执行记录](AUDIO_PREFIX12_HANDOFF_20260917.md) |
| adapter_transfer12_v2，三臂各12轮 | rank8没有实质时序收益；完整local适配收益混合；音频均值DC在608增量留出上恶化raw误差 | [最新结果](TEMPORAL_ADAPTER_TRANSFER_RESULTS_20260918.md)、[协议](TEMPORAL_ADAPTER_TRANSFER_PROTOCOL_20260917.md)、[执行记录](TEMPORAL_ADAPTER_TRANSFER_HANDOFF_20260918.md) |

## 当前能力及评价边界

- 全局情感：完整训练后的内部开发音频分类约96.79%，最新DC组合经训练motion教师读出约91.44%；后者非独立生成感知评价。当前实际训练覆盖四类，不等于八类表达已经训练完成。
- 身份：完整训练中跨参考系数基线MSE改善约28.1%；开发仅3人，身份含义是执行风格/系数基线，统一网格视频不验证人物几何身份。
- 口部：后续采用Stage3原local基座，内部开发raw MSE为0.009347868、中心动态相关约0.479，相比run12口部raw MSE低约26.4%。近期上脸实验保护其输出；没有独立唇音同步或音素闭合验收。
- 动态：研究通道为5个眉通道和4个squint/wide通道，共9维，不覆盖完整眨眼、视线和头部动作。43个其他通道保护属于工程核验，不等于整体感知质量通过。

原始池为2315 fit／405反复使用的内部开发片段，19 fit身份／3开发身份。最新增量实验沿用1707更新／608留句划分，但共享源模型见过完整2315；608只能叫新增更新的留出，不能称全模型未见句泛化。封存test未读取。

固定时钟先验沿用1053拟合／199校准／353开发回归；199和353均已被多轮设计使用，不能称独立未见测试。本轮没有读取405或封存test，历史曝光并未因此消除。四段新灰模诊断仅替换九个眉眼通道，其余43通道逐值不变；显示仍需裁剪较多基线系数，因此不能从视频推出完整52通道质量合格。

## 对应源码

- 五阶段训练：`scripts/train_full_staged.py`、`scripts/full_staged_data.py`。
- 连续前缀生成：`kinetalk_b0/models/prefix_upper_flow.py`、`scripts/train_context_mechanism.py`、`scripts/evaluate_context_mechanism.py`。
- 历史rank8模块：`kinetalk_b0/models/temporal_local_adapter.py`，训练与评价为`train_temporal_adapter_transfer.py`、`evaluate_temporal_adapter_transfer.py`。
- 最新韵律修正训练／后台启动：`scripts/train_prosody_clocked_residual.py`、`scripts/launch_clocked_motion_prior.py --prosody`。
- 最新评分与独立终点审计：沿用`train_clocked_residual_prior.py`中的评价函数、`scripts/audit_clocked_motion_results.py`独立重算已存曲线。
- 最新可视化打包：`scripts/package_clocked_prior_review.py`；完整面部诊断由`export_clocked_fullface_examples.py`导出，`render_dynamic_rig_comparison.py`渲染。

训练脚本依赖协议指定的历史源模型、独立身份参考、native音频/动作和缓存。`requirements.txt`列出基础依赖；测试需要pytest，绘图/媒体及可选音频提取依赖按使用脚本另行准备。`train.py`和`deploy.py`保留历史入口语义，不能用来代替最新实验入口。

## 下一步：活动条件与动作形状分离

依据本轮高维修正过拟合、小韵律分支未胜过干预的结果，下一项假设是音频只控制四组眉眼活动的发生概率与强度，运动先验负责具体方向、形状和持续过程。先对照既有事件探针，再固定短验证；较粗活动条件仍不能稳定预测时，不继续接生成器。该新结构尚未实现或验证，详见[最新结果文档](CLOCKED_PRIOR_RESULTS_20260918.md)。

## 历史机制诊断建议（adapter阶段记录）

先在同权重、同噪声下比较正常生成、仅一次真实起始历史、同末状态的静态起始历史；随后全程自己生成，只评价共同后缀。必要时固定起始历史、静态化局部音频，区分位置、运动惯性和音频贡献。GT仅为机制诊断，不得作为部署输入或主效果证据。

若只有均值改善，不能称动态突破；若GT运动历史有效，也不能直接推出音频能预测这些信息。可部署状态预测应先过留句验证。若初始化仍无足够收益，优先验证现有fit的完整连续上下文与原视频事件监督质量；不再仅增加同类适配器、训练轮次或loss。

相关依据：[起始状态和音频审查](NEXT_ORIGIN_AUDIO_AUDIT_20260917.md)、[连续上下文数据契约](NEXT_CONTEXT_DATA_CONTRACT_20260917.md)、[原视频跟踪核验](FIT_TRACKING_AUDIT_20260917.md)。其中历史DC/local适配建议已执行，最新结果见上表。

## 代码快照检查（2026-09-18）

本次Git快照从本地恢复基点 `3565787` 创建独立分支 `codex/experiment-snapshot-20260918`，保留当前代码与文档状态，包括此前已经删除的旧架构文档。远端main在上传前为 `f33edad`，含基点之后4个提交；本次不覆盖main，也不将那些历史修改自动混入当前实验实现。

346个Python文件语法解析通过，README与新增状态/结果文档的37个本地链接存在。全套 `python -m pytest tests -q` 结果为838通过、1失败：`test_encoder_mask_padding_and_interpolation_are_stable` 的float32最大差值2.3841858e-7略高于绝对容差2e-7。同一用例单独重跑通过，说明存在全套运行下的数值敏感性，尚未定位完整原因；未放宽容差或修改训练代码来掩盖该结果。

此记录针对仓库快照检查，与历史各实验单独完成的训练/审计测试区分。不得将本次全套测试描述为全部通过。
