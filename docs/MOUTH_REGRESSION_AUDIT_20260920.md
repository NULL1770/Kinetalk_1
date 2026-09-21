# 2026-09-20 六格视频口型异常排查

## 结论与实际状态

本轮从 SSH 读取 `/root/kinetalk_paper_20260920/full_v1/queue_status.json`：队列已完成，GPU 空闲，没有训练在运行。顶层本地 `audio_status.json`、`queue_status.json` 是较早的下载快照；最新完成记录在 `run_backup` 和远端 `full_v1`，不可把旧快照当成实时状态。

**口型确实不合格，但目前没有证据说明三条视频错配了输入音频。** 问题分为：新训练的口型基座较弱、Stage4 残差进一步损伤嘴部时序，以及一个已修复的未监督通道显示错误。只修视频不能宣布模型口型恢复。

## 输入与时钟：三条固定视频复核

对 M025/M037/M039 的 `angry_L1_001`：

- 从 SSH 直接读取 native NPZ，文件 SHA256 与 manifest 相符。
- native 与训练 prepared shard 的有效帧 motion/content/audio/times 逐值一致，最大误差全部 0；frame/channel mask 一致。
- times 精确为 `arange(T)/25`，每片只有首尾两帧无效，未删除帧或压缩时间轴。
- 远端波形、特征提取记录、视频导出波形和修正后 renderer 记录的 SHA256 一致；audio_offset 和音轨裁剪起点均为 0。
- 三片为 93/154/75 帧，音视频均为 3.72/6.16/3.00 秒。

审计文件：`artifacts/paper_training_20260920/mouth_input_audit_20260920.json`。这只是三条用户正在查看的视频的直接数据核验，不代表已逐条审完 4098 个训练样本，也不能用文件一致性代替原视频跟踪质量或感知 AV 同步验证。

446 条 validation 的 jawOpen 系数对 GT 做 ±10 帧诊断扫描，B0 与 Stage4 的最佳 lag 中位数均为 0；没有统一整段错帧证据。没有用该扫描移动任何预测，也不能据此宣称 AV 同步通过。

## 已确认的错误及修复

### 1. Renderer 忽略 channel_mask

三个原始视频 NPZ 都标记 `tongueOut=False`，但旧 renderer 仍驱动全部 52 个 shape keys。未监督的舌头输出均值约 0.69–0.80，最大值约 1.40–1.62，会被显示截断为 1；GT 的该通道是 0。

这是旧显示脚本的潜在缺陷，在新模型随机初始化后暴露。已修改 `scripts/render_dynamic_rig_comparison.py`：按输入通道名顺序读取 `[52]` 或 `[T,52]` 布尔 channel_mask，显示时未观测通道归零，未观测 NaN 不参与显示或观测统计；记录禁用通道和原始非零值。没有 mask 的旧文件明确按旧合同全通道显示。

三条完整视频已重新渲染到 `artifacts/paper_training_20260920/videos_mask_corrected/`。所有有监督通道与旧显示数组逐值一致；仅 tongueOut 被置零。rig 全帧映射误差 0，原 blend 未修改，原始预测和原始视频均保留。此修改不改变原始系数评分，也不解决有监督嘴部预测误差。

### 2. identity 阶段评估调用了尚未训练的 renderer

旧 `evaluate(stage='identity')` 调用 motion teacher 和全脸 flow renderer；在从零训练的协议中，它们此时还没有训练。这让 identity 阶段报告掺入随机残差。其 LBE=1.07767 不能用来评价身份偏置，也不能用来认定 identity 模块毁掉口型。

已修改 `scripts/train_full_staged.py`：identity 阶段只评价 `B0 + identity baseline`，articulation/identity 不调用尚未训练的教师、学生和 renderer，也不输出无效情感准确率。没有修改训练目标、checkpoint 或旧报告。旧 identity reference retrieval/cross-reference baseline 指标与这个生成评估错误不同。

## 训练问题：口型基座与后续残差

提交 `85fb20a` 改为所有 KineTalk 模块重新随机初始化，`prepare_paper_full_data.py` 创建新系统，不再加载历史 B0。完整四类 query 为 4098 条，但 Stage1 仍仅训练 715 条 neutral，每轮 45 batches、12 epochs，共 540 个更新，然后冻结。其余情感由后续残差建模。

从头训练有利于明确训练数据来源，本身不是程序错误；**但不能继承旧模型口型已经合格的结论，也不能未经基座验收就认为 12 轮已经足够。** 队列此前无阶段质量准入条件，按固定 epoch 自动完成所有后续阶段。

独立脚本 `scripts/audit_paper_mouth_regression_20260920.py` 使用全 446 validation 曲线和 6 条 hash 绑定的独立 neutral reference，重建 identity baseline。下表嘴部为通道 14:41，有效观测帧汇总、原始值不截断、seed42；速度误差是相邻帧位移 MSE。

| 路径 | 原始 MSE | 去均值 MSE | 去均值相关 | 相邻帧位移 MSE |
|---|---:|---:|---:|---:|
| 新 B0 | 0.023918 | 0.006210 | 0.304504 | 0.001107 |
| B0 + identity | 0.022774 | 0.006210 | 0.304504 | 0.001107 |
| Stage4 audio | 0.015522 | 0.006777 | 0.306986 | 0.003219 |
| Stage5 audio dynamics | 0.015522 | 0.006777 | 0.306986 | 0.003219 |

Stage4 改善了原始平均误差，但去均值误差恶化约 9.1%，速度误差增至 2.91 倍。jawOpen 的逐片零时差相关中位数由 0.4990 降到 0.4058。80 条 neutral 和 366 条 nonneutral 都出现速度误差增大，不能只归结为情感分布差。三条视频五个生成格的口型逐值相同，Stage5 upper9 分支没有进一步改写嘴。

历史保存的 teacher oracle 嘴部相关 0.8356、LBE 0.2236；audio 为相关 0.3070、LBE 0.4340。Teacher 使用目标动作条件，这不是同输入条件的模型优劣对比；它说明目标动作提供的信息在音频部署路径中没有被充分替代。不能把 oracle 的好口型作为 audio 质量证据。

Teacher 的 local 来自动作，而 audio 的 local 是另一个头；目前有全局蒸馏、flow 和间歇 rollout，没有显式 teacher-local 蒸馏。这是待验证的模态转换风险，**不是已证实的唯一根因，也不意味着加一个 local loss 必定修复**。此前已经有 local/audio 类路线，不能重回无对照重复实验。

## 下一步修复顺序

1. 当前失败的 full_v1 保留审计状态，不替换默认模型，也不在它上面继续眉眼长训。全量训练覆盖不等于每阶段质量达标。
2. 单独验收口型基座：沿已锁定 train/val，以 neutral 和 nonneutral 分组检查真实音频、静态输入、逐片口型曲线、原始范围和速度；训练内确认拟合是否充分，不能直接拿旧来源不清的 B0 作为论文基座。训练范围与预算调整需形成新的明确协议。
3. 在同 checkpoint、同输入、同噪声下拆分 B0、B0+identity、加全局条件、加局部条件、完整残差，确认是哪条路径降低嘴部时序。保护已验收的口型是后续训练准入条件，不能只检查 Stage5 的 43 通道逐值不变。
4. 对训练中始终缺失的输出通道建立 train-only support：相同 support 应用于 flow state/target/noise/output。现有 tongueOut 虽无 loss，仍可进入 flow 状态，这是另一个待因果验证的风险。不能用 query GT mask 作为部署条件，也不能把显示归零当作已修复训练。
5. 口型通路通过独立生成验收后再恢复眉眼优化；sealed test 继续封存。不得用 teacher oracle 或六格中同一错误基座相互一致替代口型验收。

## 验证与产物

- 本轮联合运行 early-stage evaluation、staged runner/resume、paper pipeline 和 renderer support 回归：19 passed。
- Renderer 修复另与旧导出/continuous latent 回归运行：20 passed（与上条有交集，不相加计数）。
- `mouth_regression_audit_20260920.json`：446 条口型分解、neutral/nonneutral、三条视频、lag 诊断。
- `mouth_input_audit_20260920.json`：三条原生输入、prepared、音轨 SHA 和时间轴核验。
- `videos_mask_corrected/render_audit.json`：修正后视频音轨/帧数、通道一致性、rig 映射。
- 本轮没有启动新训练，没有修改任何模型权重。评估/显示修复仅在当前工作树，远端完成运行的 source 快照保留不变。
