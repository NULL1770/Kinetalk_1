# Phase46 完成：部署条件适配未获得联合提升

2026-10-08。两组各完成 8 epochs / 5,456 updates，完整 1,367 条开发集评估、16 个视频和论文表格均已闭合。保留 Phase45-u 作为表达参考，不采用 Phase46，也未自动替换默认模型。没有重新训练、重新评分或读取 sealed test。

## 实际结果

下表是相同原协议的 `prior_mean/clip_all` 部署结果，均为 seed47。MBE/LBE 是本项目 ARKit 系数指标；Lip 是平均唇部几何误差，不能直接与其他论文不同定义的 LVE 比较。

| 模型 | MBE↓ | LBE↓ | Lip mean mm↓ | 主情感 F1↑ | jaw 范围 | jaw 相关↑ |
|---|---:|---:|---:|---:|---:|---:|
| Phase45-u | 0.827171 | 0.388674 | 3.320219 | **0.709448** | **0.153116** | 0.490531 |
| Phase46 mixed | 0.836943 | 0.388290 | 3.252856 | 0.702871 | 0.138563 | 0.490336 |
| Phase46 deploy | 0.842696 | 0.391550 | 3.261482 | 0.705703 | 0.135853 | 0.491785 |

GT 平均 jaw 范围为 0.175279。相对父模型，mixed/deploy 张口范围缩小约 9.5%/11.3%，Lip 改善约 2.0%/1.8%，但 MBE、主 F1 与幅度退化，时序相关性几乎不变。因此不能把较低的重建损失或 Lip 单项改善当作成功。

主情感 F1 使用原冻结 128 probe，输入是生成动作整段统计；它不是逐帧情感动态准确率，也不是音频分类头的 F1。四个 probe 分别为：

| 模型 | F1-1 | F1-2 | F1-3 | F1-4 |
|---|---:|---:|---:|---:|
| Phase45-u | 0.709448 | 0.663190 | 0.729681 | 0.668836 |
| mixed | 0.702871 | 0.639281 | 0.729664 | 0.663506 |
| deploy | 0.705703 | 0.634531 | 0.731927 | 0.667015 |

Phase45-u 的 happy F1 为 0.934097，但 neutral/fear 仅 0.468468/0.477612。Phase46 deploy 的 happy 为 0.936782，而 fear/surprise 降至 0.436364/0.596859。不能用 happy 个例推断全部情感正确。

## 本轮改动与正确性

未增加网络层、模块或 loss。只训练已有四层、宽度 192 的 response decoder，共 790,056 参数；B0、772D emotion2vec768+prosody4 学生 prior、动作 posterior、style/身份编码器、统计和语义头全部冻结。全部嘴部通道开放，重建梯度不进入学生。

两组从同一 Phase45-u 起点出发，都使用两个独立中性参考聚合；mixed 使用 `q_sample + 0.5 p_mean`，deploy 使用 `1.5 p_mean`。沿用 position + 0.5 adjacent displacement。均有父模型 24 epochs / 16,368 updates 与一次 TRAIN10,903 解析拟合的累计预算，再增加 8 epochs / 5,456 updates。mixed 每步两次 decoder 重建，deploy 一次，计算量不同。

与父模型比较同时改变 decoder 适配和训练参考聚合，不能据此断言是哪一项单独导致退化。两组之间才是在相同参考与优化步数下比较重建条件来源。

38 项针对性检查在本地和服务器通过；真实 432 帧 TRAIN batch 检查仅 decoder 梯度、HuBERT NaN 隔离及优化器/采样 RNG/batch 顺序精确恢复。收集后所有 141 个非 decoder 张量逐位一致，完整 B0 预测、指标和 probe 精确一致。

闭环重新校验两个 arm 各 205 个成员、209 个启动成员、5 个 pipeline 成员（清单存在重叠）、16 个视频、14 个表格来源。视频 SHA 与已通过的全帧解码记录一致，25fps、原生时间轴、GT/B0 一致、rig 数值误差 0。核验读取使用 `utf-8-sig`，已解决先前临时脚本错误的 `utf8-sig` 编码名；原实验证据未修改。

## 已定位的退化与下一步

眉部 mean-bias MSE 从 0.047571 增至 mixed 0.051168 / deploy 0.051690，而 centered MSE 略降；眼部均值偏差也增加。幅度缩小伴随平均姿态误差增加，说明仅看动态或训练损失会遗漏问题。尚未证明身份参考是根因。

三个开发集说话人的 MBE 全部增加：

| 说话人 | Phase45-u | mixed | deploy |
|---|---:|---:|---:|
| M025 | 0.866749 | 0.881624 | 0.894653 |
| M037 | 0.814093 | 0.817093 | 0.818400 |
| M039 | 0.778115 | 0.786400 | 0.785075 |

第 1→8 epoch，TRAIN prior position 下降，但训练内部留出说话人的 prior position 上升：mixed 0.451432→0.458963，deploy 0.450208→0.457757；留出句子略改善。这支持优先检查跨说话人泛化，而不是继续延长本轮训练。

下一轮先固定 Phase45-u 做诊断：

1. 在同一模型下比较 reference A、B 与双参考聚合，拆开上半脸均值偏差、嘴部幅度和动态误差；补足本轮缺少的固定 decoder 参考对照。参考变化不能自动证明目标身份迁移正确。
2. 检查 neutral/fear/surprise 的动作混淆与眉眼响应，核对音频 prior、教师表达目标和接收器三者各自的误差；保留 B0 中性与学生无内容入口。
3. 根据训练内部留出说话人/句子的证据制定一个明确假设与有限候选，再决定目标或结构修改。无需先叠加 loss，也不再直接延长 Phase46。

独立内容读出、目标身份迁移、多 seed、匹配预算的外部基线、未见测试和人评仍未闭合。这些开发集数值不足以声明全面 SOTA；MEDTalk/DESTalker 尚无核验训练数值，表格不补造。

## 视频、表格与恢复

[两组八类情感视频](PHASE46_VIDEO_GALLERY.md)，[17 方法对比与 13 条真实训练消融](PHASE46_EXPERIMENT_TABLES.md)。外部方法是同 rig 的适配版本，不能写成原论文官方数值。

各视频列顺序：GT / Neutral B0 / Audio prior mean / Posterior ORACLE。第三列为部署；第四列使用 query GT，只用于诊断。

原始核验凭据：`final_experiment/evaluation/diagnostics/phase46_receiver_20261008/closure_verified.json` 与 `final_experiment/paper_tables/phase46_development_20261008/closure_verified.json`。远端 `/root/kinetalk_phase46_receiver_20261008` 已完成，最新只读检查 GPU 1 MiB / 0%。不要重新启动 launch、训练、评估或 collector。

mixed checkpoint SHA：`6e58c6404f03c948378bd0217c05d4a0a6f6e1c098d285ecad007308687bb768`。

deploy checkpoint SHA：`000e62e6bb61e7a01a6e35f0d91214c9b3c5aadc0def798be909ef08a3ce424b`。
