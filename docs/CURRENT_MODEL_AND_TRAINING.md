# 当前工作模型：v5 matched control，2026-10-12

按用户“先回退上一个版本”的要求，当前选择恢复为 **v5 matched control**。v6 原型及结果保留，但队列停止，不作为当前模型。旧 Phase53/v6 说明已归档至 [回退前记录](archive/CURRENT_MODEL_AND_TRAINING_before_v5_rollback_20261012.md)。

本地默认选择为 `configs/current_model.json`，用 `python scripts/current_model.py` 严格加载核验；服务器选择为 `/root/kinetalk_current_selection/current_model.json`，用 `/root/miniconda3/bin/python /root/kinetalk_current_selection/current_model.py` 核验。历史实验 runner 仍使用显式 `--checkpoint` 复现实验，不会自动改写旧结果；后续工作应从此选择记录取得 checkpoint。

| 项目 | 当前选择 |
|---|---|
| 模型名称 | v5 matched control，实际类 `KineTalkV4Model` |
| 服务器 checkpoint | `/root/kinetalk_v5_20261012_dual/runs/control/final.pt` |
| 本地 checkpoint | `docs/v5_optimization_20261011/dual_evidence/runs/control/final.pt` |
| checkpoint SHA256 | `372dbc2a23913743cb67452bf6f5756e3be195f81c468f2a1f972afbb79c9add` |
| 原始服务器代码快照 | `/root/kinetalk_v5_20261012_dual` |
| binding | `/root/kinetalk_v5_20261011/binding.json` |
| binding SHA256 | `d30f10075b704c4cb228629549820056c952533fa5c4507666ab8640a1fa6087` |
| 训练状态 | 空闲；未启动新训练；v6 剩余队列停止 |

## 实际生成路径

1540D 缓存特征的前 768D HuBERT 经冻结 B0 生成中性动作 52D 和内容表示 128D；后 772D（emotion2vec 768D + prosody 4D）进入音频学生，输出逐帧 10D 可观察表情状态及整段情感类别、ordinal 强度。推理不输入查询动作，也不调用动作教师。

一条独立参考动作编码为 64D 风格。学生隐藏状态、表情状态和参考风格进入带持续时间的条件事件 GRU，产生事件起始、持续时间及幅度的条件分布；部署采用固定种子的采样路径。四类功能事件不对应固定情绪类别，事件诊断不是逐帧心理情感真值。

内容/B0、表情状态与类别强度、风格、两两交互及事件共同输入时序解码器，**联合输出一份 52D 残差**。几何头给出四个 rig 观测坐标的修正目标，投影器调整最终动作。它仍然有 `B0 + 联合残差`，没有拆成独立口型残差和情感残差，也没有 v5 direct/guided 的低频/高频 adapter 或 v6 因子输出头。

情感和风格可以改变全部 52D 动作，包括嘴角宽度、张口、唇紧张等嘴部表达。表示层输入职责不同，不要求在输出上用互斥嘴部/上半脸通道划分。联合生成本身合理，但不自动证明语义解耦、条件有效或性能更好。

## 教师与论文结论边界

动作教师在训练实现中保留，接受可观察动作状态和整段情感/强度监督。但所恢复 control 的真实配置是 `use_teacher=true`、`teacher_guidance_mode=none`、`state_supervision_space=raw`；这个 continuation 没有启用教师到学生的状态蒸馏项。不能把 control 的结果当作“教师优于直接学生监督”的证据。

本 control 的 256-update continuation 没有加入 HDTF（`hdtf_enabled=false`），也没有文本。HDTF 已处理的内容/风格扩充实验独立保留，不能把别的 checkpoint 的训练数据或最优指标并入此模型。

三个贡献方向仍按用户要求：残差改造、教师—学生的情感/动态学习、口型/情感/风格联合生成与解耦。当前恢复用于后续实验的较好基线，尚未达到三项贡献均有消融和优势支持的论文最终模型。

## 已有证据与本次核验

完整 1367 条开发集，sample / clip_all：mean lip 3.048290 mm，最大顶点 LVE 5.853746 mm，上半脸 EVE 2.435403 mm，closure proxy F1 .553729，生成动作情感 probe F1 .758772，angry F1 .862974。上半脸 EVE 仍差于 Phase53 的 2.237557 mm；动态及风格的充分有效性尚未建立。开发集结果不是封存测试集成绩。

本次核验 checkpoint 与 binding SHA、原始源码快照和严格加载；angry 82 有效帧、happy 103 有效帧的真实单参考推理输出有限值且形状正确，未向 predict 传入查询 GT，未调用教师，权重未改变。没有重跑完整评分或训练。详见 [回退回执](model_rollback_20261012/rollback_receipt.json)。

已渲染视频 `artifacts/v6_direct_review_20261012/{angry_render,happy_render}/comparison.mp4` 五列依次为 GT、B0、Phase53、**V5 control（恢复版）**、v6 direct。原音频、原生 25fps、rig 系数误差 0 及完整解码已核验；这是已有固定片段展示，不是新的渲染质量提升。
