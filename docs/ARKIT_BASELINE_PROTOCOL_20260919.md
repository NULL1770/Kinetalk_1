# 统一 ARKit52 评估与基线重训协议

2026-09-19。FaceFormer、CodeTalker、FaceDiffuser、KineTalk 的比较采用同一数据协议下重训的 ARKit 适配版本。本文件固定比较原则，**尚无三个外部方法的同协议训练结果**，不能把原论文数字复制进本项目主表。

## 1. 主指标的可追溯定义

使用 FaceDiffuser 官方 commit `e15f3500fdae0eda962f5d018488dfa0a1a9d552` 的 `evaluation/compute_objective_metrics_blendshape.py:main_beat`。

- MBE：每帧全区域误差向量的 L2 范数，再平均帧、采样、片段。不是逐系数 MAE，也不是每帧最大系数误差。
- LBE：同一公式，只选官方 mouth mask 对应的 12 个通道。
- FDD：`std_t(sum_upper GT²) - std_t(sum_upper prediction²)`，`ddof=0`。额外报告逐片逐采样取绝对值后的 ABS-FDD，避免正负抵消。signed FDD 以零为参考，不能按负数越小越好排序。
- 官方区域按**名称**映射，不复制另一数据集的下标。官方实现实际 51 列，未含 tongueOut；文中使用 ARKit52 这一名称不能解释为 52 个通道都被观察。

更正前期说明：FDD 没有相邻帧差分，也没有内部平方根；它对帧序置换不敏感。论文文字提到 maximal L2，但公开 BEAT 代码计算区域向量 L2。本项目明确遵循固定版本的代码定义，不声称文字与实现完全一致。

官方 upper mask 包括两侧 blink/look/squint/wide 以及 noseSneer，**不包括 brow**。官方 mouth mask 包括 jawLeft/right、dimple、funnel、pucker、roll、shrug、upperUp，缺少 jawOpen 等常用口型通道。因此另报固定的 `supp_upper9_fdd_absolute` 和 `supp_lip23_lbe`；前者含眉毛、squint/wide，不含 blink/gaze。这是事先声明的区域适配，不用补充指标冒充原版 FDD。

MBE/LBE 只使用原始 channel_mask 标明的可观测值，不将显示用填充值当 GT，不按通道数重新归一化。FDD 只使用所需区域全部可观测的帧，至少两帧。每片记录可用通道与覆盖率。缺失 tongue 不能默认为真实零值。

AV offset/confidence、Multimodality、FD、WInD 参考 SAiD，但该文使用 BlendVOCA 的 32 个 ARKit 衍生通道、60 FPS。本项目 52 维接口、25 FPS 是协议适配，不能与它的原表直接排名。

## 2. 输入条件与数据划分

- 固定原始 ARKit 通道顺序、25 FPS 时间戳、音频采样与特征提取方式、valid/channel_mask，禁止按方法调整 GT 或裁剪幅度以改善分数。
- 数据去重需要覆盖原始视频/音频、同一次录制的重叠窗口及句子身份，而不仅是文件名。全局情感/身份编码器的上游训练曝光也必须记录。
- 当前 613 fit / 206 inner-validation / 64 outer 的 64 条已经反复用于开发，**属于 development，不能重新命名为封存 test**。正式新句子测试与未见身份测试必须重新审计并封存；未完成前不开正式测试主表。
- 同输入的主比较中，所有方法必须获得相同的外部条件。若使用预计算全局情感/身份码，向全部适配基线提供同一组冻结特征及相同可见信息，记录接入方式；不能让 KineTalk 独占标签或视觉参考。
- 若选择纯音频轨道，则所有方法均只允许音频与协议规定的身份条件。标签增强轨道独立列出。不能把 GT 动态/VA/眉眼数值作为默认推理输入；oracle 只用于上限诊断。

## 3. 重训适配的边界

FaceFormer 将顶点输出头替换为 52 维系数头；保留并验证自回归/音频对齐逻辑。CodeTalker 必须在同一 fit ARKit 数据上重训运动 tokenizer，再训练音频到 code 的模型，不能直接使用另一网格的 codebook。FaceDiffuser 采用其 blendshape 路径，按本项目名称顺序及原始 mask 改接数据。

这些应在论文标注 `FaceFormer-ARKit / CodeTalker-ARKit / FaceDiffuser-ARKit (our adaptation)`，不是作者发布模型的原始结果。仍需完成三套模型的代码适配、冒烟检查和训练，本文档不是已经完成重训的证明。

固定训练种子 42/123/2026，各方法最终采样使用相同种子清单；确定性方法重复推理的多样性为零，不额外注入噪声制造随机性。当前历史随机输出只有 4 draws，论文级随机评价的采样数须对各随机方法统一，并与 SAiD 样本预算的差异写清。

预算不能只写相同 epoch：列出优化步数、有效观察帧、batch、阶段预训练、GPU 时长、参数量。主比较采用统一数据曝光/调参预算；多阶段方法的 tokenizer、运动先验、情感和身份相关训练均计入。另列按原方法推荐配置充分训练的结果，避免用不收敛的短训 baseline 证明优势。具体共同预算在吞吐量和收敛冒烟后锁定，不能按最终测试成绩选择。

checkpoint 仅按共同 validation 规则选择；测试只运行已锁定模型。逐 draw 评分再平均，不平均轨迹、不挑 best-of-K；clip 等权，并报告按句子聚类的区间和三个训练 seed 的结果。

## 4. 仍待接通的外部评价器

1. **AV offset/confidence**：同一 avatar、摄像机、口部渲染流程、音频与时间偏移、SyncNet 权重与版本。先对 GT 和人工偏移样本验证符号/单位与可用性，再运行全方法。offset 参照同协议 GT 与人工偏移校准，不能直接宣称原始零偏移一定最好。系数数组本身不能产出 SyncNet 分数。
2. **Multimodality/FD/WInD**：同一独立运动评价 encoder，仅在 fit 真实动作上训练，固定结构、窗口秒数、预处理、缺测策略与权重哈希。不能用只为 KineTalk 训练的 9 维生成 AE 替代全方法共享评价器，也不能用未经训练的特征距离冒充 SAiD。
3. SAiD 官方脚本以 BCVAE posterior mean 为特征；Multimodality 在相同人物/句子/窗口起点的多次采样之间比较；FD 比较特征分布；WInD 使用 GMM/运输距离。窗口/样本池、GMM 配置和随机重复必须固定，拟合失败须显式记录，不填零。

当前上述五个字段已在每次原生生成评估中输出为 `pending`，附具体依赖；尚未产生数值。MBE/LBE/FDD 已实现并重算历史七臂。主评估入口是 `scripts/evaluate_continuous_motion_latent.py`，额外输出 `arkit_benchmark.json`，不改变旧模型选择门槛。

## 5. 当前可以得出的结论

运动先验改善开发集运动分布；音频候选未稳定超过独立静态对照。口部通道本轮被保护，因此数值不变，不等于音画同步已通过感知验收。当前无法宣称优于外部论文方法，也无法根据输入情感/身份探针分数直接认证生成脸的情感/身份效果。

来源：[FaceDiffuser 官方评价代码](https://github.com/uuembodiedsocialai/FaceDiffuser/blob/e15f3500fdae0eda962f5d018488dfa0a1a9d552/evaluation/compute_objective_metrics_blendshape.py)、[官方通道顺序](https://github.com/uuembodiedsocialai/FaceDiffuser/blob/e15f3500fdae0eda962f5d018488dfa0a1a9d552/utils/arkit2metahuman.py)、[SAiD 官方评价代码](https://github.com/yunik1004/SAiD/blob/bb7b18018358d3156569d8051f386943744a5ea0/script/test_evaluate.py)。本地缓存和哈希位于 `artifacts/arkit_benchmark_20260919/sources/`。
