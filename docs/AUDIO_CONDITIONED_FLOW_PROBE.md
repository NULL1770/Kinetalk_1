# 音频条件下的 DiT 局部适配

执行前协议，2026-09-16。推理仍为 audio + neutral 身份参考；保留一个低率场和一个 DiT。音频不能唯一决定眉毛动作，但标准条件 flow 可以用噪声学习条件分布，无需同一音频有重复真值。确定性 audio head 的 MSE 与 flow velocity MSE 不等价；随机种子差异本身也不能证明合理性或不合理性。

本轮固定原 native RRR8/head/尺度、B0、身份、audio global、local_projection，从 `scaled_centered_v1/uniform/final_epoch008.pt` 恢复接口。只训练现有 DiT 最后一层 `cross_attention.out_proj.weight`（192×192，36,864 参数），包括该层 bias 的其余权重全部冻结。该矩阵作用于读取 noisy motion query 后的 cross-attention 输出，比再次训练确定性音频头更直接检验条件生成适应；不新增层、区域头、文本、VA 或 loss。

两个配对臂：audio_local 正常音频动态；zero_local 局部场置零。两臂有同样可训练矩阵和非零梯度。原始 uniform-full 和 uniform-zero 为不训练基线，另外评估各臂自身 zero/reverse/oracle。仅改 local 并不保证生成均值或嘴部不变，必须实测。

固定训练内19人2315 fit、旧3人405开发；不读取280/439或test目标。原模型历史暴露未消除，405已有多次探索，本轮不是独立测试。每臂8epoch、batch16、Adam lr=1e-5、gradient clip1、seed46、同batch/noise/time draws。只有 observed flow velocity MSE，沿用原20% t=0 + 80% uniform时间抽样，不用 rollout 单GT MSE、不用teacher local、不另加约束loss。motion仍提供真实训练目标和oracle诊断；无推理motion。

固定第8轮，不看开发挑轮或追加强度。12步生成，8个预指定种子 `[42,123,2026,7,19,73,211,997]`，逐一保留，不做best-of-K。每epoch保存矩阵、Adam、RNG和输入/冻结hash；默认模型不替换。

评估同时报告两种问题：

- 可预测时序：中心化R²/相关、full−zero/reverse、与frozen-full和matched zero-local训练输出比较，按句配对bootstrap5000、逐身份拆分。
- 分布：完整中心化轨迹 energy score（L2 / sqrt(观测数)，finite-K fair及empirical），相邻有效帧variogram p=.5，幅度/速度和seed方差。ES越低越好，fair VS有限样本可负；这些不替代人脸感知或情感合理性验证。

生成改善候选需非neutral眉 fair ES 相对frozen-full、同模型zero/reverse、matched zero-local训练的zero输出均有正改善95%句簇CI；可预测R²改善另列，不用其单独否定一对多。共同保护：neutral/nonneutral嘴和眼raw MSE相对frozen-full单侧90%增幅≤1%、嘴相关下降≤.005，mouth/upper速度误差增幅≤5%、非neutral upper均值MSE增幅≤1%。冻结训练teacher的情感读出只作诊断，并非独立评测。全部结果保留；任何保护失败不接默认，也不声称CCF或最终可用。
