# 受约束音频时序修正 probe

目的：检验上一轮非线性修正过拟合后，同一 rank8 场中的极小线性修正能否保住 ridge 跨句泛化。该实验只处理可预测均值，不回答 DiT 的一对多采样质量。

固定原 native ridge/std/U、输入、4 帧时钟、19 人 2315 条 fit 及三句折。令 `q=ridge(audio)/train_control_RMS`，输出为 `ridge + train_control_RMS * center(Conv(q))`。Conv 无 bias：pointwise 64 参数，居中 3-tap 时序版 192 参数；全部零初始化，初始严格等于 ridge。无新增区域、文本、VA 或逐帧自由变量。

每折两臂各 8 epoch，batch32，Adam lr=.002，seed46+fold，完全配对 batch。唯一目标为原单位中心化动作 MSE 加 `10 × 原单位预测偏离 ridge 的 MSE`；这是对同一回归的信任约束。系数 10 预先固定，不依据 OOF/开发/测试调参，不追加 epoch。目标读取只在 fold-fit 索引后发生。3-tap 版本是居中窗口，但左右 tap 独立，不施加时间反转对称约束。

沿用既有 OOF 评估及严格门槛：眉 ΔR² 句簇 CI 为正、相关提高、胜 zero/reverse、至少 10/19 人改善；nonneutral 和 neutral 的嘴/眼 MSE 单侧 90% 上界增幅 ≤1%，嘴相关下降 ≤.005。所有区域只用于审计。保存真实 checkpoint 重建复核、冻结参数 hash、配对 batch hash。通过后才考虑 renderer 小测试；失败不接入默认。

不读取 405/280/439 或 test512/15 targets，不训练 renderer/B0/identity/global，不改默认 checkpoint。三折结果属于已使用训练集上的描述性诊断，不能据此声称独立泛化、合理多样性或论文保证。

## 实测结果

v2 已完成三折、两臂、各 8 epoch。temporal 相对 native ridge 的眉 ΔR² 为 **+0.000442**，句簇 95% CI `[+0.000233,+0.000645]`，11/19 人估计改善；眉相关也提高。可是 neutral 嘴部残差预测 MSE 增加 **1.4197%**（单侧90%上界 **1.6020%**），超过1%保护线；nonneutral 嘴部反而改善 **0.1444%**，眼部保护均通过。因此 `generation_entry_pass=false`，不接入 renderer 或默认模型。这些不是已生成口型的误差，之前“嘴增加约0.8%”是错数。本轮只说明强约束小修正比上一轮自由时序网络更稳，不能证明眉毛动态已经解决。

同一轮对已有 epoch8 生成曲线进行了只读多 seed 分解。非中性眉 seed 方差占单样本中心化 MSE **13.9%**，ensemble 均值相关 **0.142**；嘴部方差占比 **6.3%**，均值相关 **0.373**。所以 DiT 确实产生了随机差异，但不能单凭该分解判定这些差异合理或不合理。分解满足 `单样本 MSE = ensemble 均值 MSE + seed 方差`，未读取封存测试目标。下一步用现有DiT的小范围audio条件flow适配及预指定多seed分布评分检验，见 `AUDIO_CONDITIONED_FLOW_PROBE.md`。
