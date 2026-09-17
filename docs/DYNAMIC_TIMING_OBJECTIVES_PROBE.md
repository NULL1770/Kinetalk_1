# 动态幅度与时序监督探针

目的：FiLM 的 heldout 时序相关为正，但 R² 仍负；验证失败来自幅度标定，还是 timing 本身不稳定。相关变正尚不能证明音频缺少语义，也不能证明归一化就能解决。

`scripts/probe_dynamic_timing_objectives.py` 保持同一 emotion2vec + content FiLM、upper-face L1 目标、stride=4、按句划分。三种 `--objective` 互斥，每次仅一个损失：

- `mse`：原连续强度 MSE，对照。
- `clip_rms`：每条训练目标除以自身 RMS，训练集 RMS 第 10 百分位设下限，防止弱动态噪声放大。仍用一个 MSE。
- `events`：目标相邻 bin 差分按训练集绝对差分第 65 百分位离散为 offset/steady/onset。预测仍只有一个 field；其差分构成固定三类 logits，使用单一 CE。不是增加三个学习输出通道或三个 loss。

推理不能读取目标 clip 的 RMS，更不能按 heldout 句子或身份用 motion 拟合幅度。因此统一报告原幅度和训练集拟合的一个非负 gain；这个固定 gain 对所有 heldout 音频相同。heldout 目标 RMS 只用于 shape 指标，不能将 shape R² 称为真实 motion 改善。

记录 full/zero/reverse、content zero/reverse/shuffle、事件 macro-F1、时序相关、原单位 R²，并保存参数、输入统计、划分、目标阈值及源码哈希。模型固定训练步数；不根据 heldout 选择 step。B0、identity、global emotion、renderer 的完整 state hash 必须保持不变。

通过条件：跨句原单位 R² 超过 zero，full 优于 reverse，收益在多个 seed 或新句子复现，然后才接 renderer；仅训练误差、shape 指标或动作幅度变大不构成成功。这里只是预测探针，不证明最终生成效果或论文录用。

运行参数与 `probe_audio_content_gating.py` 相同，追加 `--objective mse|clip_rms|events`，输出目录必须尚不存在。
