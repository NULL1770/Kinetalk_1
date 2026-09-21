# 区域接收器与音频验证协议审查

2026-09-21。此次为本地代码只读审查；未启动训练，未核实新的远端数值。本记录用于防止将接收器几何能力、oracle 条件结果和音频泛化混为一谈。

## 结论

1. `regional_intensity_gain` 是对冻结运动形状进行幅度调制的原型，不能创造先验中缺失的方向或保证移动错误的峰值。此前 `prior40` 未同时通过 ES 和 variogram，只能称运动形状先验。
2. 一轮小样本 `--smoke` 可以发现运行错误和验证输出通路；其失败不能证明接收器结构必然无效，其通过也不能证明留出数据上的接收能力。应执行预先锁定预算和条件的 oracle 训练，再独立判断。
3. `run_signed_receiver_probe.py` 使用目标动作的**未平滑四组投影**及固定提升矩阵，主要证明某个几何子空间可表示动作。它没有训练情感条件方向，也没有证明音频可以预测此投影。
4. `train_signed_audio_state.py` 实际复用旧 `RelativeAudioTiming`、stride16 慢四状态、相对声学 PCA24＋韵律4及状态 MSE，与已失败的 `train_relative_audio_timing.py` 没有实质性新监督。不能以新文件名将其作为新的区域强度方案重新长训。

## 必须先修正的定义

- **oracle 与 student 目标不一致：**signed probe 直接投影去均值 upper9；student 的 `load_context()` 使用 neutral-relative 四状态、stride16 样条投影，再去均值。若要验证 student 接收器，oracle 必须输入同一个目标定义和同一个输出通路。
- **constant 实际是 ramp：**signed probe 的 `_constant()` 生成以目标 RMS 定幅的零均值线性斜坡。应改名 `oracle_rms_ramp`；真正的时间常量输入经去均值后应与 zero 等价，不能偷偷改成斜坡来制造非零对照。
- **static 是嵌套零动态：**当前静态对照保留同一冻结基座均值、令新动态为零。它可称 `matched_mean_zero_dynamic`，不能称独立训练的静态先验。运动先验 gain=1 也不是时间静态，应称 `frozen_prior_identity_gain`。
- **reverse 不等于倒放音频：**旧 `RelativeAudioTiming(mode='reverse')` 倒排的是预测状态。它检验输出顺序，不检验网络对反序音频的响应。若保留，应准确标注；真正的 reverse 条件应反转有效声学序列、重算相对特征与差分，再推理。
- **shuffle/permutation 含混：**student 的 shuffle 是跨片预测替换；probe 的 permutation 同时跨片滚动和反序，且套用目标片 mask。它们混合了时长、补零和时间干预。应拆为同片时间 shuffle 与跨片 mismatch，保留各自原始有效长度，显式对齐到查询时钟。
- **“缺失事件覆盖率”实际是漏检帧率：**当前实现统计目标包络超过片内 q75 时，预测低于自身峰值 10% 的帧比例，越低越好。它没有先定义冻结先验漏掉哪些事件，再计算 receiver 挽回多少。用每个预测自己的峰值做阈值也不适合比较幅度。应另报固定训练阈值下的 target-event recall/precision，以及 frozen-prior-missing-event recovery；无缺失事件时后者为 null。
- **峰值延迟命名过强：**当前值是 ±15 帧内使相关最大的滞后，并非逐个峰值配对延迟。应称包络互相关滞后，标明帧率、毫秒、有效片数；常量序列无定义，不能填成 0。

新增或修改门槛之后，旧结果只能标作探索性、事后重评分，不能追溯称为预先规定的成功。新运行前保存条件定义、阈值、训练预算、样本清单及源码哈希；oracle 和 audio 的成功标志必须分开。

## 当前音频 runner 的阻断问题

- `_train_epoch(..., device)` 的选择阶段调用漏传 `device`，会直接报错。
- 训练后预测没有切换 `model.eval()`；锁定特征 dropout 继续生效，导致 OOF 与 full/reverse/mismatch 受到不同随机噪声干扰。
- calibration 采用 `held_speaker OR held_sentence`，不是全部样本同时 speaker/sentence disjoint。可以分层报告，但不能合并后称双重不重叠。
- acoustic PCA/统计按 fit 子集拟合是正确的；然而 `target_scales` 在 `load_paper_data()` 中使用全部 TRAIN 动作拟合，包含内部 calibration 动作。严格内部 OOF 应在每个 fit fold 重拟合目标尺度并重建目标；冻结基座是否见过这些身份/句子也必须披露。
- 仅有单次内部 holdout，未产生完整 cross-fitting OOF 预测。应称 TRAIN-internal holdout，或实现覆盖所有训练样本的外层折。
- `passed` 只看 full 的单个 MSE 是否小于 zero/static及两项保护，不要求显著性、reverse/shuffle/mismatch、ES/variogram；甚至未合并已经计算的 mouth protection 结果，也没有强制读取 oracle 通过证据。
- `SEEDS=(42,123,2026)` 仅改变冻结 renderer 噪声，不是三次独立 audio student 训练。不得称音频训练多 seed 稳定通过。

## 最小可执行顺序

1. **修正版 gain oracle：**锁定完整数据和预算；同时保留 identity-gain、zero-dynamic、真正 constant-gain、oracle、时间置换。输出区域包络误差/相关、互相关滞后、先验缺失事件恢复、raw 越界率和保护项。保存预测曲线；口型/non-upper 必须逐元素相同。通过只表示该 gain 接收器可控，不能称音频动态成功。
2. **若 oracle 不通过：**报告幅度、方向还是先验事件缺失的瓶颈；不自动启动 signed4/旧 timing 或随机先验续训。固定投影 oracle 可作为表示上限诊断，但必须使用与未来 student 一致的低频目标。
3. **通过后才做音频实验：**模型输入只允许部署可得音频及已授权冻结条件。保存区域目标的片段均值 `mu` 与中心化项 `delta`；不要用 query motion/intensity/emotion 构造输入。
4. **严格分组 OOF：**固定 speaker 分组和 sentence 分组；外层某个 held speaker×held sentence 单元为评估，其训练池同时排除该 speaker组和sentence组。若需覆盖所有 clip，可遍历分组组合；其余交叉单元不能悄悄回流该折训练。每折目标尺度、PCA、归一化只拟合其训练池；epoch 用内层分组验证选择或提前固定。保存每条 OOF 预测所属折和训练列表。开发集仅用于最终冻结协议后的评估。
5. **明确对照：**full、同冻结均值的 zero-dynamic、保持所预测片段均值的 constant-intensity、真正 reverse-input、同片时间 shuffle、跨片 mismatch。控制相同输出时钟、mask、全局条件及 renderer seed；静态类和分布先验基线分别报告。
6. **音频成功门：**主要检查区域时变强度在留出句子/身份上相对 matched-static 的净收益，辅以全片幅度与时间统计；报告按句子与说话人聚类的不确定性，防止用帧数夸大显著性。保留 ARKit-MBE/LBE、ES/variogram、情感/身份和视频检查。不得再将逐点 upper9 R²作为唯一成功门，也不得仅靠幅度增大认定成功。通过后才添加随机微动作。

相关事实依据：`docs/RELATIVE_AUDIO_REPAIR_20260921.md`；本次审查范围：`scripts/train_signed_audio_state.py`、`scripts/run_signed_receiver_probe.py`、`scripts/train_relative_audio_timing.py`、`scripts/train_isolated_audio_state.py`、`scripts/train_regional_intensity_oracle.py`及相应模型。
