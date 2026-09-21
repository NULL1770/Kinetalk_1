## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-22
- Verification Status: ANALYZED — real training outputs inspected; one CPU inference replay completed; full training not independently reproduced
- Version Label: native_regional_activity_v2 / native40_carrier5

# 原帧率区域活动实验：运行完成，音频时序诊断未通过

本轮交付了冻结 Stage4 上脸运动的平滑载体、25 fps 音频区域活动预测器、独立推理入口、真实权重和四类有声对比视频。它是供检查的动态候选，**尚未实现可靠的音频时序控制**。没有替换默认模型，没有读取封存测试集。原基线本身已有运动；本轮主要改善了其高频抖动，不能把这个改善归因于学到了音频节奏。

## 1. 实际路径与监督含义

```text
冻结 Stage4 原始 ARKit52 prior
  └─ upper9 有效连续段内五帧三角平滑 → 恢复原段均值 → carrier
音频特征 [25 fps, 1540D]
  └─ stride=1 区域包络网络 → 两维非负 brow / eye 活动
活动 / carrier 活动（分母下限 0.02）
  └─ gain 限制在 [0.25, 2.5] → 缩放 upper9 中心化 carrier → 再中心化并加回均值
其他 43 通道、无效帧：逐元素沿用冻结 prior
```

1540D 特征来自 content 768D、emotion2vec middle 768D 和 prosody 4D。目标是**片内中心化 upper9 的五帧平滑区域 RMS 活动**，不是绝对情感强度、眉毛方向、MEDTalk EIE，也不含 blink/gaze。目标动作保持原始数据定义，carrier 的三角平滑与活动包络的五帧平滑是不同操作。

平滑是固定后处理，不是新训练的运动生成器。对称窗口和完整有效段均值使当前方法属于离线处理。Gain 可以改变既有动作幅度，无法在零运动位置创建新的动作形状或更换运动方向。本轮未加入随机微动作。

相关代码：

- `kinetalk_b0/models/audio_regional_envelope.py`
- `kinetalk_b0/models/temporal_motion_carrier.py`
- `scripts/train_audio_regional_envelope.py`
- `scripts/infer_audio_regional_envelope.py`

## 2. 数据、来源与选择协议

| 项目 | 本轮设置 |
|---|---|
| 数据 | 四类 MEAD；4098 TRAIN / 446 development；25 fps |
| 新 adapter 选择 | TRAIN 内 speaker×sentence 留出；fit 2287 / calibration 255；1556 个混合格样本不参与选择 |
| 预算 | 40 epochs；校准 MSE 选择第 3 轮；全 TRAIN 从头 refit 3 轮 |
| 归一化 | 选择阶段仅 fit 估计；refit 重新在全 TRAIN 估计 |
| 冻结基线 | `/root/autodl-tmp/kinetalk_calibrated_20260920/run12/protected/audio/final.pt` |
| 基线 SHA256 | `be975208b978309e7eff328fb07418e413e49a8fe7f76b9bac67c341083e0fbb` |
| 数据 manifest SHA256 | `6bdc33c3ee832e968b421781267dc70654e416b8e76395f2c6ce83c859e8603e` |
| 随机种子 | 训练默认 47；最终 development prior noise seed 48；单 seed |
| 运行时间 | 约 279 秒；训练正常完成，diagnostic gate=false |

这里的留出只针对新 adapter。冻结 Stage4 已见过完整 TRAIN，不能声称整个系统完成了严格身份与句子留出。继承的 `load_context` 会预计算 development 派生张量，但它们没有用于本轮选择、拟合归一化或 oracle-only 评分。Development 在 refit 后做一次完整对照；该开发集在项目历史中已被多次使用，不是独立测试集。

选轮后训练结束，不是训练不足：选择阶段 train loss 约从 0.1343 降至 0.0345，calibration MSE 第 3 轮为 0.09042、第 40 轮恶化至约 0.11358，提示明显过拟合。

## 3. 输出侧与真实音频结果

32 条多样化 calibration 样本的 oracle 接收诊断中，真实包络经**最终组合路径**后相关约 0.893，互相关峰值绝对延迟约 0.172 帧。它支持输出路径能接收真实活动控制，不能证明音频能够预测该活动；也不是全开发集性能。

正式 development 结果按 clip 等权聚合。以下评分对象均为**最终 ARKit 合成结果再提取的区域包络**：

| 条件 | MSE | 中心化 MSE | 相关 |
|---|---:|---:|---:|
| original_prior | 0.083463 | 0.050188 | 0.02606 |
| 平滑 prior | 0.084043 | 0.046129 | 0.03770 |
| full | 0.067513 | 0.043082 | 0.08369 |
| static_gain | 0.085930 | 0.059071 | 0.03770 |
| static_envelope | 0.067507 | 0.042748 | 0.02762 |
| reverse | 0.068236 | 0.043805 | 0.06112 |
| shuffle | 0.066803 | 0.042797 | 0.03791 |
| mismatch | 0.071924 | 0.044034 | 0.06234 |

Full 互相关峰值绝对延迟约 8.59 帧（344 ms），相关仍弱。它不是模型实际运行延迟，也不能据此给所有音频统一平移 344 ms。

预测端 full 包络相关 0.12354、时变标准差 0.06023，而 GT 为 0.16423；预测端 full MSE=0.072205，反而高于 static_envelope 的 0.069190。Full 与 static 的这一预测端 MSE 差在 speaker/sentence 两种 cluster CI 下均为正；最终组合端则基本持平。

Full 相对 mismatch 的最终包络 MSE 差为 −0.004411，speaker cluster 95% CI=[−0.008268, +0.000676]，跨零；sentence CI 为负。中心化 MSE 差为 −0.000952，两种 CI 均为负，支持存在弱时序信号。Development 只有 3 个 speaker、85 个 sentence cluster，不能将这个 bootstrap 扩展成可靠跨身份结论。完整门控仍未通过。

对照含义：`static_gain` 将每条预测 gain 取片内常数，仍保留 prior 原有时序；`static_envelope` 将预测包络取片内常数，再除以变化的 prior 包络，gain 仍可能变化。两者都不是静态脸。`reverse` 反转预计算特征，不是反转波形重新提取特征；`shuffle` 在有效段内打乱特征帧；`mismatch` 使用非自身 donor，优先同人异句，并做长度归一。各对照共用同一冻结 prior。

## 4. 可见运动、抖动与保护

| 原始系数统计 | GT | original_prior | 平滑 prior | full |
|---|---:|---:|---:|---:|
| 眉部 temporal std | 0.018998 | 0.023981 | 0.013653 | 0.020911 |
| 眼部 temporal std | 0.019971 | 0.022962 | 0.012695 | 0.018487 |
| 二阶帧差分能量 | 0.000066857 | 0.00446031 | 0.000055222 | 0.000165868 |

单独平滑就降低约 98.76% 的二阶差分能量；full 相对原 prior 降低约 96.28%，但仍约为 GT 的 2.48 倍。音频 gain 恢复了部分幅度，也增加了加速度。因此“降抖”主要来自固定平滑，“更自然”仍需观看和独立主观评估。

Full 的 non-upper43 逐 bit 不变，口部保护通过，上脸均值最大漂移约 1.19e−7。它只证明保留了基线口型，不能证明基线的绝对口型质量合格。31.88% 的 gain 达到 2.5 上限，表明接收器仍受原先验与幅度限制。

原始系数评分没有 clamp。Full upper 约 12.67% 系数处于 [0,1] 外；渲染显示才 clamp 到 [0,1]，会改变边界附近的可见幅度与均值。因此原始输出的均值保护不能直接称为渲染后也严格保持均值。报告同时保留 raw/clamped 诊断。没有完成情感感知、身份感知或主观自然度验证。

## 5. 复现与推理

服务器工作目录：`/root/autodl-tmp/kinetalk_active_20260921/code`。

```bash
/root/miniconda3/bin/python -u scripts/train_audio_regional_envelope.py \
  --data /root/autodl-tmp/kinetalk_repair_20260920/prepared \
  --source /root/autodl-tmp/kinetalk_calibrated_20260920/run12/protected/audio/final.pt \
  --output /root/autodl-tmp/kinetalk_active_20260921/native_regional_20260922/native40_carrier5 \
  --device cuda --epochs 40 --carrier-window 5 --batch-size 16
```

不要覆盖已有实验目录；新复现应使用新目录。选择与 refit 日志、完整 `evaluation.json`、`protocol.json`、`status.json`、`final.pt`、`native_curves.pt` 均在上述远端输出目录。约 301 MB 的完整 curves 文件只保留远端。

本地权重及结果：`artifacts/native_regional_20260922/`。从仓库根目录运行真实权重示例：

```powershell
python scripts/infer_audio_regional_envelope.py `
  --checkpoint artifacts/native_regional_20260922/final.pt `
  --input artifacts/native_regional_20260922/mead_M025_angry_L1_001_inference_input.npz `
  --output artifacts/native_regional_20260922/angry_inference.npz `
  --device cpu
```

输入需要已有冻结 prior `[T,52]`、audio_features `[T,1540]`、valid `[T]`、times `[T]`；可附 channels/channel_mask/clip_id/noise_seed。入口仅读取白名单输入，不读取 query motion、target 或 emotion 标签。**它是预计算特征到动画的 adapter，不是 raw WAV 到完整系统的部署封装。**输出四模式为 original_prior/prior/full/static_gain，并生成保护报告。

真实权重 CPU 重放与 GPU 导出对比：original_prior、prior 最大差均为 0，full 最大绝对差约 4.05e−6，static_gain 约 6.56e−7；口型和非上脸保护通过。这是一个样例的推理重放，不是独立重训复现。

本轮相关模型、runner、carrier 和既有 regional 测试 57 项通过，独立推理 CLI 测试 14 项通过。代码测试通过与研究诊断门失败是两种不同结论。

## 6. 视频与交付范围

四类样例固定选择每类 clip_id 字典序第一条，全部来自 M025，未按指标挑最好样例：

| 样例 | 帧数 | 本地视频（仓库相对路径） |
|---|---:|---|
| mead_M025_angry_L1_001 | 93 | `artifacts/native_regional_20260922/angry_render/comparison.mp4` |
| mead_M025_happy_L1_001 | 94 | `artifacts/native_regional_20260922/happy_render/comparison.mp4` |
| mead_M025_sad_L1_001 | 124 | `artifacts/native_regional_20260922/sad_render/comparison.mp4` |
| mead_M025_neutral_L1_001 | 112 | `artifacts/native_regional_20260922/neutral_render/comparison.mp4` |

五列顺序：GT / original_prior / prior / full / static_gain。视频含原声音频，25 fps；Blender 4.3.0，原有 ARKit rig，每列 320 px、samples=8。Rig audit 检查逐帧 shape key 对应关系，原 blend 文件未修改。已检查静帧与系数曲线，没有完成整段视频自然度评审。

源代码和这份汇总随 Git 发布；真实数据、音频、视频、权重与完整评估保存在本地/既有服务器，不纳入本次 Git 提交。精简机器可读结果见 `docs/NATIVE_REGIONAL_DYNAMICS_20260922.results.json`。

## 7. 下一步必须解决的具体问题

1. **先核对监督语义。**固定 TRAIN 内 12–24 条视频，对照当前片均值 RMS、独立中性参考偏离、局部速度 RMS，并检查同一帧在不同裁切中的目标变化。当前目标依赖整片均值，例如单调抬眉可能得到两端高、中间低的目标；它未必适合表达“情感逐渐增强”。这只是结构上可解释的风险，尚未证明它是本次失败的主因。先核对可见动作与跟踪噪声，再决定监督定义；不用开发集误差挑目标。
2. **把跨人和跨句失败分开。**固定现有 fit、现目标，用低容量线性包络探针比较同人新句、新人旧句、新人新句，并配同一静态音频基线。原划分未利用的 mixed cells 可以诊断失败来自哪一轴；不再用 446 条 development 选择模型。如果同人新句有效而新人失败，再验证独立中性参考幅度校准；若三格均不优于静态，优先处理目标或音频信息不足。
3. **区分动作生成能力与幅度控制。**现有 gain 路径能保留活动，却无法生成 prior 缺失的眉峰，且大量饱和。若低维预测确认有稳定信号，再比较带零均值加性运动基的接收器与 gain-only；该比较须使用同一预测目标、同一冻结口型、同一 seed 和最终合成评分。加性路径本身不能解决音频监督弱的问题。
4. **让论文结论随证据走。**多 seed、静态/错配/反序对照和独立观看确认之后，才判断能否支持音频动态贡献；随机微动作和可见运动只用于建模一对多变化，不能冒充音频时序预测成功。当前这轮结果不能写成已成立的新贡献。

已有 `audit_prepared_audio_clock.py` 与相关审计核过 25 Hz、波形 SHA 和 audio offset 契约。严格有效相邻帧下，声学变化与 upper 速度的相关仅约 0.033–0.050；旧的 0.44–0.52 来自跨 gap 差分错误，已在 `CURRENT_SYSTEM_20260921.md` 纠正。没有新的物理错帧证据时，不进行全数据时间平移。
