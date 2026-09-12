# Restore-based KineTalk execution plan

## Goal

以 `D:\实验室项目\新实验\kinetalk_v1_baseline_restore` 已验证的 DLP + residual Flow-Matching DiT 为主干，保留其稳定口型和有效情感/风格控制，同时修正 Style 单参考迁移不稳定、参考内容泄漏和情感表示过于离散的问题。

## Hard contracts

1. 目标音频决定内容、音素时间轴和基础口型。
2. DLP/Stage1 只接收目标音频 content，不接收 reference motion 或 reference style。
3. 参考 Style 的输入必须是同一参考片段的：

   ```text
   R_ref = M_ref - DLP(A_ref)
   S_ref = E_style(R_ref)
   ```

   参考音频和目标音频可以是不同句子、不同长度、不同情感；只要求参考音频与参考 BS 在自己的时间轴上对齐。
4. Style Encoder 不复制参考时间轨迹，也不直接把原始参考 BS 当作 Style 输入。
5. Emotion 由目标音频的情感/韵律特征提供，主要作用于 residual renderer；离散 emotion/intensity 只作为辅助监督。
6. 最终输出保持完整 ARKit-52：`M_hat = B0_target + DeltaM`。

## Four stages

### Stage 1: audio-only neutral prior / DLP

```text
target HuBERT content -> DLP -> B0_target
```

- 训练目标是同一目标音频片段对应的 neutral/base motion。
- 不使用 reference motion 生成 B0。
- 主 loss：masked L1/Huber、velocity、acceleration。
- 训练完成后冻结 DLP。
- Prototype/VQ、通道硬切分和 GRL 不属于当前主路径。

### Stage 2: reference residual Style encoder + local emotion field

```text
reference audio/content -> frozen DLP -> B0_ref
reference BS - B0_ref -> motion Style Encoder -> S_ref
```

- Style 表示说话人的执行习惯：幅度、速度、联动和整体动作方式。仍然
  只有一个 motion-only Style 空间，不拆成 `s_art`/`s_expr`。
- Emotion encoder 同时输出 `E_local[t]` 和 `E_global`；局部场描述音节级
  情感变化，全局 code 描述 clip-level 情感。
- 同说话人不同句子/crop 拉近，不同说话人区分，同说话人跨情感保持稳定。
- 不做跨句子逐帧重建。

### Stage 3: audio affect + residual Flow-Matching DiT prior

```text
B0_target + target audio content + target audio affect + S_ref + noise
    -> Residual DiT -> DeltaM
```

- Stage3 的音频编码器同时拟合冻结 Stage2 的 `E_local[t]`、`E_global` 和
  intensity；分类只作辅助监督。
- DiT 训练在每条样本自己的 audio-motion 对齐时间轴上完成。
- 主要 loss：flow matching、endpoint/reconstruction、velocity、情感分类/强度、轻量 Style consistency 和 mouth-clock preservation。
- Style 从 DiT 早期 block 注入，并支持适度 style dropout。

### Stage 4: deployment and causal validation

- 训练时以同片段 residual 作为 flow matching 的数值目标；Style 条件通过
  独立 FiLM/AdaLN 路径注入。部署时替换为任意 reference residual style。
- 部署时替换为任意 reference audio + reference BS，计算 `S_ref`。
- 固定目标音频和初始噪声，只替换 Style；检查口型时间保持、情感保持、动作幅度/速度/联动变化。
- 固定 Style，只替换目标音频；检查内容、口型和情感随目标音频变化。

## Planned code changes

- 删除 Stage1 主训练中的 prototype assignment/reconstruction、bounded `B0_art` calibration 和 native attention 强制路径。
- 恢复 DLP 风格的 audio-only neutral decoder；保留现有 residual DiT 主体。
- 为 Stage2/4 明确构造 `reference_b0 = Stage1(reference_content)` 与 `reference_residual = reference_motion - reference_b0`。
- Style encoder 只编码 residual 的中心化轨迹、速度和可选加速度。
- Stage4 的 reference style 统一使用 reference residual encoder 输出。
- 旧 prototype checkpoint 与 architecture version 不兼容，必须重新训练 Stage1→Stage2→Stage3→Stage4。

## Acceptance criteria

- Stage1 32 条样本可过拟合，jaw/lip temporal correlation 不低于 restore baseline。
- 参考音频/BS 内容改变时，Style code 不复制参考口型时序。
- 固定目标音频和 noise，替换 Style 后 lip-sync 基本保持，style behavior statistics 显著改变。
- 固定 Style，替换目标音频后 emotion/activation 和口型随目标音频改变。
- 所有 checkpoint 严格检查 architecture version；旧 prototype checkpoint 明确拒绝加载。
