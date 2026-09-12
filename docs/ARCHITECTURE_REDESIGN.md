# KineTalk 重构方案与 DTW 审计记录

## 结论

当前输出平均化蠕动不是单纯的幅度缩放问题，而是三个结构性问题叠加：

1. Stage1 的内容与口型时序监督不足，测试集口部时序相关性约为 0.04，活动口部误差甚至不如全零基线。
2. Stage4 使用参考动作的 style，却要求输出严格拟合 query 自己的逐帧情感动作；跨说话人或不同句子时这个目标没有合法的逐帧答案，模型会倾向于忽略 style/emotion 并压低 residual。
3. Stage2/Stage3 把本应随时间变化的情感主要压成全局 code，无法表达音节级的张合、开合峰值和局部情感动态。

旧版 restore 值得保留的是“内容驱动的确定性口型先验 + 情感/风格残差渲染器”这一职责划分。restore 目录中有占位训练脚本，因此只借鉴结构，不把历史实现或结果当作验证依据。

## DTW 审计结论

当前 `aligned_dtw_v2` 的质量摘要为：15,220 条记录，其中 12,362 条 `ok`、647 条 `cached`、200 条 `review`、1,823 条因缺少 neutral reference 被拒绝、188 条 error；总体 pass rate 为 0.8547，`quality_gate_passed=false`。

当前构建器已有 `repeat_step_ratio`、`max_repeat_run`、`velocity_ratio` 和音画 lag，但质量门槛仍不够：

- DTW 前先把 MFCC/audio stream 重采样到 motion 长度，改变了 native timing。
- cost 主要来自 MFCC cosine similarity，不等于可靠的音素对齐。
- 路径没有充分限制局部 slope，因此存在大量重复路径。已见 review 样本的 repeat ratio 约为 0.5–0.8，最长重复段超过 100 帧。
- 没有检查 warped audio activity 与 mouth motion event 是否一致。
- `ok/cached` 只表示通过当前构建器条件，不能证明逐帧监督可靠。

审计脚本为 [audit_dtw.py](../scripts/audit_dtw.py)。它独立读取保存的 DTW path、aligned BS 和 aligned audio，重新计算：

- horizontal/vertical/diagonal step 比例；
- 最大重复路径长度；
- 局部时间 slope 分布；
- warped audio activity 与 mouth velocity 的最大滞后相关；
- 按情感统计和异常样本清单。

审计阈值先用于 triage，不宣称是最终生物学真值：

```text
repeat_ratio <= 0.50
max_repeat_run <= 20
local slope 约束在 0.25–4.0 内
audio/mouth event correlation >= 0.10
```

本轮审计的 `event_corr` 使用保存的两通道 audio proxy，不是原始音频能量/F0，也不是音素边界。因此它只能做粗筛：neutral identity path 的该 proxy 相关性也偏低，说明这个指标本身不能单独判定错位。最终训练 gate 必须补充原始音频 activity、HuBERT/phoneme boundary 和 native-time 对照。DTW 可以是跨情感监督的主要风险来源，但即便完全移除 DTW，当前 Stage1 的 neutral query 时序也没有达标，因此不能把全部问题归因于 DTW。

本轮审计结果保存于 [dtw_audit_20260910.json](../artifacts/dtw_audit_20260910.json)。在 13,209 条可读取路径中，重复比例中位数约 0.285、P90 约 0.536；最大重复段 P99 为 32、最大为 139；局部 slope 的 P10 为 0、P90 为 8。按“路径几何异常或 proxy event correlation 低”的初筛条件，共有 10,699 条进入异常清单。这个数字用于说明当前质量筛选过宽，不能直接等同于 10,699 条已经被证明错误。

### DTW v3 重做决定

`aligned_dtw_v2` 不再作为后续交叉解耦的训练源。新版本必须输出到独立的 `aligned_dtw_v3`，保留 v2 供对照，不能覆盖原始数据。

v3 的数据契约：音频特征保留 native audio frame timeline，DTW 前不把音频重采样到 motion 长度；path 在 native audio 坐标中计算；再通过 path 把源 motion、audio、content 和 affect 映射到 canonical neutral motion timeline。路径使用 repeat penalty，并保存长度、覆盖、重复率、局部 slope、cost 和事件粗筛指标。`ok` 只表示通过路径几何 gate；保存的两通道 audio proxy 相关性不作为最终硬 gate。`review` 不进入逐帧 Stage1/Stage2 teacher，最多低权重用于统计或干预训练。

release builder 为 [scripts/build_aligned_dtw_v3.py](../scripts/build_aligned_dtw_v3.py)，只写独立的新目录。在全量质量 gate 通过前，不切换训练配置，不启动 Stage1–Stage4 重训。

### 2026-09-10 数据入口与全量审计结果

先完成媒体和句子契约，再做 DTW。`artifacts/mead_media_v1` 对 16,150 个 MEAD BS 记录逐一定位原始 front video，并从视频内嵌音频抽取 16 kHz、单声道、16-bit PCM WAV。16,150/16,150 条视频音频抽取通过，16,053/16,053 条最终 QC BS 都有可读 WAV，没有覆盖原始音频、视频或 BS。

`E:\mead\list_full_mead_annotated.txt` 被用作精确文本来源。manifest 的 `content_id` / `sentence_id` 是规范化文本 hash，原来的数字编号保存在 `legacy_sentence_id` 和 `clip_number`。旧编号配对会产生 13,265 个跨情感 pair，其中 12,136 个（91.5%）对应不同文本。按精确文本分组后得到 7,846 个跨情感 neutral pair，另有 7,242 条没有同说话人同内容的 neutral，后者不能作为 neutral framewise teacher。

远程旧 WAV 与新视频抽取 WAV 不能混用：13,673 个同名文件逐文件 SHA256 对照后 0 个相同，另有 2,477 个远程缺失。因此 v3 使用本次视频抽取的 WAV，并输出到独立远程目录。

第一轮全量 DTW 审计使用 `mfcc39_cmvn`：25 ms 窗、10 ms hop、39 维 MFCC+delta+delta2、每段 CMVN。它只用于审计 native 时间轴和路径几何，不能作为最终英文 phonetic content feature。全量提取 16,053/16,053 通过；同一 pair manifest 上的 v3 DTW 共处理 8,811 对，其中 965 对 identity、7,846 对跨情感。

宽松 geometry 版错误地把 8,811 对都标为 accepted。独立分布审计发现 995 条跨情感路径存在高总体重复率或边界风险，因此增加严格 `path_quality_ok`：repeat ratio ≤ 0.35、单方向 run ≤ 3、局部 slope 在 0.25–4.0、cycle error P95 ≤ 80 ms、local valid ratio ≥ 0.8。严格 MFCC 版结果为 7,816 条 teacher-eligible、995 条 review；跨情感 teacher 为 6,851/7,846（87.3%），全量为 88.7%。

远程启用 `/etc/network_turbo` 后，英文 `facebook/hubert-base-ls960` 已下载并转换为 safetensors。修正后的提取器通过 `audio_rel + --media-root` 读取视频内嵌音轨生成的 WAV，16,053/16,053 条 native HuBERT layer 6 特征提取完成。相同 8,811 个 pair 的严格 HuBERT DTW 得到 5,880 accepted、2,931 review，其中 21 条还未通过基础 geometry。HuBERT 比 MFCC 更保守，不能因为表征更强就放宽路径门槛。

全量双特征比较中，965 个 identity 均通过；跨情感 pair 有 4,875 条被 HuBERT 和 MFCC 同时接受，1,976 条只被 MFCC 接受，40 条只被 HuBERT 接受，955 条都进入 review。两条路径在物理时间上的中位误差总体 P50 为 10 ms、P95 为 20 ms，但每条路径的 P95 误差总体 P50 为 50 ms、P95 为 300 ms，说明少数局部区间仍有大分歧。最终 consensus gate 要求两种特征都通过、整条路径中位差不超过 20 ms、P95 差不超过 80 ms，得到 4,108 条跨情感 teacher；连同 965 条 identity，共 5,073 条第一版高置信 teacher。完整结果见 [remote_dtw_v3_hubert_vs_mfcc.json](../artifacts/remote_dtw_v3_hubert_vs_mfcc.json)。

DTW 数值插值不是“Stage1 口型只有 GT 一半”的主要原因。对双特征共同通过的跨情感 pair，warped neutral teacher 相对原 neutral 的口部 P95-P05 幅度比中位数为 0.991，`jawOpen` 为 0.983；线性插值相对同一映射时刻最近邻采样的幅度比中位数为 0.995，`jawOpen` 为 0.991。局部坏例确实会衰减，因此仍需 mask；但全局约 50% 的模型输出幅度来自条件均值化、旧监督冲突和内容时序学习失败，不能靠修改插值或整体乘 2 修复。

因此后续数据使用规则分两层：5,073 条 consensus pair 通过了双特征和路径几何审计，但独立 `small.en` ASR 又发现 502 条 source/reference 内容明显不一致、460 条置信度不足。首轮安全 teacher manifest 只保留 4,111 条，其中 837 条 identity、3,274 条跨情感；460 条进入 review，502 条进入 quarantine。其余 pair 和没有 neutral 的记录只能用于 self reconstruction、音素/viseme 内容先验、事件监督或审计统计，不能通过补零、平均、单特征放行或旧编号强行进入 neutral teacher 集。完整内容审计见 [remote_asr_consensus_full.json](../artifacts/remote_asr_consensus_full.json)，三份可追溯清单见 `artifacts/teacher_consensus_*_manifest.jsonl`。

## 重构后的职责

```text
目标音频 A_q
 ├── Content Encoder ──> C_t ──> Canonical Lip Prior ──> B0_t
 ├── Affect Encoder  ──> E_global, E_local_t
 └── native timeline

参考动作 R_s ──> Motion-only Style Encoder ──> S_global

B0_t + C_t + E_local_t + E_global + S_global
       └── bounded residual renderer ──> ΔM_t

M_hat_t = B0_t + ΔM_t
```

- Content 决定音素、发音内容、开合时刻和基础口型时钟。
- `B0` 保证基本口型跟随目标音频，不读取 reference style，也不依赖 neutral audio。
- Emotion 负责目标音频上的局部表达动态、激活和全局情感方向。
- Style 负责参考人物的执行习惯：幅度、速度、左右不对称和伴随动作。
- Noise 负责一对多变化，不能承担基本口型时序。

Emotion 和 Style 可以改变嘴部幅度和形态，但不应取代 Content/B0 的口型时钟，也不应硬切 ARKit 通道。

## 各阶段修改

### Stage1：重做 native-time 内容口型先验

输入使用任意情感的目标音频，不能限定为 neutral。内容路径不读取情感类别、参考动作或 style。

音频前端之后应显式分成 `C_t = ContentEncoder(audio)` 与 `E_t,E_g = AffectEncoder(audio)`。Stage1/B0 只接收 `C_t`；Affect 分支不回流到 B0，而在后续 residual renderer 中调制。这里的“交叉解耦”是同文本跨情感的 content 一致性、情感分类从 content 中不可预测，以及固定 content 后替换 affect 的干预验收；它不是把不严格对齐的两段动作逐帧强行拉成完全相同。

建议使用冻结的 HuBERT/WavLM/wav2vec 类逐帧内容表征，保留 native timeline，替换当前仅由 32 维 MFCC 派生的内容输入。训练监督为：

```text
L_B0 = L_recon + λv L_velocity + λa L_acceleration
      + λe L_mouth_event + λp L_phoneme_or_viseme
```

可靠对齐区域可以逐帧监督；不可靠区域必须降权或跳过。对于不同情感的同内容样本，只要求开合事件和音素时序一致，不强制错误 DTW 下的每一帧数值相等。当前强 pair loss 应删除或降到只作用于事件级特征。

最小训练数据组合：965 条 neutral identity 提供无 DTW 的数值锚点；4,108 条 consensus cross pair 提供 masked neutral teacher；全部 16,053 条音频提供冻结 speech feature、音素/viseme posterior、mouth-event 和可用的 self reconstruction。没有 neutral partner 的 7,242 条记录不参与 B0 数值 neutral 回归。为防止条件均值化，batch 必须平衡 phoneme/viseme 与说话人，损失只保留 masked reconstruction、velocity/acceleration 和 mouth-event/viseme 四类，不再叠加互相冲突的全局幅度 pair loss。

Stage1 的硬验收条件：native audio、neutral audio、emotional audio 和未见说话人的 mouth event F1、jawOpen 时序相关性、峰位置误差和幅度统计都要超过 zero/mean baseline。

### Stage2：局部 affect 与全局 style 分开

当前 residual 混合了情感、说话人执行方式、发音残差、DTW 误差和 tracking noise，不能全部交给 global emotion encoder。

- `E_local[t]` 保留时间变化。
- `E_global` 表达整段音频的情感方向。
- `S_global` 只从 motion reference 提取，使用 centered motion、velocity、acceleration 和全局统计。
- Style 不读取 reference audio 的 emotion feature。

保留三类有效约束即可：same-speaker style consistency、cross-speaker separation、style swap 后的因果评估。不要继续无目的叠加 GRL、triplet、supcon、cycle 和多个 probe。

### Stage3：audio affect field

Stage3 不再只输出全局 emotion/intensity，而输出：

```text
E_local[t], E_global, neutral_gate, arousal/prosody scalar
```

neutral 使用独立 gate 或 balanced binary head，配合 neutral oversampling 和阈值校准。8 类分类只作为辅助语义监督，不能作为整个动作条件。

### Stage4：有效重建与无 GT 风格干预分开

有效的 self reconstruction：

```text
query audio + query style + query affect → query motion GT
```

这里可以使用 flow matching、mouth reconstruction、velocity 和 event loss。

跨样本 style swap 没有严格逐帧 GT，只使用：

- content/phoneme timing 保持；
- mouth event timing 保持；
- style 统计变化；
- speaker/style classifier；
- emotion classifier 不应随 style donor 改变；
- style encoder 重编码一致性。

禁止继续使用“donor style + query 原始逐帧情感动作”作为严格 flow target。

输出使用有界调制：

```text
M_hat = B0 + α(E_local, E_global, S) ⊙ ΔM
```

`α` 以小值初始化并限制范围，避免 residual 覆盖或抵消 B0 的口型时钟。

## 实施顺序

1. 运行 [audit_dtw.py](../scripts/audit_dtw.py)，输出全量分布、异常样本和按情感统计。
2. 用 annotation-based media manifest 重建独立数据目录，先做小样本 dry run，再做全量。
3. 已完成 native MFCC 与英文 HuBERT 的全量双路径审计，并完成 5,073 条 consensus pair 的独立 ASR 内容审计；下一步只对 4,111 条 safe manifest 补做 phoneme boundary 和 mouth-event gate，不回收 review/quarantine pair。
4. 先只重训 Stage1，直到 mouth event、native-time 时序和幅度达标；未通过前不启动 Stage2–Stage4。
5. 重训 Stage2，拆开 `E_local`、`E_global` 和 `S_global`。
6. 重训 Stage3，加入 neutral gate 和 balanced sampling。
7. 最后训练 Stage4，self reconstruction 与 donor style intervention 分开。
8. 用 style/emotion dropout、交换实验和事件保持指标验收，而不是只看训练 loss。

## 禁止的回归路径

- 不把推理时不存在的 neutral reference 作为必要输入。
- 不把 DTW donor 轨迹当作跨样本唯一正确答案。
- 不用整体乘 2 代替口型时序修复。
- 不用更多随机噪声或更多 DiT steps 掩盖 timing failure。
- 不继续堆叠没有因果验收的 loss。

## Safe supervision v1 implementation (2026-09-11)

The code now has an explicit optional sidecar contract for safe-pair supervision:

- `mouth_event`: five channels `[active, onset, offset, robust_signal, robust_velocity]`;
- `mouth_event_mask`: marks frames where the event stream is available;
- `boundary`: three channels `[boundary, phone_start, phone_end]` reserved for timestamped phoneme/viseme alignment;
- `boundary_mask` and `viseme_mask`: zero when no trusted aligner output exists;
- `viseme_id`: `-1` when no trusted viseme label exists.

`scripts/build_safe_supervision.py` computes the mouth event stream from neutral BS motion and applies a pair-level event agreement gate along the saved v3 path. It intentionally leaves phoneme and viseme labels masked when a timestamped aligner has not been supplied; this prevents fabricated boundaries from becoming training truth. `scripts/validate_safe_supervision.py` fails closed on missing sidecars and reports event/boundary coverage.

The dataset reads these sidecars without changing the old batch shape for existing consumers. Stage1 now has a small event head and an optional masked BCE event objective (`loss.stage1_event`), so event supervision is gated by availability rather than forced onto every frame. The default config points at the v3 release root and a separate `safe_supervision_v1` sidecar root. No training was started in this change.

The remaining remote data step is to run the sidecar builder against the remote v3 BS and pair roots, then run validation. Phoneme/viseme supervision becomes active only after a timestamped aligner output is installed and passed its own coverage and boundary sanity audit.

`scripts/audit_dtw_mouth_release.py` now checks each eligible pair instead of relying only on the aggregate median. It reports mouth P95-P05 preservation, `jawOpen` preservation, and an amplitude gate of per-pair P05 >= 0.60 and P95 <= 1.40 after excluding near-static channels. This gate is for masking pathological local paths; it is not an amplitude rescaling operation.

### Remote safe supervision run

The remote run completed on the v3 HuBERT release:

- 4,111 safe pairs produced 4,111 motion sidecars;
- schema validation passed for all gated sidecars, with event mask coverage 1.0;
- no phoneme/viseme timestamps were present, so boundary and viseme masks remain zero;
- the first cross-emotion event quality screen passed 1,238/4,111 pairs (30.1%). This is a screening result, not evidence that the other 2,873 DTW paths are wrong: the screen compares a motion-derived binary activity stream across different emotional executions, so it is intentionally conservative and can reject valid expressive timing.
- per-pair amplitude audit read all 4,111 pairs. Mouth P95-P05 ratio percentiles were P05 0.835, P25 0.960, P50 0.999, P75 1.016, P95 1.145. `jawOpen` ratio percentiles were P05 0.819, P25 0.958, P50 0.996, P75 1.015, P95 1.135. 3,778/4,111 passed the 0.60–1.40 amplitude gate.

Conclusion: the amplitude gate confirms that DTW does not globally halve the mouth. The event result is now implemented as a local quality/mask signal: all 4,111 safe rows remain numerical teacher eligible; the 1,238 high-quality rows are only a diagnostic subset, and local mismatching frames are masked through `pair_masks/*.npz`. Event agreement is never used to delete an otherwise safe pair.

Event gate purpose: it detects local cases where the DTW path maps a mouth-active frame to a mouth-inactive frame, or vice versa. It is useful for lowering the weight of unreliable local supervision, but it cannot define emotional equivalence and must not be used as a global cross-emotion acceptance test.

## Stage1 contract correction (2026-09-11)

An audit of the actual v3 artifact and Dataset found a critical target error. In `build_aligned_dtw_v3.py`, `canonical_motion` is the **source emotional motion sampled on the reference/canonical feature timeline**. It is not neutral ground truth. The neutral teacher is the reference clip's neutral motion, which must be materialized on that same canonical timeline. `neutral_teacher_on_source` is a separate array on the source motion timeline and cannot be paired directly with `canonical_content` without an explicit time-axis conversion.

Therefore the current Stage1 checkpoint trained with `canonical_content -> canonical_motion` is invalid as a neutral B0 result. Its low loss is compatible with learning a contaminated emotional target and does not validate mouth amplitude or timing. It must not be used to initialize Stage2.

The correct Stage1 path is exactly one branch:

```text
source audio of any emotion
  -> native HuBERT content features
  -> canonical content timeline C_t
  -> content encoder/decoder
  -> B0 neutral motion
```

The target must be:

```text
reference neutral motion sampled on canonical_times
```

For identity pairs this is the neutral motion itself. For cross-emotion pairs it is the neutral reference motion, while the source emotional motion is reserved for Stage2 residual training. Stage1 must not use `emotion_pair`, `pair_loss`, or source emotional motion as its neutral target. Accepting emotional audio as input remains correct; it is the target and the content branch that must be separated.

The minimal Stage1 objective is two terms:

```text
L_B0 = masked weighted Huber(B0, neutral_target)
     + lambda_v * masked weighted L1(delta B0, delta neutral_target)
```

The first term preserves absolute mouth values and the second preserves opening/closing timing and direction. Mouth channels receive a fixed channel weight in the first term. No event loss, cross-emotion output matching loss, global amplitude loss, classification loss, contrastive loss, or adversarial loss belongs in the first clean Stage1 run. Mouth events are evaluation diagnostics until their target is generated from the same canonical neutral timeline. Acceleration can be added only after a small overfit test shows velocity supervision produces jitter; it is not required initially.

Before retraining, a Stage1-specific materializer must add `canonical_neutral_target` and `canonical_neutral_mask` to each pair artifact or write an equivalent sidecar. A 32-sample overfit test must first reach high mouth timing correlation and an amplitude ratio near 1.0. Only then should the full Stage1 run be accepted.
