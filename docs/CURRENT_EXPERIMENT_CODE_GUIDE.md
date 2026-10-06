# KineTalk 当前实验代码与数据流说明

> 审计时间：2026-10-03。本文按当前工作区源码、已下载的训练协议和 sealed-test 报告整理。重点区分“当前源码”“2026-09-23 已完成正式模型”“2026-09-24 未完整训练的 ours_v2”，避免把不同快照混为一谈。

## 1. 先说结论

当前远端实际运行的正式主线是四阶段：

```text
MEAD 原始视频/音频
  -> 25 FPS ARKit52、HuBERT/content、声学特征
  -> emotion2vec 中层特征 + 4维 prosody
  -> articulation(B0)
  -> identity(中性参考身份偏置)
  -> teacher(运动教师语义 + 音频时序条件训练残差流)
  -> audio(音频学生替代运动教师)
  -> 52维 ARKit 系数序列
  -> 固定 ARKit rig 转顶点
  -> 系数、顶点、独立情感 probe 指标
```

模型的核心分解是：

```text
最终运动 M = B0(语音内容/口型) + I(中性参考身份偏置) + R(随机残差流)
```

- `B0`：由逐帧 768-D content 特征预测音素/口型基础运动。
- `I`：由同一人的独立中性 enrollment 片段提取，整段保持静态。
- `R`：DiT flow-matching 生成的随机表情/其余运动残差，由身份、全局情感、强度、音频时序条件控制。
- 训练时 motion teacher 可看真实运动；部署时只使用音频和中性参考，不读取 query 真实运动。

本地已有 1,622 条八情感 sealed-test 的主模型和大多数消融/基线报告，但严格最终表仍不完整：`no_residual` 报告缺失，`final_experiment/paper_tables/` 只有旧 development 内部表。

### 先回答 B0 的问题

你的理解在“模型接口目标”上是对的：B0 应该把任意情感语音中的**内容/音素信息**转换为情感无关的 canonical articulation（中性口型基础），情感表情再由 residual 分支补上。`Stage1Model` 的类注释也明确写着：

```python
# kinetalk_b0/models/model.py
class Stage1Model(nn.Module):
    """Emotional or neutral content audio -> the same neutral articulation B0."""
```

但“接口设计”和“Stage1 实际训练样本选择”不是一回事。当前远端实际 recipe 是 `articulation_scope: neutral`，Stage1 优化只取 `emotion_id == 0` 的 715 条 TRAIN query；之后 B0 会对所有 train/validation query 前向计算并缓存，后续 teacher/audio 阶段也会用到这些 B0。也就是说：

```text
Stage1 参数更新：只有 neutral 音频/内容 -> neutral motion target
B0 前向使用：train/validation 的所有情感 query 都会经过 B0
模型接口意图：任意情感内容 -> canonical articulation
```

所以不能把现状表述成“B0 用各种情感样本训练成中性口型”；准确说法是“B0 的输出契约是情感无关的，但当前正式 recipe 只用 neutral 子集拟合 B0”。如果要让 B0 真正看到八类情感，必须运行 `--articulation-scope all-emotions`；远端 recipe 没有这样做。

## 2. 三个容易混淆的“版本”

| 对象 | 状态 | 关键区别 |
|---|---|---|
| 当前工作区源码 | Git 有未提交修改 | 四阶段；默认仅保护 15 个发音口部通道；新增生成结果情感一致性损失和 DiT 输出侧情感驱动；`--articulation-scope` 可选 all-emotions |
| `sealed_20260923/ours` | 完整训练并完整测试 | 4 阶段各 12 epoch，共 154,860 step；Stage1 只用 715 个 neutral clip；实际命令启用了 `--protect-mouth`，因此 14..40 全部口部通道不允许残差修改；当时没有当前新增的生成情感一致性项 |
| `sealed_20260924/ours_v2` | 训练未完成但按用户要求评估 | checkpoint 只到 audio 9/12 epoch，recipe 明写 `training_complete=false`；不能当作完整正式训练结果或用于挑模型 |

因此，“当前代码会训练什么”和“已有 `ours` 报告评估的是什么”不完全相同。复现实验必须以 checkpoint 保存的 `training_protocol.json`/`recipe.json` 为准，不能只看今天的源码。

## 2.1 9 月 15 日之后 B0/Stage1 的 Git 历史结论

我检查了 2026-09-15 之后当前 Git 历史中与 B0、`articulation`、`emotion_id`、`neutral` 相关的提交：

| 提交 | 日期 | 对 B0/Stage1 的事实 |
|---|---|---|
| `46c7d1d` | 09-18 | 已经是五阶段旧架构；`articulation` 的样本选择是 `(emotion_id == 0)`，即 neutral-only；Stage1 loss 计算 `system.stage1(...)["b0"]` 对真实 motion |
| `85fb20a` | 09-20 | 迁移到 full native MEAD/audio residual flow；仍是 neutral-only Stage1；主要改训练协议和条件路径，没有把 B0 改成 all-emotion 输入 |
| `a260aa6` | 09-21 | 继续整理四/五阶段条件和口部保护；B0 的 `Stage1Model` 结构没有改变 |
| `164a063` | 09-22 | 新增 `--articulation-scope {neutral,all-emotions}`；默认仍为 `neutral`。这是唯一明确改变 Stage1 数据选择接口的提交 |
| `c7a39eb` / `cb9abd8` / `596e4ba` | 09-22 | 只改 smoke 选 shard、加载进度和 rollout 内存；没有改变正式 B0 样本选择 |

关键 diff 是：

```diff
- if stage == 'articulation':
-     items = (data['splits']['train']['emotion_id'] == 0).nonzero(...)[0]
+ articulation_ids, articulation_scope_info = articulation_selection(...)
+ if stage == 'articulation': items = articulation_ids
```

而 `articulation_selection()` 的实际代码是：

```python
ids = (labels == 0).nonzero(as_tuple=True)[0].long() \
      if scope == 'neutral' \
      else torch.arange(len(labels), dtype=torch.long)
```

因此，9 月 22 日以后“允许 all-emotions”是新增选项，不是远端已训练模型的实际选择。当前可见历史从 9 月 18 日开始；在这条分支中没有一个 9 月 15--17 日的 B0 训练提交可以证明曾经使用过 all-emotion Stage1。更早的设计文档/旧实验很多，但不能冒充当前正式 checkpoint 的训练来源。

### 是否改过 B0 的输入数据？

结论分两层：

1. **B0 的张量接口没有改。** 从 `46c7d1d` 到远端 `source_20260923`，`Stage1Model` 一直是 `content: [B,T,768] -> b0: [B,T,52]`；`model.py` 的 `AudioContentEncoder`、`NeutralArticulationDecoder`、`stage1_native_context=9` 接口没有被替换。当前 `prepare_paper_full_data.py` 也一直校验 native `content` 为 768 维、motion 为 52 维。
2. **B0 的训练样本集合改过/可配置过。** 9 月 18--21 日代码直接固定 `emotion_id == 0`；9 月 22 日新增 `--articulation-scope`，但默认值仍是 `neutral`。因此实际远端训练不是“输入特征维度变了”，而是“Stage1 更新参数时选哪些 clip 变成了可配置，但正式 recipe 仍只选 neutral”。

同时，完整训练数据准备从旧的逐 shard 读取改成了 packed variable-length memmap；这改变的是加载/存储方式，不是 B0 的 content 数值接口：`content` 仍为 native 25 FPS 的 `[T,768]`，`audio_features` 的 1540 维扩展只供情感学生。

## 3. 数据接口

### 3.1 单个原生 clip

`scripts/prepare_paper_full_data.py::read_native()` 从每个 `.npz` 读取：

| 字段 | 形状/类型 | 含义 |
|---|---|---|
| `motion` | `[T,52] float32` | 目标 ARKit52 blendshape 系数 |
| `content` | `[T,768] float32` | 与 25 FPS 对齐的语音内容特征，供 B0 使用 |
| `audio` | `[T,83] float32` | 原有声学特征；被保存，但当前主模型不直接用它训练情感分支 |
| `times` | `[T] float64` | 视频帧时间戳，要求相邻约 0.04 秒 |
| `mask` | `[T] bool` | 有效帧 |
| `channel_mask` | `[52] bool` | 本 clip 哪些 ARKit 通道真实可观测 |
| `provenance` | JSON | 音频路径、hash、25 FPS 时钟证据等 |

所有无效帧/通道在进入缓存前清零，但评估仍使用原始布尔 mask，不能把填充零当真实标注。

### 3.2 当前实际送入训练的音频特征

query clip 额外调用 `scripts/extract_predictable_audio.py::extract()`：

```text
content 768
+ emotion2vec layers(2,4,6) 中间表示的固定平均 768
+ prosody 4 (log pitch, log RMS, periodicity, voiced flag)
= audio_features 1540 维
```

- emotion2vec 是冻结的 `emotion2vec_plus_base`，不在本实验中更新。
- 特征先按真实音频卷积中心时间插值到视频原生 `times`。
- `SlowStateAffect` 的 mean/std 只在 TRAIN query 有效帧拟合。
- packed cache 将 1540-D 音频以 FP16 存盘，每个 batch 转 FP32；motion 为 FP32，times 为 FP64。

需要严格区分 B0 和 audio affect 输入：

```python
# train_full_staged.py 的实际调用
base = base_forward(system, b['content'], b['valid'])
audio_output = audio_affect(audio, b['audio_features'], b['valid'])
```

也就是：

- B0 使用 `b['content']`（768-D content），不是直接使用完整 1540-D `audio_features`；
- teacher/audio 情感路径使用 `b['audio_features']`（1540-D）；
- `b['motion']` 只在训练时作为 B0/残差 flow target 和 motion teacher 输入，部署时 query motion 不存在。

### 3.3 数据规模与划分

已完成正式训练协议记录：

| 划分 | clip 数 | 身份用途 |
|---|---:|---|
| train query | 12,536 | 拟合模型和所有统计量 |
| validation query | 1,367 | development 诊断，不用于 sealed-test 拟合 |
| sealed test query | 1,622 | 3 个未见身份、8 类情感；只在预测封存后读 GT |
| neutral enrollment | 每个身份至少 2 条 | 推理时允许使用的独立中性身份参考 |

TRAIN 的 TongueOut（索引 51）全部缺失，所以所有方法统一只评分 0..50 共 51 个受支持通道；TongueOut 在预测和 GT 两侧都固定为中性。

### 3.4 batch 字典

训练器主要读取：

```python
{
  "audio_features": [B,T,1540],
  "content":        [B,T,768],       # audio_features 的前 768 维
  "motion":         [B,T,52],
  "valid":          [B,T],
  "times":          [B,T],
  "channel_mask":   [B,52],
  "anchors":        [B,52],          # neutral enrollment 统计锚点
  "anchor_valid":   [B,52],
  "speaker_id":     [B],
  "emotion_id":     [B],             # 0..7
  "intensity_id":   [B],             # neutral=0，其余通常 1..3
  "intensity_valid": [B]
}
```

## 4. 数据阶段用了哪些脚本

### A. 划分与 manifest

1. `scripts/expand_mead_full_emotion_manifest.py`
   - 读取 TRAIN/VAL 元数据，不读取 test 数组。
   - 验证身份互斥、hash、帧数、八类情感及强度覆盖。
   - 输出带 canonical SHA256 的 train/val/test 元数据 manifest。

2. `scripts/rebuild_mead_eight_emotion_sealed_manifest.py`
   - 从真实 MEAD test 元数据重建 1,622-query 八情感 sealed role。
   - 每个测试身份固定选择两条 neutral enrollment。
   - 本阶段只碰元数据，不打开 query motion。

3. `scripts/audit_sealed_test_manifest.py`
   - 只审核 manifest：身份交叉、role hash、enrollment、情感覆盖。
   - 输出协议记录，不是模型评估。

### B. 特征与训练缓存

1. `scripts/prepare_paper_full_data.py`
   - `validate_manifest()`：阻止 test 数据进入准备过程。
   - `read_native()`：校验每个 native artifact 的路径/hash/形状/25 FPS/mask。
   - `prepare()`：保存每 clip 的 motion/content/audio/times/mask，并为 query 提取 emotion2vec/prosody。
   - `pad_clip()`：只为当前内存 batch 补齐，padding 永不计入观察。
   - `load_paper_data()`：组装 train/validation、neutral refs、train-only feature stats 和 motion scales。

2. `scripts/extract_emotion2vec_pilot.py`
   - 校验官方冻结权重和卷积时钟；最终输出 768-D emotion2vec 表示。
   - 独立脚本版本也能生成 `train.pt/heldout.pt`；主准备流程主要复用其 `load_extractor()`、对齐和 hash 工具。

3. `scripts/extract_predictable_audio.py`
   - 提取 layers 2/4/6 的中层 768-D 表示以及 4-D prosody。
   - 不读文本、情感标签或目标运动。

4. `scripts/build_trainval_cache.py` → `scripts/packed_trainval_cache.py`
   - 把大量 per-clip shard 转成共享的变长 NumPy memmap。
   - `PackedSplit.batch()` 只取本 batch、按本 batch 最大长度 padding，避免把 13,903 个 query 全部展开成巨型 padded tensor。
   - 输出 `metadata.pt`、`refs.pt`、各 split 的 flat `audio_features.npy/motion.npy/times.npy` 及固定字段。

## 5. 模型组成与代码位置

### 5.1 B0：内容到基础发音运动

代码：`kinetalk_b0/models/model.py::Stage1Model`

代码的真实接口（`Stage1Model.forward`）是：

```python
def forward(self, content, mask=None,
            reference_motion=None, reference_mask=None):
    content, lag, native_pos, weights = \\
        self.native_aggregator(content, return_aux=True)
    h0 = self.content(content, mask)
    canonical = self.neutral(h0, mask, content)
    # affect mouth coefficients are zeroed at the B0 boundary
    canonical[..., affect] = 0.0
    return {"h0": h0, "b0": canonical, ...}
```

对应的数据流是：

```text
content[B,T,768]
 -> NativeFeatureAggregator(context=9)
 -> AudioContentEncoder
 -> h0[B,T,128]
 -> NeutralArticulationDecoder
 -> b0[B,T,52]
```

- 配置中 `neutral_output_indices` 是 14..40 加 51。
- 当前代码固定的主要发音通道为 15 个：`14..22,31,32,37..40`。
- smile/frown/dimple/stretch/shrug/press 等 affect mouth 通道在 B0 输出边界被清零，交给残差分支。
- 注意：`reference_motion` 和 `reference_mask` 虽然出现在函数签名中，当前 B0 forward 并不使用它们；身份只在后面的 `encode_identity()` 进入生成器。
- “中性”不是把输入音频波形变成 neutral 音频，也不是删除输入的情感声学特征；B0 的输入仍是 query 的 content 特征，情感去耦主要由 Stage1 的训练数据选择和输出通道契约实现。

当前远端正式 run 的 Stage1 训练代码原文是：

```python
# scripts/train_full_staged.py
if stage == 'articulation':
    out = base_forward(system, b['content'], b['valid'], gradients=True)['b0']
    cc = list(system.stage1.art_indices)
    m = mask[..., cc]
    scale = scales[cc].clamp_min(.05)
    raw = huber(out[..., cc] / scale, b['motion'][..., cc] / scale, m)
    pair = m[:, 1:] & m[:, :-1]
    velocity = huber(
        (out[:, 1:, cc] - out[:, :-1, cc]) / scale,
        (b['motion'][:, 1:, cc] - b['motion'][:, :-1, cc]) / scale,
        pair)
    loss = raw + .1 * velocity
```

这里的 `b` 来自当前 `items = articulation_ids`；远端 recipe 明确记录 `emotion_counts: {"0": 715}`。所以训练 target 是 715 个 neutral query 的发音通道，而不是八情感全部 query。
- TongueOut 因训练集无观测而最终被 `motion_support` 清零。

### 5.2 身份分支

代码：`kinetalk_b0/models/neutral_affect.py::NeutralIdentityEncoder/NeutralAffectSystem.encode_identity`

输入是 neutral enrollment 的：

```text
reference_residual = reference_motion - reference_B0
形状 [B,R,T,52]
```

每条参考计算逐通道 mean/std，经 MLP 得到 reference code，再对参考等权平均：

```text
identity code: [B,128]
identity baseline: 0.5 * tanh(Linear(code)) -> [B,52]
```

它表示 ARKit 系数的静态身份/中性执行偏置，不是人物网格几何。

### 5.3 音频情感学生

代码：`kinetalk_b0/models/slow_state_affect.py::SlowStateAffect`

实际 forward 的关键代码是：

```python
# kinetalk_b0/models/slow_state_affect.py
clean = torch.where(mask, features, self.feature_mean)
normalized = (clean - self.feature_mean) / self.feature_std
hidden = torch.where(valid[..., None], F.silu(self.input(normalized)), 0.)
for block in self.blocks:
    hidden = block(hidden, valid)
global_code = self.global_head(masked_mean(hidden, valid))
local = torch.where(valid[..., None], self.local_head(hidden), 0.)
return {"global": global_code, "u_a": local,
        "emotion_logits": self.emotion_classifier(global_code),
        "intensity_logits": self.intensity_classifier(global_code)}
```

```text
audio_features[B,T,1540]
 -> train-only mean/std 标准化
 -> Linear + 4 个 dilated temporal blocks (1,2,4,8)
 -> pooled global[64]
 -> emotion_logits[8], intensity_logits[4]
 -> u_a[B,T,64]
```

`u_a` 是音频时序上下文，不是“逐帧眉毛真值”。文件里仍返回的 slow-state/local 旧字段只为旧 checkpoint/诊断兼容；当前四阶段生成主线只消费 `global + u_a`。

### 5.4 运动教师

代码：`kinetalk_b0/models/neutral_affect.py::LowRateAffectEncoder`

实际调用代码是：

```python
# scripts/train_full_staged.py
residual = torch.where(
    obs(q), q['motion'] - q['b0'] - identity['baseline'][:, None], 0.)
teacher = system.encode_motion(residual, q['valid'])
```

所以 teacher 明确读取真实运动，而 audio student 不读取真实运动：

```text
motion residual = GT motion - B0 - identity baseline
 -> LowRateAffectEncoder(motion=True)
 -> teacher global / emotion logits / intensity / low-rate control
```

输出全局情感向量、8 类 logits、4 级强度 logits 及低率控制。当前 runner 在 teacher 阶段使用教师的全局语义，同时使用音频学生的 `u_a` 作为 renderer 时序条件；不会把 motion teacher 的逐帧轨迹当作音频唯一目标。

### 5.5 残差生成器

代码：`kinetalk_b0/models/dit.py::ResidualDiT` 和 `NeutralAffectSystem.flow/generate`

配置：DiT width 192、4 blocks、6 heads、residual scale 0.25。条件包括：

- `h0[B,T,128]`：内容上下文；
- `global affect[B,64]` + `intensity[B,1]`；
- `identity code[B,128]`；
- `u_a[B,T,64]`：音频时序条件；
- `x_t[B,T,52]` 和 flow time。

flow matching：

```text
r = (motion - B0 - identity_baseline) / 0.25
x_t = (1-t)*noise + t*r
velocity_target = r - noise
```

DiT 预测 velocity；推理时默认用 12 步 Euler 从随机噪声积分得到 residual，再与 `B0 + identity baseline` 相加。

## 6. 四个训练阶段具体做什么

统一入口：`scripts/train_full_staged.py`。

| 阶段 | 样本/输入 | 更新参数 | 主要损失 | 输出 |
|---|---|---|---|---|
| 1 `articulation` | 默认 TRAIN neutral query；`content, motion, valid, channel_mask` | `system.stage1` | 归一化 Huber(position) + `0.1*Huber(velocity)`，只算发音通道 | 新 B0/h0；随后重算所有 query/ref 的 base cache |
| 2 `identity` | 同一身份的两组互补 neutral references | identity encoder + identity bias | 跨参考 baseline MSE + `0.05*style_contrastive` | 身份 code 和静态 52-D baseline |
| 3 `teacher` | 全部 TRAIN query；真实 motion residual、audio_features | motion teacher、audio student、renderer | flow MSE + `0.1*(emotion CE + intensity CE)`；当前源码另加生成结果情感一致性 | 运动教师全局语义 + 音频 `u_a` 条件下的 renderer |
| 4 `audio` | 全部 TRAIN query；只把 audio student 送入 renderer | audio student + renderer；motion teacher 冻结 | flow MSE + `0.1*semantic CE + 0.5*global distill`；当前源码另加生成结果情感一致性 | 可部署 audio-only 情感/时序条件模型 |

当前源码新增的生成情感一致性由 `generated_emotion_consistency()` 完成：把可微生成运动重新送入 motion teacher，计算 emotion CE、global cosine 和 intensity CE，默认总权重 `0.2`。它解决“音频分类头准确，但 renderer 忽略情感”的问题；2026-09-23 完整 `ours` 并未使用这一新版损失。

### 6.1 远端实际训练命令/recipe（SSH 审计）

远端实际目录为：

```text
/root/autodl-tmp/kinetalk_final_20260922/
  code/source_20260923/
  checkpoints/kinetalk_refactor_memmap_20260923/
  checkpoints/kinetalk_v2_20260924/
```

正式完整 `ours` 的保存 recipe 记录：

```json
{
  "stages": ["articulation", "identity", "teacher", "audio"],
  "epochs": 12,
  "batch_size": 2,
  "seed": 47,
  "articulation_scope": "neutral",
  "protect_mouth": true,
  "paper_data": "/root/autodl-tmp/kinetalk_data/packed_trainval_20260923",
  "fit_clips": 12536,
  "development_clips": 1367
}
```

`ours_v2` 的远端 `status.json` 明确写着：

```json
{"status":"stopped_for_evaluation",
 "reason":"user_requested_evaluation_without_resume",
 "stage":"audio", "completed_epochs":9,
 "checkpoint":"last.pt", "test_loaded":false}
```

它的 recipe 与完整 `ours` 的关键差异是：

```json
{"protect_mouth": false,
 "emotion_consistency_weight": 0.2,
 "emotion_global_weight": 1.0,
 "emotion_intensity_weight": 0.1,
 "articulation_scope": {"emotion_counts": {"0": 715},
                        "clip_count": 715}}
```

因此，当前 SSH 上“实际采用的流程”不是旧的五阶段 `dynamics` 流程，也不是 `run_paper_full_queue.py` 中写的 `--condition-mode` 流程；那两个是历史脚本。远端正式主模型使用四阶段 runner，B0 Stage1 仍是 neutral-only，后续 teacher/audio 才遍历全部 12,536 train query。

本次 SSH 检查时（2026-10-03）远端没有仍在运行的 `train_full_staged.py` 进程；`kinetalk_v2_20260924/status.json` 是“用户要求先评估、未 resume”的停止状态。因此下面写的是远端实际采用过并已写入 recipe/checkpoint 的流程，不是一个当前仍在后台推进的任务。

### 参数冻结逻辑

- 每个阶段开始先冻结 `system` 和 `audio`，只解冻本阶段列出的模块。
- 每阶段结束验证未解冻参数 hash 没有漂移。
- checkpoint 固定保存 system/audio/optimizer/RNG/recipe hash；正式 stage 最终保存 `final.pt`。
- validation 中间结果只作诊断，协议声明固定使用最后 epoch，不按 sealed-test 选 checkpoint。

### 口部支持的关键差异

当前代码默认：残差禁止修改 15 个发音通道，但允许修改 affect mouth；只有显式 `--protect-mouth` 才禁止修改 14..40 全部口部。

已完成 `ours` 的训练协议实际用了 `--protect-mouth`，所以该 checkpoint 的口部主要来自 B0 + 静态身份偏置，情感残差不能产生 smile/frown 等口部变化。这一点不能用当前默认代码的行为去解释旧 checkpoint。

## 7. 推理接口

部署时需要：

```text
query: content[T,768] + audio_features[T,1540] + valid[T]
identity: 至少两条独立 neutral enrollment motion/content/mask
随机种子: 控制一对多 residual draw
```

逻辑接口：

```python
base = system.base(content, valid)                   # b0, h0
identity = system.encode_identity(reference_residual, reference_valid)
affect = audio(audio_features, valid)               # global, u_a, logits
result = system.generate(
    content, valid, identity, affect,
    initial_noise=noise, steps=12, base=base
)
prediction = result["motion"]                       # [B,T,52]
```

输出既可保存 ARKit52 系数，也可通过固定线性 rig：

```text
vertices[T,V,3] = neutral_vertices[V,3]
                  + Σ_c coefficient[T,c] * blendshape_delta[c,V,3]
```

## 8. sealed-test 阶段与脚本

### 8.1 先准备输入

`scripts/prepare_sealed_inputs.py`

- 读取 query 的音频/content/times，不读取 query motion。
- 读取允许作为推理输入的 neutral enrollment motion。
- 复用 TRAIN 已冻结的特征统计和 emotion2vec 提取器。
- 输出 flat query input、offset、metadata、`enrollment.pt` 和完整 hash 绑定。

### 8.2 冻结 checkpoint 推理、封存、再评分

- KineTalk、FaceFormer、FaceDiffuser及旧适配入口：`scripts/evaluate_sealed_models.py`。
- 修正后的 VOCA-core/EmoTalk-core：`scripts/evaluate_core_sealed_models.py`。

严格顺序：

```text
加载固定 checkpoint
 -> 只读 sealed inputs 做全部预测
 -> 保存 predictions.npy
 -> 写 predictions_sealed.json 和 SHA256
 -> 第一次打开 query motion
 -> 计算指标并写 report.json
```

报告绑定 manifest、test role、checkpoint、输入缓存、neutral references、rig、probe、代码 hash；不在 test 上拟合统计量或选 epoch。

## 9. 评估指标到底是什么

### 9.1 ARKit coefficient 指标

实现：`scripts/evaluate_arkit_literature_metrics.py`，汇总：`scripts/arkit_benchmark_report.py`。

- **MBE**：每帧在全部受支持系数上计算预测与 GT 的 L2 norm，再平均帧、随机 draw、clip。单位是原始 coefficient。
- **LBE**：同样公式，但主表使用 FaceDiffuser/BEAT 映射的 Lip12；另有 `supp_lip23_lbe`。
- **FDD absolute**：先算指定 upper region 每帧 `sum(coeff^2)`，分别取 GT 和预测的时间总体标准差，再取二者差的绝对值。主 region 是 BEAT Upper16（眼/鼻，不含眉）；另有 Upper9 补充版。
- FDD 对帧顺序置换不敏感，只反映能量分布差异，不能证明音频-眉眼时序同步。

聚合始终是“每 draw 先算指标 → draw 平均 → clip 等权平均”，不是 best-of-k，也不是先把多个生成轨迹求均值。

### 9.2 固定 rig 顶点指标

实现：`scripts/evaluate_vertex_lve.py`；rig 为 `final_experiment/rig/fixed_arkit_rig_20260923.npz`。

- **LVE (mm)**：每帧在 281 个 lip vertices 中取最大欧氏距离，再对帧、draw、clip 平均。
- **EVE (mm)**：每帧在 351 个 eye/forehead vertices 中取最大欧氏距离，再平均。
- **vertex FDD (mm²)**：在固定区域对相对 neutral 的顶点位移平方能量做时间标准差，再比较预测与 GT 的绝对差。
- 另存 mean-region LVE/EVE 和 FaceDiffuser squared-LVE 诊断；它们不能和主表的 max-Euclidean 定义混用。

### 9.3 情感指标

`scripts/fit_packed_emotion_probe.py` 只在真实 TRAIN motion statistics 上拟合一个独立 probe，以 validation macro-F1 选 probe epoch；之后冻结。

sealed test 的 `emotion_macro_f1` 是 probe 对“生成运动”的八类 macro-F1：1,622 clip × 3 个随机 draw，共 4,866 个观察。它不是音频学生分类头的 accuracy。旧主模型音频头 validation accuracy 约 0.884，但生成运动独立 probe macro-F1 只有 0.127，说明问题在 renderer 的情感可读性，不是音频分类本身。

### 9.4 t-SNE 与尚未实现项

- `scripts/evaluate_emotion_tsne.py` v3只对 train/validation 的emotion2vec mean和真实upper9的mean/std/分位数/相邻速度统计做描述性t-SNE，不是生成质量指标。旧v2的centered-upper均值在数学上为零，相关旧输出不能作为情感分布证据。独立F1判定器使用的motion_features一直是正确的完整统计，未被本修复改动。
- AV offset/confidence、FD、WInD、multimodality 在当前报告里明确是 pending；不能把 `pending_metrics=[]` 误读成所有文献指标都完成。

## 10. 基线脚本

| 方法 | 训练/模型脚本 | 实际性质 |
|---|---|---|
| VOCA-core | `baseline_training.py` + `baseline_core_arkit.py::VocaCoreARKit` | 保留 window/strided conv/expression residual 结构，但用共享 1540-D frozen audio，输出改为 ARKit52 |
| EmoTalk-core | `baseline_training.py` + `baseline_core_arkit.py::EmoTalkCoreARKit` | content/emotion/level/person 分支及 cross-attention decoder 的 ARKit 适配；不是官方 3D-ETF checkpoint 复现 |
| FaceFormer | `faceformer_arkit_adapter.py` + `faceformer_arkit_model.py` | 保留自回归、PPE/ALiBi/历史运动，改用共享缓存和 ARKit52 |
| FaceDiffuser | `train_facediffuser_arkit.py` + `kinetalk_b0/models/facediffuser_arkit.py` | 扩散模型 ARKit 适配，使用冻结 KineTalk 条件；不是官方 BEAT 全流程复现 |

所有论文比较都必须标注为 ARKit/shared-condition adaptations，不能称为官方原数据集端到端复现。

## 11. 当前本地结果状态

### 11.1 完整正式 `ours`（不是当前源码 v2）

| 指标 | 值 |
|---|---:|
| MBE | 0.919714 |
| LBE | 0.466377 |
| coefficient FDD abs | 0.108821 |
| LVE | 9.570863 mm |
| EVE | 2.448974 mm |
| vertex FDD | 131.563912 mm² |
| independent emotion macro-F1 | 0.127030 |

### 11.2 未完整训练的 `ours_v2`（audio epoch 9/12）

| 指标 | 值 |
|---|---:|
| MBE | 0.895043 |
| LBE | 0.472897 |
| coefficient FDD abs | 0.121104 |
| LVE | 8.842567 mm |
| EVE | 2.453040 mm |
| vertex FDD | 131.032477 mm² |
| independent emotion macro-F1 | 0.220141 |

该 v2 数值只能称“用户请求时对 epoch-9 checkpoint 的固定评估”，不能称最终模型；也不能因为 test 更好就反向选择它。

### 11.3 消融/表格完整性

- `no_teacher`：完整报告存在。
- `no_mouth`：完整报告存在；实际含义是开放 mouth residual，不是删除输入 mouth。
- `no_residual`：本地没有 sealed 完整报告。
- VOCA-core、EmoTalk-core、FaceFormer、FaceDiffuser：完整 1,622-clip test 报告均存在。
- 严格 `build_final_paper_tables.py --require-test` 仍缺 Table 3 的 `no_residual`，所以当前没有完整最终 Table 0/1/2/3。

## 12. 最终论文表的脚本

`scripts/build_final_paper_tables.py` 是当前最终表入口：

- Table 0：五种方法统一 ARKit52 的 MBE/LBE/FDD abs；
- Table 1：固定 rig 的 LVE/EVE/vertex FDD；
- Table 2：EmoTalk、FaceDiffuser、Ours 的 ARKit MBE/LBE；
- Table 3：Full、w/o residual、w/o mouth、w/o teacher 的消融；
- `--require-test` 会拒绝 `test_loaded!=true`、缺指标、缺方法以及 legacy VOCA/EmoTalk 报告。

输出为 CSV、LaTeX 和带 provenance/hash 的 `paper_tables.json`。

## 13. 哪些脚本不是当前正式主线

- `scripts/run_paper_full_queue.py` 已过时：仍写“五阶段”、`dynamics` 和 `--condition-mode`，而当前 `train_full_staged.py` 只有四阶段且没有该 CLI 参数；不要直接运行。
- `scripts/aggregate_paper_tables.py` 是旧的通用 main/ablation/emotion/dynamic 聚合器；最终论文表应使用 `build_final_paper_tables.py`。
- `evaluate_paper_coefficients.py`、`evaluate_staged_arkit.py`、`evaluate_empirical_upper_motion.py` 等主要是旧 development/动态诊断，不替代 sealed evaluator。
- `UpperInnovationFlow`、upper9 slow-state、Stage5 helpers 仍留在代码中供旧 checkpoint/测试兼容，但当前四阶段 runner 不实例化、不训练、不保存 Stage5。
- `configs/paper_full_v1.yaml` 的 data/model 维度仍有效；其中 `training.identity_steps/teacher_steps/audio_steps` 不被当前 `train_full_staged.py` 使用。当前预算来自 CLI 的 `--epochs/--identity-epochs/--batch-size`。
- 根目录大量 `.tmp_*.py`、`.codex-finalizer/`、诊断 render 和旧 development 表不是可复现实验入口。

## 14. 建议的可信入口顺序

```text
expand_mead_full_emotion_manifest.py
prepare_paper_full_data.py
build_trainval_cache.py
train_full_staged.py
fit_packed_emotion_probe.py
rebuild_mead_eight_emotion_sealed_manifest.py
audit_sealed_test_manifest.py
prepare_sealed_inputs.py
evaluate_sealed_models.py / evaluate_core_sealed_models.py
build_final_paper_tables.py
```

复现时应先把当前源码提交成一个固定 snapshot，并重新生成 training protocol；否则当前工作树的新增 loss/口部策略与 2026-09-23 已有 checkpoint 不匹配。

## 15. 最重要的源文件索引

| 目的 | 文件 |
|---|---|
| 总训练入口 | `scripts/train_full_staged.py` |
| 数据准备/加载 | `scripts/prepare_paper_full_data.py` |
| 变长 memmap | `scripts/packed_trainval_cache.py` |
| B0 模型 | `kinetalk_b0/models/model.py` |
| 身份、teacher、总系统 | `kinetalk_b0/models/neutral_affect.py` |
| 音频 global/u_a | `kinetalk_b0/models/slow_state_affect.py` |
| Flow DiT | `kinetalk_b0/models/dit.py` |
| sealed 输入边界 | `scripts/prepare_sealed_inputs.py` |
| sealed 推理/评分 | `scripts/evaluate_sealed_models.py`、`scripts/evaluate_core_sealed_models.py` |
| ARKit 指标 | `scripts/evaluate_arkit_literature_metrics.py`、`scripts/arkit_benchmark_report.py` |
| 顶点指标 | `scripts/evaluate_vertex_lve.py` |
| 独立情感 probe | `scripts/fit_packed_emotion_probe.py` |
| 最终表 | `scripts/build_final_paper_tables.py` |
| 当前正式协议 | `final_experiment/docs/FINAL_PROTOCOL.md` |
| 远端运行记录 | `docs/remote_training_20260923.md` |
