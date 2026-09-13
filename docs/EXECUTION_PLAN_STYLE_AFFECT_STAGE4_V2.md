# Style/Affect Identity and Stage4-v2 Execution Plan

日期：2026-09-13；状态：v2 架构修改完成，本地测试通过，准备远端重训。

## 目标

修复三个现象：Style swap 的 speaker 动态 identity 不明显、Style 携带 emotion、Stage4 劣于 B0 且产生负 BS。禁止把 Style 硬切成“只管上半脸”；真实 execution style 可以同时影响口型幅度、眉眼联动、左右不对称和速度。

需要区分动态 identity/execution style（BS 动作统计，应由 reference Style 控制）与 neutral BS 中的静态 speaker 偏置（若数据保留，应由 Style/neutral identity 路径控制；若预处理清零，则需要单独 geometry/3DMM identity）。

## 已确认的证据

- Test Style swap 与 reconstruction 平均差异约 `0.0505`，上半脸差异约 `0.0635`，说明 Style 不是失效，但变化更像局部扰动。
- Style speaker probe balanced accuracy 约 `0.803`；Style emotion probe balanced accuracy 约 `0.440`，高于随机 `0.125`。
- Stage4 all-52 MAE Val/Test 约 `0.1060/0.0998`，B0 为 `0.0879/0.0834`；mouth MAE 约 `0.1057/0.0963`，B0 为 `0.0702/0.0590`。
- 独立真实 BS emotion probe 对 Stage4 生成仅 Val `0.000`、Test `0.029`；内部 factor cosine 约 `0.99` 不能替代真实动作验证。
- 当前 DiT 直接预测无界 residual：`M_hat=B0+DeltaM`，因此出现大量 `<0`。

## 核心待验证假设

Stage2 emotion encoder 输入完整 residual BS，可能学到 `emotion + speaker execution + upper-face identity`；Stage3 又从 audio 蒸馏这个被污染的 teacher，导致 audio affect 分支携带 query speaker identity，压过 reference Style。该结论必须由 probe 证明。

## 诊断阶段 D1-D4（不覆盖现有 checkpoint）

### D1 数据 identity 审计

按 speaker 的 neutral/reference BS 统计 eye/brow/cheek/nose 的均值、std、P95-P05、左右差值、eye-brow correlation，并比较同 speaker 跨句方差与跨 speaker 方差。输出 `identity_neutral_statistics.json/csv`。若跨 speaker 方差明显更大，说明 BS 中存在可迁移 identity signal。

### D2 latent probes

按 speaker-disjoint split，对 Stage1 `h0`、Stage2 emotion global/local、Stage2 Style、Stage3 audio global/local 分别训练独立 speaker/emotion probe。记录 balanced accuracy、shuffle control、norm 和 checkpoint provenance。

### D3 renderer sensitivity

固定 query content/audio/noise，只替换 Style、audio affect、content/h0，计算 `Delta_style`、`Delta_affect`、`Delta_h0`，按 upper/lower/mouth/eye-brow coordination 和每通道报告。若 `Delta_affect` 或 `Delta_h0` 显著大于 `Delta_style`，则 Style 被其他路径压制。

### D4 独立真实动作评估

保留 independent real-BS emotion probe；新增不共享 Stage2 Style encoder 的 real-motion speaker/style evaluator，用于 cross-style 输出评估，避免 encoder-generator 自洽闭环。

## 论文原则映射

DiffPoseTalk 的公开摘要支持连续 reference style embedding、diffusion 条件和 style guidance；本项目对应增加 style-strength/guidance ablation，并把 Style swap 作为一等评估。DeSTalk/MeDTalk 类方法可迁移的共同原则是因素交换、跨条件重建、内容时间轴保持和独立识别器评估；不采用固定通道归属。

统一因素定义：

```text
content c  -> phonetic timing and B0 mouth prior
affect a   -> target-audio emotion/prosody
style s    -> reference speaker execution dynamics
identity g -> optional neutral geometry/BS bias if present
```

目标是可交换/条件独立，而不是通道正交：swap(s) 改变 execution statistics，swap(a) 改变 affect，swap(c) 改变 timing。

## 架构 v2（仅在 D2 确认后实施）

### 1. 净化 affect teacher

对可靠 neutral partner 构造：

```text
r_emotion = M_emotion - B0_emotion
r_neutral = M_neutral - B0_neutral
delta_affect = r_emotion - aligned(r_neutral)
```

用 `E_affect(delta_affect)` 学 global/local affect；用 `E_style(r_neutral)` 和 emotional/neutral invariant statistics 学 Style。无可靠 neutral partner 的样本不参与 affect residual 逐帧 teacher。

### 2. Emotion-path speaker adversary

对 Stage2 affect global/local 加小权重 speaker GRL（初值 `0.02–0.05`），同时保留 emotion classification。Stage3 只蒸馏净化后的 affect teacher；只有 D2 证明 audio latent 仍泄露 speaker 时，才对 Stage3 加 adversary。

### 3. 连续全脸 Style

Style 继续编码完整 residual，不做硬通道切分；增加 neutral/emotional invariant statistics、same-speaker cross-emotion positive、different-speaker negative、独立 style-statistics recovery 和 style-strength intervention。Style 可影响嘴部与眉眼，但 B0/content 保持 mouth timing。

### 4. B0-preserving bounded Stage4

优先实现 logit-space bounded residual：

```text
z0   = logit(clamp(B0, eps, 1-eps))
zhat = z0 + alpha(c,a,s) * DeltaZ
Mhat = sigmoid(zhat)
```

`alpha` 小值初始化并有上限，输出天然在 `[0,1]`。Self branch 增加 raw BS Huber/L1、velocity/acceleration、mouth event/timing 和 gate regularization；cross branch 只做 affect preservation、donor style statistics recovery、独立 evaluator 和 mouth timing，不做错误的 donor-style/query-motion 逐帧 strict target。初始 self/cross 比例 `0.75–0.85 / 0.15–0.25`。

如果当前 renderer 改动破坏 checkpoint 兼容性，则创建新的 Stage4-v2 architecture version，旧 checkpoint 只读保留。

## 实施顺序

1. 新增 D1-D3 诊断脚本和统一 JSON schema。
2. 运行诊断，把结果写入 `findings.md`；先判断泄露发生在 emotion、audio、h0 还是 Style 使用不足。
3. 若 D2 支持，实施 affect-residual teacher 与 emotion speaker GRL。
4. 实施 bounded Stage4-v2、raw reconstruction、range checks、style-statistics recovery、style-strength evaluation。
5. 本地 compile/pytest/synthetic contract；再做小规模 overfit/smoke。
6. 远端备份旧 checkpoint，上传并按 Stage2→Stage3→Stage4 重训。
7. 运行 held-out fidelity、independent emotion、style leakage、speaker/style swap 和 range audit。

## 已实施的 v2 代码变更

- `Stage2Model.encode_factors` 新增 `emotion_residual`，训练时使用
  `query_residual - neutral_residual` 作为 affect 输入；Style 仍编码原始
  reference residual，保持部署输入契约。
- `Stage4Model.bounded_motion` 使用 data-dependent bounded residual gate，
  输出严格落在 `[0,1]`，并加入 self reconstruction loss，防止 flow-only
  residual 偏移导致 Stage4 劣于 B0。
- cross-style 分支同样经过 bounded renderer；factor consistency 在有效
  输出上计算，避免把未约束 raw residual 当作最终 BS。
- 配置默认 `style_grl_lambda=0.15`、`stage4_initial_residual_gate=0.10`、
  `stage4_max_residual_gate=0.6`、`stage4_self_reconstruction=1.0`。

## 本地验证记录

- `python -m compileall kinetalk_b0 train.py`：通过。
- `pytest -q`：27 passed。
- 旧 v1 checkpoint 不覆盖；Stage2/3 继续使用 architecture version 8，
  Stage4 新 checkpoint 使用 version 9。

## 验收标准

1. Stage4 self MAE ≤ B0 的 `1.05x`；mouth temporal/velocity correlation ≥ B0 的 `0.95x`。
2. 输出系数越界率为 0。
3. Independent real-BS emotion 明显高于随机，不能退化到 `0.000/0.029`。
4. Style emotion probe 下降，speaker probe 保持可用。
5. 多 donor 的 upper-face execution statistics 可区分，Style swap 不破坏 mouth timing。
6. 若声称大小眼/静态左右眼差异，必须先证明 neutral BS 存在跨 speaker 差异；否则需单独 geometry identity 分支。

## 禁止回归

- 不把 Style 硬编码为上半脸独占通道。
- 不整体放大 residual、不用更多 DiT steps 掩盖 timing failure。
- 不把内部 factor cosine 当成真实 emotion/style 证明。
- 不用硬 clamp 作为唯一范围修复。
- 不覆盖可回退的 v1 checkpoint。
