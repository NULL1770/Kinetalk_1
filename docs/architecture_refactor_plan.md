# 当前实验架构重构记录

日期：2026-09-23  
范围：当前 `scripts/train_full_staged.py` 主实验路径、其实际依赖，以及与论文架构图一致的实现契约。

**实施状态（2026-09-23）**：默认训练路径已完成重构并固定为
`articulation → identity → teacher → audio`。Stage5/dynamics 已移出默认训练和推理；旧
upper-flow helpers 仅作为 legacy 兼容保留。PPT 架构图已按同一契约更新。

## 先固定两个论文事实

前一版记录把 MEDTalk 误写成 EmoTalk，已废弃。

- **MEDTalk**（[arXiv:2507.06071](https://arxiv.org/abs/2507.06071)，[官方代码](https://github.com/SJTU-Lucy/MEDTalk)）先用运动序列做 content/emotion 解耦，并冻结 motion encoder/decoder。音频分支不是预测完整的逐帧表情向量，而是用 emotion2vec 加 ASR 文本预测一个逐帧强度标量，再调制一个全局情感方向。强度伪标签是选定控制器的活动量。它因此绕开了“每个音频帧对应唯一眉眼向量”的问题，但模型本身是确定性的，没有 flow/diffusion/random latent，不能说真正覆盖了 one-to-many。
- **SubtleTalk**（[arXiv:2608.06408](https://arxiv.org/abs/2608.06408)，[项目页](https://molly-ding.github.io/SubtleTalk/)）用 DMP 稳定口型/内容先验，再用 residual flow matching 生成弱相关上脸和头部的随机偏差；prosody、VA、区域强度等条件负责时序、幅度和情感趋势。它仍有速度、平滑度和区域时间标准差等序列辅助损失，不能概括为“完全没有逐帧监督”。论文明确给出了音频到 VA 的 VADP；五个区域强度在音频-only 路径中的默认生成细节没有充分说明，不能直接照搬成已解决的接口。

对本项目的启示是：借 MEDTalk 的“全局方向 + 可选低维活动包络”，借 SubtleTalk 的“随机残差流”。前者不是逐帧眉眼蒸馏，后者才承担一对多。包络先作为明确的消融项，不预先把它写成主路径必需输入。

## 当前实现中必须处理的问题

1. Stage4 的 motion teacher 输出 `global_m`、rank-8 controls→`local_m`，而 audio student 由同一 TCN trunk 输出 `global_a`、native-rate `local_a` 和 `state_a`。主线只蒸馏 `global_a`，renderer 却同时更新，所以没有保证 `local_a` 与 `local_m` 同义；decoder 确实可能把音频 local 当作任意辅助编码。问题是接口和训练目标，不是共享 trunk 本身。
2. `state_a` 有损失但不进入 full-face renderer，只服务于 Stage5；它不应继续作为主情感接口。
3. Stage5 复制完整 audio encoder，却只拿 local/state 接管 upper9；其余 43 个通道复制 Stage4，不能解决完整面部动态，也不能修复嘴部。local、state 和 unrestricted residual 还可以重复解释慢变化，动态归因不清楚。
4. Stage4 的随机 rollout 对单个 GT 终点加 Huber（当前训练代码约在 `train_full_staged.py:615`），可能压缩采样多样性。FM 的 velocity MSE 本身并不等于均值回归，需单独消融这项 rollout loss。
5. 当前 motion teacher 的 global 是运动 hidden 的池化投影，不自动等于纯情感；它可能携带幅度、说话内容或身份信息。不能把任意 64 维码当作情感真值。
6. `system.audio_encoder` 没有被当前主 runner 使用；Stage1 的 articulation loss 也硬编码了嘴部通道，未真正使用配置的 `art_indices`。这些是接口清理项。
7. identity 分支输出的是中性系数偏移和 reference style code，不应在论文中称为已证明的几何身份。缺失通道统计需要按 channel mask 处理；应做跨句子/留一参考的稳定性和情感泄漏检查。

## 已采用的主架构

删除 Stage5 的默认路径，保留旧实现只作历史 checkpoint/ablation。保留 Stage1 articulation、Stage2 neutral-reference identity 和一个完整 residual generator：

$$\hat M_{1:T}=b^0_{1:T}+b_{id}+F_\theta(\epsilon;h^0_{1:T},u^a_{1:T},g^a,z_{id}).$$

- `h0`/`b0`：Stage1 的内容和口型基线。
- `z_id`/`b_id`：静态 reference style 与中性系数偏移，不携带动态。
- `u_a[T,D]`：情感分支中唯一一条时序声学条件，供 flow 生成口型之外的时序运动；称 acoustic condition，不称逐帧情感状态，也不要求它等于 teacher latent。`h0` 仍是 Stage1 的内容/口型时序条件，两者职责要在消融中分开。
- `g_a[E]`：clip-level affect direction/semantic code。
- `F_θ`：对完整 residual support（按现有 mouth protection 配置决定是否保护嘴部）做 conditional flow matching。随机初始噪声提供分布建模能力，但不能单凭随机性宣称生成了合理多解，需用多样性和活动统计验证。

`u_a` 是唯一的时序声学条件，`g_a` 是由同一 trunk 池化得到的全局语义条件；不再维护 `global_a [B,64] + local_a [B,T,64] + state_a [B,T,4]` 三套情感接口。若后续加入 `s_a`，它只能是从 `u_a` 派生的一个可选统计 head，而不是第三条独立动态路径。

## BS 运动教师还在，但职责改变

BS 教师仍然保留在训练期，不进入推理：

1. 用 motion-only encoder 产生运动侧的全局情感表征和类别/原型监督；它可以提供语义参照，但不再用带有 GT local 的 renderer 作为 audio 的 privileged decoder。
2. 可选地从去除 `b0 + b_id` 的运动残差计算归一化上脸活动统计 `s_m`（眉/眼窗口 RMS、速度能量或标准差）。这些统计只用于单独消融的辅助损失，不是 upper9 输出分支，也不代表九个通道的真实划分。
3. audio 只学习 `g_a` 的类别/弱语义对齐；默认不使用 \(\lVert L^a_t-L^m_t\rVert\)，也不做 native-rate 眉眼轨迹蒸馏。若实验启用包络 head，使用部署时同样可获得的 `s_a`，并与 `s_m` 做低权重窗口级 robust 对齐；若目标不稳定就移除该 head。

这样教师指导的是“情感方向和活动量”，而不是规定音频何时抬哪一根眉毛；何时、哪些通道以及细节幅度由条件 flow 从分布中采样。教师不是为了显得重要而保留，而是提供运动侧定义的语义锚和先验初始化。

## 已采用的训练阶段

1. **Articulation**：训练 `b0/h0`，继续用有明确通道定义的口型/速度损失；清理硬编码与配置不一致。
2. **Identity**：只训练 reference encoder 与 neutral offset；修复 partial-channel pooling，做跨参考稳定性检查。
3. **Motion teacher**：motion-only 训练 global affect encoder/classifier（可选统计 head），冻结它；不预训练一个接收 GT `local_m` 的 renderer 供 audio 继承，也不把 `local_m` 宣称为逐帧情感真值。
4. **Audio condition + generator**：同一个 audio trunk 输出 `u_a` 和池化 `g_a`。Stage 3 使用 motion teacher 的全局语义与 audio 的 `u_a` 共同训练 flow；Stage 4 冻结 teacher，用停止梯度的全局向量 MSE 对齐 `g_a` 与 `g_m`，同时继续用 `u_a` 训练 renderer/DiT。包络 head、逐帧 teacher local、单 GT 随机 rollout Huber 和 Stage5 upper 接管均不在默认路径中。`u_a` 通过 DiT 的 temporal context 进入 renderer。
5. **Inference**：只给 audio、content、identity 和随机噪声；不需要 teacher motion、GT envelope、`state_a` 或 upper9 代理分支。

## 必须做的判别实验

- 固定 audio/identity/content，改变 flow noise，报告眉眼轨迹多样性、速度频谱和区域活动统计；不能只报告单样本重建。
- 固定 noise，分别将 `u_a` 静态化、反转、打乱，检查动态和包络统计是否改变；否则说明 decoder 没使用时序声学条件。
- 同时静态化 `h0` 与 `u_a`，固定 global/reference/noise，测量全部时序声学依赖；只干预 `u_a` 不能把 h0 的贡献误算给情感分支。
- 对比 `global + flow`、`global + envelope + flow`、以及无 teacher semantic loss；判断教师是否带来真实收益。
- 去掉 Stage4 rollout Huber 后检查 diversity 与质量，避免把单个 GT 拉成条件中心。
- 增加 native-frame/relative-time 干预或对照；现有 flow time 不是视频帧位置。
- 分别报告口型同步、上脸活动量/速度、随机多样性和 identity 稳定性；不要用 teacher-oracle 条件的结果代替部署结果。

## 不再保留的主路径变量/模块

- `Stage5 dynamics`、`UpperInnovationFlow/AudioResidualFlow` 的 upper9 接管、`state_a [B,T,4]`：移出默认训练和推理，保留为显式 ablation。
- 自由 `local_a` 与 teacher `local_m` 的“同语义”叙述：删除；若代码保留 native-rate feature，只命名为 `u_a` acoustic condition。
- 未使用的 `NeutralAffectSystem.audio_encoder`、无损失的 `target_intensity/regional_window_activity`、诊断字段：逐一确认调用方后删除或隔离。

## 实现核对

- `scripts/train_full_staged.py` 的 `STAGES` 为四阶段，schema 为
  `full_staged_teacher_audio_temporal_v2`。
- `NeutralAffectSystem.flow/generate` 优先读取 `affect['u_a']`，并通过
  `temporal_condition` 传入 `ResidualDiT`；`local_emotion` 只保留为 checkpoint/API 兼容别名。
- `SlowStateAffect.forward(..., include_legacy_state=False)` 可只返回部署所需的
  `global`、`u_a`、情感分类与强度输出；旧 `local/state` 字段仅在 legacy 调用中返回。
- identity encoder 按有效帧和 `reference_channel_mask` 做逐通道统计；缺失通道不会以零值污染
  `z_id` 或 `b_id`。

这份记录描述的默认四阶段重构已实施，并于 2026-09-23 同步至 SSH。训练结果仍需以实际 checkpoint 和评估报告为准；MEDTalk 或 SubtleTalk 的论文结果不属于本仓库实验结果。
