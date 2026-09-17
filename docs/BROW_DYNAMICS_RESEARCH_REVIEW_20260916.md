# 眉毛动态审查与后续研究方案

审查日期：2026-09-16。范围为当前 v10 中性身份／时序情感场实验及其继承链；不把旧 v5/v9 演示和最新 probe 的成绩混为一谈。本轮实际连接 SSH，核对代码、训练文档、manifest 和权重，查阅官方论文与公开实现；未训练、未改默认模型、未读取封存测试目标。

**判断：目前不足以支持“眉毛动态已经解决”的论文主张，也不能确认身份、口型、全局情感都已通过。最应先补的步骤是：在足够覆盖的数据上，让有足够可训练容量的生成器学会接收真实音频条件，并验证最终输出；此前大量大数据训练只更新了很小的接口。** 这与音频眉部信号弱同时存在，不能保证补上一步就一定成功。

## 1. 本轮实际核查

远端代码：`/root/autodl-tmp/kinetalk_b0_residual_train`。

当前结果：`/root/kinetalk_runs/teacher_schedule_v1/audio_flow_v1`；正式投影训练：`/root/kinetalk_runs/formal_predictable_v1`。

本地／远端以下五文件 SHA256 完全一致：`docs/ARCHITECTURE_NEUTRAL_AFFECT.md`、`docs/AUDIO_CONDITIONED_FLOW_RESULTS.md`、`kinetalk_b0/models/neutral_affect.py`、`kinetalk_b0/models/dit.py`、`scripts/train_audio_conditioned_flow_probe.py`。因此以下源码分析对应服务器实际文件。

| 资产 | 实查结果 | 解释 |
|---|---|---|
| native clips | 23,379 个 NPZ | 文件数，不是全部都进入当前实验 |
| native train / val metadata | 18,756 / 2,158 段；MEAD+CREMA-D | 本轮未读 native test 内容 |
| 当前 fit | 2,315 段、19 人、67 句；251,584 帧 | 只含 neutral/angry/happy/sad |
| 当前内部 dev | 405 段、3 人、56 句 | 56 句与 fit 共享；是留身份开发，不是身份和句子联合未见测试 |
| enrollment | 79 段 neutral、22 人、4 句 | 给目标身份提供参考；不等于编码器在22人重新训练 |
| 抽查 native 格式 | motion[T,52]、content[T,768]、audio[T,83]、times、mask、channel_mask；25Hz | 按原生时钟；不同任务不能混用旧 DTW 标签 |
| 计算资源 | RTX4090 空闲；autodl 盘约409MB、root盘约16GB空闲 | 后续产物须显式选空间充足目录，不能默认继续写满数据盘 |

运行元数据和来源已下载到 `artifacts/research_audit_20260916/`。传输返回 `TRANSFER_OK 6` 后 SSH 被远端关闭；所需只读审查已完成，无远端任务留下。当前未获得用户所指“目前效果”的具体视频／checkpoint，本文结论针对已确认的最新实验链，不能当作对未提供演示视频的观看评价。

## 2. 最关键的新证据：大数据训练不等于主体生成器训练

继承链是：

```text
run01: 4人 / 56段
  身份模块 200步；motion teacher + DiT 1000步
       ↓ renderer / identity 保持
run09: 224段 / 原4人，仅重新训练 audio encoder
       ↓
formal: 2720段 / 22人，主要训练512参数 local_projection
       ↓
19人2315段的日程 / rollout / centered / loss尺度试验
  仍主要只训练512参数接口
       ↓
latest audio_flow: 只训练最后cross-attention的out_proj.weight
  36,864参数，其余条件读取／生成参数冻结
```

本轮在服务器直接加载 `run01/teacher.pt` 和 `run09_emotion2vec_probe/audio.pt`，逐张量检查：

| 模块 | 张量相等数 | 张量元素数 |
|---|---:|---:|
| renderer | 84 / 84 | 4,428,148 |
| identity_encoder | 4 / 4 | 44,864 |
| identity_bias | 2 / 2 | 6,708 |
| motion_teacher | 22 / 22 | 129,012 |

216个非audio张量全部相同。原 pilot manifest 实际为56段、M003/M005/M007/M009四人。最新 compact checkpoint 存储一个 trainable 矩阵及冻结来源，provenance 将它链接回该链。

这意味着：最近的失败主要证明了“弱音频场＋小样本预训练后长期冻结的接收端＋小范围适配”仍不够，**没有完成对扩大数据后的完整条件生成器能力的充分验证**。

不能说以前从未训练过完整 renderer：run13/14、run32确实尝试过。尤其 run32 在1200段上各600步，同时训练 audio head 和 renderer，audio upper R² 本身从 .02354 降到 .01193。这种联合退化没有单独排除“冻结可靠学生、只充分训练生成器”的有效性。补这条基线属于修复研究设计，不是论文创新。

来源：[正式来源记录](../artifacts/research_audit_20260916/remote_formal_provenance.json)、[audio来源记录](../artifacts/research_audit_20260916/remote_audio_provenance.json)、[pilot来源记录](../artifacts/research_audit_20260916/remote_pilot_provenance.json)、[最新来源记录](../artifacts/research_audit_20260916/remote_latest_provenance.json)。

## 3. 眉毛动态失败发生在哪些环节

当前主链为：

```text
audio ─→ B0内容/口型 ─────────────────────────────┐
audio ─→ frozen global情感 ───────────────────────┤
audio多层特征+韵律 → 去均值 → ridge8 → neutral门控 → 共享local投影 → DiT残差
neutral多句参考 → S_id / 恒定P_id ────────────────┤
noise ──────────────────────────────────────────┘
最终52维动作 = B0 + P_id + DiT残差
```

这里的 renderer 指残差动作生成器，不是把人脸渲染成视频的图形引擎。`train.py` 仍是历史v9入口，不能据它复现当前方案。

### 3.1 音频眉部时序弱：已证实

训练内三折 OOF，原native基下眉 R²=.020650、相关约.14382、预测幅度约GT的13.79%。放大同一条预测曲线的最优R²上限约.020685，已接近现值。因此，**给当前眉毛曲线乘2或乘3基本不能改善正确性**。

基定标将真实motion的眉投影R²从 .36303 提到 .55654，audio却从 .02065 变为 .02004；更大的 pointwise／temporal 头提高训练拟合，跨句反而变差。这些结果反对“只要rank更大、眉loss更重、TCN更深就会好”，但不证明音频没有任何眉部信息。

### 3.2 生成后同口径时序误差更大，支持检查接收端

同 stride4 时钟诊断中，眉部 audio直接投影 R²约+.03133、GT投影+.31078；经生成器分别约−.08298、+.12834。数值来自该诊断自己的数据和口径，不能与最新八seed分数直接相减。直接投影与随机完整输出的差异可能混合条件均值误差、seed方差和U子空间外动作误差；这支持检查生成接收端，尚不能单独证明信息被丢弃。

最新八seed输出：眉 R² −.279269→−.283526；fair energy score改善仅约0.037%，邻帧variogram反而恶化约1.15%；正确local优于反转，但仍未胜关闭local。眼部 full-zero 有收益，不能替眉部报喜。

而且，眉原动作MSE从 .03525189 降到 .03410909，中心化时序MSE却从 .00104755 升到 .00105103。改善来自平均表情，不是眉毛随时间动得更对。

### 3.3 条件并未完全断开，时间信息也不是完全缺失

最新只读诊断中，非中性 activity gate均值=.981，audio local/content RMS=.477，cross gate非零；同状态oracle引起的眉速度响应约audio的3–4倍。DiT虽没有独立帧PE，h0已有TCN和正弦PE，local还逐帧加进tokens。

因此“开关没开”“没有任何位置编码”“把音频放大就行”都不是当前证据支持的解释。真正待查的是：弱条件中的时间信息是否足够，以及生成器能否按正确时间位置使用它。

### 3.4 数据尚未完成视觉真值级认证

本轮查看源视频抽帧与眉曲线；既有8段训练样本的音画时间检查、raw/native检查没有发现整批错位或大幅抹平：最大配帧误差13.392ms，native/raw眉动态能量比.939–1.043。

但这些样本来自已看过的小候选池；稀疏帧不能认证每个眉峰，raw/native一致也可能是两者继承同一个tracker错误。必须区分：原视频本来不动、tracker没提取到、生成模型没学到、导出/rig没显示出来。最后一种目前也缺当前版本的视频级验证，不能凭系数图排除。

### 3.5 neutral身份尚未真正证明动态校准作用

`P_id` 是恒定偏置，有：

`center(M − B0 − P_id) = center(M − B0)`。

因此目前中心化ridge收益不能证明neutral身份改善了动态。S_id仍能经DiT影响动态，但编码器只在四人训练，目标主要是neutral均值重建与身份对比；“neutral参考能校准不同人的眉部响应”尚未得到因果验证。

相关结果详见 [最新flow结果](AUDIO_CONDITIONED_FLOW_RESULTS.md)、[基定标诊断](SCALED_MOTION_BASIS_PROBE.md)、[时序头结果](TEMPORAL_AUDIO_REFINER_RESULTS.md)、[原视频监督复查](../artifacts/teacher_schedule_v1/temporal_refiner_v1/supervision/REVIEW.md)。

## 4. 身份、口型、全局情感是否已经没有问题

| 项目 | 已有证据 | 还不能推出什么 |
|---|---|---|
| 身份 | 独立neutral query检索4/4，中性系数偏置可学 | 不是几何脸型身份；不是大范围新身份泛化；动态风格贡献未证实。原run01的训练视图100%不能混作独立检索，独立结果来自后补identity_heldout.json |
| 口型 | 有冻结B0；若干适配组相对既有模型满足工程保护 | 最终52维残差仍能改变嘴；早期jaw相关低于B0；缺最新生成视频的独立lip-sync/感知认证 |
| 全局情感 | 一些小开发集audio分类较高；最新冻结teacher读出约95.7% | 训练teacher打分不等于独立生成情感识别；当前动态训练仅四类，不代表八类或跨库都好 |
| 眉动态 | GT条件能动，有弱音频信号；眼部局部获益 | 最新眉full未胜zero，跨人不稳，不能称核心贡献已成立 |

可以说“有可复用基础”，不能写“这三项已经没问题”。若目标是同一avatar上的个人表达习惯，应评价行为风格；只用同一mesh或人脸识别分数无法证明动作身份保留。

## 5. MEDTalk、DEITalk、SubtleTalk值得借鉴的真实部分

| 方法 | 与动态直接相关的机制 | 与当前任务的区别／核验边界 |
|---|---|---|
| [MEDTalk](https://arxiv.org/html/2507.06071v2) | 指定rig的逐帧强度伪标签；emotion2vec+转写文本预测强度；调制情感embedding；最终动作强度有监督 | 外部情感label/图/描述指导，174维MetaHuman。不是当前纯audio未知情感协议；不是固定52维ray乘一个标量。Eq.8–12及官方audio_semantic.py已核 |
| [DEITalk](https://doi.org/10.1145/3664647.3681359) | 摘要可核：speech+情感style标签、局部动态强度 | 很可能是用户所指DEITalker；正文403，官方仓库code/dataset占位，不能编造其loss/指标细节 |
| [SubtleTalk](https://arxiv.org/html/2608.06408v1) | 清洗后的视觉motion监督；多层WavLM+韵律直接进生成器；DMP+RFM；VA及窗口眉眼/头动态条件 | 训练59.76小时/2456身份；§4.5有audio-only用户研究，但Table3自动控制来源未充分披露；官方代码仍Coming soon |

共同启发是：**监督中的动作确实存在，时序音频在训练期就进入有学习能力的生成器，最终输出确实保留这些动态。** 不等于必须复制VA、Whisper、标量强度或增加多个区域输出头。

MEDTalk的EIE评价强度时序；其FRD和SubtleTalk的FDD主要评价运动幅度统计，反转时间并不会改变std。我们中心化ARKit52残差R²与这些指标不同单位、不同表示和数据，不能直接说差了多少，也不能仅凭负R²断言随机生成不可用。应同时评价时序依赖、动态分布和最终视频。

详细公式和条件边界见 [原文指标核对](DYNAMIC_PAPER_METRICS_COMPARISON.md)。

## 6. 后续优化：先补基线，再试一个创新机制

### 阶段A：把目标、输出和显示链分清（不做长训）

从训练池按metadata预先抽取有／无明显眉事件、不同人和情绪的短段，至少形成一个比既有8段更广的独立诊断集。事件类别由人工盲看或独立视觉方法确认，不能只按当前模型误差选“漂亮例子”。

用同一rig和相机渲染 `GT系数 / B0 / frozen基线 / audio / oracle / zero-local`，固定noise和时间轴；检查通道名、左右方向、scale、clamp、重采样。若GT系数都不显眉动态，先修提取或显示；若GT正常、oracle失败，先修接收端；若oracle正常、audio失败，重心转向条件信息。

为区分事件跟踪与tracker伪动，验证一小批连续片段的眉部距离／AU1/2/4代理，控制头姿。独立AU也是估计量，不命名为心理情感真值。这里是质量门槛，不包装为算法贡献。

### 阶段B：补齐“冻结学生、充分训练生成器”的匹配对照

先沿用2315 fit和405内部dev做短诊断，避免立刻消耗最终test。B0、现有global、U和audio head固定；参考身份先固定，以免多个模块同时变化。

| 对照 | 可训练部分 | local条件 | 要回答的问题 |
|---|---|---|---|
| A0 | 无新训练 | 现有audio／oracle | 当前冻结基线 |
| A1 | 同一套renderer及共享local投影 | 真实motion投影oracle | 扩大覆盖后，有容量的生成器能否传递可表示的眉事件？ |
| A2 | 与A1完全相同 | 固定audio预测 | 学生不漂移时，生成器能否学会部署条件下的运动分布？ |
| A3 | 与A1完全相同 | zero-local，保留content/global/identity | 收益是否只是更多renderer训练，而非新的local条件作用？ |

A1/A2/A3同初值、batch顺序、noise、flow time、单一既有flow目标，先各8epoch，固定第8轮；足够容量可以直接开放原renderer，先不再逐个矩阵试验。8epoch是诊断预算，不是收敛保证；不能把阴性结果解释为理论不可能。

训练audio条件需要接近部署时的误差分布，可用训练句内交叉拟合预测。每折的U/尺度和坐标必须有清晰契约；如果使用全fit的U，只能称student交叉拟合，不能声称整条路径严格OOF。也可固定一个fit-only子集拟合head、另一子集训练renderer，以更直观代价换取干净诊断。所有拟合不得用405目标，更不得碰封存测试。

每臂都评 full／zero／reverse／oracle、多seed原始输出、眉眼嘴分报、均值／时序分解；oracle臂只是诊断。A2还需优于匹配训练的A3；推理关local可能构成条件分布变化，不能代替A3。若A1明显改善、A2没有，进入阶段C查音频条件；若两者都不行，先查监督、表示与接收端训练。只有A2在最终动作上有可复现净收益，才进入扩大训练和新验证。run32同时破坏学生的混杂在这里被移除。

最后层Q-only可作为短小定位实验，但不应再成为长期“换一个矩阵再试”的主路线。也不建议继续放大曲线、重复rank/尺度扫描、无依据堆loss或延长512接口训练。

### 阶段C：针对“条件均值瓶颈”检验，而非再换特征名字

当前专门的local动态支路把强音频特征压成ridge对动作的8维线性预测均值；flow还接收含时序的h0以及global、identity，不能说全部声学条件都经过ridge。local预测均值弱，不表示音频里与事件概率、节奏、变化分布相关的信息全部不存在；该压缩接口也不保证保留条件生成需要的信息。

A1证实接收端能传递oracle动态后，即可比较“原ridge动作均值场”与“同预算、同一共享接口内保留声学时间事件信息的场”；不要求A2先成功，否则可能把真正的条件瓶颈挡在验证之外。不新增眉／眼／嘴三个动作decoder，不采用文本主分支，不强迫一条全脸标量解释所有区域。保持单一残差flow。

先测小时间平移／局部条件脉冲下的响应时刻，再看同noise full-zero及reverse。注意力熵、非零梯度只能证明运行，不能证明眉部时序正确。

## 7. 更有研究价值的创新候选及其反证

建议聚焦：**neutral参考能否校准个人对同一声学动态的运动响应，并结合训练内可预测性和监督可信度，让一个共享flow同时生成可控节奏与合理自主细节。** 这目前是待验证假说，不是已有成果或已证明的新颖贡献。

具体机制应超过现有恒定P_id：从多句neutral参考提取稳定执行统计，通过共享、有界、向群体均值收缩的映射校准动态场响应；音频条件提供时间变化，noise提供条件未唯一决定的细节。对参考中几乎不动的眉通道，不能直接除以其很小的std，也不能假定neutral能揭示所有情绪下的眉响应。

训练内跨句／跨人可预测性和独立tracker一致性可作为校准依据，但两者要分开：预测残差大可能是行为随机，也可能是标签噪声或模型弱，**单GT不能把三者自动辨识开**。未证明之前，不把高残差简单变成更大的随机幅度。

最少需区分这些组件的贡献：

1. 无neutral／仅静态P_id／neutral动态响应校准；固定相同audio/global/noise。
2. 同人不同参考组应稳定；跨人参考应改变个人执行方式，而非破坏情绪与发音。
3. 原ridge／保留声学条件／加入候选校准；同参数量或明确报告增量。
4. 确定性与随机生成；既测正确音频干预收益，也测公平energy score、variogram、多样性与自然度。
5. 无可靠性处理／只处理标签质量／只处理可预测性／两者结合，验证机制究竟带来什么。

如果neutral校准仅改变均值、全局情感或tracker偏差，而没有改善新身份动态与口型保持，就应拒绝“身份驱动动态校准”的主张。若效果主要来自重新训练renderer，应诚实将其记为基线修复，而非新模块贡献。

已查到的近邻工作必须承认：

- [MeshTalk，ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/html/Richard_MeshTalk_3D_Face_Animation_From_Speech_Using_Cross-Modality_Disentanglement_ICCV_2021_paper.html) 已做音频相关／无关运动分离，涉及眉与眨眼。
- [DEEPTalk，AAAI2025](https://arxiv.org/html/2408.06010) 已做概率音频／动作情感嵌入及生成情感一致性。
- [KSDiff](https://arxiv.org/html/2509.20128) 已做韵律、多尺度、关键帧预测及频谱动态损失。
- [ARTalk](https://arxiv.org/html/2502.20323) 已有多尺度运动代码与参考motion风格条件。

所以“确定性＋随机”“加韵律”“加事件loss”“加概率头”均不足单独构成创新。上述候选只有在明确机制、近邻对照和独立泛化收益齐备后，才可能形成论文贡献；仍需继续系统查新。

## 8. 投稿判断与完成条件

**现阶段不建议以“已解决细粒度眉部情感动态”为主贡献直接投稿。** CCF-C没有统一数值门槛；研究问题、与已有方法的差异、公平比较和结果可靠性都需要成立。也不能因为目标档次是C就省略核心证据。

建议论文至少形成如下证据闭环：

- 独立锁定的新句／新身份测试，记录B0、global和其他预训练资产的历史暴露；反复使用的280/439/405都标开发集，不重命名为新测试。至少分别报告新句和新身份，并检查联合泛化。
- 明确是四类还是八类任务；跨库验证必须核验表示、强度等级与标注可比性，不能把CREMA-D直接混入便称泛化。
- 在统一输入条件、表示、FPS和划分上复跑可用强基线；label-guided、参考motion-guided与纯audio分栏。不能拼论文原表值和自身ARKit52指标排名。
- 生成级口型、身份执行和情感验证：适合当前3D表示的口部轨迹／事件评价，加同rig渲染视频盲评；如使用SyncNet等2D评估器，注明合成域适用性。不能只靠训练teacher和系数总MSE。
- 眉与眼分别报告；可预测时序、运动分布、视频自然度三类证据并列。多seed不能当独立人物；按句／人聚类统计，并报告逐人、逐情绪失败。
- 清晰的机制消融、多训练seed复验、预先固定展示样本和可复现训练脚本。禁止best-of-K挑最好噪声。

目标应是一个可以清楚回答的研究问题和一条可靠证据链，而不是先断言三个模块已好、把所有剩余误差压到眉毛头上。当前最值得先做的是阶段A/B；如果它们成功，再让阶段C和neutral动态校准承担创新验证。

## 附：本轮交付

- 本综合报告；官方文献详细审查 `tmp/research_20260916_literature.md`。
- 源码审查 `tmp/research_20260916_architecture.md`；独立数值审查 `tmp/research_20260916_experiments.md`。
- 远端provenance与metadata副本 `artifacts/research_audit_20260916/`。
- 本轮没有改训练代码、没有新增训练、没有覆盖或删除权重。以上新方案均为建议，不能记为已验证结果。
