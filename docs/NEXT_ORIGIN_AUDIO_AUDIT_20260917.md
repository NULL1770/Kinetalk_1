# Context12 后：静态原点与音频时序的独立审查

日期：2026-09-17。本文件为只读分析和下一轮建议；没有修改训练源码、启动训练或读取封存 test。依据 context12 三臂完整405开发集/3seed、固定128 fit诊断、此前实验报告以及当前源码。它不承诺效果或论文新颖性。

**建议先验证“片段均值外置、内部前缀续接保留”的最小组合，再对独立上脸音频编码器做同预算适配。暂不重建一套慢状态样条＋随机残差。** Context12 已经改善了接收端，当前更需要把“平均表情位置错”与“时间点错”分开。仅给现有结构添加另一条 audio→4D 慢状态路径会重复已完成且失败的实验。

## 1. 实际存在的缺口与不能重复的结论

当前 `scripts/train_history_upper.py:43` 的 `cache_audio` 先从冻结 Stage4 audio 得到逐帧四状态，再对整段有效帧求均值，按 `lift_slow_state` 加到独立 neutral anchor，缓存 `[B,9] static_upper`。随后 `normalized_target` 在第59行将上脸定义为 `(motion_upper - static_upper) / source_scales`。

`scripts/train_prefix_upper.py:140` 的 `load_context` 加载已有上脸/声学来源；`local` 是冻结的 `SlowStateAffect` 副本，加载 history12/no_history 的 local 权重，所有帧的 `prefix_local` 预先缓存且无梯度。Context12 的 `context_batch`（第68行）只更新上脸 flow；不是本轮重新训练了 audio-state 或 audio-local。全局情感、B0、identity、state均保留，当前不存在全局情感丢失，但也没有新增在线 motion→audio 蒸馏。

`PrefixUpperFlow` 当前硬编码 `use_state=False`。每层 local 注入和已知前缀仍有效；它不是没有音频或时序位置，而是该音频表示冻结、没有针对本次成功的接收机制重新学习。`TemporalUpperFlow._LayerConditionedDiT` 同时在 token/context 与每层投影中接收 local，已无需再增加一个名字不同但作用重复的加法条件。

已完成的相近尝试：

| 既有方案 | 结果与本轮启示 |
|---|---|
| `FULL_STAGED_SPLINE_PROTOCOL_20260917.md` / run12：4D raise/down/squint/wide 样条状态＋硬 P/Q 随机残差 | 已正式训练。慢状态预测 centered corr .0989、R² −2.814，真实state oracle眉corr .5105，但audio动态失败。不能再当作没实现的“缺少慢状态”。 |
| `TEMPORAL_REPAIR_PROTOCOL_20260917.md`：free9D direct/soft-state每层条件 | direct/soft无独立4D收益；说明仅换为soft注入也已尝试。 |
| 同协议 residual12：冻结zero-noise aligned全轨迹＋随机完整差量 | 已失败；不能简单重提“一个deterministic轨迹再加noise flow”。 |
| `CENTERED_TEMPORAL_PRIOR_PROTOCOL_20260917.md`：整段去均值动态、baseline mean，white/AR1 | white有连贯收益，AR1未优于white；后来state_white用同一audio4D窗口均值替换静态原点，corr仍弱。DC外置不是首次提出，新的实测价值在于保留context12已学成的续接。 |
| `DIRECT_AUDIO_DYNAMICS_RESULTS_20260917.md` / `OUTPUT_MOTION_DYNAMICS_RESULTS_20260917.md` | 完整rich audio+renderer、中心化L1、velocity/std都已训练；结果有平均化、幅度过冲、口部均值漂移。不能重复用“从前只训小头/缺眉loss”解释。 |
| `artifacts/temporal_repair_20260917/centered_prior/state_mean/RESULTS.md` 的句分割ridge | 1707fit/608留句，4/8/16帧低通目标眉R² .019/.016/.002，眼 .023/.024/.007；简单变慢并未提升可预测性。这不是非线性音频的信息上限，但足以否定未经验证就强制慢状态路径正确。 |

## 2. 现有数值足以先做一个无训练分解

相同mask、按有效帧定义各clip各channel均值时，raw MSE = centered MSE + 按观测数加权的均值误差。它不是未加权clip均值误差，必须沿原评分口径重算。

| context12部署三seed | raw MSE | centered MSE | 两者差：均值误差 | 占raw比例 |
|---|---:|---:|---:|---:|
| teacher眉 | .02254560 | .00192220 | .02062339 | 91.47% |
| teacher眼 | .00987195 | .00201807 | .00785389 | 79.56% |
| whole眉 | .02357851 | .00200670 | .02157181 | 91.49% |
| whole眼 | .00953638 | .00195684 | .00757954 | 79.48% |

旧state_white动态被严格去均值，因此其raw−centered可描述它实际使用的audio静态均值误差：眉 .01673727，眼 .00557248。若核验其 static_upper 与 context12 当前缓存、anchors、scale及mask确实逐值同源，保留teacher当前动态但重新使用这个均值，理论raw MSE应约 **眉 .01865948、眼 .00759054**，相对当前降低约17.24%/23.11%。这是由既有报告得到的可验证预测，不是已经重新合成的实测结果。

同时，teacher的时序相关 .0704/.0587、RMS比1.339/1.263和centered误差不会因常量平移而改善。把预测自身均值换成GT均值后raw误差恰等于其centered误差，是oracle数学界，不是可部署能力或独立创新。

需要保留四个输出：原始pre-DC；用旧static_upper的可部署DC；用训练得到的新audio均值的可部署DC（如果后续有此分支）；用GT均值的独立oracle。旧static与新head不能只挑对每条更好的那个。

## 3. 最小结构：显式片段均值＋保留前缀的随机形状

令 `o_old(a,id)` 是旧冻结 `static_upper`，`s` 是当前fit尺度，`u = context12_teacher(a, local, id, z)` 为完全按已有坐标逐块生成的9D残差。其未合成输出为：

`y_pre(t) = o_old(a,id) + s * u(t)`。

只在完整clip内部生成结束后，使用有效帧均值组成：

`y_final(t) = mu_audio(a,id) + [y_pre(t) - mean_valid(y_pre)]`。

其余43通道及无效帧精确保留原基座。第一步 `mu_audio = o_old`，不需要新训练；第二步再评估是否需要clip均值预测器。

这个设计作为**明确写入模型定义的离线合成**是合理的；它不是渲染器偷偷调整gain、平滑或挑GT，也不读取query目标。但是它依赖整个3.84秒窗口的生成结果，不能声称流式因果；若未来延长片段、滑窗或拼接长视频，均值定义会改变，必须另立协议。

它只消除DC分量，允许其余全部时间频率与左右差异存在，没有run12的分组样条P/Q限制。因此它不会强制“audio4D慢路径之外的残差必须与其正交”，也不会把随机运动的低频全部拿走。相对state_white的新组合是“已经验证有用的clean-prefix autoregressive形状＋显式audio均值”，不能单靠这句话宣称论文新颖。

### 必须保持的坐标与历史语义

1. 内部生成仍按旧 `o_old` 和旧scales定义；known使用原始GT严格过去残差或自己已生成的残差。不把GT整段均值用于normalize known，不给前缀引入未来目标统计。
2. 先完整生成内部形状，再对所有有效输出统一加同一个clip/channel平移；不要每块分别减均值，否则会重新造缝并改变低频形状。
3. 部署full时，全部generated块及其前序端点同时平移，邻帧位移、拼接缝、时序相关、RMS、centered ES/VS在舍入误差外保持不变。若这些指标大变，应先查实现。
4. oracle实际输入是**未平移的GT前缀**。不能拿DC后的首预测−原GT端点声称“实际接收变差/变好”。保存pre-DC，oracle actual严格用pre-DC对真正供给端点；如展示final坐标接收，GT端点也须加同δ并明确这只是坐标变换。原GT同端点诊断另列，不能相互替换。
5. 一个受监督的均值头可以读取GT整段均值作为训练标签，这不是推理泄漏；但是这个标签不能再作为generator条件或known坐标。训练与部署给generator的条件都只能来自audio/独立identity，GT均值oracle只在单独诊断使用。

## 4. 下一轮训练应隔离什么

### A. 先只做DC离线合成验证

使用现有context12完整三seed曲线，锁定均值来源与mask，验证raw=mean+centered、43通道、invalid、连贯指标不变，并测raw域内外与显示clamp前后活动。这里不需要重新训练。它回答“自由flow多出来的DC是否有害”，不能回答“音频timing是否学会”。

### B. 独立上脸音频路径适配：最有区分度的下一配对

两臂都从同一个context12/chunk_teacher最终权重起步，使用相同旧static坐标、相同最终DC合成、相同12epoch预算/RNG/clip级noise与flowtime、同严格GTpast训练规则：

- 继续训练上脸flow、冻结现有local；这是额外训练本身的匹配对照。
- 继续训练上脸flow，同时开放**独立副本**的local声学trunk＋local_head。全局audio/state/identity/B0/全脸基座不共享更新。

新local副本step0输出必须与context12缓存精确或规定数值容差一致，不能重新把local head置零后声称只改可训练性。不能继续从预缓存 `prefix_local` 取值；训练与推理都要用新local模块在线得到 `local`。如果只用`torch.no_grad`缓存新local或加载器末尾`.requires_grad_(False)`没恢复，新实验会虚假“开放audio”。

在当前source里，local其实是有global/state无用头的`SlowStateAffect`副本；训练可只注册input/blocks/local_head参数，或抽出等价的local-only模块。必须核对功能等价，不必让无用state/global计算得到梯度。明确记录哪个trunk被复制、哪些权重冻结、哪路local进入DiT。

先保留context12全程GTpast训练方式，而不是同时重新引入之前失败的scheduled-history；当前部署generated是否泛化仍独立实测。对比旧prefix12不能把全部改善归因teacher schedule，因为本次还改变了完整noise与clip flowtime处理。若以后检验generated-history训练，再单独配对。

主目标保持当前unknown-only FM。它仍监督pre-DC完整残差，所以可能把精力用在最后会被丢弃的常量分量；应报告FM中的均值/centered分解作为效率诊断，但第一轮不因此混加新的逐GT rollout L1/std。先取得音频开放的实际净收益再改目标，避免一次更换条件、目标与均值头无法归因。

### C. clip均值校准可以做，但与timing单独验收

若固定旧static均值仍限制raw，可训练独立 `mu_audio`：输入原生audio的clip汇总、冻结global和独立identity code；目标为训练clip有效upper均值。它只预测每clip一个9D向量，而非每帧9D动作。九个静态数值允许左右差异，信息量与逐帧96×9近目标条件完全不同；“9维”本身不决定学术性，透明说明其为动作统计回归即可。

先用旧static为零初始化残差输出，保证step0可退回；参数和归一化只fit训练集。新head若在同一audio上预测条件均值，不应宣称所有one-to-many静态表情都可唯一确定。可先用4group头和9D头在训练内留句检验，但不在405上网格挑最优；若仅需最小实验，固定一个9D clip头比同时启动两组新状态设计更清楚。

该head的监督是均值重建，不应以情感类别准确率代替；报训练内留句/fit/dev均值MSE与旧static、identity-only、audio-global-only对照。clip/sentence共享的记忆与new-identity校准问题必须分别说明。若没有超越旧static，保留旧static作为正式来源，不把失败head融到生成器里稀释结果。

## 5. “audio预测慢状态路径＋随机残差”是否仍有空间

可以作为以后结构，但当前缺的是**新证据和新的功能约束**，不是这个概念。run12已有4D路径，soft-state也试过。现在更合理的是：

- 将一个低率audio路径用作**soft引导或flow条件起始分布的均值**，输出仍由可修正的full9D前缀flow生成；不再把lift(state)锁成不可纠错输出，也不将Q限制到旧样条零空间。
- 这种条件起始分布必须正确改写FM插值/velocity目标和所有Euler初态；同时要把历史动作保持在同一绝对坐标。它改变了采样难度和warmstart分布，风险大于单纯开放现有local，因此不建议和DC/local适配在同一最小实验同时引入。
- 只有新audio路径在训练内留句（必要时留身份）对低率目标具有实际增益，并能在不显著损害raw/域合法性的情况下影响生成事件时间，才值得进入随机残差组合。增加低通尺度、PCA/VQ/latent名词本身不能证明可预测。
- 不要求所有瞬时眉事件都能由audio确定；随机支路负责分布内变化，audio分支的条件影响用static/reverse及配对分布分数衡量。不要以近GT的9D逐帧teacher条件驱动成功作为跨模态证据。

## 6. 必要诊断与明确停点

1. **分清origin、首块和块内轨迹。** 原始静态均值误差、pre-DC生成均值误差、final均值误差分别保存；GT均值替换是评分oracle，真实首8/16帧初始化是另一个oracle，不相互替代。GT首段启动后后续全部generated且不再GT reset，可定位“初始状态带偏”；不得用其部署图。
2. **保留整段GT reset与正逆prefix。** Context12在295共同覆盖clip中，oracle首块眉raw .022884、后块 .00255–.00324，首2帧 .00060–.00094，说明接收后仍有块内离开目标。新分支应报块首2/4、块尾4、块内速度、mean change和首块到末块，而非只报缝。
3. **local适配必须有同训练预算的冻结local臂。** 二者同source、同DC/均值头、同seed；保证step0功能一致。部署三seed与fit128诊断都保留，不能只取最好seed。
4. **静态干预、音频反序、GT正逆只改变声明的通路。** 新local的static/reverse是否连h0一起干预应固定；要隔离local时保持h0/global不变，另保留joint-audio干预。预测均值在timing干预时固定，避免把状态变化混成timing效果。
5. **跨模态是否有净收益。** 看raw/centered/相关/幅度、fair ES/VS及越界，正序优于反序不足以证明优于冻结local；不能只用更大动作或更低FM判成功。
6. **DC合成保持连续性但不保证系数合法。** 不clamp评分、不调gain。报告越界距离、每通道全片负值比例、clamp后有效活动以揭示“原始动了但rig仍不动”。固定9例/3视频、不挑好看的样例。
7. **全局身份口型。** 43通道逐位保护和冻结hash只说明继承；感知自然度、唇音同步、身份仍需独立验证。该轮新local副本不能回写全局audio、teacher或mouth renderer。

若DC组合改善raw而timing不变，这是预期的阶段性结果，不能当整体失败也不能当动态成功。若开放local仍不优于同预算冻结local、GT首段oracle也无法保留后段目标一致性，应优先审查训练片段覆盖/真实音视频tracker质量与表征可预测性，而非继续堆head和loss。任何扩展数据、句子/身份划分或未来长片段都另建协议，不混入当前2315/405对照。

## 7. 关键实现接口与风险清单

| 接口 | 可复用职责 | 新结构最易出错处 |
|---|---|---|
| `scripts/train_history_upper.py:43` `cache_audio` | 冻结global/intensity、旧static_upper | 新mean输出不能覆盖旧内部原点而不重定义known与target。 |
| `scripts/train_history_upper.py:59` `normalized_target` | 无query GT均值的原始残差坐标 | 不在这里塞GT整段center。 |
| `scripts/train_prefix_upper.py:35` `window_batch` | 严格过去8槽、mask、known | 更新音频local时仍按真实native位置，禁止跨gap压缩或看当前GT。 |
| `scripts/train_prefix_upper.py:99` `patch_loss` | unknown-only FM、past detach | audio生成的local保留梯度；detach只针对source motion。 |
| `scripts/train_context_mechanism.py:41` `decode_context` | GTpast训练对应的generated部署续接 | 先保存内部raw输出，整段结束才DC；不要逐块中心化。 |
| `scripts/train_context_mechanism.py:68` `context_batch` | 完整noise＋clip flowtime、一次optimizer更新 | 替换缓存local为在线副本，匹配随机流与训练预算。 |
| `kinetalk_b0/models/prefix_upper_flow.py:52,71,89` | joint clean known attention、每步clamp | known精确约束在pre-DC内部坐标；final合成不是同一外部观测坐标。 |
| `kinetalk_b0/models/mean_preserving_upper.py` | 现有mask安全的baseline均值＋centered动态 | 当前接口取baseline均值；传明确audio均值需新函数或构造透明baseline，不能偷偷复用错误基座均值。 |
| `scripts/evaluate_context_mechanism.py:27` `supplied_history_endpoints` | 正逆生成/GT真实端点 | DC后对oracle记录pre/final来源，不修改旧helper解释去掩盖坐标差异。 |
| `scripts/evaluate_context_mechanism.py:58` `evaluate` | oracle隔离、三seed、43保护 | 必须额外保存pre-DC/final；部署与oracle的actual端点分别按真实输入定义。 |

建议只新增源码/协议和隔离输出，不修改已绑定的旧实验文件。最小下一步的主张应是“在保持可复核前缀接收的前提下，分离可部署均值与随机形状，并检验针对性声学适配的收益”，而不是“我们终于补上一个别人已有的动态loss”。
