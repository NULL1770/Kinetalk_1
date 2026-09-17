# 动态监督与共同训练：执行记录

本轮继续优化audio→动态，保留B0、neutral身份和原音频global路径。**以下是开发实验，不是已达成效果的架构承诺。** 原28条和后加26条都已经查看过，不能再叫未触碰测试集。

## 1. 目标是否适合跨句预测

data224内部按句子固定seed45划分190训练、34保留片段。学生从相同新初始化开始，四组各1600步；原83D输入先还原缓存归一化，再只用新训练片段fit输入统计。PCA、目标尺度只fit训练句子。teacher/B0可能见过内部保留句子；此处测试新音频学生泛化。

| 目标 | train R² | heldout R² | heldout时序相关 |
|---|---:|---:|---:|
| 原8维teacher controls | .4540 | −.1797 | .1264 |
| 平滑teacher controls | .3777 | −.1171 | .1309 |
| 8维实际残差PCA | .4120 | −.2148 | .0908 |
| 平滑残差PCA | .3676 | −.1716 | .0826 |

R²以各clip去均值后的零动态为基准。恢复原尺度能量权重仍四组全负，排除neutral也仍全负；不能仅靠更换坐标、平滑、重新加权宣称成功。PCA解释82.98%残差训练方差，但其中76.25%在口部、19.04%眼部、4.70%眉部。普通PCA不是情感监督的充分替代。

保留集中neutral为12/34，训练中20/190，类别比例不同；分组分析已单报。原controls在neutral上的预测RMS/目标=.932，pool corr=.081，说明包括“幅度有了但时序错误”，不能只解释为不动。详细结果在`target_predictability/axis_group_analysis.json`。

## 2. 保持global与零动态行为的局部适配

从run07同checkpoint开始，seed46，224条各1600步，同批次、flow噪声、teacher局部条件混用概率。使用原flow+.1位移差分，global CE与global蒸馏因为冻结成为常量，不加入优化。

- run10：只训练audio.control_head。
- run11：同时训练已有renderer.local_emotion.weight，bias固定，teacher/local_projection固定。
- run12：在run11基础上启用零初始化32维三层时序control_refiner，仍只输出原rank8场，额外15848参数。

三组完整global/logits/intensity均bitwise保持；B0、identity和所有非白名单参数保持，local=zeros的生成结果bitwise保持。开启动态仍可能修改嘴部，不能把零动态保持等同于口型已保护。

| 模型 | full MSE | mean MSE | 眼corr | 眉corr | jaw corr |
|---|---:|---:|---:|---:|---:|
| run07原声学 | .010491 | .010686 | .06896 | .05690 | .40191 |
| run10 head-flow | .010787 | .010686 | .06569 | .02684 | .42580 |
| run11 joint接口 | .010907 | .010686 | .05128 | .03006 | .41150 |
| run12时序refiner | .010987 | .010686 | .05787 | .02018 | .40603 |

3噪声42/123/2026、纯噪声Euler12步。三组动态幅度增大，但full均差于自身mean；未通过。run10/11的audio侧只有776可训练参数；run12虽增加local时序容量，仍固定原renderer主体，因此这些结果不等于完整共同训练已被否定。

## 3. 进一步开放renderer与teacher动态头

run13/14已启动：同run07+zero-refiner、seed47、各1600步。两臂audio control/refiner及整个renderer训练，run14额外训练teacher.control_head；teacher共享特征和global均冻结。使用同一个flow+.1位移+.1control对齐；对齐中的teacher target detach，teacher只从teacher条件flow获得更新，避免两边仅相互收缩。

前半训练teacher局部条件50%，后半10%；global/intensity始终用冻结audio输出。此时不再要求zero-local生成保持，需直接评估最终mouth和表情。只要未达到验收，默认模型不替换。

## 4. 单标量 upper-face energy 联合训练（run15）

为检验“明确动态目标 + renderer 梯度”是否能修复方向，新增可选 `upper_l1` 探针：从 neutral-anchored motion residual 计算上脸强度，在线性映射后只对现有 rank 控制施加 0.05 对齐；没有新增区域通道，B0、identity、global 和推理接口保持不变。run15 使用 emotion2vec 768D 音频、coupled 模式、800 steps。

结果：heldout audio→teacher control MSE 从原 run09 的 0.04072 降至 0.03276，但 zero-local 基线为 0.03101；renderer 的 audio full 相比 zero 的总 MSE 改变量为 **−0.000128**，upper-face 为 **−0.000147**，brows 为 **+0.000091**。teacher full 仍明显优于 zero（总 MSE 改善 0.002244）。

该实验尚未让 audio 动态在 heldout 上超过静态基线。run15 同时改变了 teacher，0.04072→0.03276 的目标坐标不固定，不能单独据此宣称控制预测改善。energy loss 未按训练 RMS 归一，训练日志中约 0.0002–0.0007，再乘 0.05 后很小；此结果也不足以否定有效强度监督。保留为实验开关，不替换原 checkpoint。

## 5. 冻结 emotion2vec + 多尺度 TCN probe（run16）

在 emotion2vec 768D 帧特征上加入轻量 dilation=1/2/4 的时序卷积头，只训练该 temporal head，目标仍为 upper-face L1。训练集 R²=0.0855、时序相关=0.191；heldout R²=−0.0237、时序相关=−0.0827。后续审查发现该版 GroupNorm 包含 padding 时轴，且 readout 有饱和风险，因此不能用此结果排除 TCN 架构；暂不接入主 renderer。现已改成逐帧 LayerNorm，并补充 padding 不变性测试；旧权重对应旧归一化语义，不能用新代码静默重评旧结果。

## 6. energy + velocity dense probe（run17）

将同一 upper-face energy 增加一阶差分轴，形成 rank=2 的稠密目标。acoustic 1200 steps 后 train R²=0.3205、时序相关=0.4639；heldout native R²=−0.0652，仍低于 zero baseline。它在训练集明显增强动态拟合，但跨句泛化失败，因此暂不进入 renderer 主路径。

## 7. 内容-音频 FiLM 门控 probe（run18/19/20）

用低参数 FiLM 将 content 768D 压到 16D 后调制 emotion2vec 动态表示。结果：film heldout R²=−0.1143、temporal corr=0.1378；同设置 audio-only heldout R²=−0.1035、temporal corr=−0.0639；content-only heldout R²=−0.2684、temporal corr=0.0319。FiLM 提高了时序相关，但没有降低误差，仍未超过 zero baseline，因此暂不接入 renderer。内容是否提供有效 timing cue 仍是假设；这些结果不能证明数据量或幅度标定是唯一原因。后续实际检查发现 audio-only 读出存在整段饱和，需要先修正再比较。

## 8. 幅度归一与事件目标（run21/22）

针对 FiLM 的 timing correlation 与幅度 R² 不一致，新增 train-only clip RMS 标定和事件目标。emotion2vec + FiLM 的 clip_rms heldout R²=−0.4828、时序相关=0.0411；校准后 R²=−0.3743。事件目标 heldout R²=−0.5791、时序相关=−0.0256，事件 macro-F1=0.2898，略高于 zero 的 0.272，但 onset/offset F1 仍只有 0.18/0.21，且 reverse 条件并未稳定变差。两组实现均未通过，不能将问题只归结为幅度尺度；暂不接入主模型。

## 9. 数值与目标审查，读出对照（run23/24）

实际 data224 为 51 个句子标识；seed45 划分 41 句/190 段训练、10 句/34 段保留。“6句”是之前 audit26 的规模，不能套用到 data224。

- run19 有 37.23% 有效帧 `abs(logit)>3`、65/224 段整段饱和；heldout 为 51.99%、15/34 段。`3*tanh(logit)→clip-center` 可以把整段饱和变为近零动态并使梯度消失。新增 `--readout linear` 保持零点斜率、去掉此陷阱；默认旧读出用于复现。FiLM 只有 1.55% 饱和，不能据此解释其全部泛化失败。
- emotion2vec 标准化后片内 RMS=.235、片均值 RMS=.990；content 为 .980/.210。新增 `--audio-input clip_centered` 只用当前音频去 DC，无 heldout motion 标定。
- 旧 TCN GroupNorm 在时间轴含 padding；已改逐帧 LayerNorm，padding 不变性回归通过。旧 TCN checkpoint 不能用新归一化静默重新评估。

同初始化、相同1200步及 minibatch、唯一 MSE、固定 seed/split45：

| 分支 | 读出/输入 | train R² | heldout R² | heldout corr |
|---|---|---:|---:|---:|
| audio | 旧 tanh/raw | .1313 | −.1035 | −.0639 |
| audio | linear/raw | .1549 | −.0976 | −.0363 |
| audio | 旧 tanh/去DC | .4444 | −.2951 | −.0923 |
| audio | linear/去DC | .4453 | −.3176 | −.1006 |
| FiLM | linear/raw | .9403 | −.3526 | .0463 |
| FiLM | linear/去DC | .9547 | −.3871 | .0245 |

消除饱和没有解决泛化；去 DC 与更自由输出在此设置下加重过拟合。未推广到主模型。

训练集目标方差的协方差贡献（各组与总标量的 covariance / total variance，可相加；不是独立方差比例）：blink 23.59%、gaze 43.38%、squint/wide 12.51%、brows 20.53%。原 upper_l1 明显包含视线/眨眼，不能当作密集真实情感标签。新增单标量 `upper_expression_l1` 排除 gaze/blink，仅作训练标签对照，不加生成通道。

## 10. 控制过拟合的基线与说话动作正控制（run25/26）

`scripts/probe_dynamic_ridge.py`：float64；bin→clip-center；每个内训练折独立拟合输入 std；3 折按句子交叉验证选择 ridge α（mean Gram + αI）；固定后才预测外层 heldout。仅一个线性标量输出、一个 MSE，无目标/身份/句子泄漏标定。jaw17 直接预测 motion，不减 B0，作为正控制。

seed45结果：

| 输入 | 目标 | 内部选α | heldout R² | corr | reverse R² |
|---|---|---:|---:|---:|---:|
| emotion2vec | upper_l1 | 100 | .0010 | .0013 | .0045 |
| emotion2vec | 排除gaze/blink | 100 | .0004 | .0552 | .0019 |
| emotion2vec | jaw17 | 10 | .0070 | .0609 | −.0090 |
| content | upper_l1 | 10 | .0130 | .1093 | −.0093 |
| content | 排除gaze/blink | 10 | −.0039 | .0402 | −.0082 |
| content | jaw17 | 1 | .3508 | .5825 | −.3759 |

content 基线另两组句子划分复核：

| split seed | upper_l1 R² | corr | reverse R² | jaw R² |
|---|---:|---:|---:|---:|
| 44 | .0126 | .1056 | −.0141 | .3061 |
| 45 | .0130 | .1093 | −.0093 | .3508 |
| 46 | .0211 | .1281 | −.0024 | .2543 |

三个划分来自同一224段且相互重叠，不能视为三个独立测试集。上脸 R² 按句子 bootstrap 的95%区间均跨0；收益很小，未解决情感动态。jaw正控制明显通过，排除“这条数据管线所有时序均不可预测”的说法，但不代表独立 audiovisual lip-sync 已验证。

当前可用结论：保留 ridge 作为跨句基线，停止仅凭训练相关增加时序容量；目标筛除 nuisance 本身也未达标。先在训练句内部检验声学/标签的时间分辨率及正则化，再决定是否引入更可靠的视觉表达监督。B0、identity、global、renderer 默认权重未替换，不增加新loss/主模型通道。

结果与源快照：`artifacts/dynamic_audit_run23_26/`（含各run summary/provenance/config、ridge权重、`input_target_audit.json` 与 `ridge_split_comparison.json`）。本地完整测试133通过。

## 11. 全局情感方向 × 有符号音频强弱（run27）

新增 `emotion_ray.py`、`emotion_conditioned_dynamic.py` 和独立 `train_emotion_ray_pilot.py`。训练句内按人等权拟合 neutral-relative 类别方向，排除blink/gaze；用 motion 的中心化残差在该方向上的投影作为固定有符号标量。小头仅有一个32维共享hidden、静态FiLM与线性标量输出，无tanh，无新区域输出和附加loss。每4帧分箱后去句均值；唯一MSE使用训练非neutral RMS归一化。

4组使用相同seed45、minibatch、600步、AdamW lr=.0003/wd=.01；不按heldout选模型。真实标签仅为oracle；实际条件为run09原归一化下冻结audio global softmax。输出方向用概率加权ray、不单位化，不利用GT中性门控。所有组均对同一GT-ray投影及同一真实残差评分，neutral单独报告。按句簇bootstrap，非逐帧/逐clip伪独立统计。

224段为51句，训练190段/41句，内部留句34段/10句；外部开发28段/19句。run09 global见过完整224段，内部留句只检验新动态头；原28段亦多次用于开发，不称未触碰测试集。实际train enrollment与query、内外sentence/clip不重叠已由runner断言。拟合ray、输入统计及target尺度只用190段。

| 600步小头 | train标量R² | 内部留句标量R² | 外部开发标量R² | 外部开发实际方向proxy R² |
|---|---:|---:|---:|---:|
| emotion2vec，无global条件 | .1708 | −.1076 | −.0253 | −.0312 |
| emotion2vec，真实标签oracle | .1702 | −.1566 | −.0451 | 标签输入仅诊断 |
| emotion2vec，冻结audio global | .1673 | −.1927 | −.0538 | −.0582 |
| content，冻结audio global | .9975 | −.2532 | −.0650 | −.0553 |

外部global分类27/28=96.4%；错误类别不是主要充分解释。content实际组外部scalar corr=.321，但幅度误差使R²仍负，不能仅凭相关性采用。内部emotion2vec实际组reverse proxy R²=−.1008，优于full的−.1921，时间关系未通过。

表达方向的GT强度重建也有限制：内部留句全表达残差解释R²=.3783，但上脸表达为−.0707，mouth=.4942；外部为全表达.3022、上脸.2799、mouth.3056。全维投影是全维最优，局部眉眼未必最优，不能把嘴部主导的总收益当作稳定情感动态。需要独立target geometry复核及受正则约束的可预测性对照。

四组权重、曲线、数据hash、源快照、按句/按类/zero/reverse/shuffle结果存于 `artifacts/emotion_ray_run27/`。`run27_emotion_ray_audit`仅加载同一小头补公平actual方向与独立性检查，不重训或改目标。主系统全部参数训练前后SHA256一致，未接入renderer、未替换默认checkpoint、未删数据；不能据此声称最终口型已有新收益。完整本地测试151通过。

### 几何复核与正则预测（run28）

`audit_emotion_ray_geometry.py` 使用真实motion作诊断：内部留句22个非neutral片段仅覆盖5句；同一ray改用上脸维度自身最优的GT scalar后，上脸重建R²从−.0707变为+.3525。眉眼与嘴部各自最优scalar的时序corr在train/internal/external为.1120/−.1125/.1899。happy ray的93.56%平方权重在嘴部；全脸共享scalar对happy眉眼的GT重建在三组均为负。说明“全脸同一强度”包含时序冲突；即使audio学会这个目标，也不能保证眉眼改善。逐片GT上脸PCA rank1/2解释84.1%/94.9%仅是motion表示上限，不是audio可预测性或最终架构建议。

`probe_emotion_ray_ridge.py`：每个内训练句折重拟ray及feature尺度，3折仅按非neutral MSE选择alpha(.01/.1/1/10/100)，固定后评估原outer-ray目标。一个共享线性标量；neutral仅参与方向锚定，实际推理不GT门控。content选alpha=1；emotion2vec选100，后者输出近零。此阶段读取真实缓存到本地CPU拟合，不是模拟数据。

| content ridge，实际audio方向 | 内部留句 | 外部开发 |
|---|---:|---:|
| 固定动态proxy R² | .1267 | .1457 |
| proxy R²句簇95%CI | [−.1065,.2523] | [.0652,.1853] |
| reverse proxy R² | −.1582 | −.2100 |
| shuffle proxy R² | −.1084 | −.0370 |
| 标量时序corr | .3579 | .3852 |
| 真实上脸残差R² | −.1147 | −.1001 |
| 真实嘴部残差R² | .0899 | .0669 |

这个目标上有音频可预测成分，强正则明显优于600步自由头；但眉眼受损，不能用proxy正R²推进renderer。外部开发CI不替代未触碰数据复验。结果存于 `artifacts/emotion_ray_run28_ridge/`，几何审计在 `artifacts/emotion_ray_run27/geometry_audit.json`。

### 排除嘴部主导的单标量对照（run29）

同一ridge脚本新增 `--supervision upper_expression`：仅在诊断监督中使用9个眉眼表达controller，不新增生成区域通道。外层/每个内折分别从训练均值重拟ray，复制bundle后重新构造方向、scalar、proxy；不以全脸ray截断代替重拟，不更改原数据。实际仍使用原冻结audio概率，无GT类别/neutral推理门控。

content内折选alpha=10，actual proxy R²内部=.0050，外部=.0113（句簇95%CI=[−.0079,.0253]）；真实眉眼动态R²内部=.0018、外部=.0062。外部reverse proxy R²=−.0039，方向关系弱但幅度/解释力不足；emotion2vec外部proxy仅.0008。两个目标定义不同，不把run28→run29数值下降当作同一指标退步，也不能说去除嘴部主导后学会了眉眼情感。

本轮结论：已有语音内容特征确有部分表达时序信息，强正则能防止小样本记忆；然而全脸单强度假设存在组间时序冲突，现有特征对眉眼目标的可预测性仍弱。暂不训练renderer接口，不替换主checkpoint。后续优先审计视觉逐帧表达监督的稳定性、时钟与音频时间特征，确定可跨句预测的共享动态目标，再考虑小规模时变基；不凭GT PCA覆盖率直接增加rank，也不追加一串loss。

run29产物在 `artifacts/emotion_ray_run29_upper_ridge/`；本轮完整测试最终为156通过。真实GPU训练4×600步；两轮ridge各两输入来源在下载的真实缓存上完成。没有进行新的人脸渲染/口型感知检验，也没有宣称CCF录用或动态成功。

## 12. 无文本音频输入与共享可预测motion基（run30/31）

本轮实现并实际拟合四种输入：旧emotion2vec final、真实中间层2/4/6逐帧LayerNorm后等权平均＋韵律（temporal）、已有HuBERT content、content＋temporal。韵律来自原始未归一化波形的40ms窗口F0/log RMS/periodicity/voicing；F0为自相关估计，不是真实标注。官方emotion2vec checkpoint共8个Transformer blocks、10个extra tokens；提取时去extra tokens并按已审计卷积中心和native时间对齐。没有文本、VA、生成区域头或新增loss组合。

`kinetalk_b0/predictable_motion.py`以train-only ridge reduced-rank regression（RRR，既有方法）拟合共享基U；motion目标是 `center(motion-B0-P_id) U`，始终来自真实motion。与motion-PCA同rank2/4/8、同ridge输入预测器及全维direct ridge比较。每内折按训练句重拟feature RMS、U和alpha；共同的native全表达MSE（含neutral）选参，然后冻结评价。该选择目标可能被嘴部主导，不能称眉眼最优。PCA是motion-only低秩基线，不是已严格重训的原神经motion teacher。

run30沿用原224片段和190/34句子划分，外部开发28条。inner-selected RRR upper R²：旧final=.00149、temporal=.00941、content=.01195、合并=.00858；CI均跨0。相同rank8下RRR与PCA持平，未形成可靠优势。

run31元数据先锁定12人，train1200条/65句、validation280条/25句、48条neutral参考，三者句子/clip隔离。非中性validation240条/22句。另锁定new_test15条/3句，仅13neutral和2sad，未读取或评估；不足支撑正式泛化测试。B0/预训练模块的历史暴露未排除，验证身份均已登记。run30→31同时改变人/句子/验证构成和正则选择，不能归因于单纯增加样本数。

| run31输入，inner-selected RRR | rank | 非neutral upper R² | 95%句簇CI（5000次） | reverse R² |
|---|---:|---:|---|---:|
| 旧final emotion2vec | 2 | .00023 | [−.00065,.00103] | 见完整JSON |
| temporal | 8 | .01601 | [.00933,.02577] | −.01733 |
| content | 8 | .02026 | [.01516,.02779] | 见完整JSON |
| content＋temporal | 8 | .02354 | [.01564,.03476] | −.02251 |

合并输入的眉毛R²=.01899，CI[.01067,.03183]；眼部表达R²=.03319，CI[.01976,.04788]，均胜zero/reverse/shuffle。这里upper排除blink/gaze；评估分组不构成独立生成通道。upper有19/22句正收益，12个身份汇总均正，brows有3个身份负值。预测/真实RMS仅upper13.75%、brows12.11%、eyes16.71%；GT投影oracle upper R²=.55051，仍存在巨大audio预测缺口。

配对句簇bootstrap：合并输入RRR8相对PCA8 upper ΔR²=.00486，CI[.00304,.00760]；相对direct ridge仅+.00039，CI跨0；相对content RRR8为+.00328，CI[−.00164,.00869]。只能说RRR对同维PCA有小幅开发集优势，不能说比直接回归更好，也不能宣称temporal在已有content上可靠增益。探索性多组统计未做多重校正。

**明确负结果**：neutral40条上的mouth残差R²=−.2978（CI[−.5190,−.1495]），jaw=−.3115（CI[−.3967,−.2334]）。这不是已有renderer最终口型的退化测量，但足以拒绝把预测场直接相加到动作。静态身份偏置在中心化目标中抵消，本实验也不能证明身份贡献或严格情感解耦。

当前决策：保留原默认模型，以隔离生成小训练检验弱动态条件是否有用，特别检查neutral误激活和嘴部。不得将正R²改写为已解决强情感动态。代码/产物：`scripts/extract_predictable_audio.py`、`lock_predictable_motion_data.py`、`prepare_predictable_motion_bundle.py`、`probe_predictable_motion.py`、`summarize_predictable_motion.py`、`audit_predictable_motion_predictions.py`；`artifacts/predictable_motion_run30/`和`predictable_motion_run31/`保存结果、权重、曲线、manifest、hash和配对统计。

## 13. 共享动态场接入生成器（run32）

已执行RRR8/PCA8/无local三臂各600步，Adam lr=.0002、batch16、同seed46，同批次、flow噪声/time/teacher选择的hash一致。B0、身份、audio global、motion teacher全部冻结；训练原renderer＋local_projection和一个以ridge初始化的无bias线性audio head。真实motion目标固定U和训练RMS，前半teacher概率.5→0、后半纯audio。仅flow+.1坐标对齐，zero臂仅flow，无区域头。

原始final特征按run09保存的两层统计归一化得到global，不能用去均值bins恢复global。缓存为原FP32内容计算B0/h0、content半精度仅存储；每次生成必须使用缓存base。重建缓存batch16曾使GPU B0数值与原bundle差4.8e-5，改回原batch32后全部1480片段通过原严格residual断言，未放松容差。缓存391MB置于/dev/shm；正式诊断bundle、权重和曲线保留。

三noise42/123/2026、纯噪声12步生成280条validation；先按clip平均各seed误差统计，再5000次按句簇bootstrap，不能把三个noise当三倍独立样本。原checkpoint生成逐值相同，所有冻结参数hash相同。

| 三noise平均，非neutral | upper动态R² | mouth动态R² | neutral mouth原动作MSE |
|---|---:|---:|---:|
| 原run09 | −.14637 | −.08317 | .003598 |
| RRR8 joint full | −.46249 | −.21023 | .004651 |
| 同一RRR模型zero-local | −.20491 | −.19274 | .004265 |
| 同一RRR模型reverse | −.45747 | −.38625 | .004769 |
| PCA8 joint full | −.51303 | −.22110 | .004766 |
| matched无local训练 | −.56350 | −.37589 | .004642 |

动态R²针对原帧中心化真实motion residual，和run31低率投影的R²不直接横比。RRR优于PCA或matched无local训练，但比原模型明显差；同模型full比zero-local upper ΔR²=−.25758，CI[−.35261,−.18834]，full与reverse的动态差值CI跨0。oracle upper也差于zero，不能只怪audio。全脸低误差不能掩盖此问题：非neutral upper原动作MSE确实从.02539降到.01822，但动态RMS从真实的40.94%升至73.65%，时序相关只从.02597到.05424，且动态误差变大。

neutral mouth MSE相对原模型恶化.001053，句簇CI[.000756,.001392]；mouth相关.7342→.6429，故未通过口型非退化。冻结motion teacher对生成表情的类别准确率从约92.1%降至69.9%，它不是独立评测器，但也不能宣称global质量保持。

只读诊断`audit_predictable_head_drift.py`：joint后audio头自身native upper R² .02354→.01193，PCA .01868→.00682；既有音频可预测性被削弱。后续仅做一个有依据的固定预算修正：冻结已验证head和整个renderer，零初始化原local_projection单独训练，仍不替换默认模型。run32产物与配对统计在`artifacts/predictable_renderer_run32/`。

## 14. 固定音频预测器和renderer，仅适配共享接口（run33）

已完成RRR8/PCA8各600步，只训练原 `local_projection.weight`（8×64=512参数），零初始化，无新增层/区域头。renderer、ridge audio head、U、输入尺度、身份、B0、global全部冻结并核验hash；训练目标只有已有flow MSE，删除在此模式下为常量的alignment。teacher→audio条件比例与run32相同，后半300步纯audio。复用同一数据、seed/batch/noise/hash及三noise12步生成，不按验证挑checkpoint。

| 非neutral三noise平均 | upper动态R² | mouth动态R² | upper时序corr |
|---|---:|---:|---:|
| 原run09完整audio | −.14637 | −.08317 | .02597 |
| 冻结原renderer，zero-local | −.08619 | −.03284 | .02584 |
| RRR8新audio接口 | −.07357 | .01967 | .07349 |
| RRR8 reverse | −.10505 | −.06863 | .02200 |
| PCA8新audio接口 | −.08468 | .02533 | 见JSON |
| RRR8真实motion oracle | .31207 | .54401 | 仅诊断 |

RRR full相对自身zero-local upper ΔR²=.01262，句簇95%CI[.00542,.02174]；相对reverse为+.03148，CI[.01952,.04599]；相对PCA8为+.01111，CI[.00671,.01737]。三种noise的full均优于zero/reverse。oracle明显有效，表明这一冻结解码接口能利用真实motion动态，比run32共同训练更稳妥。run32无local训练臂不是run33新增训练臂；它仅作为历史匹配预算参照，关键对照是同一冻结renderer的zero-local。

**不能省略的边界**：upper绝对动态R²仍负；相对zero-local眉毛单项ΔR²=.00374，CI[−.00157,.01090]跨0，眼部+.03064才有明确收益。原模型→新接口改善的一部分来自去掉旧audio local，因此必须保留zero-local对照，不能全部归功于新情感场。

中性mouth原动作MSE从原模型.003598降至.003311，mouth时序corr .7342→.7683；但仍差于冻结zero-local的.003212/.7760。neutral upper动态也比zero-local差ΔR²=−.04548，CI[−.07667,−.02148]，误激活仍在。冻结motion teacher对生成类别读出约92.14%→92.02%，接近原模型，但不是独立情感质量或口型同步认证。全局表示固定与最终全局效果固定必须分开描述。

因此当前成果是“音频已有弱可预测成分，冻结生成器后的共享接口能小幅利用它”，不是强动态成功，也不足将新权重设默认。下一步优先在训练内验证已有global对动态幅度的条件化是否能抑制neutral误激活，避免重新开放整网破坏预测器；需补正式balanced新句测试和原视频监督质量核验，再考虑更长训练。没有增加文本、VA、硬区域输出或新loss堆叠。

本轮新增生成训练共3000步（run32三臂＋run33两臂）。185项完整测试通过；测试验证梯度/冻结、double normalization、时钟/目标配对、padding和noise统计，不是效果证明。run33 `projection_adapter.pt`约65KB，包含共享接口和固定audio预测器；只有原checkpoint hash匹配才能恢复，默认禁用。所有远端曲线/来源保留，本地结果在`artifacts/predictable_renderer_run33/`。示例按每类clip ID字典序第一条、固定noise42选择，非挑最佳；只是系数曲线，没有人脸视频感知验证。未删除训练数据。


## 15. 音频全局概率门控与两种子复核（run34/35）

不新增网络头/损失，沿用冻结audio global，`g=1-softmax(emotion_logits)[neutral]`，对整条共享8维控制乘同一片段标量。g是非中性置信度，不是新的心理强度/时序监督。teacher和audio、reverse都使用同一个audio gate；GT标签仅作分组，不把neutral真值动作归零。

run34不训练：原run33 full/门控/训练平均g常数/zero/reverse/oracle/original，固定三noise。门控upper相对zero ΔR²=.012767 CI[.005588,.021920]，neutral mouth MSE相对zero +1.217%、单侧90%上界1.969%，比未门控+3.08%减少；neutral upper原动作MSE -0.237%。门控与常数缩放的若干差异CI跨零，不宣传它显著优于通用幅度抑制。开发neutral分类77.5%，仍有漏门控。

run35将相同门控纳入已有flow训练：seed46/47各600steps、batch16、lr.0002、从原checkpoint零初始化512参数接口，head/U/renderer/B0/identity/global冻结。三noise逐clip平均误差后按句bootstrap5000：

| 指标 | seed46 | seed47 |
|---|---:|---:|
| 非neutral upper动态R² | -.073523 | -.073950 |
| upper full−zero ΔR² | .012665 | .012237 |
| 上述95%句簇CI | [.005340,.021934] | [.004646,.021845] |
| 眉 full−zero ΔR² | .003789 | .002974 |
| 眉95%CI | [-.001682,.011177] | [-.002645,.010035] |
| neutral mouth原动作MSE比zero | +1.274% | +1.368% |
| 上述单侧90%上界 | +2.061% | +2.220% |
| 冻结teacher生成类别读出 | .92024 | .92024 |

两seed均通过预先记录的动态、眉毛容忍、中性原动作/口型/速度/global工程准入。此为扩大受监控训练的依据，不是眉毛显著改善或强动态成功。默认模型未替换，zero-local从冻结renderer保证同值。产物`artifacts/predictable_renderer_run34/`和`run35/`，`audit_projection_readiness.py`保存逐seed完整checks。

正式数据已锁2720train/22人/67句（保留原1200全部），原280跨句开发不变；native val439/3人另作跨身份同句开发，native test512仅元数据封存。真正新句覆盖不足，不能称全系统未见测试；B0历史暴露仍待认证。16条按hash预选训练源视频与native曲线稀疏抽检未发现整段错配，仍有慢变/近饱和眉毛与眨眼干扰；不据此筛除或改标。详细协议和限制见`FORMAL_TRAINING_READINESS.md`。

正式准备路径`/root/kinetalk_runs/formal_predictable_v1`，用root盘持久缓存避免autodl仅411MB。先仅训练句3fold重拟rank8坐标和alpha，固定方法检查原280可预测性后，从原冻结renderer重新训练接口，不将旧projection接到新U。
