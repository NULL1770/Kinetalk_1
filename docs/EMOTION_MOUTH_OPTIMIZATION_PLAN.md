# 情感口型与 Emotion-F1 优化记录

更新时间：2026-10-04（Asia/Shanghai）

最新恢复摘要：`docs/CURRENT_OPTIMIZATION_STATE.md`。本文旧章节保留实验历史；以最后结果和拒绝/选择决策为准，不重复已失败的实验。

## 使用规则

上下文压缩、重新进入任务或开始新一轮实验前，先阅读本文件，以及项目根目录的 `task_plan.md`、`findings.md`、`progress.md`。本文件是当前任务的事实基线和实验顺序记录；不得因为上下文丢失而重新从头审计。

## 用户目标

- 情感应能影响口型大小，尤其是 `jawOpen`、唇部张合和情感口型；内容/音素时序仍应保持。
- MBE 从当前约 0.90 降到约 0.70 或更好，其他指标也继续下降。
- 生成动作的独立 emotion macro-F1 目标至少约 0.60；当前很低，需要先定位是条件、监督、数据质量还是评估问题。
- 先讨论和诊断，逐步修改；避免盲目完整重训和重复分析。

## 已确认的当前架构

### B0 / articulation

- `kinetalk_b0/models/model.py::Stage1Model` 生成 `b0`，输入是 query 的 content 特征，输出契约是中性 speech articulation。
- 远端 `b0full_gate` recipe 的 requested `articulation_scope` 是 `neutral`，其中 715 条 neutral clip；safe-DTW 分支实际覆盖 2,583 条获准配对，effective articulation 集合共 3,298 条。不能把 requested scope 的 clip_count 当作实际训练集合总数。监督是 native neutral teacher，嘴部另有局部 event mask，而非情感动作直接监督 B0。
- `neutral_output_indices` 配置为 14..40 加 51；B0 decoder 当前输出完整 neutral coefficient vector。
- 固定的 articulatory mouth indices 为 `(14,15,16,17,18,19,20,21,22,31,32,37,38,39,40)`。

### 情感/残差路径

- `NeutralAffectSystem` = frozen B0 + neutral identity baseline + stochastic residual flow。
- 情感输入包括：
  - `global[64]`：全局情感语义；
  - `intensity_value`：4 类 intensity logits 的 softmax 期望值；MEAD 约为 neutral=0、表达强度 1/2/3；
  - `u_a[T,64]`：音频逐帧时序条件，不是逐帧情感标签；
  - neutral enrollment 得到的 identity/style code 和静态 baseline。
- motion teacher 从 `motion - b0 - identity baseline` 提取 global/emotion/intensity；audio student 从 1540-D audio features 预测同一接口；renderer 使用 DiT flow matching。
- `generated_emotion_consistency()` 只对最终可微 endpoint 做 motion-teacher 的 emotion CE、global cosine、intensity CE；它没有显式保证 mouth/jaw 振幅或强度单调性。

### 关键通道边界

- 远端 `b0full_gate` recipe 的 `protect_mouth=false`，但 `residual_support` 仍把上述 15 个 articulatory mouth 通道设为 false。
- 因而情感残差不能修改 `jawOpen`、`mouthClose`、`mouthFunnel`、`mouthPucker`、`mouthLeft/Right` 和部分上下唇通道；这些主要来自 B0 + identity baseline。
- 允许情感残差修改的 affect mouth 主要是 smile/frown/dimple/stretch/shrug/press 等通道。该边界足以造成“GT 张口大、生成张口偏小”。
- 只有显式 `--protect-mouth` 才会把全部 14..40 口部都锁死；旧完整 `ours` recipe 确实使用过 `--protect-mouth`，不能与当前 `b0full_gate` 混为一谈。

## 已核实证据

来源：远端 `/root/autodl-tmp/kinetalk_final_20260922/`，以及本地 `final_experiment/evaluation/rendered/safe_dtw_b0full_gate_happy_20261003/`。

- 最新 `b0full_gate` audio evaluation：audio emotion accuracy `0.8778346745`，motion teacher accuracy `0.6730065838`，生成 teacher readout（非独立）约 `0.879–0.883`。
- 同一 run 的生成 mouth RMS ratio 约 `0.717–0.720`，说明动作能量明显低于 GT。
- sealed coefficient metrics：MBE `0.9034527614`，LBE `0.4827744897`，supplementary lip23 LBE `0.6026846100`。
- sealed 独立 real-motion statistics probe：accuracy `0.2336621433`，macro-F1 `0.1555114561`。
- 独立 probe confusion 显示 disgust 大量过预测，fear/happy 几乎不被预测；happy F1=0，fear F1=0。故不是单纯 audio classifier 低，而是情感没有稳定落实到生成动作。
- 旧记录中，audio head 的约 0.88 是“音频标签分类准确率”，不能当作生成动作 emotion-F1；生成动作的独立 probe 才是当前 F1 指标。
- 预览视频 `comparison.mp4` 为 248 帧，其中 103 帧有效、145 帧由末帧填充；后半段停住是可视化 invalid-frame policy，不是完整动态证据。
- 视频某一有效帧可见 GT(happy) 口部张开明显，KineTalk 生成较小，B0 neutral articulation 更小，和通道边界分析一致。

## 主要假设（按优先级）

1. **结构性 mouth bottleneck（高可信）**：情感残差不能改 articulatory mouth，直接限制了情感引起的张口幅度。
2. **renderer 情感落实不足（高可信）**：flow loss 允许生成 endpoint 偏中性；global/audio 分类头高准确不代表动作可读。
3. **监督与评价耦合（高可信）**：生成 consistency 使用同一 motion teacher 作为 critic/target 体系的一部分；其非独立 readout 会虚高，独立 probe 才能判断泛化。
4. **强度未转成幅度（高可信）**：intensity 只作为分类/标量条件，没有 pairwise ordinal mouth-energy 约束。
5. **数据质量污染（中高可信）**：safe-DTW 高质量 mouth gate 主要约束 neutral articulation teacher；teacher/audio 阶段仍遍历全部 fit clips，低质量或情感标注动作可能进入情感残差监督。
6. **B0 neutral-only 导致幅度基线偏中性（中等可信）**：需要通过对照实验确认，不能仅凭视频下结论。

## 后续实验顺序与验收门槛

### Phase 0：数据/指标复核（只读）

- 固定使用同一 validation clip、同一 neutral identity、同一 content、同一 noise seed。
- 明确比较 valid frames、all mouth、articulatory mouth、affect mouth；不把 invalid 填充帧计入动态指标。
- 保存每个实验的 config、checkpoint SHA、输入 SHA 和输出 JSON；sealed test 不用于选模型。

### Phase 1：oracle / zero / shuffle 条件诊断（已完成）

对现有 checkpoint 做条件替换：

1. predicted audio affect（baseline）；
2. oracle GT motion-teacher global/intensity；
3. zero global/intensity；
4. emotion label shuffle；
5. intensity 1/2/3 替换或反转；
6. fixed noise 与多 seed。

每项测：independent probe macro-F1/accuracy、mouth RMS ratio、`jawOpen` energy、affect-mouth energy、MBE/LBE、条件间 paired delta。

**判断门槛：**

- oracle 与 zero 几乎无差异：renderer/输出 support 没有真正使用情感，先修模型结构；
- oracle 能明显改变动作但 predicted 差：audio student/teacher 蒸馏或音频特征有问题；
- intensity 替换不改变 mouth energy：必须新增强度幅度约束；
- shuffle 仍有相同 F1：评估 probe 或 identity/content leakage 需审计。

#### Phase 1 已完成（2026-10-03）

只读诊断脚本：`scripts/phase1_condition_diagnostic.py`。远端输出：`/root/autodl-tmp/kinetalk_final_20260922/diagnostics/phase1_condition_20261003.json`。使用 audio-stage checkpoint SHA `069708add8f3ec5fe3b1555e9d2e5bc3a64605473cbfacf06fd31afd16abe932`、validation 1367 clips、固定 noise seed 42 和独立 probe。

| condition | MBE | LBE | mouth pred RMS / GT | jawOpen pred RMS / GT | independent F1 |
|---|---:|---:|---:|---:|---:|
| audio_pred | 0.9446 | 0.5147 | 0.0974 / 0.1525 | 0.0881 / 0.1195 | 0.1457 |
| teacher_oracle | 0.7906 | 0.5139 | 0.1054 / 0.1525 | 0.0881 / 0.1195 | 0.1320 |
| zero_all | 1.0816 | 0.5023 | 0.0723 / 0.1525 | 0.0881 / 0.1195 | 0.0689 |
| zero_global | 1.0667 | 0.5107 | 0.0779 / 0.1525 | 0.0881 / 0.1195 | 0.1820 |
| shuffle_global | 1.1494 | 0.5143 | 0.0976 / 0.1525 | 0.0881 / 0.1195 | 0.1476 |
| intensity_low (0) | 0.9451 | 0.5140 | 0.0964 / 0.1525 | 0.0881 / 0.1195 | 0.1690 |
| intensity_high (3) | 0.9430 | 0.5152 | 0.0983 / 0.1525 | 0.0881 / 0.1195 | 0.1391 |

Interpretation: `teacher_oracle` changes global/full-face behavior and improves MBE, but cannot change `jawOpen` at all; this is direct evidence of the residual support bottleneck. Intensity 0→3 has negligible mouth-energy effect, so current scalar intensity path is not an amplitude controller. The independent probe F1 is non-monotonic under interventions (zero-global is higher than audio/oracle), so it is not a sufficient causal sensitivity metric; retain it as a frozen external readability metric and add paired condition deltas, region energy and class confusion.

### Phase 2：开放 mouth residual（最小结构修改，进行中）

- 新训练默认让情感残差可修改全部已观测 mouth 通道，包括 `jawOpen`、`mouthClose`、`mouthFunnel`、`mouthPucker`、上下唇相关通道；`--protect-mouth` 和 `--articulatory-mouth-only` 变成显式 legacy ablation。
- DiT output-side affect route 现在覆盖 14..40 全 mouth 加上 upper-face expression；旧 checkpoint 的 state dict 仍可加载，因为该 mask 是 non-persistent buffer。
- B0 仍保留 content timing；新增/保留 content-preservation、velocity 和 phonetic alignment 约束，避免情感改变说话内容。
- 做小规模 smoke 和 validation ablation：old support vs full mouth support；先看 mouth amplitude/F1，不直接上 sealed test。

Phase 2 当前只完成 support/routing 的最小代码改动；内容保持损失和 intensity ordinal loss 留到后续独立步骤，避免一次改动混入多个因果因素。

Phase 2 smoke（远端 `phase2_fullmouth_smoke_20261003`）已完成四阶段、每阶段 1 epoch、8 clips/8 steps、`test_loaded=false`。配置验证：`residual_support_mode=full_mouth`，51 个可观测通道中 `jawOpen` 和全 mouth 均 active，DiT affect mask 覆盖 36 个 mouth/upper-face 通道。该 smoke 从随机初始化开始，validation 数值不作为质量结论；只证明训练、保存和评估链路可运行。

#### 当前执行边界（2026-10-04）

- 本阶段下一项是 validation 规模的旧 support vs `full_mouth` 对照。必须固定 validation manifest、neutral identity/enrollment、content、noise seed 和 evaluator；不得加载 sealed test 选择模型。
- 对照首先回答一个结构问题：`teacher_oracle` 条件下 `jawOpen` RMS 是否随 residual support 开放而改变，以及 mouth RMS 是否向 GT 接近。只有这个问题得到肯定，才继续加入内容保持损失。
- 本阶段暂不同时加入 ordinal intensity loss、数据清洗和 class-balanced consistency，避免无法判断收益来自哪个因素。
- validation 对照最低记录：MBE、LBE、all-mouth RMS ratio、articulatory-mouth RMS ratio、`jawOpen` RMS ratio、affect-mouth RMS、paired oracle-minus-zero delta、内容时序误差/速度误差、独立 probe macro-F1 和每类 F1。
- 若 `full_mouth` 仍不能改变 `jawOpen`，停止继续训练并回查 support mask、DiT output mask、checkpoint recipe 绑定和 inference 路由；若能改变但内容误差上升，再进入内容保持约束设计。

#### Phase 2 continuation 对照设计（2026-10-04）

- 使用已完成的旧 `audio/final.pt` 作为共同 warm start；不覆盖原目录。
- `legacy_support` 保持原 articulatory-only residual mask；`full_mouth` 通过显式 `--allow-residual-support-expansion` 只扩展 residual support，禁止任意缩小或重排 support。
- 两个 continuation 使用相同 validation manifest、训练数据、seed、batch size、decode steps 和 continuation epoch 数；不加入 intensity ordinal、数据清洗或新的 consistency loss。
- 每个输出单独保存 provenance、checkpoint SHA 和 validation curves。结果只用于 development ablation，不进入 sealed 表格。

#### Phase 2 continuation 结果（2026-10-04）

远端两个 continuation 均从同一个旧 audio checkpoint 开始，训练 2 个 audio epoch，固定 validation 1367 clips、noise seed 42、decode 12，`test_loaded=false`。输出：

- legacy：`/root/autodl-tmp/kinetalk_final_20260922/checkpoints/phase2_legacy_support_continuation_20261004/audio/final.pt`，SHA `93bc6a50b03d572ab93304d092efa55511dde8b100b47f08eb303015d2a098ed`。
- full-mouth：`/root/autodl-tmp/kinetalk_final_20260922/checkpoints/phase2_fullmouth_continuation_20261004/audio/final.pt`，SHA `60ac835ab7212877539519d9b7cbb9fd433e1e89aa42b1ac139a180f7ac53bfd`。

Validation generation summary（下表使用 seed 42；RMS ratio 是去均值后的动态能量比，不是绝对系数幅度比）：

| support | centered mouth RMS ratio | mouth raw MSE | mouth frame-displacement MSE |
|---|---:|---:|---:|
| legacy articulatory-only | 0.703 | 0.02105 | 0.00186 |
| full-mouth | 0.972 | 0.01729 | 0.00344 |

固定条件诊断的核心结果：

| support / condition | MBE | LBE | mouth RMS | `jawOpen` RMS | independent macro-F1 |
|---|---:|---:|---:|---:|---:|
| legacy / audio | 0.9556 | 0.5146 | 0.0955 | 0.0881 | 0.1852 |
| legacy / teacher oracle | 0.7896 | 0.5116 | 0.1053 | 0.0881 | 0.1637 |
| full-mouth / audio | 0.9053 | 0.4553 | 0.1280 | 0.1395 | 0.1503 |
| full-mouth / teacher oracle | 0.6944 | 0.3806 | 0.1581 | 0.1492 | 0.1270 |

结论：support 扩展解除 `jawOpen` 无法变化的结构限制，并改善幅度；并不表示幅度、时序已全部修好。Oracle MBE 约 0.70 依赖真实目标动作提取的 global/intensity，不是可部署成绩，也不是音频可达到的已证实上限。该差距表明条件路径值得研究，但不能仅凭 oracle 证明音频蒸馏有缺陷，因为 teacher 也可能编码音频无法确定的动作细节。独立 F1 没有随口部能量自动提升。

当前还需补充 articulatory mouth 的速度/时序误差。如果 full-mouth 的幅度恢复伴随明显速度误差上升，先加入内容保持项，再增加独立语义一致性；如果速度误差可接受，则优先修 audio-global distillation 和 class-balanced semantic consistency。

paired same-noise delta 进一步确认：legacy 的 oracle/audio `jawOpen` delta 为 0；full-mouth 的 oracle/audio `jawOpen` delta RMS 为 `0.0459`，zero-all/audio 为 `0.0739`。也就是说，full-mouth 已具备真实的情感条件控制，但 intensity 0/3 对 `jawOpen` 的 delta 仍只有约 `0.013`，强度排序监督仍需单独处理。

内容诊断（3 seeds 平均）显示 full-mouth articulatory-mouth frame-displacement MSE 约 `0.00484`，legacy 约 `0.00199`；全序列展平后的生成/B0 速度相关约 `0.50`，不是逐帧方向平均。B0/GT 速度相关自身只有约 `0.089`，因此保留 B0 不能等同于保留正确音素时序；新增约束只能是待验证的弱正则。旧 ad-hoc JSON 的 cosine 有浮点误差且被同名覆盖，后续统一审计脚本替代它。

#### Timing 对照（当前进行中）

- `phase2_fullmouth_timing005_20261004`：从 full-mouth checkpoint 再训练 2 轮，方向约束权重 0.05，已完成；SHA `94904b4c013f9fd315dc2b0ac867fc5e3c8eeb12abf680a59f9a37b19bc2cc2b`。
- `phase2_fullmouth_timing000_20261004`：完全相同起点/源码/预算，方向约束权重 0，对照已启动。必须与此对照比较，不能将额外两轮训练收益全部归因于 timing loss。
- `content_timing_alignment` 默认权重 0；只约束发音通道速度方向，不匹配 B0 的绝对张口幅度。保留所有口部 residual 通道。
- 独立 probe 固定不参与训练。下一步完成 matched timing ablation 与 GT-probe sanity check；不提前扩大 loss 或启动长训练。

#### Timing 对照结果（2026-10-04，development only）

两次 continuation 都从同一个 full-mouth audio checkpoint、相同 2 epoch 预算、相同 validation manifest、seed 42 decode 和 3 个 noise seed 开始；唯一差异是 `content_timing_weight` 为 0 或 0.05。两者均 `test_loaded=false`。

| weight | checkpoint SHA | audio MBE | audio LBE | centered mouth RMS | mouth frame-displacement MSE | independent macro-F1 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `e17659536a6fcaaa2ec9d22f99403fe4d9f1c8690c3cd182583894545f5ab45` | 0.9024 | 0.4503 | 0.1274 / 0.1525 | 0.003179 | 0.1698 |
| 0.05 | `94904b4c013f9fd315dc2b0ac867fc5e3c8eeb12abf680a59f9a37b19bc2cc2b` | 0.8931 | 0.4387 | 0.1244 / 0.1525 | 0.002384 | 0.1830 |

三次生成的 mouth centered RMS ratios 分别为 0.971/0.973/0.980（weight 0）和 0.910/0.912/0.918（weight 0.05）。因此 0.05 的方向项改善了逐帧位移误差、MBE/LBE 和整体独立 F1，但没有增加嘴部动态能量；它不是强度控制器，也不能把 F1 推到 0.6。固定条件下两者的 `teacher_oracle` MBE 也由 0.6798 降到 0.6675，说明这是小幅整体优化而非单独嘴部修复。

每类 F1 仍显示 happy=0；weight 0.05 的 audio condition 为 neutral 0.109、angry 0.401、contempt 0.300、disgust 0.301、fear 0.085、happy 0、sad 0.184、surprise 0.085。oracle condition 同样 happy=0，说明独立 probe 低分不能归因于 audio classifier 一项。

本轮产物保存于 `final_experiment/evaluation/diagnostics/phase2_timing_ablation_20261004/`，包括两个 condition JSON、checkpoint provenance 和 evaluation JSON。后续默认保留 timing weight 0，除非更大规模实验确认其内容时序收益可重复；不把这两轮 continuation 作为最终模型。

### MEAD 强度/质量审计（2026-10-04）

- 训练 query 共 12,536 条，neutral=715（intensity 0），表达样本 11,821 条；validation query 共 1,367 条，neutral=80。
- 同一 speaker、sentence、emotion 的 intensity 1/2/3 完整组：训练 3,778 组（11,334 条），validation 413 组（1,239 条）。MEAD 三级强度配对覆盖充足，适合做 ordinal supervision；不能只依赖 batch 内随机碰巧出现的配对。
- Stage1 标记不是质量 mask：训练只有 715 条 neutral 全部 Stage1，表达样本另有 3,316 条 Stage1；validation 为 80 条 neutral 和 203 条表达 Stage1。当前 `articulation_scope=neutral` 仍只用 neutral B0，情感阶段使用全部 query。
- safe-DTW gated manifest 有 4,111 个 teacher-eligible pair；`mouth_event_gate=true` 仅 1,238 个，`amplitude_gate` 3,778 个。未通过 mouth event gate 的样本仍可给 upper-face teacher；训练代码只把 event-local mask 应用于 Stage1 target，不会从情感 renderer supervision 中排除低质量 paired motion。
- safe-DTW mouth amplitude ratio 中位数 0.999，jawOpen ratio 中位数 0.996；5% 分位数分别为 0.835 和 0.819。这说明多数配对幅度接近，但长尾质量问题足以污染强度排序。ordinal loss 应先使用 same-speaker/sentence/emotion 完整组，并绑定 finite/channel/valid mask；暂不把所有 low-quality pair 直接送入新 loss。
- 对 packed train/validation motion 做了只读目标能量审计：以每个 speaker 的 neutral anchor 为参考，在 mouth 14..40 上计算 uncentered RMS。训练中 intensity 1→2、2→3 的顺序正确率分别为 0.803/0.803，1→3 为 0.898；validation 分别为 0.823/0.668，1→3 为 0.864。jawOpen 单通道顺序更弱（train 0.603/0.612，validation 0.729/0.729），所以第一版 ordinal loss 应使用 whole-mouth energy，并把 target ordering 不一致的 pair 跳过，而不是强行约束 jawOpen 单调。

结论：强度配对数据本身不是稀缺原因；当前 renderer 只把 intensity 当分类/标量条件，且随机 batch 不提供成对约束。下一步先实现可审计的 pair index 和 mouth residual energy ordinal loss，再做小规模 smoke/validation，不立即长训。

### Phase 3：MEAD 强度监督

- 使用 intensity 1/2/3 的同 speaker、同 sentence 配对。
- 对 mouth/jaw energy、区域 RMS、峰值或开口相关统计加入 ordinal/pairwise loss：高强度不应系统性小于低强度。
- 检查 neutral=0 的处理，避免 neutral intensity CE 造成错误排序。

### Phase 4：情感一致性与类别平衡

- 保留独立 motion probe 作为冻结评估器。
- 训练 critic 与生成器更新解耦；增加 class-balanced emotion loss，报告每类 recall/F1，不只报 accuracy。
- 对 renderer endpoint 增加 class/global/intensity consistency，并单独记录 mouth-region semantic effect。

### Phase 5：质量加权与重训

- 统计 safe-DTW/event_local/source_observation mask 的覆盖率和每类有效帧数。
- teacher/audio 对低质量样本做 mask/权重或排除；保证 train/validation 同一规则。
- 先在 validation 完成 ablation 和 checkpoint 选择，再执行 sealed test 一次。

### Phase 6：最终评估

- 目标：MBE 约 `<=0.70`，独立 emotion macro-F1 约 `>=0.60`，并同时检查 LBE/LVE、嘴部 RMS ratio 和内容保持。
- 若未达到，按 failure mode 回退到 Phase 1/2 诊断，不用 sealed test 结果反向调参。

## 重要路径与命令线索

- 当前工作区：`D:\实验室项目\新实验\kinetalk_b0_residual_train`
- 训练入口：`scripts/train_full_staged.py`
- B0：`kinetalk_b0/models/model.py`
- 情感系统：`kinetalk_b0/models/neutral_affect.py`
- DiT renderer：`kinetalk_b0/models/dit.py`
- emotion probe：`kinetalk_b0/emotion_probe.py`、`scripts/fit_packed_emotion_probe.py`
- sealed evaluator：`scripts/evaluate_sealed_models.py`
- 当前视频：`final_experiment/evaluation/rendered/safe_dtw_b0full_gate_happy_20261003/comparison.mp4`
- 当前远端 run：`/root/autodl-tmp/kinetalk_final_20260922/checkpoints/kinetalk_safe_dtw_20261003_b0full_gate/`
- 远端 recipe 真值：该目录下的 `provenance.json`。

## 上下文恢复检查清单

1. 先读本文件的“当前状态”和最后一个 Phase 状态。
2. 再读 `progress.md` 最后 80 行，确认最近命令、失败和输出 SHA。
3. 读 `findings.md` 中最新诊断，避免重复审计。
4. 查看 `git status --short`，不要覆盖用户已有未提交修改。
5. 任何新实验先写入本文件的 Phase 日志/结果，再继续下一步。

## Phase 3 ordinal condition diagnostic (2026-10-04)

Remote ordinal checkpoint `/root/autodl-tmp/kinetalk_final_20260922/checkpoints/phase3_ordinal_mouth_w02_20261004/audio/final.pt` (SHA `f7d336638c8f1aafe411b2cc2a1feeeb1e89db1a411d0cc881684f2481c03671`) was evaluated on the fixed 1,367-clip validation split with the same independent probe, noise seed 42, shuffle seed 20261003, and 12 decode steps as the timing controls. No sealed/test data was loaded.

| condition | MBE | LBE | mouth RMS / GT | jawOpen RMS / GT | independent macro-F1 |
|---|---:|---:|---:|---:|---:|
| audio_pred | 0.9171 | 0.4633 | 0.1272 / 0.1525 | 0.1358 / 0.1195 | 0.1980 |
| teacher_oracle | 0.6852 | 0.3747 | 0.1529 / 0.1525 | 0.1462 / 0.1195 | 0.1644 |
| intensity_low (0) | 0.9235 | 0.4686 | 0.1243 / 0.1525 | 0.1265 / 0.1195 | 0.1693 |
| intensity_high (3) | 0.9122 | 0.4584 | 0.1285 / 0.1525 | 0.1448 / 0.1195 | 0.2233 |

Paired intensity deltas remain small (`jawOpen` about 0.013; mouth about 0.009), comparable to the timing controls. The ordinal objective is therefore wired and finite, but one epoch at weight 0.2 does not provide a strong extra amplitude controller. Keep this checkpoint as a development diagnostic only. The next isolated change is class-balanced generated-emotion consistency; do not combine it with quality filtering or sealed evaluation.

## 2026-10-04 class-balance and probe-critic diagnostics

A matched 2-epoch continuation from `phase2_fullmouth_timing000` compared no class balancing with tempered inverse-frequency emotion CE (`power=0.5`). Results were effectively unchanged: audio macro-F1 0.1585 vs 0.1563; MBE 0.9081 vs 0.9080. Class-frequency weighting is not the missing signal and is not selected.

A frozen train-only independent motion-probe critic was added as an explicit opt-in. Smoke with weight 0.2 produced probe CE 34.9 and gradient norm 562, so that weight is rejected as unsafe. A low-weight 0.005 continuation completed with finite gradients. Its original probe readout is audio macro-F1 0.8214 (MBE 0.9124), but a separately fitted hidden-64/seed-777 alternate probe reads macro-F1 0.6037; surprise F1 is 0. This is evidence of probe-specific optimization, not a valid 0.82 emotion result.

The alternate probe itself reaches validation macro-F1 0.641 on real motion and is bound to the same train manifest with no test selection. Next, train one matched continuation using the alternate probe as the frozen critic and evaluate with both probes. Keep only if both independent readouts improve without MBE/mouth regression. No current probe-critic checkpoint is promoted.

## 2026-10-04 probe ensemble cross-validation

The alternate-probe critic continuation and the original-probe critic continuation did not pass dual-probe validation. The original-probe critic checkpoint scored alternate F1 0.604; the alternate-probe critic checkpoint scored original F1 0.552.

A two-probe ensemble critic (original hidden-128 seed-42 plus alternate hidden-64 seed-777, mean CE, weight 0.005) completed a matched two-epoch continuation. Fixed validation condition diagnostics:

| condition | MBE | LBE | original probe F1 | alternate probe F1 |
|---|---:|---:|---:|---:|
| audio_pred | 0.9107 | 0.4560 | 0.7836 | 0.6767 |
| teacher_oracle | 0.6716 | 0.3646 | 0.7565 | 0.6488 |
| intensity_low | 0.9174 | 0.4592 | 0.7449 | 0.6282 |
| intensity_high | 0.9063 | 0.4527 | 0.7633 | 0.6647 |

This is more robust than a single probe but still misses the alternate 0.70 target and worsens MBE relative to the timing=0 control (0.9024). Do not promote this checkpoint. Next work must recover coefficient error and mouth fidelity while preserving dual-probe gains; no sealed test has been used.

## 2026-10-04 endpoint reconstruction control
- Added opt-in `--endpoint-weight`, a residual-scale-normalized Huber loss on the differentiable flow endpoint at sampled training time. It does not change residual support or mouth routing; default remains 0.
- Smoke with weight 0.05 passed with finite loss/gradients (`endpoint=0.0136`, grad norm 2.80). The matched validation continuation is the next isolated test, starting from timing000 without probe critic or class balancing.

## 2026-10-04 有效帧口型视频复核

使用现有 `scripts/render_dynamic_rig_comparison.py` 和固定 `arkit2.blend` 对 `mead_M025_happy_L3_005` 进行显示验证。旧输入保留 248 个 native frame，但只有 103 帧 valid；本轮通过 `--max-frames 103` 只渲染 valid prefix，避免 nearest-valid 末帧填充被误看作动态。

产物：
- `final_experiment/evaluation/rendered/phase2_fullmouth_happy_valid103_20261004/comparison.mp4`：GT、full-mouth continuation seed42、B0；
- `final_experiment/evaluation/rendered/phase2_fullmouth_audio_seeds_happy_valid103_20261004/comparison.mp4`：GT、full-mouth seed42/123/2026、B0；
- `final_experiment/evaluation/rendered/phase2_timing005_happy_valid103_20261004/comparison.mp4`：GT、timing005 seed42/123/2026、B0。

视频复核结论：full-mouth/timing005 在 GT 的张口峰值帧能明显打开，B0 仍偏小，确认情感口部 residual 已经能作用到 jawOpen；但生成序列仍有 seed 间差异和非峰值过冲。当前不能宣称口型完成，下一步应优先做范围/稳定性与内容时序约束，并用同一 validation clip 的 paired 曲线检查闭唇时机。

## 2026-10-04 F1 单片段复核与下一步判别

同一 `mead_M025_happy_L3_005` 的 independent train-only probe 结果：GT 被判为 happy（约 0.999），full-mouth 三个 audio seed 被判为 contempt（约 1.0），B0 被判为 neutral（约 0.998）。这解释了视觉与 F1 的表面矛盾：视频只展示局部帧的笑嘴，F1 使用全段 51 通道的 mean/std/quantile/速度统计，要求八类动作的整体协同分布正确。

当前 timing005 audio 全量 validation 中，happy 预测数为 0，contempt/surprise 过预测；GT validation 的同一 probe macro-F1 约 0.660，而生成约 0.183。故首要问题是 generated-motion distribution shift/class collapse。另一个可疑但可直接验证的因素是输出范围：full-mouth 三个 seed 的有效帧全部系数越界约 16%–22%，mouth 越界约 20%–28%，GT 无此问题。

在改训练前先做一个 train-free range ablation：对已保存 validation curves 的 raw、全通道 `[0,1]` clip、仅 mouth `[0,1]` clip 做同一 independent probe 与 MBE/LBE 评估。若 clip 后 F1 明显回升，优先修输出参数化/范围；若几乎不变，才把主要原因锁定为情感类别条件没有转成真实的全脸协同运动，再设计单一语义对齐改动。当前不直接追加更多 loss。

## 2026-10-04 train-free range ablation结果

对固定 validation 曲线做 raw、仅 mouth `[0,1]` clip、全通道 `[0,1]` clip：raw F1 约 `0.138–0.146`，mouth clip 约 `0.179–0.186`，全通道 clip 约 `0.183–0.192`；MBE/LBE 约由 `0.90/0.46` 降至 `0.88/0.43`。范围越界是重要误差源，但 clip 后 happy 仍为 0，因此 F1 低的主因仍是完整情感动作分布与类别协同未对齐真实 motion。

下一模型改动应保持单一因果：先把输出范围修正为训练/推理的一致参数化或保守输出投影，再只加入一个 class-specific affect alignment 目标；不同时叠加 ordinal、probe critic、statistics 和新约束。

## 2026-10-04 endpoint result and next isolated diagnostic

Endpoint-only two-epoch continuation completed: audio MBE/LBE/F1 0.908011/0.452838/0.158536; matched no-endpoint phase4-unbalanced control 0.908069/0.452917/0.158536. Do not compare its causal effect against timing000 directly (different training budget). No useful effect; do not select or increase weight. Control recipe directory is no longer present remotely, so the available comparison remains diagnostic. Raw outputs still have range issues.

Next: frozen-checkpoint sampling-gap diagnostic on timing000, without training: fixed noise, native masks, audio vs teacher oracle; Euler 12 vs 48 steps; one-step endpoint at fixed times 0/0.25/0.5/0.75/0.9. Interpolated endpoints with t>0 contain GT and are explicitly nondeployable. Evaluate raw and clipped outputs, GT sanity, original and alternate probes. No probe used as critic on this selected checkpoint. Determine whether solver accuracy or target-containing training states explain the internal/external score gap before altering training.

Correction: t-SNE and probe use the same statistics, so t-SNE is not independent evidence. Previous t-SNE-plane kNN values are transductive descriptive figures, not recognition benchmark scores. GT probe validation F1 around 0.64–0.66 limits assessor reliability, not a mathematical ceiling. Both ensemble critics are training assessors once used in loss; their scores cannot be called independent certification. SOTA requires same-protocol baselines, not arbitrary absolute thresholds.

## 2026-10-04 full flow sampling diagnostic

The frozen timing000 checkpoint was evaluated on all 1,367 validation clips with fixed noise seed 42, Euler 12/48, audio versus teacher-oracle affect, zero-noise, GT-assisted endpoints, clipping variants, and both train-only independent probes. Report: `final_experiment/evaluation/diagnostics/flow_sampling_timing000_full_20261004/report.json`.

The phrase “pure-noise inference” means only the initial residual state: `ResidualDiT.decode()` starts from a random tensor on the supported motion channels and integrates from `t=0` to `t=1`. Audio content, global affect, intensity, identity, and temporal audio condition are still passed at every vector-field evaluation. This is the standard flow-matching sampling path; it does not mean the model receives noise instead of audio.

Results rule out “just use more Euler steps” as the primary fix: audio Euler12/Euler48 are MBE/LBE `0.9024/0.4503` and `0.9178/0.4608`, with independent probe F1 `0.1698/0.1468` and `0.1706/0.1455` (original/alternate). Zero-noise is also not a fix (`MBE=0.9349`, F1 `0.1969/0.1538`). Teacher-oracle affect improves coefficient error (`MBE=0.6798` at Euler12) but F1 remains `0.1531/0.1476`; this confirms that global affect conditions influence the output while the generated motion distribution is still not aligned with real emotion statistics.

GT-assisted endpoint rows are diagnostic only because `flow()` receives the real target motion. At `t=.9`, audio MBE is `0.0917` and oracle MBE `0.0898`, yet F1 is only `0.1612/0.1228` and `0.1849/0.1416`. Therefore the low F1 is not explained by solver discretization alone, and the endpoint numbers must not be treated as deployable performance. They show that even small full-face/statistical deviations can move samples outside the probe's real-motion decision regions.

The independent probe itself has 27 of 306 feature scales clamped to `1e-4`; generated raw features are tens of normalized standard deviations away from GT on mean/quantile/velocity groups, partly because generated coefficients leave `[0,1]` on about 25.5% of observed values. Clipping raises audio F1 only to `0.1859/0.2197` and happy remains absent. Thus range correction is necessary but insufficient, and an isolated semantic alignment loss or sampler change cannot be selected from F1 alone yet. The next model experiment must first use paired coefficient/region metrics and a fixed, independent readout; do not tune toward the endpoint rows or the single-probe score.

### Sampler smoke (train-free)

The diagnostic now also implements midpoint and Heun integration and checks that its Euler path matches `ResidualDiT.decode()`. On a fixed 16-clip smoke at 12 steps, audio Euler/midpoint/Heun MBE is `0.8880/0.9042/0.9064`; F1 does not improve consistently. Oracle MBE is `0.5804/0.6040/0.6070`, again with no reliable F1 gain. Keep Euler for protocol comparability. A sampler swap is not the next correction.

## 2026-10-04 residual gain and assessor robustness (supersedes earlier causal claims)

Full-validation audio residual gain .75/1/1.25 yields MBE .9210/.9024/.9500 and original F1 .1721/.1698/.1679. Neither amplitude scaling nor sampler changes solve the issue; keep defaults.

Real-validation perturbation audit completed on 1,367 clips, with no fitting or sealed-test access. Both frozen probes score .6603/.6412 on clean GT. Adding Gaussian coefficient noise sigma .001 to GT reduces F1 to .1928/.2673; perturbing only four train-derived near-constant channels (cheekSquintLeft/Right, noseSneerLeft/Right) reduces it to .1728/.3005. Each selected channel has all six feature scales at the 1e-4 floor. Thus a large part of the F1 collapse can arise from assessor sensitivity, even without an emotion change. Earlier statements that clipping's failure proves missing category semantics were too strong: clipping does not remove artificial motion on near-constant channels.

This does not clear the generator: residual features outside these dimensions remain shifted, and its MBE/LBE/range issues are real. Preserve original metrics and do not optimize against this perturbation audit. Before more model changes, fit a separately named stability probe using only real train motion and the preregistered rule: retain feature dimensions whose real-train std exceeds 1.0001e-4. Use the original capacities/seeds (128/42 and 64/777), unchanged 50-epoch budget and real-validation earliest-best selection. Freeze and audit clean/perturbed GT before replaying generated features. No generated score may choose mask, normalization, training budget or seed. This is auxiliary diagnosis, not a replacement F1 protocol or critic loss.

### Stable-feature probe result and next experiment

The fixed rule retains 279/306 features. Clean validation F1 is .6640/.6459, matching original reliability. With sigma .001 GT noise it is .5568/.5274, substantially more robust; sigma .005 still collapses (.0830/.0919), so the new readout also remains sensitive to artificial frame jitter. Constant-only perturbations have exactly zero effect, as expected from the fixed mask. Probes are frozen and never used as loss/critic.

Frozen timing000 audio Euler12: stable F1 raw .1772/.1753; clip_all .4424/.4134. Happy F1 becomes .360/.288 after clipping, with fear still .057/.147. Original F1 stays .1698/.1468 raw and .1859/.2197 clipped. Clipping MBE/LBE .8860/.4297 vs raw .9024/.4503. `render_dynamic_rig_comparison.py:81` already clips for display, explaining why rendered smiles look better than raw F1. Both evaluator fragility and generator range/dynamics mismatch are present; neither alone explains everything.

Next isolated experiment: an opt-in [0,1] output projection applied consistently to the reconstructed flow endpoint used by existing semantic objectives and to generated final motion. Keep vector-field loss, sampler, support, noise, architecture parameters and all existing weights identical. Default remains unprojected for old checkpoint compatibility. No critic/statistics/ordinal/endpoint/timing addition. Preserve raw pre-projection motion for inspection. Matched two-epoch continuations from timing000; compare both branches with and without the same final clip, original+stable probes, region/velocity errors and mouth range. An improvement from projection alone must not be mislabeled as learned emotion improvement.

### Projection continuation result: rejected

Both runs completed from timing000 with seed47, batch16, 2 epochs/1568 optimizer steps. Source/data/warm-checkpoint hashes match; args differ only by bounded_output. New remote artifacts use `/root/kinetalk_projection_20261004` (autodl-tmp disk nearly full). Local artifacts and paired bootstrap comparison: `final_experiment/evaluation/diagnostics/phase6_projection_matched_20261004/`.

| trained branch / final policy | MBE | LBE | original / alternate F1 | stable128 / stable64 F1 |
|---|---:|---:|---:|---:|
| control raw | .9081 | .4529 | .1586 / .1396 | .1528 / .1752 |
| control clip | .8916 | .4319 | .1783 / .2427 | .4740 / .4492 |
| bounded raw (pre-projection) | .9107 | .4574 | .0656 / .0959 | .1583 / .1699 |
| bounded clip | .8950 | .4371 | .0385 / .0585 | .2134 / .2226 |

Equal-final-clip paired bootstrap95% intervals: bounded minus control MBE [.00275,.00386], LBE [.00490,.00558]; stable F1 differences [-.288,-.233] and [-.253,-.199]. Mouth centered RMS .06773 vs .06786 and displacement MSE .002530 vs .002553: amplitude is essentially retained, but error/emotion worsen. Do not promote or raise projection training weights. The switch remains default-off and is retained only to reproduce this rejected ablation. Training-time clipping changes how the existing semantic critic receives gradients; it is not equivalent to a safe final inference clamp.

Next: frozen train-only emotion/intensity centroids as a diagnostic of audio-vs-motion global-code mismatch, with audio-predicted labels for deployable cases and true validation labels only for labeled oracle rows. Keep native temporal audio, identity, noise, renderer and all mouth support fixed; fit centroids only on real train motion and do not train/update any module. This tests whether pooled teacher codes contain hard-to-predict motion variation and whether semantic prototypes help. Do not treat label-oracle rows as audio-only performance. Final inference projection remains a separate evaluation policy; no new loss stack.

### Global prototype full result

1367validation seed42, frozen timing000, centroids only on12536 real train clips. Same final clip: audio MBE/LBE .8860/.4297, stableF1 .4424/.4134; audio-predicted class prototype .8726/.4307, stableF1 .5044/.4858; audio-predicted class/intensity .8714/.4320, stableF1 .4697/.4509. True-label class oracle stableF1 .5877/.5643 (not deployable). Class prototype mouth RMS .1347 vs audio .1249 (GT .1525); displacement MSE worsens .002601->.002781, and original F1 worsens .1859/.2197->.1727/.2046. No default selected. This isolates a benefit in categorical global conditioning, but does not establish a complete lip/semantic correction. Class means explain .4972 of train teacher-code variance; remaining variation may include legitimate continuous expression, not all nuisance.

Next fixed diagnostic: real-train ridge mapping frozen audio global64 -> frozen teacher global64, standardized using train input mean/std with1e-4 floor, lambda=.001 (predeclared, no sweep). Audio temporal/intensity/content/identity/noise remain unchanged. This preserves sample-specific code variation while testing systematic coordinate mismatch. No fine-tuning, new loss, label-oracle input, generated fitting or test selection. Compare full-validation code MSE and same-noise output/clip metrics; do not select based only on stable F1.

### 2026-10-05 global calibration complete and next single change

Fixed train ridge map reduces train global MSE .3844->.3212, but full validation .3.2133->3.2549 worsens (roughly8.4x train baseline). Train audio/teacher mean-square norms5.224/5.383 vs validation3.917/7.129. Same final-clip MBE .8860->.8853, LBE .4297->.4286, stable F1 .4424/.4134->.4420/.4147; originalF1 changes mixed. Do not select/sweep ridge. This indicates a continuous global-code generalization problem rather than merely a fixed linear coordinate mismatch. Do not label the cause specifically identity leakage without controlled evidence.

Next preregistered two-epoch match from timing000: keep default unbounded training, renderer/audio trainability, seed47/batch16/all weights/intensity/u_a/identity/content fixed; change only the target of the existing .5 global MSE distillation from per-clip frozen teacher global to frozen real-TRAIN emotion class mean. Existing losses unchanged, no additional prototype-CE/critic or constraints. Frozen class means computed after B0/identity cache on train only, with all8 classes required, saved with recipe/manifest. No prototype selection from validation. Warm-start audio-only experiment restriction. Default `--global-distill-target clip`; new `class-prototype` is opt-in. 19 targeted tests pass; smoke and full matched runs next. This tests reducing hard-to-predict continuous targets while leaving explicit intensity and native audio timing intact; it may lose useful within-class variability and must pass motion/diversity metrics.

### 2026-10-05 target-only continuation completed; provisional benefit
Both12536-train/2epoch/1568step continuations and1367validation diagnostics completed. Shared source/data/warm hashes match; args differ only global_distill_target. Equal-final-clip MBE/LBE .891630/.431854 -> .880332/.422237; original F1 .178335/.242093 -> .179711/.268576; stable auxiliaryF1 .474020/.449852 -> .498508/.468667. Paired bootstrap confirms coefficient benefit, but first original and second stable F1 CIs include zero. No promotion or target achievement. Next replay existing three-seed curves, allclasses/speakers/intensities/regions, jaw timing and diversity; fixed previously used clips for visual comparison. Distinct noise allocation in native curves versus batch-trimmed diagnostics must remain explicitly separated.

### 2026-10-05 confirmatory result: do not select class-prototype training
Training seeds48/49 full12536*2epochs and1367validation finished with sole target difference. Same final clip: seed48 control/prototype MBE .878764/.881068, stableF1 .480073/.460151 -> .430065/.439858; seed49 MBE .891381/.892777, stableF1 .427349/.391329 -> .436763/.433750. Original first probe declines in both; alternate improves only48. LBE and mouth displacement improve more consistently but do not satisfy joint criteria. Preserve default clip target; no weight or seed sweep.

Next single change removes existing generated-endpoint teacher consistency (emotion-consistency-weight .2 ->0) using its existing flag, without code/model changes or new losses. Teacher readout of generated motion is around .89 whereas real validation GT teacher accuracy .673, so it may reward classifier-compatible artifacts; this is a hypothesis, not yet proven. Keep global per-clip distillation .5, flow matching, audio semantic loss, mouth support/u_a/intensity/noise unchanged. Train seeds47/48/49 fromtiming000 for2epochs/batch16, compare to exact existing controls with source/data/warm/seed equality and sole consistency-weight arg difference. No probes in training; fixed final checkpoint and all seeds, no test. Do not promote on teacher accuracy alone.

Curve audit initial display export assertion assumed valid-prefix masks for all emotions; actual masks can have leading invalid frames/gaps. Scoring correctly used valid masks but report was not yet written. Fixed export to longest contiguous native-valid interval, tie earliest, maintaining native timestamps; all valid frames still scored. Scores now persist before export. Rerun read-only audit PID7702 in curve_audit_v2; original incomplete directory preserved, no training repeated.

### 2026-10-05 existing endpoint consistency removal result: reject
Three fixed trainingseeds47/48/49 complete, identical source/data/warm/2epoch/batch16; only emotion_consistency_weight .2->0. Allfourprobes decline in everyseed with paired95%CIs below0. Means same finalclip MBE .887258->.888117, originalF1 .203014/.253359->.071138/.109557, stableF1 .460481/.433777->.339544/.328099; LBE slightmean improvement .430275->.428132 does not rescue semantics. Keep.2, do not increase it or stack losses. Next read-only decomposition of same-input three stochastic draws into conditional mean error and seed variance; compareGT/B0, velocity and native jaw. Averaged draws are diagnostic only, not selected inference. No test.

### 2026-10-05 finite-draw attribution and next receiver check
Timing000 saved3draws full1367: finalclip jaw speedRMS1.89xGT, finite-draw velocityvariance fraction meanperclip .4101; three-draw meanjaw correlation .3851vsB0 .3985, velocity .1488vsB0 .1659. Randomness contributes but shared speech timing is also poor. Means are diagnostic only, no averaging/gain/zero-noise deployment. h0 already has position, u_a already native-rate. New frozenonlyu_a mean/reverse/shuffle/zero comparison preserves allglobal/content/style/noise and validmasks;4probes/regions/pairedRMS, full1367 and oldbaseline parity. PID1933, no newtraining.

### 2026-10-05 u_a order result and next isolated architecture trial
Full1367 frozenreceiver: same-run finalclip audio jawcorr .31072, reverse .24635, shuffle .22714. Shuffling increasesjaw displacementMSE .004709->.011750 andMBE .886043->.905561. Native temporal condition is used. Static reducesjawRMS .08105->.05718 (GT .07450) andstableF1 .44182/.41480->.37195/.38533; zero reducesstableF1 to.25136/.25099. Keepu_a. Reverse can improve stableF1 despite worse timing, confirming F1alone cannot certifyphonetics.

Exact clip/label/GTfeature parity against olddiagnostic passes, but generatedcross-script strictnumeric parity fails (maxfeature3.43e-4, maxclipMBE4.31e-5, F1minorchanges); cause not isolated. New within-run conditions are validmatchedcomparisons, not newmodel improvements. Preserve failure and donot relaxassertions to call exact reproduction.

Next single architecture change: opt-in zero-initialized learned3frame Conv1d residual onDiT motiontokens beforeattention. It introduces directlocal-motion interaction for noisyresidualstates; currentDiT hasattention andpositionalinformation inh0, but no localmotion convolution. No addedloss, smoothing/filter afteroutput, gain, support/mouthmask orcondition change. Zero-start preserveswarm behavior; exactmissingnewweight guard foroldcheckpoint. Defaultdisabled. Sameprior/noise, allfullmouthchannels retained. Match3trainingseeds47/48/49,2epochs/batch16/finalepoch/fromtiming000, sharedupdatedsource inbothbranches. Requireallprobes/region/jaw/amplitude andrepeatability; rejectonregression. No test orpromotionbeforefullvalidation.

### 2026-10-05 phase9 completed: reject temporal adapter

All six runs complete; same source/data/warm/support and sole renderer_temporal_adapter flag difference. Three-seed mean, same final clip: control/adapter MBE .887252/.888203, LBE .430268/.432680, mouth displacement MSE .00256784/.00223728; original F1 .202728/.253864 vs .205920/.260176; stable F1 .461032/.434015 vs .449752/.428017. Displacement improvement12.9% does not establish correct timing or emotion. Both coefficient errors increase for every seed with paired95% CIs excluding zero; stable F1 improves only seed47 and declines48/49. Default stays off. No new selected model, no goals achieved; all jobs finished. Results: `final_experiment/evaluation/diagnostics/phase9_temporal_adapter_20261005/replicate_summary.json`.

Next work: read-only GT/B0/generated static-bias and timing decomposition, plus precise B0 cross-emotion supervision/DTW quality-mask trace. Use saved curves; do not rerun the rejected adapter or scan filters. Distinguish teacher/global generalization from B0 speech timing before preregistering another isolated change. No sealed-test selection.


### 2026-10-05 B0 supervision audit active
Read-only audit scripts/audit_b0_supervision.py added; local and remote2 targeted lag/mask tests pass. Uses provenance-bound timing000 saved curves and hash-matched safe-pair artifacts, no training/probes/test. Reconstructs actual native mouth-mask coverage, follows warm-checkpoint lineage, compares GT/B0/three draws by emotion/speaker/intensity and fixed -5..5-frame jaw lag profiles with identical contiguous anchors across lags. Lag profiles are diagnostic only; no shift/gain selection or inference changes. Remote fresh /root/kinetalk_b0_supervision_20261005, copied frozen phase9 source (old source untouched), PID4775, launch.json/audit.log, output audit/report.json. Current audit pending. Recipe actual articulation coverage3298=715 neutral+2583 pairs; requested clip_count715 was misleading as a total.


### 2026-10-05 B0 audit complete; phase10 preregistration
PID4775 completed, report/NPZ/launch downloaded to b0_supervision_20261005. Artifact/manifest hashes match original recipe and lineage reaches original four-stage b0full_gate. Coverage715 exact native neutral +2583 eligible pairs (297 full-gate,2286 partial local-gate); paired mouth valid frames148140/267521=55.375%, no pair with zero retained mouth frames. Neutral validation80 clips: B0 jaw centered correlation .44148, three generated draws .28519/.27133/.28148; B0 displacement MSE .00174621 vs draws .00322737/.00319809/.00304404. Common-anchor -5..5-frame profiles peak at zero for B0 and every draw; no global fixed delay inferred or applied. Emotional GT comparisons alone are insufficient to assess neutral B0 amplitude.
Existing full-validation oracle reused, no generation repeated: true-motion global+intensity (audio timing held fixed) clipped MBE .659867 vs audio .886042, LBE .345671 vs .429659, but displacement MSE .00270539 vs .00260116. GT-informed oracle is not deployable and does not certify phonetic accuracy. Brow mean-feature MSE .0553983 -> .0157411. This supports condition prediction as a material static-expression bottleneck alongside receiver timing errors.
Next isolated trial: freeze existing renderer weights during warm-start audio-only continuation, training only audio student. Current audio stage jointly updates student and renderer; freezing tests whether a moving receiver worsens transfer/generalization. This causal hypothesis is unproven. No new loss, weight changes, support mask, output filter or noise changes. Default joint training remains. Warm timing000, same shared source/data/fullmouth, seeds47/48/49,2epochs/batch16/final epoch; args differ only freeze_renderer_audio. Existing per-clip .5 global MSE and .2 endpoint consistency retained; probes never in training. Full1367 original+stable F1/geometry and neutral timing must improve jointly and replicate before promotion; fail -> reject, no sweep. Local gradient/frozen-weight checks and remote smoke precede launch. No sealed test.


### 2026-10-05 phase10 implementation, verified smoke and formal launch
Added --freeze-renderer-audio defaultFalse, restricted to isolated warm-start audio-stage/ablationnone. audio_parameter_groups changes optimizer membership only; no no_grad around fixed renderer, same evaluation/dropout mode and RNG.19 targeted local tests passed, remote2 gradient/frozen checks passed. Real remote smoke loaded8train/8validation; existing metadata selection used4train/2steps, not a performance estimate. Finite first loss .210382/gradient2.12652; all84renderer state tensors exactly matchwarm and26audio state tensors changed. Smoke complete.
Fresh shared source /root/kinetalk_frozen_renderer_20261005/code, actual formal full12536train/1367validation, seeds47/48/49, eachcontrol/frozen_renderer; wrappers5515/5516,5517/5518,5519/5520. Allsix states training and base-cache progress observed. Only freeze_renderer_audio differs; default baseline remains joint. Future checkpoints not selected.
One-shot pipeline postprocessor PID6370 waits for the six existing jobs, then checks sole args/source/data/warm and exact frozen system weights, replays four frozen probes in fixed original128/original64/stable128/stable64 order, runs existing2000 paired bootstraps, and writes per-seed comparison.json plus replicate_summary.json. No fitting/test/automaticpromotion. Track postprocess_state.json/log; failures preserved. Frozen oracle stableF1 .41645/.35569 vs audio .44242/.41340 despite oracle MBE .65987: recoveringstaticgeometryalone is not proof of independent semantic/timing correctness.
Next restore: read CURRENT first, inspect states without relaunch; after completion download reports/features/provenance/summary, check full-validation multimetrics; if preliminarygate passes extend native curve audit sole-arg guard for frozen_renderer and review allclasses/speakers/intensities, neutral mouthtiming, amplitude anddiversity before promotion. If it fails, record rejection and keepdefaultjoint. No sealed report edits.

Phase10 all six training arms reached2epochs/1568steps; full validation evaluation ongoing. Postprocessor import failure was repaired with standalone compare.py outside frozen training source. New postprocessor PID7290/postprocess_retry1.log; original PID6370/log preserved. Do not restart training.


### 2026-10-05 phase10 completed: reject frozen-renderer continuation
Allsix wrappers5515–5520 and repaired one-shot postprocessor7290 complete. Final full12536train/2epochs/1568steps,1367validation; sharedsource/data/warm, solefreeze_renderer_audio False->True; all frozen-system tensors exactly matchwarm, student updates verified. Local phase10_frozen_renderer_20261005 contains replicate_summary, comparisons, eachreport/features/provenance/complete/state.
Three-seed finalclip means control/frozen: MBE .887259/.888053; LBE .430274/.429358; displacement MSE .00256784/.00265037 (+3.21%). Original F1 .202999/.253360 vs .203372/.232490; stable F1 .460481/.433700 vs .456273/.407653. Everyseed displacement error increases with paired95%CI>0. Stable benefits onlyseed49, firstoriginal benefit only47. Jointgate fails: do not select, defaultjoint retained; freeze flag remainsdefaultoff for reproducibility, no longer-training orseed/weight sweep. This2epochwarm trial does not rule out all possible staged distillation designs.
No newmodel selected, goalsnot achieved, no new training stillrunning. Currentbest reference remains timing000; never substitute oracle .65987MBE for inference or call stableprobe a replacementprotocol. Native B0 diagnostic establishes neutral residual timing damage without globaldelay, but B0itself is also imperfect.
Next work not launched: isolate held-outspeaker continuous condition/generalization or B0train/validation gap (andexisting same-protocol validationbaselines ifavailable). Do not repeat phase7–10 or stack losses. Keep mouth amplitude freedom and testsealed boundary. Read CURRENT before anysource audit aftercompaction.

Frozen affect-generalization audit launched PID1264 at /root/kinetalk_affect_generalization_20261005; local2tests pass; fixedtiming000 alltrain/validation code andneutralB0 extraction; launch.json/audit.log. Nooptimizer ornewmodel. Do not duplicate.


### 2026-10-05 affect generalization completed; phase11 preregistration
Frozen timing000 all12536train/1367validation complete; source/data/checkpoint bound. Local affect_generalization_20261005 report+perclipcodes downloaded. Global MSE train .384369 vs validation3.213282; speakermean fractions4.70% vs31.53%, jointspeaker/emotion/intensity30.72% vs69.34% (descriptive attribution, no inferred leakage or deployable correction). NeutralB0jaw train715corr .70190 vs validation80 .44148; MSE .003354/.006533. ConditionreadoutF1 audio .98591train/.84560val vs teacher .80824/.63644. Crosshead applying frozen motionteacher classifier toaudio global stillvalF1 .80400, so severe category-coordinate mismatch not supported.
Trainteacher agrees realemotion labels10066/12536=80.30%; disagrees2470=19.70%, whileaudio iscorrect on2367ofthose. All8classes retain >61.7% and>=648correctteacher observations. Current globaldistill MSE supervises everyclip regardless of this labelconflict. Validationteacherwrongcaseshave3.184globalMSE vs correct3.227: conflict does NOT explain the fullgeneralizationgap.
Next singleexperiment tests reliability-gated existingglobaldistill, not a claimed fix: existing .5perclip MSE multiplied by fixedtrainteacher-argmax==realtrainlabel mask, retaining fullbatch denominator so acceptedclip weights do not increase. No confidence threshold/sweep/newloss; teacherfrozen, botharms use sameflow/semantic/.2endpointconsistency/clipglobaltargets/u_a/identity/noise/fullmouth andsameall12536trainmembership. Badteacher globaltargets skip onlydistill, not flow or labeled emotion/intensity supervision. Gate calculated onTRAIN only and allclasscoverage recorded. Defaultall retained. Warmtiming000 audioonly, seeds47/48/49,2epochs/batch16/finalcheckpoint/full1367 andfourfrozenprobes; preregisteredjoint geometry/timing/semantic replicate gate beforeadoption. No sealedtest.


### 2026-10-05 phase11 implementation and formal launch
Defaultall global MSE unchanged, optionalteacher-agreement multiplies perclipMSE by detached frozenTRAINteacher argmax agreement with realTRAINlabel, fullbatchnormalization; allrejected batchzero gradient valid. Gateallowedonlywarm isolatedclip-targetaudio stage. No newloss, weight increase, modality/mouthmask or flowtarget changes.19targeted localchecks pass, remote3gate checks pass; actualsmoke4train2steps finite loss .556713mean/grad2.18679, retainedfraction.75. FrozenB0/identity/teacher equality to warm confirmed; bothaudio andrenderer update.
Freshphase11 /root/kinetalk_teacher_agreement_20261005/code frozen shared snapshot; fixedseeds47/48/49,2epochs/batch16/1568steps/fulltrain12536/fullvalidation1367,sole global_distill_gate difference. One-shot finalizer has strictsource/data/warm/arg/frozenmodules guards andfixedfourprobe/pairedbootstrap outputs, no automaticpromotion. launch/state/logsaved; currenttraining active, waitforvalidation before any conclusion.

Phase11 durable identifiers: wrappers2100/2101(seed47),2102/2103(seed48),2104/2105(seed49), finalizer2106. Verified optimizer batch progress and finite gradients on all6. While waiting, checked training-mode hypothesis: reference configured dropout=0, all renderer MHA/MLP probabilities0, so inactive dropout does not explain gap; no change made. Do not repeat. Initial log append had an unmatched triple quote and failed before writes; fixed literal.


### 2026-10-05 phase11 complete: small geometric benefit, no adoption
Allsix wrappers2100–2105 andfinalizer2106complete; fixedsame-source/data/warm/fullmembership, sole global_distill_gate all->teacher-agreement. AllfrozenB0/identity/teacher matchwarm.2epochs/1568steps/full12536train and1367validation. Localphase11_teacher_agreement_20261005 report/features/provenance/complete/state/comparison/summary downloaded.
Three-seed finalclip means: MBE .887248->.886433, LBE .430270->.430011, displacement MSE .00256781->.00255738 (-.406%). OriginalF1 .202978/.253830->.198520/.256667; stableF1 .461332/.433743->.465081/.442713. MBE lowersallseeds butseed47CIincludes0; displacementlowersallseeds with95%CIs<0. Stable128gainonly47, stable64gain47/49, original128seed49declines significantly. Jointsemantic repeatability gate fails; rejectdefaultpromotion. Defaultgateall andsame timing000reference retained; no sweepingclassconfidence/weight or selectingseed47. Badteacherlabel conflicts are real but skipping onlyglobaltargets is not sufficient. No nexttraininglaunched; goalsstillunmet.
Next mechanism priority: B0 andcontinuousconditiongeneralization. Savedfrozenconditions allow correlation to independentneutralreference styles or fixedspeaker-heldout prediction tests without reextracting12536clips; onlyiftrain-heldout evidence supports it consideridentity-conditionedglobal prediction. Do not alter outputgain/sampling/timingmask or claimlabelrecognition absent. Semanticcategorycrosshead .804 alreadyreadable; targetamplitude andspeakerdependence aredifferent. Allpriorfailedtrials remainreproducible defaultoff.


### 2026-10-05 reference predictability check complete and compact handoff
Fixedridge .001 with fit-onlystandardization/std1e-4; reusedrealcodes andneutralidentitybaseline, audio-predictedcategory only. Fivefixed TRAIN speakerfolds measureextra mapping only (frozenencoderalreadyseenallTRAINspeakers). Audioonlycode mapcrossfoldMSE .336421, addingreference×category .382262; allfivefolds worse. Alltrainfit realvalidation3.255055 vs3.370405. No sweep/generation/inference change/adoption or nonlinearidentityadapter; lackofbenefit islimitedtothisfixedmapping. Saved reference_code_predictability_20261005/report.json+predictedcodes.
Phase11complete/noadoption andall artifacts verified/downloaded. CURRENT_OPTIMIZATION_STATE.md rewrittenascompactentry; detailedhistory remains mainplan/findings/progress, so no analysisfactsremoved. No active newtraining orselectedcandidate; goalsunmet. Next independentdesign must address proven B0 andcontinuousmotionconditiongeneralization, not repeatphase7–11 or forceF1 withcritic.

### 2026-10-05 B0 mask-normalization audit preregistration
Continue with B0 before another affect loss trial. Source ResidualTCN applies ordinary GroupNorm across channels/time before masking each block, and TemporalBackbone does not zero its input projection before the first convolution. Thus missing frames inside true native length can influence valid activations. This is a code fact, not yet evidence that it explains generalization or F1. First count within-true-length missing frames in actual TRAIN/validation (all clips and neutral), then if present run a frozen B0-only counterfactual with masked normalization/input masking. Same weights/native positions/masks, no compression/re-timing, no generated GT conditions, no training/test. Require exact unchanged all-valid path, padding/missing-value invariance and finite gradients before real audit. Compare neutral jaw timing/amplitude/displacement and region error; no promotion or full-system mixing because B0 h0/identity/residual coordinates also change. If no material issue, record and reject rather than launch an unsupported retrain.

### 2026-10-05 B0 audit decision and Phase12 preregistration
Actual packed masks: every clip has one missing frame inside its native span (one TRAIN clip has two). Missing fractions train .9054%, validation .9810%; neutral .9282%/.9792%. This does not establish a material explanation for the train/validation gap. Do not implement masked-GN counterfactual or silently change existing B0 coordinates now.
Phase12 isolates standard Stage1 training dropout, justified by train/validation neutral jaw correlation .7019/.4415 and current dropout0. Fixed probability0.1 versus0.0 control, no sweep. Both arms explicitly Stage1 train-mode during optimization and eval-mode during caches/scoring, no change to any other modules/config/dropout. Only existing Stage1 nn.Dropout and MHA dropout probabilities change; no input time masking, feature noise, new loss, gain, output clipping, mouth support or target policy. Same warm timing000 system, actual3298 eligible TRAIN clips, same715 neutral+2583 gated native targets, existing Huber+.1velocity, LR3e-4, AdamW1e-5, batch16,4epochs/final only, seeds47/48/49. All Stage1 parameters trained; B0/identity/teacher/renderer architecture shapes unchanged. Record frozen non-Stage1/audio tensor equality and sole-argument/source/data/warm match.
B0-only gate on all80 validation neutral clips: joint improvement in jaw correlation, mouth/ jaw displacement MSE and absolute mouth error with replicated direction; RMS must not collapse, closure frames reviewed if gate passes. Record all715 TRAIN neutral controls and speaker breakdown; no oracle emotional GT as neutral targets. No original/stable emotion F1 claimed at this stage, no test access. The old identity and residual models use the old B0 coordinates, so this checkpoint is diagnostic-only: a passing B0 requires a matched later-stage adaptation and full validation four-probe/geometry review before adoption. Local mode/gradient/state-dict tests and real remote smoke precede formal six-run launch.

### 2026-10-05 fixed semantic-direction error diagnostic (while Phase12 runs)
Reuse frozen12536/1367codes, extract only fixed motion-teacher emotion/intensity linear head weights from boundreference. Center each head across outputclasses (softmax-invariant commonlogit direction), SVD ofcombinedweights infloat64, fixedrelative ranktolerance1e-10. Orthogonalrowspace P exactly preserves BOTH teacher-head probabilityreadouts onanyglobalcode; decompose audio-teacher squared errorintoPerrorand(I-P)error. TRAINteachermean anchors projectedcodes fordescriptivevarianceonly. No newmappingfit, labelprototypes, perqueryGTinference, generation, testortraining; noadapterselected. This willtestwhethercontinuousconditionerrorislargelyoutsidefixedclassifierdirections, but discarded directionsmustNOTautomaticallybecallednuisance:renderer mayneedthemfornoncategoricalexpression. Save findingsbeforeanyprojectiontrialdecision.

### 2026-10-05 Phase12 completed: reject fixed B0 dropout
All6armsandfinalizercomplete;4epochs/828steps,715train/80validationneutral,source/data/warm/safe/frozenweightsverified. Three-seedvalidationfinalclip control/dropout mouthMSE .01270565/.01284185, mouthdisplacementMSE .00117130/.00118543 (+1.21%); jawcorr .438899/.436686, jawdisplacementMSE .00180911/.00183583(+1.48%). Everyseed displacementworsens andjawcorrslightlydeclines;corrspeakerCIsincludezero, so do notclaimsignificantcorrelationdrop. Bothcontinuationsimprovetrainfitbutvalidationreferencewarmjawcorr.44148andjawMSE.006533becomecontrol.43890/.007608, reinforcingoverfit; notaloneproofofaspecificcause. NoB0candidateadopted,no downstreamadaptationstarted. OriginaldefaultNoneandreferencecheckpointretained. Alldownloadedphase12_stage1_dropout_20261005,withsummaries/provenance/completions/nativecurves. Currentnoactivephase12trainers.

### 2026-10-05 Phase13 coordinate-distillation necessity preregistration
Fixedteacherheaderrorattributionshows91.78%validationglobalMSEinsemantic/intensityrowspace;classificationargmaxremainsreadable(.804crosshead),butcontinuousmargin/amplitudenotmatched. Causalnecessitytest: existingperclipglobalMSE weight .5->0 only, while frozenmotionteacher, labeledaudioemotion/intensityCE, generatedmotionteacherconsistency .2(allglobalcos/emotion/intensitycomponents), flow/noise/u_a/identity/B0/fullmouthallremain. This differsfromno_teacherablation whichremovesteacherdesign, andfromrejectedclassprototype/agreementfilter whichstillforcecoordinateMSE. It doesnotassumemismatchisclassifier-invisiblenoise, anddoesnotproveMSEharmbeforecomparison. NoKL/prototypes/newloss/critic/weightscan.
Samewarmtiming000,isolatedjointaudio+renderer2epochs/batch16/full12536train/1568steps,finalonly,seeds47/48/49;soleglobal_distill_weight .5vs0. Default.5unchanged. Same1367validation/Euler12/noiseseed42,raw/clip,original128/64andstable128/64frozenreal-trainprobes,matchedgeometry/displacementand2000pairedbootstrap. Requirejointrepeatablesemantic/geometrybenefit;passingpreliminarygatestillrequiresnativejaw/class/speaker/intensity/diversityauditbeforeadoption. No testread. Localdefaultalgebra/gradientandCLIisolationchecks,thenrealremote smoke,frozenB0/identity/teacherverificationbeforelaunch. No automaticpromotion.

### 2026-10-05 Phase13 completed: dynamic benefit, no adoption
Allsix wrappers6453–6458 andfinalizer6459 complete,full12536train/1568steps/1367validation,four frozenrealtrainprobes,matchedsolearg/source/data/warm/frozenB0/identity/teacher verified. Three-seedfinalclipcontrol/noMSE: MBE .887255/.885541,LBE .430272/.428144,displacement .00256785/.00247556(-3.59%); originalF1 .202676/.253124→.205185/.251481,stable .460030/.433935→.478992/.451577. DynamicMSEallseed improveswithCI<0,stableF1directionsallimprovebutseed49CIcontains0;original128seed49andoriginal64seed47CI<0. Seed48MBE/LBE directionworsens(CIcontains0). Preregisteredjointgatefails:keepdefaultdistill.5/reference, noautomaticpromotion/weightscan/seedpick. This confirmsa limiteddynamicbenefit ofremovingcoordinateMSEunder2epochwarmtrial, notfullsolutionoruniquecausalroot. Reports/features/provenance/complete/state/summary/comparisonsdownloadedphase13_no_global_distill_20261005. No newtrainstillrunning; goalsnotachieved.
Latestfrozenintensityaudit:audiointensityF1train .8296/val.4515;realvalL2/L3meancontinuousprediction1.375/1.898 vs train2.065/2.833,teacherneutralval1.042. No scalarcalibrationorordinal lossselected. Next mechanismnotyetimplemented: inspectdecoupledintensitypredictionrepresentationfromglobalmotioncoordinatesandhighspeakerholdoutbias, preserveemotion/stylemouthamplitude; clearstructuralcontrolbeforetraining. ReadCURRENTaftercompaction, no reextracts orrepeatphase12/13.


### 2026-10-05 Phase14 preregistration: frozen intensity/global factorial diagnostic
Recovered CURRENT plus planning files; Phase12/13 rejected, no live training, remote GPU idle and /root7.9GB free. Source confirms audio intensity is a linear classifier of the same 64D global code; renderer receives both global and its expected intensity scalar. A separate intensity head is not yet justified by label-prediction bias alone.
Before structural training, fixed frozen timing000 validation-only five-condition audit: original audio; teacher scalar only; exact MEAD label scalar only; teacher global only; teacher global+scalar. Hold u_a/content/B0/identity/noise and Euler12 fixed, same1367validation/batch16/noise42, four frozen real-TRAIN probes, raw/clip, region geometry/timing/amplitude and emotion/intensity/speaker groups. No fitting/test, no inference calibration or promotion; teacher/label conditions explicitly GT-informed diagnostics. Scalar-vs-global crossed interventions distinguish their contributions and interaction; label scalar is not a guaranteed continuous motion amplitude target. Preserve clips/features/provenance and paired bootstrap, no sweeping scalars.
If scalar-only corrections have negligible/negative benefit while global-only improves static geometry, do not train an isolated intensity branch as the next repair. If scalar benefits jointly, investigate independent pooled acoustic intensity representation with unchanged label CE and a matched control. This is a diagnostic decision rule, not a metric-based deployment choice.
Recovery command errors: rg used nonexistent models/ then corrected to kinetalk_b0/models; PowerShell literal .codex-finalizer/*.py unsupported by rg, use directory plus glob. No files changed by these failed reads. Skill session-catchup reports no unsynced context.


### 2026-10-05 Phase14 complete: scalar-only repair rejected
Full1367 frozen validation five-condition diagnostic completed PID8093; source/warm/manifest/fourprobe/8clip smoke and frozen output bindings checked. Report/perclip/state/launch/smoke downloaded phase14_intensity_factorial_20261005. No training or new default.
Finalclip audio MBE/LBE .886043/.429660, dynamicMSE .00260116, fourF1 .185877/.219977/.441821/.414801. Teacher scalar only .881039/.426183, dynamic .00261140; labelscalar .880680/.428082, dynamic .00260652, stableF1 .41877/.39920. Bothscalar interventions significantly increase displacement error; labelscalar significantly lowers bothstableF1. Do not train anisolatedintensitybranch as the nextrepair.
Teacher globalonly MBE .659836/LBE .345109 vs teacherboth .659867/.345671; static jawmeanbias .0060226->.0028928, browmeanbias .0553982->.0162693. Globalonlydynamic .00269251 and fourF1 .13518/.18036/.42153/.37645, so matching staticshape doesnotfix independentreadout/timing. Nearly alloracleMBEbenefit residesglobalcode, scalarrepair islimited. Oraclevalues are GT-informed, neverdeployable.
Readoutscore minorcrossrun differences alreadydocumented (batchGPU floating behavior); use same-runpair comparisons. GT frozenprobes F1 .66034/.64118/.66395/.64589. Audio mean jawq90-q10 .19637 vsGT.17528 and correlation .31072: do notclaim universal mouth amplitudecompression; emotion/channel/speakerbias andtiming differ. Next evaluate prior vs TRAIN residual variation as a hypothesis, not source-noise rootcause proof. Current isotropic source becomes .25 rawresidualstd in every channel; Stage1 .05targetscalefloor is unrelated. No covariance/newloss implementedyet.

AfterPhase14 scalar-only failure, preregistered TRAIN-only frozen residualspread audit. Full12536nativeobservations minusfrozenB0/independentTRAINneutralidentity; exactperchannelpopulationstd, counts, means, currentisotropicrawsource std.25. No validationmotionread/generatedfitting/modelchange. scripts/audit_flow_source_scale.py and2masked-statistics tests;2localpass. Fresh /root/kinetalk_source_spread_20261005 containscopyofphase14source,fit.log/train_source_stats.json. Scale skew is hypothesized capacity/SNR issue, notprovenrootcause; if large, considersingle priorcovariance trainingcontrol, neveruntrainedzero-noise orposthocoutputgain. Missing/constantchannelsretainpositiveGaussianstd,no mouthmask.


### 2026-10-05 Phase15 preregistration: TRAIN diagonal source covariance
TRAIN-only frozen12536clips/1372104observedframes/channel stats complete (2local/2remote maskedmomenttests). Rawresidualstd jawOpen .14630,mouthsmile .19912/.19588,brows .24142/.23091/.32049,cheekPuff .0002323,cheekSquint .000009, noseSneer .000005-.000006; currentisotropicrawsource std.25. Severeprior/targetspreadskewisreal, but notcausalproof. This is not the Stage1 .05floor (unrelated), outputscaling, variance-based supportmask, or previouslyrejected untrained zero-noise.
Next singlechange: standard source N(0,I) versus N(0,diag(s_c^2)), s_c=frozen TRAIN populationstd(motion-B0-independentidentity)/.25, normalizedpositive floor1e-4 onlyfornondegeneracy, unobservedsupportexcludedchannels retain1. No mean shift, cap/gain tuning, mouthmask, newloss, projection, orprobecritic. Gaussianremainsnonsingularallobservedchannels. Source scaling applies before trainingx_t/velocitytarget and at rolloutinitialstate, withsame underlying whitenoise; rendereroutputs/endpoints neverrescaled. Savefixed52scalesincfg andTRAINstats hash/warmmanifest/sourceguards. Controlsamefileandargs except source mode; bothsamefullmouthtargets/u_a/audio/teacher/.5MSE/.2consistency.
Warmtiming000 audio-stagejointcontinuation, threefixedseeds47/48/49,2epochs/batch16/full12536/1568steps/finalonly. Full1367fixedbatch16/Euler12/noise42/rawclip/fourfrozenprobes/fixed2000pairedbootstrap. Requirejointreplicatedgeometry/displacement/semanticbenefit beforeconsideringcandidate; nativejaw/class/speaker/intensity/diversityreview beforeadoption. Local defaultnumeric parity, bothpathnoise/targetalgebra, mask/finitegradient, configstrictcheckpoint roundtrip, sourcefilebindings andCLIisolation; realremote smoke exactfrozenB0/identity/teacherandchangedaudio/renderer. No sealedtest, outputoverride orautomaticpromotion. Untrainedrescalednoise notusedasresult.


### 2026-10-05 Phase15 verified formal launch
Correctedimplicit-noise9checks pass locally/remotely; freshsmoke_retry1 complete4samples/2steps firstloss.238463/grad2.419279,84renderer/26audiotensorschanged; allnonrenderersystemidenticalwarm. Source/cfg/stdfile/fullmouth/manifest guards pass. Verifier initiallyreadtotal_steps frominference_only final.pt (field intentionallyabsent); correctedstrictstepcheck toactualepoch_complete log, no trainingrestart. Originalsmoke andfailurespreserved. Training-source fingerprint fixedaftercorrection; launcher verifiesit/statshash beforedispatch.
FormalPhase15 root /root/kinetalk_source_spread_20261005,soleflow_source_noise standardvs train-residual-std usingbothsamefixedstatsfile. Sixseed47/48/49 warmjointaudio-stage2epochs/batch16/full12536+1367validation,finalonly; launcherregisteredjobsandone-shotpostprocessor. Checklaunch.json forPID thenexistingstate, neverduplicate. Fourfrozenrealtrainprobes+fixed2000pairedbootstrap; sourcevectorconfigstrictload, allothermodulefreeze andsoleargguards. No source-scaledmodeladopted; defaults unchanged. Currentnewtrainingrunning; objectiveunmet. Waitcomplete/downloadreports/features/provenance/summary beforedecision; passingcandidate stillneedsnativejaw/diversity/class/speaker/intensity review.


### 2026-10-05 Phase15 full validation complete: replicated F1 gain, no adoption
All6wrappers9929–9934/finalizer9935 complete. Three-seedclipmeans control/source: MBE .887251/.888017,LBE .430269/.432589, mouthdisplacement .00256787/.00266068(+3.61%); originalF1 .202691/.253625 -> .255245/.327616, stable .460048/.434124 -> .527568/.505934. Everyprobe/everyseed F1paired95%CI>0; thisisindependentreadoutgain withoutprobecritic, but not>.7 norSOTA. EveryseeddisplacementregresseswithCI>0; seed48 MBE/LBE andseed49 LBE regresssignificantly. Jointgatefails; no modelpromotion, standardpriorremainsdefault. Do notselectseed47 orscanstd/floor.
All fullreport/features/provenance/complete/state/comparisons/summary downloadedphase15_source_spread_20261005. Native3draw/region/allclass/speaker/intensityaudit dispatched as specificunresolvedmouthtradeoff despitepreliminarygatefailure; no retrain/newfitting/defaultchanges. Freshnative_audit_state.json/native_audit.log in same root, scriptoutsidefrozen training snapshot. Nextrestorecheckstate anddownloadnative reports; cannotinferwhichsourcechannelscauseF1benefituntilregion/timingreview. Likely nextisolateddesign mustpreservephonetic dynamics whileimprovinggenerateddistribution, notscalarcalibration ortrainingcritic.


### 2026-10-05 Phase15 native and tradeoff completed; no running tasks
Allnative3seed/3draw audits PID11025 complete and downloaded (3reports/3npz/24fixedemotionrenderinputs/state/launch), all6formal files verifiedcomplete. NativefourF1 .200324/.248031/.458105/.433186 -> .245990/.317126/.515900/.496013; jawcorr .297506->.303797, range .196547->.198559, neutraljawcorr .271629->.277761. Mouthcenteredcorr .287141->.287774, butdisplacement .00258796->.00268350, browsdisplacement .00028368->.00034192. No collapse, no generalmotion improvementclaim.
Exactfinite3draw partition downloadedmouth_tradeoff.json/npz: mouthvelocitydelta .00009554144, stochasticvelocityvariance .00006322810 (66.18%), meandrawvelocityerror .00003231334 (33.82%). Jaw totaldrawvariance lowers whilevelocityvariance rises; source-channelmarginal matching alone fails temporal statistics. This is finite-draw attribution, not population noise orproof anAR sourcewillfixit; three-drawmean neverinference. No modeladopted, standard source/timing000 retained, goalsunmet.
Nextcandidate NOTimplementedorlaunched: TRAINwithin-clip/demeaned residual adjacent-frame covariance audit, then onlyif temporalcorrelation supported matched nondegenerate temporallycorrelated Gaussian source withsamechannelstd, fullmouthsupport, identicaltraining/rollout/native-gap transform; rho0 strict compatibility; reference standard anddiagonalcontrols,3seeds/fullvalidation. No extra loss/meaninfer/outputfilter/weightscan. Source mean andglobalcontinuous-condition generalization remain unresolved, so temporalprior alone cannotbeclaimedcomplete solution.
CURRENT/phase15README/native_replicate_summary updated; allwork thisround complete, currentno trainer/postprocessor running. Nextresume readCURRENT then latestplan; don'trelaunchPhase14/15 orrediscoverstats. Detailed priorpage archivedcurrent_before_phase15_20261005.md.


### 2026-10-05 user reaffirmed paper/SOTA objective
User goal remains overall metric superiority over the declared comparison methods for a publishable paper. No candidate/default accepted merely because one probe or one metric improves. Phase15 supplies replicated semantic-readout gains and an exact temporal tradeoff mechanism, not a paper-level final improvement.
Next priorities: (1) build/verify a validation-only, same-protocol baseline gap matrix for VOCA-core, FaceFormer, EmoTalk-core, FaceDiffuser versus current reference/candidate; distinguish ARKit/shared-audio adaptations and frozen KineTalk conditions from original-method replication, lock split/input information/masks/native clock/rig/metric definitions and comparable training/selection budgets. Local final tables remain publication-gated; historical sealed reports may exist but must not be used to tune current runs. Do not claim test is never previously accessed. New candidate has no completed full comparative LVE/EVE/FDD table yet.
(2) TRAIN within-clip residual lag-stat audit and, if supported, matched nondegenerate temporal Gaussian source trial retaining Phase15 diagonal marginal spread, standard/diagonal controls, fullmouth support, same training/rollout transform and three-seed multimetrics. Not implemented/launched.
(3) continuous global-condition generalization: audio class recognition already .8456, but validation global-coordinate MSE3.213 vs train.384; teacher-global GT-informed MBE.660 is diagnostic evidence of a material static-shape bottleneck, not an attainable-audio guarantee. Inspect acoustic temporal distribution/pooling representation on TRAIN speaker-held-out folds before selecting a new structure; no scalar-only, prototype, ridge/reference-map repeats or loss stacking.
(4) B0 speaker generalization/phonetic fidelity: neutral TRAIN/validation jawcorr .702/.442 remains a separate issue. Retain only audited paired supervision and affect/style influence on mouth amplitude; no mouth masking or unmotivated training changes.
Once a candidate passes joint validation, evaluate locked model and baselines under the final test protocol, report BS MBE/LBE/FDD and fixed-rig vertex LVE/EVE/FDD with exact definitions, emotion metrics, robust perclass visual review/user study where appropriate, and component ablations. t-SNE is descriptive, not a replacement for comparative results. MBE~.7 and F1>.7 remain optimization goals; SOTA is a relative same-protocol claim, not certified by these absolute thresholds. Do not promise allmetrics best before evidence. All last-round jobs are complete; current exchange is progress/planning, no new training started.


### 2026-10-05 Phase16 preregistration: temporal source statistics and validation baselines
User authorized continued paper/SOTA work. Restored CURRENT and planning files; allPhase15 taskscomplete, GPUidle,/root4.8GBfree. Baselineadaptation contract sayssharedaudio/ARKitadaptations, notoriginalpaperreproduction; sealedreport metricswillnotbereadfortuning.
Read-only temporal audit frozen timing000 realTRAIN12536clips. Compute perclip/perchannelmean usingobservednativeframes then accumulate lag1 crossmoment/left/rightenergy anddifferenceenergy onlyforadjacentvalidnativeframes. Perchannel normalizedrho=cross/sqrt(leftenergy*rightenergy), fixedclip to[-1+1e-4,1-1e-4] onlytokeepnondegenerateinnovation; absent/zeroenergychannelrho0. FullTRAINspeaker/emotion breakdown testsdirection, notvalidationselection. RetainPhase15populationsourceSTD unchanged andwarm/statshashbinding. No model/noiseinferencechangebeforeauditresult. MissingframesmustresetfutureARchain; trainingandrolloutusewholeframevalidmask, neverqueryGTchannelmasktochooseprior. No staticclipmeans intocorrelationfit.
Inparallel metadata-only inventoryofbaselinecheckpoint/validation native_predictions/provenance/complete, excludingsealed/testdirectories. Nextsameprotocolvalidation comparison will useexistingnativepredictions ifbound; otherwise fixedcheckpoint inference only. Do notreadhistoricaltestmetrics or silentlymixreportdefinitions. No newtraininglaunchedyet.


### 2026-10-05 Phase16 audit complete and matched AR trial preregistration
TRAIN audit complete12536clips;1,359,567adjacentobservations/channel, within-clipdemeanedlagrho .82728–.97187; jaw .85691,smile .91695/.91582. All8emotiongroups positivehighcorrelation. Two localandtwo remote statistics tests pass. Artifacts phase16_temporal_source_20261005/train_lag_stats/report.json/per_clip.npz; remote same under/root/kinetalk_temporal_source_20261005. Remote python bare name missing; corrected to explicit/root/miniconda3/bin/python, no duplicateaudit.
Next single source mechanism: standard/control, TRAINdiagonal/source_spread, sameSTD+TRAINrho/source_temporal. Same warmtiming000,three seeds47/48/49,2epochs/batch16/full12536TRAIN/1367validation,finalonly,allloss/.5distill/.2consistencyunchanged. Primary comparison ARvsdiagonal,secondary ARvsstandard. No automaticpromotion: require allthree seeds originaltwoF1 pairedCI>0 vsstandard, neitheroriginalF1 significantlyworse vsdiagonal; mouthdisplacement pairedCI<0 vsdiagonal and no significantregression vsstandard; MBE/LBE no significantregression vsstandard. Fourprobes rawandclipreported; stableF1 neverreplacesoriginal. Candidatepassingthisscreenstillneedsnative3draw/groupamplitude/correlation/diversityandfixedriggeometryreview.
AR normalizedstate stationary initialization at each nativevalidsegment, z_t=rho*zprev+sqrt(1-rho^2)*white_t; invalidframezero/resetschain; samepositiveSTD in trainingflowandrolloutinitialstate. Wholeframevalidmask only, neverqueryGTchannelmaskforARprior. No outputfilter/averageinfer/newloss/weightscan/seedselection. Rho0 strictlegacycompatibility, finitegrad/padding/gap/observationalmaskindependence/stationaryvariance/checkpointconfig/Eulerreplaytests before realsmoke. Changes are opt-in; defaultsandhistoricalkeys preserved. Stats loader mustbind warm/manifest/STDfile/support/full12536/no-validation and recompute rho from savedmoments.
Baseline metadata confirms completed bound12536/1367validation predictionsforVOCA-core80epoch,EmoTalk-core80epoch,FaceFormer100epoch; baseline adaptations are notofficialend-to-endreproductions. FaceDiffuserdirectoryisfacediffuser_full_20260922, inspectitsvalidationmetadata next. No historicaltestmetrics read.


### 2026-10-05 Phase16 smoke verified and formal launch
Real4sample/2steps ARsmoke passed: loss.42778283,grad3.05466437,84rendererand26audio tensors changed; allother system parameters exactlymatchfixedwarm. ConfigSTD/rhoandTRAINstats/sourcemanifest hashesverified;local35checks/remote23checks passed. STDhashfc9645c7...,lagreporthash0b5b81c7.... Formal3arms(controlstandard/source_spreaddiagonal/source_temporalAR)×3seeds47/48/49 queued by24GBGPUlimit3concurrent,2epochs/batch16/full12536/1367,finalonly. Root/root/kinetalk_temporal_source_20261005;launch.json/mode_state.json/postprocess_state.json are recoveryentry,neverrelaunch. No defaultpromotion. PrimaryARvsdiagonal andsecondaryARvsstandard,allfourfrozenprobesandrawclip/pairedbootstrap/geometry-displacementjointscreen preregistered. No newloss/mouthmask/outputfilter. Waitfullresults while validationbaselinegapwork proceeds.

Validationbaselinegap task launchedCPU-only/root/kinetalk_validation_gap_20261005(state.json/run.log). Reuses all1367nativepredictions:VOCA-core80epoch,FaceFormer100epoch,EmoTalk-core80epoch;FaceDiffuser100epoch/3draws has frozenKineTalkglobal/identity/intensityconditions;currenttiming0003draws. Fixed632vertexrigandfourTRAINprobes,rawandclippolicies,coefficientMBE/LBE/FDD,mean/max/squaredvertexLVE/EVE/FDD,mouthdisplacementandjawcorr/range,allmethodsoneprotocol. Scriptbuild_validation_gap_table.py and1nativecontracttest passed. Checkpoint/nativeclock/target/manifestcontract required. Historicalbaselineartifacts lackcheckpointsha: deterministicmethodsverifyfirstoriginalbatchCPUvssavedGPU(maxabs<=1e-3),FaceDiffuser recordsbindinglimitation pendingfullresampling. Timing000curveshascomplete.json checksum binding. This is validationdevelopmentgap evidence,notfinalpapertable;no test metricsornew fitting.


### 2026-10-05 Phase16 full comparison complete; next pooling audit preregistration
Nineformalrunsandcomparisonscomplete,allsource/solearg/frozenparameterbindingspassed. Meansstandard/diagonal/AR: MBE.887258/.888016/.908438,LBE.430272/.432583/.444092,mouthdisplacement.002567814/.002660729/.001720026. ARreducesdisplacement33.02%vsstandard/35.36%vsdiagonal,all3seedCI<0,butMBEandLBEregressall3seedCI>0;originalF1.202964/.253380 -> AR.233189/.309935,belowdiagonal.255627/.327671. OriginalF1128benefitvsstandardnotreplicatedseed49;bothstableprobeslowerthan diagonalallseeds. Jointscreenfailsall3,noadoption/noheadselection. Native3drawdiagonalvsARaudit stillactive;baselinegapcomplete.
Validationgap baselines(raw/clip and allvertexvariants saved),clip: VOCA-coreMBE.7467/LBE.3324/originalF1.4977/.4845;FaceFormer1.2968/.5443/.1238/.1760;EmoTalk-core.7450/.3315/.5919/.6862;FaceDiffuser.7927/.3231/.1545/.2701;currenttiming000.8847/.4313/.1825/.2155. All1367clips/shared1540audio/native masks/fixed632rig/4probes. Baselinegeometryandexpressiongap is material,notonlyclassifierartifact. Localvalidation_gap_20261005 report/npz/README/complete/state downloaded;noofficialbenchmarkSOTAclaim.
Next fixeddiagnostic(no generatortraining): reusefrozen timing000audioandcachedrealteacher64codes from/root/kinetalk_affect_generalization_20261005/audit. IMPORTANTactualstagedstudentisSlowStateAffect(hidden128,4blocks),notunusedsystem.audio_encoder. Capturepre-global-headmean128andlastblockhiddenstats without reimplementingnetwork; compare mean128 vs mean128+maskedpopulationstd128. Twofixedridge .001/stdstandardizationfloor1e-4, fiveTRAINspeaker_id%5 mappingfolds;frozen encoderalreadytrainedallTRAINspeakers,folds evaluate extra-readoutgeneralization only. FitallTRAIN and report validation separately,no validationfit/labelsasinputs/no sweep. Cachedtarget/code/ids/manifest/warmbinding required, meanprojectionmustreproducecachedaudio64withinrounding. No extraGTmotion/B0/identityextraction. Ifstd doesnotimproveeveryTRAINspeakerfold, do nottrainmean+stdglobalbranchmerelyfromvalidationbenefit. No deployableridge/prototype/neutralmaprepeat. Recordnormalizedmeanbiasandsemantic/intensityheadrowspaceerror ifboundheadsavailable; totalMSEandfoldconsistencyprimary.

APIpatchedmodule22remotechecks(includingCUDAmodel+CPUnoisebit-exactcompatibility)pass;frozenformaltrainingcodeunchanged. Nativeartifactdownloadfirststoppedonassumeddisgusted/surprised filenames;actualdeclaredclassesdisgust/surprise,readreportandresumedonlymissingartifacts. Removedonlynewlycreated0bytefailedlocaldisgusted.npz,allhistoricaldataunchanged. Poolingauditcomplete: mean-only TRAINout-of-foldMSE.331713964 vsmean+std.3334265,ALL5speakerfoldsworse;validation3.323732 vs3.319506 negligibleandworseoriginal3.213. Preregisteredgatefails,doNOTtrainstd-poolingbranchorscanridge. GPUcodecapturematchesoriginalcachedaudio64bit-exact(train/validationmaxdifference0);no teacher/B0re-extraction. Downloadingpoolingreport/featuresandremainingnativeartifacts.


### 2026-10-05 round complete (restore CURRENT)
Phase16nineformalruns/threefullnativeaudits, five-methodsameprotocolvalidationgap, frozenaudio-poolingpredictabilityauditALLcomplete; no activejobs/noadoption. Allninecomplete/report/checkpointhashesmatched, all3native1367/24renderinputschecked, gapreport+NPZhashesverifiedlocally. ARvelocityerrorbenefitdoesnotimprovejawcorrelationandgeometry; stdpoolingfailsall5TRAINfolds, no newtraining. Fullresults/limitations/nextprioritiesnowinCURRENTandphase16README. Remote/rootfree.394GB, no largeexperimentuntilstorageplanned. Usergoals SOTA/MBE~.7/originalF1>.7unmet. No sealedtest tuning. Currentdefault unchanged.

### 2026-10-05 Phase17 budget chain verified; raw-emotion diagnostic preregistered
Fresh12stages + fullmouth2audio + timing0002audio warm SHA chain matches. B0:12epochs/19788steps/3298samples; teacher12/75216steps; audio-global16/100288steps; renderer+audio-temporal28/175504steps. Baselines80-100epochs; no equivalence of epochs/losses/step cost inferred. Small64clipdevelopment shows audioheadaccuracy improves .875->.921875 butbrowMSE .045353->.049889; no unconditional more-training conclusion. ActualEmoTalk ignoresprosody4 but preservesemotion832frame memory/256local features versusstudenthidden128/global64/u_a64. Report+metadata savedtraining_budget_20261005.
NextfixedTRAINspeakerfold diagnostic: cachedhiddenmean128 vs hiddenmean128+pretrainedemotionmean768; teacher64target reused,lambda .001/std floor1e-4,all5folds+totalMSEmustimprove. FrozenencoderhasseenallTRAINspeakers; readout-only evidence. AllTRAINfitvalidationseparate; no sweep/generation/defaultchange. Maskedrawmean finitevalid-only check andbothscripts compile passed. Serverroot376MiB/autodl1.1GiB/shm60GiB; noactivejobs. Needverifiedbackup before large persistentwrites. No sealedreport read.

Phase17 rawemotionmean diagnostic complete PID6248: hiddenmean128 TRAINfold MSE.331713964 -> hidden+raw768 .345320243; all5foldsworse. Validation3.323732->3.315553 stillworseoriginal3.213. No globalrawmean bypass implementation/no lambda sweep. Next distinct preregistered physicaltarget check reusesrawmeans andhiddenmeans without repeatedaudio extraction: realclipobservedmean52 minusindependentenrollmentanchor, fixed51TRAINsupport, samefiveTRAINspeakerfolds/lambda.001/std1e-4. Gate allfoldtotalMSEimprove plusoverallmouth/brow no regression; validationseparate. This addresses targetcoordinate vs physicalshape; it cannot retroactively claim theteacher-code trial passed. Diagnosticmaskcheck passed; caughtregionindicesbeforelaunch, correctedbrows41:46/eyes0:14. Missingconstants.py searchlogged; exactregions fromexistingnativeaudit.
Two completed rejectedroots (frozen_renderer/no_global_distill) verifiednoactiveprocess andcompletestate; recoverytar+perfileSHAmanifest nowin/dev/shm/kinetalk_recovery_20261005. No historicalfile removedyet; nextdownload andverifyeverymemberbefore storage reclamation.

### 2026-10-05 Phase17 complete; Phase18 identity-budget preregistration
Teacher-code raw-mean trial and physical-mean trial bothfailedall5TRAINfolds; no bypassadopted. Physicalmean MSE .006695457->.006921893,mouth .004020796->.004192108,brows .024452508->.0253462; validation .011132035->.011798391. Complete reports/NPZ downloaded. Teacher-code validation reduction alone isnot justification. Two complete rejectedexperimentroots archived6.12GB/858files,tarSHA+everyfile+dirmanifest locallyverified; before reclaim remoteoriginals unchangedreverified. Recoverablelocal fulltar andremoteRAMcopies/pointers retained. Rootfree6.1GB, baseline/data/currentwarm/conditions untouched.
Cachedneutralquerymean identitydiagnostic (no inferencechange):715TRAIN allbiasMSE .0121933->.0111696, mouth .000281455->.000257853,jaw .000519647->.000331649;80validation all .0118050->.0110211,mouth .00757676->.00765705,jaw .00197909->.00220194. Doesnotproveunderfit, but132originalidentitysteps andlittlebaselinefit warrant a small reference-onlybudgettrial.
Phase18 fixed12vs120ADDITIONALidentityepochs,seed47/48/49,batch2,originalAdamWlr1e-4/wd1e-5/clip1,exactoriginalsymmetricreferenceMSE+.05contrast. Only22TRAINcomplementaryenrollmentpairsfit; allquerydata evaluationonly. B0,teacher,audio,renderer/support/sourcefrozen; no generation/newloss/newarchitecture. ReferenceB0extractiononly, cachedqueryGT/B0meansreused. Sharedfirst12trajectorymustbehash-exact;frozenstatesmustmatchwarm. Gate everyseed TRAIN/developmentcrossrefMSEimproves andall/mouth/brow/jaw neutralquerymeansnonregressing inTRAIN/validation. Final120only,noepochselection/sweep. Passingrequiresfullseparatedownstreamretraining/evaluationbeforepromotion, becauseidentityinterfacechangesresidualteachercoordinates. Currentdefaultunchanged.

### 2026-10-05 Phase18 complete; Phase19 native-target B0 preregistration
Sixfixedreference-onlyidentityrunscomplete. Everyfrozenstate unchanged andshort/longfirst12hash-exact. Additional12->120epochs three-seedmean crossrefTRAIN MSE .00824018->.00155632,development .00958919->.00421686. Neutralquery allMSEtrain .00898162->.00170843/validation .00938158->.00503696;validationmouth .00777020->.00656686 butTRAINmouth .000252222->.000328690(+30.32%) andvalidationbrows .00193709->.00212587(+9.75%). All3jointgatesfail; no identitypromotion/downstreamadaptation. Identityunderfit real forglobalstaticshape butlongerfit isnot a universalfix. All6states/report/summary downloadedandhashverified.
NextPhase19isolatedB0 target/membershiptrial responds tooriginalneutralized-supervision concern: control3298=715native-neutral+2583approvedsafeDTWneutralteachers, candidateall12536TRAIN eachwithitsOWNsynchronousnativeGT(no cross-emotion pair lookup). ExistingB0 outputs fullneutral_output_indices, so candidate is an AUDIO MOTION BASE, notsemanticallyneutral. No newmodule/loss/mouthmask/identitychange. Bothsamewarmtiming000,sameAdamWlr3e-4/wd1e-5/clip1,batch16,seeds47/48/49 andEXACT1568optimizerupdates. Control8partialpasses vs candidate2fullpasses; actualclipinputs24990 vs25072(.328%difference from partialbatches), explicitlyreported. Nativeinvalidframes andchannelobservations remainmasked. Finalonly, fourfrozenprobes/rawclip/nativegeometry timing afterboth; B0scoresdiagnosticonly until separate fullteacher/audioadaptation.
Addedopt-in--articulation-updates onlyforisolatedwarmB0paperstage. None retainsoriginalepochs/batches/RNG; exactfinalpartialepoch recordedcorrectly. 9budgettests passed (bothscopesexact1568,partialepoch/resume/exhaustion/typeguards/defaultpath). Candidatealreadyusesexistingall-emotionsscope andno-safeDTW option,controlretainsoriginalteacherqualitygate. Notscheduleduntilremote2-update realdata smoke verifiesfinitegradientandonlyB0changes. No sealedtest/epoch/seedselection. Formalgate preregistration: all3seeds nativevalidation fullcoefficient/mouth MSE lower, neutralmouth displacement nonregressing/jaw centeredcorrelationnonregressing; fixedoriginalprobes reportedbutnotforcedwithcritic. Ifpasses, nativeperemotion/speaker/intensityauditbeforeanyfulladaptation; neverreplacefullgeneratorbasedonB0alone.

### 2026-10-05 Phase19 smoke + formal launch
31localtargetedbudget/runner/resume/dropoutchecks pass,9remotebudgetchecks pass. RealnativeGTsmoke4TRAIN/2updatesfinite completed;101B0statetensors changed; ALL other system andaudio bit-exactwarm. Smoke manifest/provenance/explicitupdatebudgetpass. Fresh/root/kinetalk_native_b0_target_20261005 frozencommoncode,threepairworkers7679/7680/7681 seeds47/48/49 sequentialcontrol/native,max3GPUtrainingconcurrent. Exact1568updates each;control3298approvedsupervision8partialpasses/nativedirectall12536twofullpasses. Finalcurvesstored,stage1only. Finalizer7682 waitsall6 thenonefullnative1367validationauditofgeometry/timing/fourfrozenprobes,715TRAINneutral timing. No defaultpromotion. Neverduplicate launch;canonicallaunch.json/state. Auditprobes classifystage1baseonly, nevercallthisfullgeneratorF1.
Phase18all6localidentitystatehashes checked againstreport; no identitycandidate adopted. All priorreports/data preserved; completedrejectedtwooldroots archivedintofullverifiedlocaltars/remoteRAMtemporarycopies andpointerfiles;restoreinstructionsinremote_archives/20261005/README.md.

### 2026-10-05 Phase19 complete: positive B0 evidence, no generator promotion
All6formaltrainruns and3fullnativeaudits complete; everyseedpassesregisteredgeometry/timinggate. Three-seedclip B0 control->native mouthMSE .0233636->.0148477(-36.45%), mouthdisplacement .00138916->.000974281(-29.87%), mouthcorr .264962->.438176;neutraljawcorr .443933->.472812,neutralmouthdisp .00119174->.000778675. FullcoeffMSE .0376162->.0331078, B0MBE1.336885->1.254975/LBE.512225->.393427. B0originalF1 .03556/.03441->.15224/.12072 is NOT fullgeneratorF1. Everyseedall8emotions/all4intensitiesmouthMSElower. Target/masks/times/channel tensorsbit-exact acrossarms;onlyB0changes,allotherstatesfrozen. Sixfinal/provenance/complete/summary downloading;allreports/perclip/summaryalreadydownloaded.
Amplitude not fullysolved: fullmouthq90-q10 .05479->.05997 vsGT.09199;neutraljawrange .11193->.09810 vsGT.11188. No fullgeneratorpromotion, noSOTA/F1>.7claim. Target+membership+effectiveframemaskcoverage changedtogether; same1568updatesbutcontrol24990vsnative25072inputclips. This is causal recipe evidence, notisolated proofneutralizationalonecausedgap.
Next fixedsame-membership study preregisteredconceptually: reuse3298approvedclipselection,samesafeobservationmasks,samebatchordering,seeds47/48/49,1568updates,warm/source/optimizer; onlyreplacealignedneutralteacher VALUES withthatclip'sownnativeGT VALUES. ReuseexactcontrolfromPhase19afterinitialtrajectory/sourceparity verification. No additionaldata/frames/weights/newloss/mouthmask;thisseparatesde-neutralizationfromcoverage. Implementation/launchNOTdoneyet. Aftercausalauditonlythenconsiderfullidentity/teacher/audioadaptation; newB0changestheteacher residualcoordinate system. Nativebasecontent768F1stilllow;fullaudioaffectpathremainsneeded.

Phase19all6downloadedfinalcheckpointSHAand1568updatessummaryverified;local_checkpoint_verification.json saved. Noactivejobs. Followupsinglefactor CLI implementedlocally only: --articulation-target-values safe-teacher(default)/native-query, nativequeryallowedonlywarmisolatedsafeDTWB0withupdatebudget. Afterexisting safe_target_batch producesexactoriginalmask, replacesONLYtarget_motion values withown b.motion; clipselection/mask/scales/loss/batchorderunchanged.31localrunner/budget/resume/dropoutchecks passagain. Phase19frozensource untouched. Phase20NOTlaunched; next validate real2stepdefaultparityvsPhase19/targetmask invariants, thenmatched3seedcontrols(nativevalue differs onlyoneflag). Do not prematurelyreuseoldcontrolwithoutparityproof. Recoveryentry updated. An accidentalfunctions.wait with exec_commandsession60291 returnedcellnotfound; actualwrite_stdin hadalreadyconfirmedtransfercomplete,alllocalfilesverified; no transfer repeated.

### 2026-10-05 Phase20 complete
See CURRENT and phase20README. Same3298safeclips/observations/1568updates; actualsample/observation/noise hashes equal; targetvaluesonlyswitch. Sixarms/three1367validation+715neutralTRAINaudits done. MouthMSE.0231544->.0179866(-22.32%),mouthdisplacement-11.44%,corr.2671->.3599;everyseedall8emotion/4intensityMSElower. Neutraljawcorr.444223->.441229:47/48regress,49pass;gatefails,notpromoted/noseedselection. Amplitude.058828vsGT.091991 remainscompressed. B0originalF1.0772/.0684 isnotgenerationF1. Allsixfinals downloaded/SHAverified. NextfullmatchedadaptationofpassedPhase19 B0recipe; nonewloss/parameterorbudgetscanning. Target/data/framecoverageinteract, can'tsubtractPhase20fromPhase19 to assign causalmembershippercentage.

### 2026-10-05 Phase21 matched full-path warm adaptation launched
Threepairedseeds47/48/49allPhase19control/nativeB0s. Separateidentity12epochs/batch2/132updatesthen2teacher+2audiofullTRAINepochs/batch16/1568each. No newloss/source/mask/module/probecritic. B0frozen; identityrefitnewbaseresidualcoordinates; teacher/renderer/audio adapted ratherthanunverifiedoldrenderercomposition. Pairedactualinput/observation/noise hashesrecorded; fullnative3draw/fourprobe/group+amplitudeaudit afterallsix. Budgetisfixedinitialadaptation, notclaimofconvergence. Presetallseedjointgate andnoautomaticpromotion. SeeCURRENTforPIDs/root/storage. Phase20fulltarverified/reclaimedbeforelaunch; allhistoricalresults recoverable.

### 2026-10-05 Phase21 verified complete

## Phase21 complete / Phase22 冻结全局条件诊断

三 pairworkers5030/5031/5032 和 finalizer5033 全部结束；六臂/三完整审计成功，无 OOM。18完整阶段权重已下载，SHA和132identity/1568teacher/1568audio更新数核验通过。实际训练sample/GT/mask/noise流跨臂完全相等；B0全程冻结，identity拟合后冻结，teacher在audio阶段冻结，teacher/audio/renderer确实更新。主训练/模型没有新增修改。

完整1367validation，native-padded三draw42/123/2026、raw/clip_all、四冻结probe。clip_all三seed均值control→native_B0：MBE .882626→.866997，LBE .432473→.417398；嘴部MSE .0154841→.0145246，位移MSE .00251557→.00187616（−25.42%），centered corr .292511→.331791；全嘴q90−q10 .100725→.0931905，GT .0919912。范围总体接近GT不代表所有情感/通道/中性jaw正确。原probeF1 .187394/.250608→.182096/.235712；辅助stable .494992/.472235→.496938/.484648，不可冒充原协议。联合门槛47false/48true/49false，不能挑48、不能推广；默认timing000保持。

native中性jaw corr47 .276929→.268940，48 .270571→.279711，49 .251959→.248467。native音频类别accuracy .874177/.874177/.874909（不是生成F1）；teacher accuracy .655450/.635699/.656181。固定暖适配pilot不是充分收敛或从头重训。本地compact_summary、三report/perclip、contracts及18checkpoint齐全；远端每seed有固定8emotion render_inputs未下载/渲染。Phase20完整归档468文件、2.899GB，本地tarSHA ae6d26878ec2e9a8784855f9534b9248ef895b409d15f6c94a2f051b4407dd90，可恢复；Phase21完成/root约2.3GiB。

下一项Phase22：全部三个native_B0最终audio模型冻结、相同validation/nativeclock/mask/三draw/batch16/12Euler，仅把audio global替换为该片段真实motion teacher global；u_a、intensity_value、logits、B0、identity等不变。先重放正常audio并与Phase21原生整体验证曲线核对，不能混入trimmed-batch噪声。仅validation+独立enrollment重建B0缓存，避免重提取12536TRAIN。保存四probe/几何/时序/情感强度speaker分组及配对区间。GT-informed仅定位，不能部署/SOTA/替换默认；不新增loss、不挑seed、不读sealed成绩。


### 2026-10-05 Phase22 local-ready, remote connection pending

本轮没有新增训练/模型/loss修改。Phase22三个native_B0最终模型、原训练源码、manifest、四probe及Phase21曲线/审计SHA已本地绑定，preregistration SHA69a652b9bdf47c0c817c3437cd7d7e8ad9309935aefa7fa8cc17463cd3af00b4。全1367、batch16、12Euler、native-padded三draw42/123/2026，仅换global，其他条件共享原对象。正常audio全部三draw先重放核对，GT/time/mask/B0严格一致，prediction atol2e-6/rtol0，并报告是否bit-exact；全部通过才oracle解码，不混trimmed-batch协议。无需保存新大curves，只保存perclip统计/特征/分组/paired speaker CIs。所有模型状态前后相等，fresh输出保留失败且不自动重跑。诊断GT-informed，不能部署或计为SOTA。

本地12targeted tests通过（全padded噪声/RNG独立性、single-global条件隔离、nativeGT/clock/mask/B0/clip漂移拒绝及原factorial）。另用三个实际Phase21 perclip数组+四冻结probe复算raw/clip_all，全部分数与原report一致到1e-12；zero-paired-bootstrap通过。run/launch脚本编译通过。注意这只是评分闭合与单元检查，**实际生成重放和oracle还没有跑**。

SSH connect.nmb1.seetacloud.com:11473两次paramiko/TCP核验及第三次TCP复核均连接拒绝（Windows10061），DNS仍116.136.52.182；不是训练报错。已异步询问机器是否停机或地址变更。没有启动/上传远端Phase22，不要声称有running PID。待用户恢复连接后，先确认Phase21路径及磁盘，再建独立/root/kinetalk_frozen_global_diagnostic_20261005、上传prereg/local_smoke/run/launcher；使用新目录launch，不覆盖Phase21冻结代码。入口 tools是本地.codex-finalizer/run_phase22_global_diagnostic.py和launch_phase22_global_diagnostic.py。本地README/prereg/local_scoring_smoke齐全，后续不重复分析/测试。

对现有三个最终模型的全部三draw分组描述（非新训练成绩）存Phase21/error_decomposition.json和class_f1_decomposition.png：原128happyF1 .018、64 .114；stable128happy .616、64 .640，stable总体仅.497/.485，fear .020/.077。原128预测75.2%片段为contempt，属于读出分布塌缩；历史GT低方差敏感性已证明，不能把当前视觉笑容低F1全部解释为缺失happy语义。stable对fear也很差，仍有实际情感差距。

眉部MSE .053670，其中mean_bias .051771（96.46%），占51观察通道加权平方误差31.71%；嘴部占46.33%，眼部21.94%，这是coefficient SSE贡献不是MBE贡献。眉部corr .0555，fear眉部MSE .0889/corr−.002，surprise .0878/.063。全嘴范围/GT：happy1.149、surprise .705；jaw范围/GT：happy1.376、surprise .779、sad .894。不能继续统一放大嘴部。下一项机制定位仍是global-only oracle，看表情平均姿态以及fear/surprise/全部类是否改善，再决定唯一修改点；不能直接凭meanbias加loss。Default timing000未替换，用户目标未达成。


### 2026-10-05 SSH restored; Phase22 generation replay failed safely
用户已重启服务器，旧Phase21/root/data完整、/root空闲2.232GiB。新独立/root/kinetalk_frozen_global_diagnostic_20261005，orchestrator1233/workers1234/1235/1236完成并失败，不能当running或重启原输出。所有绑定和nativeGT/time/mask/B0严格校验通过，但第一batch正常audio生成与原曲线最大差47 .000412107、48 .000385284、49 .000347137，超过预登记2e-6，全部在oracle前停止，无训练/模型修改。原state/logs/launch已下载到Phase22/failed_replay，远端原失败目录保持不覆盖，prereg未放宽。
下一项firstbatch数值路径诊断：固定seed47，不涉及效果/模型选择；比较原trainer evaluate入口、同一冻结forward重复、原audio阶段requires_grad标志、MHAfastpath和TF32路径，以查明接口/数值差异，不通过挑较好score绕过重放。check_phase22_replay_paths.py执行中，原oracle尚未跑。


### 2026-10-05 Phase22 v2 running, canonical replay bit-exact
Firstbatch路径诊断完成：frozen default/repeat maxvs原 .000412107、repeat与自身完全一致；恢复原audio阶段requires_grad标志后maxvs原0，原trainer evaluate入口max0。MHA slow/TF32 false都不复现原路径，不采用。这里只是诊断实现此前把所有标志关闭导致数值dispatch变更，未改训练或权重，未放宽2e-6。原失败root+logs保留，本地failed_replay含完整路径证据。
修复：run依然整体@no_grad、没有optimizer/backward/参数更新，恢复audio和renderer的原阶段requires_grad标志，其他模块false；结束逐tensor比较权重/状态。新增guardtest，本地13targeted checks和真实三seed评分闭合通过。dispatch_addendum绑定原prereg和路径证据SHA；原prereg69a652.../噪声预算/probes/输入/容差均不变。
新root /root/kinetalk_frozen_global_diagnostic_v2_20261005，orchestrator1790/workers1791/1792/1793，canonical launch/state在该root，勿重启/勿看旧failedroot为当前。三个模型完整1367validation×三draw重放全部bit-exact，maxdiff0；已进入global-only GT-informed解码/评分。真实oracle结果尚未完成，不声称改善或SOTA。Phase21被冻结，默认timing000保持。


## Phase22 complete / Phase23 TRAIN flow-units audit planned (2026-10-06)
Phase22 v2三个worker1791/1792/1793、orchestrator1790全部成功结束。1367validation×3draw×3模型正常audio重放全部bit-exact；GT/time/masks/B0严格相同，所有模型state前后完全不变。三report/perclip/state/replay/summary已下载并核验SHA，13localtargeted checks通过。旧first-replay失败保留，因原audio阶段requires_grad标志影响dispatch；修复仅诊断flags，容差仍2e-6，无训练或权重修改。当前无Phase22运行任务，不要重启。

仅global换成真实query motion teacher，clip_all三seed均值audio→oracle：MBE .866997→.620782，LBE .417398→.326710；嘴部MSE .0145246→.00977143；嘴部meanbias .00771852→.00326642，眉部meanbias .0517715→.0122579（−76.32%）。嘴部位移MSE .00187616→.00192276（+2.48%，每seedclusterCI跨0），jaw corr .322320→.315969。原probeF1 .182096/.235712→.122878/.175071，三个seed配对原F1 CI均<0；辅助stable .496938/.484648→.514020/.494310，CIs跨0，不宣称情感稳健提高。teacher-global是GT-informed，.621不是部署成绩/默认/SOTA。

Teacher-global也未解决情感：辅助fear .0196/.0769→.1236/.2121；happy .6159/.6403→.6691/.6715；sad/surprise反而下降。眉部动态corr仍 .0686，静态姿态改善远大于真实动态时序。说明至少有两个问题：连续全局几何预测缺口，以及生成动作分布/动态与评估敏感性的剩余差距。不能只提音频类别准确率或增大全嘴范围。

额外只读冻结head核验（GPU同batch16，teacher logit重放bit-exact）：audio自己head macroF1 .8514/.8557/.8497；teacherhead(audio_global) .8048/.8156/.7830；teacherhead(teacher_global) .6201/.5984/.6154。这些都不是最终生成F1，证实音频编码已有类别语义，不能把瓶颈简单归因类别识别或学生/teacher语义坐标错位。validation global MSE2.4331/2.3976/2.2479，cosine .6340/.6213/.6534；全局误差中约29.8%–33.1%为整体均值偏差。CPU teacherhead初始strict类别重放失败仅1/0/1样本、最大logit差 .00778/.00507/.00679；未混用CPU/GPU分数，改为原GPUbatch16后所有logits exact，失败事实保留在progress。

下一步Phase23：三个native最终audio模型冻结，在全部12536真实TRAIN逐observed native帧核验motion与residual通道populationstd，以及固定t=.5/noise seed20261006的原flow向量误差/GT-informed桥endpoint误差。保留原audio阶段计算标志但no_grad/nooptimizer/backward，所有支持通道不遮蔽。直接用原system.flow，现有objective不改；不评分sealed/validation、不按F1挑权重、不重复旧warm source-only trial。这是为判断新的residual坐标是否仍有严重单位/精度失衡，不是修复效果或部署评估。若尺度证据支持，再设计保持同物理先验、条件、初始输出的单变量归一化对照；不得提前把source-only失败当作归一化成功，也不能直接堆loss。归一化尚未实现/训练。


### 2026-10-06 Phase23 complete / Phase24 opt-in loss units local-ready
ThreefullTRAIN12536audits2413/2414/2415+orchestrator2412success; allmodelstatesunchanged, nativeGT/observation/noisecontracts exactall3, reportsdownloaded/SHAverified. t=.5桥endpoint为GT-informed训练路径，不能当rolloutF1或推理MSE。
TRAIN GTstd cheekSquintLeft2.00e-6/Right9.24e-7/noseSneerLeft4.60e-6/Right2.47e-6；新B0/identity坐标残差std约1.2e-5–2.6e-5；scalarrawsource仍.25。t=.5原flow桥endpointRMSE这些通道 .0080–.0189，GTstd倍数约1738–20457。jawOpen GTstd.1678、bridgeRMSE.0757–.0761；browInnerUp.3208、.0385–.0389。因此近constant通道精度不足真实存在，不能靠原0.25固定单位MSE衡量每通道相对误差；并非mouthmask。没有实际调参/模型promotion。
Phase24隔离设计调整：不声称做可逆模型坐标变换。两个matched臂都固定相同TRAIN残差STD物理diagonalsource（reuse existingmechanism），唯一变化是原flowMSE scalarunits→error/stdunits；既有loss项/权重、全51通道支持、forward和输出单位不变，不新增loss。不和旧Phase15warm control混比。单变量仅能归因relativeunits effect GIVEN相同diagonalsource；相对standard的推广还需全metric独立对比。stats直接从Phase23fullTRAINmoments派生，每seed绑定自己finalSHA，eps沿用既有normalized1e-4，不扫描。
本地opt-in --flow-vector-units scalar(default)/train-residual-std 只允许既有隔离warm-audio+boundstats+diagonalsource路径。新helper默认直接调用原mse，单位1loss+gradbitexact，NaN未观察排除、invalidstd拒绝、单位变换不变量checks；28targetedrunner/source/unitschecks通过。原训练源码默认行为不变，历史Phase21snapshot不覆盖。实际smoke准备脚本做旧源码diagonal/newscalar/newrelative三个2update，绑定其他source不变、actualsample/mask/noise相等、非renderer冻结state、defaultstate/lossbitexact。尚未正式训练，不能把smoke当改进。空间约2.15GiB，6fulltrialcurves会超预算；正式trial必须先完整验证归档或规划artifact保存，不擅自删除历史结果。


### 2026-10-06 Phase24 both loss-only variants rejected; Phase25 coordinates planned

Balanced fresh root /root/kinetalk_flow_balanced_units_20261006 preparation3246 complete. Default old/new diagonal-source scalar states/losses bit-exact, actual TRAIN sample/GT/observation/noise exact, frozen B0/identity/teacher guards pass. Eighteen evidence files downloaded and SHA verified at phase24_flow_balanced_units_20261006. No formal Phase24 run, no running jobs.

Balanced total loss is finite (~.483 average), but first-update four near-constant channels take99.5528% of flow loss and99.9949% of flow-output derivative squared; mouth derivative norm is .00019237× scalar, brows1.3387e-7×. This is analytical dL/dphysical-vector-output, NOT full parameter-gradient attribution. With unchanged model coordinates it exposes severe mouth/brow downweighting. Reject balanced formal trial as well; no mean-weight/STD-floor/lambda scan. Default remains timing000. Objective-only alternatives do not constitute channel normalization.

Phase25 next design: coherent TRAIN-centered channel coordinates inside the DiT. Input y=(x_t-t*mu)/s; physical velocity v=mu+s*network(y,t,conditions), where mu=TRAIN residual mean/.25 and s=bound normalized residual std (existing1e-4floor). Same physical diagonal source in matched arms. Control uses centering and scalar s=1, existing scalar MSE; candidate uses channel s and MSE in those same channel units. All output/support/native-clock/content/affect routes retained, no extra loss or mouth masking. Both arms reset ONLY renderer output weight/bias to zero, thus initial physical velocity is exactly the same mu. Shared trained trunk/audio/teacher/B0/identity inherited identically. Reset is necessary because a warm physical head has huge normalized outputs in tiny units; this compares refitting from a shared mean field, not continuation from unchanged timing000.

Implement opt-in flow-coordinate-system default scalar / train-centered-scalar / train-standardized. Save mu/s in config with nonpersistent buffers for strict old checkpoints; inference must integrate the same transformed physical vector field. Bound fullTRAIN moments to each Phase21 final SHA, finite/missing-support checks. New architecture branches stay off by default, old/new default real smoke bit-exact required. Smoke must inspect normalized head gradients (physical-output derivatives alone cannot judge the reparameterized model), actual stream equality and identical initial mean field. If passed, preregister all3 seeds,2audioepochs/1568updates/batch16/finalonly/full1367×3draw/fourfrozenprobes and joint geometry+originalF1+mouthtiming gate before large launch. Complete verified archives/reclaim required first; preserve dependencies and no sealed-test tuning. No performance or SOTA claim yet.


### 2026-10-06 Phase25 smoke complete; archive before formal launch
Fresh v2 root /root/kinetalk_channel_coordinates_v2_20261006 preparation3794 complete (first root3717 failed AST before training and is preserved). Three intentional training-source changes only: trainer, DiT coordinate transform, neutral config call. AST default equals frozen Phase21; all other snapshot files exact. Existing local AR CPU-noise API fix is intentionally excluded from this isolated diagonal-source snapshot, remains in local file. Local39 targeted tests pass. Twenty-two smoke evidence files downloaded/SHA verified and matched recipe has exactly one coordinate-system flag difference.
Real2update four-arm smoke: old/new default system/audio/loss bit-exact; all sample/GT/observation/noise streams identical; centered control/standardized initial physical mean field and x_t hashes exact. B0/identity/teacher frozen, audio/renderer updated. Standardized flow~1.895; total~2.579, finite. Actual total output-head gradient square shares standardized: mouth54.19% then43.80%, brows21.57% then26.79%, four nearconstant .532% then .605%; no low-variance monopolization. This is actual head parameter gradient, unlike Phase24 physical-output derivatives. Smoke is not validation improvement.
Three fixed seed47/48/49 matched2audioepochs,1568updates/batch16/full12536 TRAIN and1367validation,12Euler/3draws/4frozenprobes preregistered locally and remotely; source/conditions/head-reset common, no selection. Full originalF1/geometry/jawtiming joint gate plus class/intensity/speaker/meanbias/range/constantprecision/CIs planned. No formal launch yet. Finalizer/helpers uploaded; original/stable scores separate, no sealed-test tuning/default promotion.
Disk~959MiB, so first preserve full completed Phase21: three native final inputs/fourprobes/legacy source copied and SHA-bound independently into v2 root. Complete archive made:746files/4,040,861,183original bytes, tar4,042,362,880bytes SHA8bb170c8647ff55c42cf00f98d5cf621ac7343a23f46a61006e73a8803e99469. Download to final_experiment/remote_archives/20261006 in progress; no reclaim before full local archive/every-file/every-directory verification and unchanged remote recheck. RAM copy alone is not durable. Current pending local exec session69202 is archive download, not training. Default timing000 unchanged; goals unmet.


### 2026-10-06 Phase25 formal running; Phase21 durable archive verified
All746files/directories and complete4.042GBtar locally verified, remote originals unchanged rechecked before reclaim. Local persistent archive final_experiment/remote_archives/20261006/kinetalk_native_b0_adaptation_20261005.tar SHA8bb170c8647ff55c42cf00f98d5cf621ac7343a23f46a61006e73a8803e99469; original/root path now ARCHIVED.json pointer, NOT missing data. Localverified/reclaimed/ARCHIVED records saved. RAMtar temporary only. Phase25 has independent3final inputs,4probes,legacy code/SHA binding before reclaim. /root free4.8GiB after reclaim; no pending local exec/archive session.
Formal fresh launch /root/kinetalk_channel_coordinates_v2_20261006: pairworkers4449/4450/4451 seeds47/48/49, finalizer4452. PreregSHA6bde2ce64ace585c9ed7c35cad3d07e4d54efcd105315a8ad5587b51949ff4b0. Never relaunch. Canonical launch.json, seed*/control_state.json/standardized_state.json, postprocess_state.json. Both arms2audioepochs/1568updates/batch16/12536TRAIN, full1367nativevalidation×3draw/4frozenprobes. Current3controls active epoch1batch326/351/326, finite losses/gradients, GPU5.5GiB/24GiB, no OOM. Candidate queued after each control. No complete validation results yet; smoke is not benefit.
Finalizer checks frozen B0/identity/teacher, actual sample/GT/observation/noise exact, checkpoint/config/source/curve bindings; full raw+clip/all emotion/intensity/speaker native audit, original/stable probes separate, paired speaker CIs for geometry+all4F1 (reuse Phase22 statistic function with new branch labels), nearconstant precision and fixed8renderinputs. Fixed joint gate unchanged; no selection/promotion/sealedtest. Default timing000 and user goals unmet. Continue by checking canonical state/results; do not reopen/reanalyze old phases.


### 2026-10-06 Phase25 complete; strong F1 gain, joint geometry/timing gate failed
All6formal arms1568updates and3complete native audits finished, workers4449/4450/4451/finalizer4452 ended. All93finalartifact files/6completefinals (~303MB) downloaded and SHA verified after interruption; no pending local exec51569 (unknown session, filesystem evidence allcomplete). Default/initialfield/actualstream/frozen/config/source checks pass. Full1367validation/all3draws/raw+clip/fourprobes/classes/intensity/speaker/precision/CIs saved.
Matchedcenteredcontrol->standardized3seedclip means: MBE .873153->.855830; LBE .419426->.416625; mouth MSE .0145754->.0144379; mouthdisplacement .002202119->.002201256 (essentially unchanged); mouthcorr .328318->.321403. Jawcorr .321086->.307054, jawMSE .0154882->.0162794. Neutraljawcorr47 .269745->.242856/48 .276766->.247924/49 .270859->.257264, allworse. Threejointgatesfalse, no defaultpromotion.
Original final GENERATED independent F1 .272033/.286390 -> .778258/.665628; auxiliary stable .414684/.337615 -> .791925/.749272. OriginalF1gain all3seed/speakerCIs strictlypositive; original128 surpasses .7, original64 remainsbelow .7. Do not claim allF1target/SOTA. Fournearconstant motion MSE decreases from1e-4/1e-5 to3e-11–1.22e-10, predicted std now sameorder GT~1e-6. Stable gain substantial too, not only original-sensitive classifier. OriginalGT .66/.64 known; F1 alone cannot certify fidelity.
Remaining眉meanbias .0454809/.0479510MSE≈94.85%; mouthmeanbias .0072374/.0144379≈50.13%. Fullmouth range .098382 vsGT.091991, jaw .190668 vsGT.175279; avoid global amplification. Geometry near nativebaseline and jawtiming worse. Twoepoch TRAIN flow .94–.96 -> .50–.51, still strongly decreasing acrossall3seeds; head refit not yet converged. No loss/architecture addition warranted from those results.
NextPhase26 fixedbudget control: same Phase25 source and initialmeanfield/coordinate arms, train12 total audioepochs (original base stage budget), all3seeds, batch16/9408updates. Fresh matchedruns using identical initial/checkpoint/std source, no reset-after2 or invalidresume; compare first1568updates actualstream and full system/audio state to Phase25 before continuing. Only epochbudget changes; no source edits/default/loss/condition changes, no selection. Formalfixedfinal12/fullnativeallprobes/geometrytiming/precision/CIs required; 2 vs12 within eacharm and centered vsstandardized12. STORAGE must verifiedarchive/reclaim completed Phase25 after copying independentinputs/source/probes/stats AND pilotreference weights/contracts, keepfullcurves historical. Do not relaunch Phase25. No sealedtest tuning. Userasked architecture/data/branchtraining explanation: create explicit docs using true checkpoint config/code/provenance rather than stale cfg.training fields or unused old Stage5.
