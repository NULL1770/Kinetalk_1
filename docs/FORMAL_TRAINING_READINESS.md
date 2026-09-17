# 无文本动态场：正式训练准入与评估协议

更新：2026-09-16。此文是 run31–33 的独立审计和后续执行约束，**不是训练已启动或效果已达标的记录**。用户已授权继续优化，在收益稳定后启动正式训练，无需重复请求授权。

## 1. 当前能确认什么

- run31：1200 条训练、280 条开发、12 人；RRR8 音频预测 upper R²=.02354，眉=.01899，眼=.03319；相对同维 PCA 有小幅优势，对 direct ridge 或 content-only 的优势未确定。真实目标固定，音频预测幅度仍弱。
- run32：开放整个 renderer 和 audio head 的三臂各 600 步共同训练，最终动态、neutral 嘴部和情感读出退化，不能作为正式训练初始化。
- run33：固定 head 和 renderer，只训练 512 参数原接口，两臂各 600 步。RRR full 相对 zero-local upper ΔR²=.01262、相对 reverse 为 .03148，句簇区间为正；眉单项收益区间跨零。neutral mouth MSE 比原模型改善，却比 zero-local 差约 3.1%，neutral upper 也有误激活。

因此可以继续验证同一共享场的幅度条件化；目前不足替换默认模型。绝对动态 R² 为负不自动禁止扩大训练，但非零幅度、训练拟合、优于退化的 joint 组也不构成准入证据。

## 2. 为什么现有 new_test 不能承担正式测试

`lock_predictable_motion_data.py` 默认只读 native `train.jsonl`。它先把历史训练、开发、登记句子全部排除，再按 hash 抽 fresh 句子的 15%，最后按身份×情感×强度裁剪片段。**最后的平衡裁剪不能补回抽句阶段缺失的情感。**

已保存 manifest 的直接审计结果：

| pre-run31 未使用句子的去向 | 句数 | 片段 | 情感分布 |
|---|---:|---:|---|
| run31 train | 15 | 169 | neutral 41、happy 91、sad 37 |
| run31 validation | 0 | 0 | 无 |
| 保留 new_test | 3 | 15 | neutral 13、sad 2 |

fresh 共 18 句，当时没有 angry 候选；现在其中 15 句已用于拟合。不能重新打散这 18 句，再把选出的部分称为未触碰测试。保留的 15 条包含 13 条 `stage1=true`，其是否实际参与 B0 训练尚需训练来源核对；已有排除规则从未保证全系统未见。

native 句子标识是规范化真实文本的 hash，不是视频编号。增加 MEAD 身份可能只增加同一脚本的重复朗读；这能评价新身份，不等于新句。文本 hash 仅用于数据隔离，不引入文本推理分支。

### 必须先做的 metadata inventory

只读 native 根目录实际存在的 train/val/test manifest 和构建来源，不载入其 motion 或模型输出，输出一份带来源 SHA256 的覆盖表：

1. 去重后的 dataset、speaker、canonical sentence、emotion、intensity、valid_frames、artifact、source split、stage1 eligibility；重复 clip 的冲突必须报告。
2. 每个源 split 的身份/句子交集；每个 sentence×emotion×intensity 的身份数和片段数。把“新身份旧句”“旧身份新句”“新身份新句”分开。
3. 排除最新完整暴露台账之后的覆盖，而不是只排除 data224/audit26。台账至少包括 run31 的训练、开发、登记；保留 test15 标记为 locked、不纳入训练。historical `selection.json` 中的 excluded 列表不等于已评价结果，保守排除需要保留来源和原因，不能为凑数量自动解除。
4. B0、global、identity、motion teacher 的已知训练 manifest 单独核对；无法核对时明确“动态模块新句/新身份”，不写“全系统未见”。

若确有新的完整候选，在读观察值前按句子组锁定，优先覆盖 neutral/angry/happy/sad 及非中性 L1/L3，建议至少 20 个独立非中性句簇、每个非中性情感至少 5 个句簇、多于一个身份。元数据层面的分层选择可行；不能通过查看预测或追求某个 R² 选样。语料不支持这些数量时报告实际覆盖，不伪造完整平衡格。

若全局脚本已基本耗尽，保留两类诚实协议：原 280 条继续作开发；按原始身份划分锁定未参与调参的新身份集合，允许共享句子，但明确是**跨身份、非跨新句**测试。若 CREMA-D 或其他原始数据有未暴露句子，先统一时间、可观测通道和 neutral 登记规则，再独立作为跨数据集测试，不能为扩样直接混入当前锁定集合。没有平衡新句测试不必永远停止训练，但必须降低正式评估的泛化声明。

### 已执行 inventory 的结论

`scripts/audit_formal_data_coverage.py` 已只读三个 native manifest，结果和 SHA256 在 `artifacts/formal_readiness/metadata_coverage.json`。未读 motion/音频/视频、未提特征、未改 test15：

| 原始 split | MEAD 全部片段/身份 | MEAD 当前四类 L1/L3 + neutral、≥32帧 | CREMA-D 全部片段/身份 |
|---|---:|---:|---:|
| train | 12959 / 22 | 4265 | 5797 / 72 |
| val | 1435 / 3 | 470 | 723 / 9 |
| test | 1659 / 3 | 546 | 806 / 10 |

- 三 split 身份、clip 两两完全互斥，MEAD 句子两两重叠 141 个；CREMA-D 两两重叠 12 个。MEAD val 人员为 M025/M037/M039，test 为 M022/M028/W037。
- 排除历史 80 句、run31 新使用 15 句及封存 3 句后，所有 MEAD split 当前四类候选只剩 **1 条、1 个新句**（train 的 W029 angry L1）；val/test 为 0。当前资产无法形成平衡、全情感、真正新句最终测试。调整随机种子不能补足这个缺口。
- MEAD val/test 的 470/546 条是未用于当前动态试验的独立录制与新身份，可以在保留独立 neutral 登记后提供正式跨身份评价；同句不代表同一录制。此路径比继续消耗 test15 更合理。选用原 val 调参就必须将它记为开发，原 test 保持封存。
- 删除原登记 4 个句子及封存 3 个句子后，val/test query 上限降为 439/512 条，尚未锁定。原登记句子的合格 neutral 参考覆盖分别是 val M025=4、M037=2、M039=2；test M022=4、M028=4、W037=2。当前身份编码器支持变长参考数，可以预先声明最少 2 条、同协议评价所有新身份；若坚持 4 条，就须另外按元数据固定参考，并把新增参考句子从所有 query 排除，不能复制片段冒充 4 个独立参考。
- 历史 `stage1_fidelity.json` 把 3248 条标为 train-set diagnostic，其中与当前 MEAD val/test 四类分别重叠 57/140 条。它未保存确切 checkpoint hash，不能断言现在 B0 必定拟合了这些条目，也不能宣称 B0 未见。需回溯 run01 指向的 stage1 checkpoint（SHA256 `4e419db1ac1b9bf8ff99752e172b9d25845a5338f2f35a91584518a8ab67475b`）原训练 manifest；在此之前只做“动态模块跨身份”声明。
- CREMA-D 四类、不限制强度的合格片段为 train3827/val475/test529。若照搬 MEAD L1/L3 则仅剩 1276/157/176 且主要是 neutral，因大量非中性强度为 −1，不能静默丢弃或映射。当前动态 pilot 无 CREMA-D 记录，但 B0 的外部暴露未认证，句子 code 与 MEAD text hash 也不能直接比较。先作为单独域、保留未知强度标签的候选评价，不能混改 audio/global 归一化。

## 3. 本轮唯一优先干预：已有 global 调节共享场幅度

固定 run31 的输入统计、RRR8/U、head、B0、identity、global 和 renderer，继续训练既有局部接口。用冻结 audio global 的 neutral 概率对**整个共享动态场**做平滑幅度调节，训练和推理都用相同 audio 条件；不以 GT 类别替代，不设眉眼嘴独立输出，不叠加分类/速度/情感一致性等新损失。

用同 batch/noise、同初始化、同预算的无门控对照，先回答：neutral 误激活能否降低，同时保留非中性 full 优于 zero/reverse 的效果。保留 oracle motion 条件，区分音频限制和接口限制。若门控仅把所有动态缩到零，则该干预失败。门控函数或温度只用训练句内划分选，不能在 280 条上试多个阈值后只报最优。

该试验是当前最有信息的一步。它通过后再增加训练预算；不凭 run32 的失败盲目开放整网，也不将新 gate 称为解决了音频中不存在的信息。

## 4. 进入正式训练的可执行门槛

以下为未来实验的预先记录工程容差，不是从 run34 结果反推的显著性标准，也不等于论文充分证据。沿用 280 开发条目进行决策，最终测试保持封存。至少两个训练 seed、各三个相同生成 noise，先对每片平均 noise，再按句簇做配对统计；noise 不能增加统计样本数。

| 项目 | 准入判断 |
|---|---|
| 非中性时序利用 | 两个训练 seed 的 upper full−zero、full−reverse 均为正；合并前分别报告。主 seed 的 full−zero ΔR² 至少 .005，句簇 95% 区间下界大于 0；不得仅因全脸/嘴部平均改善过关。 |
| 眉毛 | 平均点估计 full−zero 大于 0，95% 区间下界不低于 −.005；若仍跨零，只允许以“眼部已证实、眉部待加强”进入受监控训练，不能写眉眼都成功。 |
| 中性与嘴部保护 | 对原模型和同模型 zero-local 都报告。neutral mouth 原动作 MSE 相对 zero 的单侧 90% 区间上界不超过 +3%；全体 mouth 时序相关下降不超过 .01。neutral upper 原动作 MSE 不超过 +3%，速度误差不出现超过 5% 的系统性增加。若置信区间因覆盖不足不确定，先补开发覆盖，不用“不显著”宣布非退化。 |
| 全局与身份 | 冻结参数 hash 一致；global 教师读出开发准确率相对原模型下降不超过 1 个百分点，并单报均值偏移。该教师不是独立情感评测器。neutral 独立参考不与 query 重叠。 |
| 基线与可重复性 | 保留原 checkpoint、zero、reverse、同维 PCA；RRR 若仅持平 PCA，仍可验证共享场，但不能宣传“可预测基优于 PCA”。两训练 seed 不出现方向相反的主要退化。 |
| 数据与资源 | 锁定正式 train/dev/test 角色及允许的泛化声明；完成下述资源预算和监督抽检。 |

门槛可在看到新实验结果前依据样本覆盖修订一次并留痕，不在看到失败结果后随意放宽。准入通过允许开始更长训练，**默认模型替换仍需正式训练后的完整生成验收**。若暂时只满足动态项而 neutral 失败，继续局部优化，不启动昂贵全量训练。

## 5. 正式训练应具体做什么

正式训练是固定方法后的完整数据、固定选择规则和可恢复运行；不等于必须解冻全部模型。512 参数的接口也可以做有效的参数高效适配，但再跑一次 600 步不能包装为正式训练。

1. **锁定正式数据版本。** 扩展至所有审核合格的训练身份/片段，保留原 source split 和历史开发隔离。现有 1200 预算前有 1647 条可用，是否进一步增人/增句由 inventory 决定，不提前承诺数量。新身份须有独立 neutral 参考；登记片段不能当 query。记录每个角色的 manifest/hash/时钟/通道布局。质量抽检只用训练和开发：每类固定 hash 选择 4 条、至少覆盖 4 人，对原视频音频检查同步、眉眼跟踪、眨眼视线污染和系数边界；当前系数曲线不能替代原视频检查。
2. **训练内确定音频目标与坐标。** 按已经固定的输入/方法和句子 CV 重拟正式 train 的标准化、正则和 U，保存后冻结；所有 oracle 来自真实 `center(motion-B0-P_id) U`。开发/测试 motion 不参与尺度、基或 gate 标定。PCA 基线用相同训练数据、预算、gate 和评估；不重复搜更多架构。
3. **完整共享接口适配。** 从原 run09 冻结生成器重新建立接口，使用通过准入的 gate 和固定 audio head，仍只用 flow MSE。以完整训练 epoch 计预算：预设最多 40 epoch，至少完成 10 epoch 后才允许早停；前 5 epoch 将真实 motion 条件概率从 .5 线性降到 0，之后只用 audio。每 2 epoch 在固定开发集评价，连续 5 次无可行改进早停。初始 lr 沿用已验证值，训练内确定的日程写入 config；不临场加 loss。
4. **checkpoint 选择有保护约束。** 只在中性、嘴部、全局约束通过的 checkpoint 中，选非中性 upper full−zero 最好者；并保存 final/last 和可恢复 optimizer、RNG、epoch。记录浏览过多少次开发结果。若无任何 checkpoint 可行，判本轮失败、保留默认模型，不能退而选择最低全脸 loss。
5. **正式复核。** 最终候选以三个训练 seed 复训/复核，并报告相同预算 PCA 和 zero 对照；固定原帧 full/zero/reverse/oracle 三 noise 评价，报告按类、按身份、句簇区间、眉/眼分项及嘴部。只在方法与候选锁定后打开新的最终集合一次。看过后它立即成为已评价集合，不再用于改 gate/选 rank。

若更长的冻结适配饱和而 oracle 仍明显好于 audio，下一步是补训练覆盖/表达监督或改善 audio 学习；不是再次解冻 renderer。若 oracle 自身也差，才针对共享条件接口做新的独立干预。两种情况都不能把失败归结为“8 维一定不够”。

实现入口 `scripts/train_formal_predictable_projection.py` 已完成：每 epoch 无放回遍历完整输入 manifest，10–40 epoch、每 2 epoch 三 noise 开发、5 epoch teacher 退火、patience5；checkpoint 选择额外保护上述四组速度误差不恶化超过 5%。只训练原 512 参数共享接口，gate 固定，无附加 loss。`last.pt` 保存 epoch、optimizer、Python/NumPy/Torch/CUDA/采样器 RNG、best/patience、全部输入和源码 hash、可恢复 best 快照；恢复用原参数加 `--resume OUTPUT/last.pt`，路径、配置或输入变化会拒绝。`best.pt`、`last.pt` 都不是默认模型。默认仅保存紧凑 checkpoint 与指标，`--save-final-curves` 只在末期保存完整 full/zero/reverse/oracle 曲线。运行入口存在不等于已准入或已启动；实际训练集大小、seed 和日志由启动记录确认。

## 6. GPU、存储和运行约束

父任务最新只读资源查询：`/root/autodl-tmp` 约 411 MB，root overlay 约 25 GB 可用，正式输出可另建 `/root/kinetalk_runs`，无需删除已有资产。原 renderer cache 约 391 MB 在 `/dev/shm`，诊断 bundle 约 321 MB。临时 RAM 缓存不持久，机器重启后不能当可恢复训练依据；root overlay 仅对当前实例持久，也不保证跨实例保留，关键 checkpoint 仍须下载备份。启动前再次测量真实余量。

- 启动前查询 GPU 占用、显存、host RAM、`/dev/shm` 和目标盘真实可用空间，避开已有任务；用实际 batch 的 20–50 步 smoke 测峰值和吞吐，估算整个预算。
- 分块读/提取，避免一次复制全量 tensor 或全量中间层。用可重建特征 shard 和 manifest/hash，不保存重复原始训练数据。
- 先测 checkpoint 大小 S 和数据缓存增量 C；可用空间至少覆盖 C + 4S + 1 GB 日志/故障余量，否则先归档已确认可重建缓存到可用盘或缩减重复输出，不能带着不足 1 GB 启动扩量长训。只保留 best/last/resume，不每次评价落盘全部曲线。
- 原始数据、现有模型和唯一实验结果不得为腾空间直接删除；记录实际 PID/log/output/checkpoint 恢复路径，启动成功以日志进入训练且写出首个可恢复 checkpoint 为准。

代码测试通过只证明接口和冻结约束正确。正式结论还需要真实生成视频、口型与表情评测及与已有方法的公平比较；本协议不保证 CCF 录用。

## 附录 A：已锁定的正式数据元数据版本

`scripts/lock_formal_predictable_data.py` 已于 2026-09-16 在本地 native manifest 副本上执行，输出 `artifacts/formal_readiness/formal_lock_v1/selection.json` 和各角色清单。此次只打开 JSON/JSONL 元数据，未加载任何 native motion、audio、video、特征或模型预测；封存 test 未物化。锁定不代表正式训练已经开始。

| 角色 | 片段 | 句子 | 身份 | neutral 登记 |
|---|---:|---:|---:|---:|
| 正式 train | 2720 | 67 | 22 | 79 条，按人 2–4 条 |
| 原 validation | 280 | 25 | 12 | 复用原每人 4 条 |
| 新身份 validation，native val | 439 | 80 | 3 | 8 条：4/2/2 |
| 封存 test 元数据，native test | 512 | 80 | 3 | 10 条：4/4/2 |
| 原保留 test15 元数据 | 15 | 3 | 12 | 保持原锁定 |

训练保留原 1200 条并新增 1520 条；四类分布为 neutral496、angry691、happy798、sad735，52 个非中性句簇，身份×情感×强度覆盖 154/154。合格 native train 四类 4265 条中，按互斥角色排除历史开发句 1278 条、保留测试句 20 条、登记句 247 条，得到 2720 条；未从开发或原 native val/test 补训练。原四个全局登记句已经覆盖所有 22 个训练身份，无需新增登记句，原 12 人的 ID 和 48 条参考不变。17 人各 4 条参考、M040 为 3 条、W028/W029/W036/W040 各 2 条，没有身份因不足 2 条被纳入或放宽规则。

原 `validation.jsonl` 原样按字节复制，SHA256 仍为 `49fb7c9344a01351fb884a295f77dd0629e5eaa1887dbda285482241465d4e2d`。原 reserved test15 同样原样复制。正式 train SHA256 为 `8d53a4d4d21541bbd56436a680abac6746948640de1320f12143894fd03ab6b8`。所有角色、来源和准备脚本 SHA256、排除原因、逐身份参考数、缺失格均见 `selection.json`；12 项隔离检查全通过。

新身份 validation/test 各覆盖 21/21 个身份×情感×强度格，都是 **跨身份、共享脚本** 协议，与正式 train 各重叠 57 个句子；不能称为全局新句测试。它们各自的 neutral enrollment 单独保存，所有 query 均排除四个全局登记句与三个旧 reserved 句。旧 test15 仍只有 neutral13、sad2，其全情感覆盖缺口没有消失。native val 是单独开发集，若用于调参必须按开发暴露记录；native test 仅封存元数据，固定方案和最终候选前不得读取目标。B0/global 的外部预训练暴露仍未认证。

情感 `emotion_id` 继续使用原八类合同中的 neutral=0、angry=1、happy=5、sad=6，与冻结 global 和旧 280 条兼容；不将 happy/sad 静默改成 2/3。后续 materializer 必须显式使用最少 2 条登记的合同，仍保留旧人的 4 条，不能为凑齐 4 条复制参考。7 个元数据锁定测试覆盖角色隔离、固定身份映射、缺参考排除、旧训练冲突拒绝、原验证字节不变、重复 metadata 拒绝和不读取不存在的 native artifact。

复现命令：

```text
python scripts/lock_formal_predictable_data.py --native-root artifacts/formal_readiness/native_metadata --locked-root artifacts/predictable_motion_run31/locked_manifests --historical-development artifacts/neutral_affect_pilot_20260916/audit_available26/selection.json --output <新的空目录>
```


## 附录 B：本轮执行与启动记录

run35两训练seed46/47均通过第4节预设检查；neutral mouth误差单侧90%上界分别2.061/2.220%，upper full-zero ΔR² .012665/.012237且CI下界为正。16条训练原视频/native时间采样与系数抽检、嵌入音频对比已完成；波形最低相关.99999983，duration一致。稀疏抽检并非独立口型/感知评估，慢变眉毛/眨眼污染仍存在，见本地supervision_samples/REVIEW.md。

正式run已实际启动于`/root/kinetalk_runs/formal_predictable_v1`，持久bundle约650MB/cache793MB，root盘仍23GB，无数据删除。只用正式train2720重拟输入RMS/U与alpha，固定rank8，句内3fold选alpha=1；原280开发upper R²=.02201 CI[.01505,.03265]，reverse=-.01300，oracle=.52003，可预测性没有消失。不得将旧适配器接此新基。

正式seed46进程PID6781已完成18epoch/3060步，由预设patience早停，选中epoch2。每epoch完整2720样本；全程已验证冻结hash。已保存`rrr_seed46/{last,best}.pt`（约172KB/86KB）、recipe/source/input SHA、Adam/全部RNG，final曲线sidecar绑定模型。`launch_seed46.json`含精确启动与恢复参数。入口`train_formal_predictable_projection.py`，用相同参数另加`--resume OUTPUT/last.pt`恢复；来源代码/input变化会拒绝。

该正式seed46在原280复核upper ΔR²=.012925 CI[.006855,.021346]，brows ΔR²=.005843 CI[.002072,.011657]；这280已参与epoch选择，区间仅选择后的描述性结果。默认模型未替换。seed47/48和相同预算PCA46复核随后启动，native-val439另做跨身份评价；原native-test512及保留15未读target。
