# 完整 fit：前缀接收与整窗生成机制的固定终点诊断

日期：2026-09-17。本协议在本轮三臂训练之前固定。上一轮 prefix12 已完成，未获得足够的眉眼连续动态证据；本轮依据这一已知结果设计，属于后续机制定位，不是对上一轮结论的重新包装。本轮不承诺模型成功、论文创新或泛化改善。

## 1. 本轮回答的问题

1. 使用完整 fit、始终供应真实过去训练时，生成器能否在独立噪声完整求解下利用过去动作，改善未知区的续接？分别检查选定 fit 与内部开发数据。
2. 同一 teacher 模型改用自己生成的过去时，接收收益能否保留？真实过去成功与可部署生成成功必须分开。
3. 在相同目标、冻结条件和更新预算下，一次生成 96 帧是否优于反复生成 16 帧的分块机制？

第三项同时改变运动注意力范围、可访问的声学 token 范围、后续帧位置和求解重启。因此只能解释为**整窗机制对照**，不能把全部差异单独归因于噪声重采样或接缝。对所有未知帧使用共同的一次性初始噪声，求解每一步不重新抽噪声。

本轮不新增 loss、监督数据成员、标签、文本、时序覆盖或随机 crop；不扩展情感类别、L2、人物、台词、窗口长度对应的原始数据，也不读取封存 test。默认模型不替换。

## 2. 数据、坐标和来源

严格复用当前 **2315 fit / 405 内部开发**及独立 neutral enrollment。每条保持固定 96 帧、原生 25Hz、原 times/valid/channel mask 和 clip 顺序；缺失帧不压缩、不跨 gap 或跨 clip 接续。405 条是已经反复用于开发的数据，不称最终测试，不据此声称未见身份或未见句子泛化。

三臂均从同一个原始 `history12/no_history` checkpoint 的共享上脸 backbone 初始化，复用其冻结 local 条件网络和配置，**不加载 pilot、prefix12 或本轮其他臂的更新后参数**。共享参数逐项一致；known/unknown embedding 统一零初始化。旧 history 压缩器的弃用键须显式枚举并检查，不能静默忽略额外 missing/unexpected keys。记录来源权重、recipe、数据与初始化 SHA。

目标坐标沿用来源：

`normalized_upper9 = (GT_upper9 - frozen_audio_predicted_static_upper9) / source_fit_scale9`

其中静态部分仍由独立 neutral anchor 与冻结音频状态预测构成，不减 query GT 均值，不按 chunk 重新中心化，不使用当前或未来 GT 估计原点，不额外乘旧全脸 renderer 的 residual_scale。完整 clip 内所有块共享同一音频预测原点。尺度复用来源 fit 尺度，不能在开发、诊断子集或本轮输出上重估。

只更新九维上脸 flow 及其 known/unknown embedding。`system`、`audio`、`local`、身份路径和静态音频预测均冻结，训练前后校验 hash 且不得产生参数梯度。全局情感条件保留；本轮不新增在线 motion→audio 蒸馏。九维范围为五个 brow 与四个 squint/wide 通道，blink/gaze 不属于本次新生成变量。

正式全脸合成输出保留同输入、同 seed 的原冻结全脸基座：其余 43 通道与无效帧逐位复制，并逐位审计。来源曲线或新算的冻结基座应记录来源与 metadata 绑定。fit 接收诊断若仅生成九维上脸、以零占位另外43维，必须保存 `nonupper_placeholders=true` / `nonupper_scored=false`，该文件只用于有效上脸指标，不能冒充完整全脸输出或直接用于52维视频。系数保持不等价于独立验证了唇音同步、人物几何身份或情感自然度。

## 3. 三臂的精确定义

| 臂 | 训练输入与未知区 | 部署推理 |
|---|---|---|
| `chunk_empty` | 每次 8 个 invalid 前缀槽 + 当前 16 帧，共 24 槽；不读任何过去动作 | 六块各独立求解；过去 8 槽始终 invalid |
| `chunk_teacher` | 每次严格过去 8 个原生位置 + 当前 16 帧，共 24 槽；有效过去为干净 GT known，仅当前有效 unknown 计 FM | 首块无过去；后续仅用本次此前生成的动作作 known |
| `whole` | **8 个 invalid 前缀槽 + 当前完整 96 帧，共 104 槽**；全部有效当前帧为 unknown；无 known 动作 | 一次求解全部 96 帧，去掉最前 8 个 padding 槽得到输出 |

`whole` 不是 96-token 输入。其当前帧位置为 8–103，前 16 个当前帧位置与两种 chunk 输入的 8–23 一致；chunk 后续块重复使用 8–23，whole 后续继续使用 24–103。首块位置匹配不能消除后续位置、长程注意力和上下文范围的差异。

过去动作进入与当前未知动作相同的 DiT attention。GT known 只允许来自严格之前的有效原生位置，首块过去全部 invalid；没有有效紧邻前帧时不伪造连续边界。训练所有 flow time 和推理所有 Euler 步都保持 known 干净且逐位固定，unknown 由一次抽样的初始噪声积分。known 和 padding 不计 FM 分子或分母。

`chunk_empty` 删除整个过去 token，因此同时移除该位置的动作和声学 attention 上下文。与 `chunk_teacher` 的臂间差异不能仅归因于动作值；同 teacher 权重的 GT 顺序/逆序诊断用于进一步检查动作值是否被使用。

音频条件及全局状态仍可依赖离线窗口信息，whole 也有完整当前窗口 attention；本轮不声称在线因果生成。

## 4. 固定训练计划与随机流

| 项目 | 固定设置 |
|---|---|
| 训练臂 | `chunk_empty` / `chunk_teacher` / `whole` |
| 预算 | 每臂固定 12 epoch；每轮遍历 2315 fit 一次 |
| batch / 更新 | batch16；145 次 optimizer 更新/epoch，1740 次/臂 |
| 随机 seed | 83；三臂一致的 Python、NumPy、Torch 与独立训练 generator 初态 |
| 优化 | 沿现有标准 FM 优化设置：AdamW，lr=1e-4，weight decay=1e-5，梯度范数 clip=1；非有限 loss/梯度报错 |
| FM | 每 clip 一个标量 flow time；每 clip 一份 `[96,9]` 标准高斯初始噪声 |
| 推理 | 固定 12 步 Euler；固定 seed42/123/2026，不做 best-of-K |
| 终点 | 固定 epoch12，不按开发/oracle/视觉效果挑中间 checkpoint |

每个 batch 三臂使用同一 clip 顺序、同一 `[B,96,9]` 噪声和同一 `[B]` flow time。两个 chunk 臂只切取对应当前 16 帧噪声并填入 24 槽的未知区；其六块共用该 clip 的一个 flow time。whole 将同一噪声置于 104 槽的当前 96 帧，前八槽无效。不能为某一臂额外抽样而改变后续 generator 流。

FM 仅在有效 unknown 上计算标准平方误差。chunk 六块先分别计算，再按有效未知帧数加权合成一个完整 batch loss；whole 直接对相同有效帧求平均。**每 batch 只调用一次 optimizer.step**，不能每块更新一次。九个目标通道均须有效，否则应明确拒绝不满足既有数据契约的输入，而不是悄悄改变分母。

`chunk_teacher` 每次训练都使用真实过去，无 scheduled sampling、self-rollout、生成历史扰动或额外接缝损失；`chunk_empty` 与 `whole` 训练不读取 GT 历史条件。三臂均使用 GT 当前目标进行标准 FM 监督，这与把当前 GT 当条件不同。

记录初值 hash、每轮 clip/noise/time 随机流 hash、更新数、loss、用时、checkpoint/optimizer/RNG。共同更新数和未知帧数不是等算力证明：104-token attention 与 6×24-token forward 的计算不同；省去上一轮训练 self-rollout 是否更快以实测时长为准，不预先宣称速度优势。

## 5. 在新输出之前锁定 fit 诊断子集

固定选择 128 条现有 fit 用于训练前与固定终点接收诊断。选择只读 clip metadata，不使用动作幅度、模型误差、oracle 收益或任何本轮生成结果：

1. 按 `(speaker_id, emotion_id)` 分组，组按整数元组升序排列；每组 clip 按 `clip_id` 字符串升序。
2. 第一轮从每个非空组取第一个 clip，第二轮取第二个，依次轮转；不足的组跳过，累计到 128 条立即停止。
3. 保存有序 clip_id、原 fit index、分组字段、具体排序规则，绑定 selection 文件 SHA；训练前锁定，step0/epoch12 完全复用。

分层只控制 metadata 覆盖，不保证 128 条代表全体 2315 的运动分布；报告总组数、各组入选数和有效前缀覆盖。该子集均属于训练曝光数据，不另立新数据成员或外推泛化。训练仍完整使用原 2315 条。

为控制重复曲线体积，本轮 step0 固定为上述 **128 fit、seed42**，不把可选 32-dev step0 作为必须完成项。终点评估仍包括全部 405 dev。

## 6. 训练前 step0 与位置偏移诊断

在更新前、同一共享权重下，128 fit seed42 至少保存三臂 `full` 的完整求解结果，以及 `chunk_teacher` 的 GT-history oracle 接收结果。各臂使用相同当前噪声、静态音频条件、身份和基座；记录没有任何优化步。step0 是机制基准，不用于按表现选择初始化或替换固定训练计划。

另做不训练的 `position_offset_probe`：用同一权重、相同有效当前 16 帧、当前音频/全局/身份和相同 `[16,9]` 噪声，比较：

- 直接 16 槽、所有有效当前帧 unknown、位置 0–15；
- 8 个 invalid 槽 + 同一当前 16 帧、位置 8–23。

两者均不供应 GT 或生成历史；只比较两种张量布局对同一当前块的影响。沿固定128 fit的既有6块及原mask计算，空块跳过，保存比较数与噪声绑定。该诊断量化现有 warmstart 从原位置到空前缀布局的变化，不增加训练臂。若差异明显，后续结果需披露该长度/位置迁移，不能假定“同权重”就等价于“同函数”。

whole 前16帧虽位置一致，但看见其余当前帧；它与 chunk step0 的差异不能当成纯位置偏移诊断。whole 的新位置覆盖和上下文分布也不自动因为 sinusoidal 编码而获得训练适应性。

## 7. 固定终点评估与 oracle 边界

epoch12 在全部 **405 dev、seed42/123/2026** 上评价三臂部署 `full`；在已锁定 **128 fit、seed42** 上评价接收与部署。fit 与 dev 分表，不把 fit 接收改善解释成音频跨模态泛化。

同模型 seed42 的重点干预为：

- `full`：`chunk_teacher` 只用此前生成动作；其余臂没有动作前缀；
- `empty`：清空过去 token，保持当前布局、音频、global、身份和噪声；
- `oracle_history`：**只有 `chunk_teacher` 允许读 GT 严格过去**；
- `oracle_reverse_history`：**只有 `chunk_teacher` 允许**，把实际有效 GT 过去动作按原有效槽逆序，保留 known mask、时间槽和过去声学条件；不逆序声学输入，不引入当前/未来 GT。

如保留现有 generated-history 逆序、local-audio static/reverse 等诊断，使用明确独立名称与既有语义，不能把 generated-history 逆序冒充 GT-history 逆序。`chunk_empty`/`whole` 的所有推理模式均不得为了方便代码路径而读取 GT 历史；无意义的 oracle 模式应拒绝或显式标为未使用历史的空操作，不能让该结果进入接收论据。

oracle 表、曲线和来源须明确标为 **GT-conditioned diagnostic only**。主部署表、三 seed 分布指标、主视频不混入 oracle，不比较“挑最好 seed 的 oracle”与普通部署输出。FM loss 下降、known 复制精确、GT过去下输出幅度增大，都不能单独证明 audio 能预测目标动态。

## 8. 质量、连接和分布指标

眉五通道与表情眼四通道分别报告 raw MSE、centered MSE/相关、动态 RMS/参考、越界率、帧间位移；dev full 三 seed 沿现有 fair ES/VS 分布评价。neutral/nonneutral 与有效样本数沿既有协议列出。仍用训练 motion teacher 读出的情感分数必须标为非独立。

连接相关指标严格区分：

1. **生成轨迹相邻块边界**：对两个 chunk 臂是实际解码拼接；oracle 是不同真实前缀重置后的独立预测块拼接。
2. **实际输入前缀的续接**：当前首预测减实际供应的 known 末 token。普通 oracle 用GT端点，GT逆序用逆序后的实际末 token，generated/full用生成端点；empty与whole不适用。
3. **共同GT端点评分**：所有臂都可在评分时计算当前预测首帧相对GT前帧的误差，但这不表示所有臂输入过GT。普通oracle下该位移误差等价于首未知帧 endpoint MSE，不作为另一个独立成功证据。

上述边界只计前一原生位置和当前首位置均有效的对；不跨 gap 寻找最后有效帧代替。GT逆序的实际已知末 token若来自较早历史，应明确其来源，不能把它误称原时间的相邻GT前帧。

whole 没有每16帧的解码接缝。可沿相同 16 帧网格计算位移用于公平位置比较，但必须标为 **固定网格诊断，非解码拼接**；whole 的实际 supplied-prefix continuation 一律不适用。

同时保留每块 raw均值、均值误差、首2/4帧误差、块内位移、整块动态与越界，以及相同mask覆盖规则下的共同片段 profiles。接缝下降若来自静态化、幅度崩塌或整体偏置，不算续接解决。单GT逐帧相关也不是一对多自然度充分判据，多样性大不自动代表正确动态。

展示沿预先固定样本、同音轨和rig。曲线/评分不做gain、平滑、滞后拟合或clamp；渲染为显示若截断到[0,1]，须单独报告，并保留原始52维数组和输入/音频/视频hash。统一rig不代表独立身份评价。

## 9. 审计与解释边界

必须验证：三臂共享参数初值一致；共同 clip/noise/time 流与1740更新匹配；known逐步精确保留；过去索引严格小于当前窗；GT逆序仅影响允许的过去动作值；whole/empty推理无GT输入；首块各历史干预在同权重下相等；缺失/padding不污染损失与推理；冻结模块无权重/梯度变化；其余43通道和无效帧逐位复制。

保存协议与源码hash、source/data/selection绑定、初始/最终权重与曲线hash；固定终点完整报告。运行错误可以修复并披露，不能看过结果后静默改数据覆盖、训练预算、噪声、坐标或统计口径。

结果按证据解释：若GT-history在fit已有收益而dev没有，只有选定训练片段接收证据；若GT-history有收益但generated没有，部署过去分布差异仍未解决；若whole改善，只能支持本固定设置下整窗机制更合适；若三臂均弱，报告当前来源、参数化与固定预算未建立目标动态，不能宣称理论上音频无信息或模型必然无效。

本轮固定12轮后停止，不按开发结果临时续训、替换默认、增幅或改判据。后续若要扩展时序覆盖、数据契约、音频预测条件或概率建模，须另立可审查实验。工程诊断成功本身不保证论文创新、CCF-C录用或最终系统完成。
