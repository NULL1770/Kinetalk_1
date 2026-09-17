# History12 结果核验与报告约定

本文件在 history12 正式训练运行期间编写，依据 `scripts/train_history_upper.py`、`kinetalk_b0/models/history_upper_flow.py` 和 `docs/HISTORY_CONTEXT_PROTOCOL_20260917.md` 的已实现接口。它规定如何审查最终产物，不代表训练已经完成或任何指标已经改善。不改训练代码，不接触封存测试集，不用开发结果选择 epoch、噪声种子、输出增益或时间偏移。

本轮的核心问题是：在同一 chunk flow 参数化、训练预算与音频条件下，训练和部署使用前序动作是否改善自回归生成。主比较只能是 `scheduled_history/full` 对 `no_history/full`。与之前 `state_white`、`state_aligned` 的比较用于衡量候选效果变化，不能单独归因于 history。

## 1. 产物完整性与来源

- [ ] 两臂均有 `provenance.json`、12份 `epochNNN.json`、`last.pt`、`final.pt`、`evaluation.json`、`curves.pt`、`complete.json`；根目录有完成状态和 `matched_audit.json`。不把 smoke 产物当正式结果。
- [ ] 检查 `schema=scheduled_history_upper_v1`，两臂实际完成12 epoch、固定最终 checkpoint、完整405个内部开发片段、`test_loaded=false`，最终种子列表严格为 `[42,123,2026]`。
- [ ] 重新计算 final/curves 的 SHA256并对照 `complete.json`；canonical recipe hash与 provenance、checkpoint、complete一致。检查源脚本hash对应本次运行版本，不能用后来修改的代码解释旧曲线。
- [ ] 比较两臂 `initial_sha256`、12个 epoch 的 `batch_noise_time_decision_sha256`、总step与预算。初始 upper/local 参数必须相同；随机批次、训练噪声、自rollout噪声、flow time及teacher选择随机数必须相同。
- [ ] 验证 system/audio 冻结hash与来源checkpoint一致。`no_history`参数中也包含历史编码器，但历史参数不接收梯度；其teacher选择计数只是“选择过的来源”，不能写成实际使用GT历史。
- [ ] 本轮每个完整epoch预期 `ceil(2315/16)=145` 次更新，12轮1740次；若recipe改变了样本数或batch，按实际recipe重新计算，不硬套此数。
- [ ] teacher概率以recipe和epoch记录为准。当前12轮实现从第8轮开始为0，即最后5轮全generated；检查这5轮 `teacher_selected_count=0`。
- [ ] 新旧及两臂 `clip_id`顺序、target、valid、channel_mask、times、emotion_id、speaker_id一致；b0也需核对。检查 native 相邻时刻差为0.04秒，所有九维上脸可观测。若不一致，停止直接做逐片配对比较。

`curves.pt`记录 target、valid、times、channel_mask、b0、标签、predictions、seeds、decode_steps，但不独立记录 arm、chunk/history长度、static_upper、源checkpoint或每个mode的oracle标记。必须同时绑定同目录recipe/evaluation/complete。仅凭曲线文件名不能建立来源或部署语义。静态参考值若需要精确重建，应沿recipe中的冻结audio和独立身份参考；不要把新的GT均值补作输入。

## 2. 部署与oracle隔离

| 曲线键 | 输入语义 | 报告位置 |
|---|---|---|
| `42/full`、`123/full`、`2026/full` | 只用已生成过去动作；音频/global可含整句上下文 | 正式部署主表、三seed分布评分 |
| `42/empty` | 所有chunk清空历史；音频保持 | 固定seed历史作用消融 |
| `42/reverse_history` | 每次已生成前8帧中有效动作逆序；mask时刻不动 | 固定seed历史顺序敏感性 |
| `42/static`、`42/reverse` | 改audio local+h0时间结构；global/identity/基座保持 | 音频时序条件依赖，不是物理事件因果证明 |
| `42/oracle_history` | 每个chunk使用真实前8帧，禁止当前和未来GT | 明确标红的ORACLE诊断，不能进入部署主表或主视频 |

- [ ] `full`推理不读取query motion/emotion，也不以GT全序列均值构造history条件。GT前序只来自 `target[:, max(0,start-8):start]`，归一化减固定audio static并除train尺度。
- [ ] `normalized_target`不以当前chunk或完整GT序列中心化。否则前序GT数值会携带未来信息，当前oracle的解释将失效。
- [ ] 同一臂、同一seed的 `full/empty/reverse_history/oracle_history` 首chunk必须完全一致，因为开头没有历史。`static/reverse`不受这一断言约束，它们改变当前音频条件。
- [ ] 对 `no_history`，`full/empty/reverse_history/oracle_history` 整段输出应逐值一致。若不同，优先查条件或噪声是否被额外改变，不直接解释为模型效应。
- [ ] `scheduled_history` 的 `full`优于`empty`才支持部署时历史有帮助；仅“输出发生变化”只证明通路被使用。对历史逆序有响应说明顺序敏感，但不自动证明是正确动作时序。
- [ ] oracle与full差距单列。oracle改善而full未改善，应解释为真实历史可利用、生成历史存在误差累积/分布差距，不能按oracle成绩宣称成功；oracle也非理论上限，异常差结果应检查GT和生成历史分布、归一化与时钟。

`evaluation.distribution.single_seed_interventions`复用通用汇总函数，因此包含oracle项。导出程序必须显式分开ORACLE和deployable；不要因为它与其他mode共用字典就合并展示。只有full具有3seed，不能把seed42消融伪装成3seed均值或best-of-3。

## 3. 主要指标与比较单位

- [ ] 对brows五维、eyes_expression四维分别报告raw MSE、中心MSE、中心相关、中心R²、RMS/GT、相邻帧位移误差、越界率；同报全部405、neutral74、nonneutral331，必要时按三位开发身份检查是否收益集中于一人。
- [ ] 主表保留每seed和三seed均值；不只报幅度或相关中的最佳一项。中心R²仍小于0时，说明配对平方误差尚未胜过每条GT时间均值的诊断性常数参考，不能据幅度接近1宣布动作恢复成功。
- [ ] ES同时看raw和centered，VS看相邻时序；fair和empirical都保留。lower更好，3seed估计方差较大，公平VS可为负，不能简单截到0。ES是整条轨迹按有效维数归一后等clip平均；MSE通常按有效帧/通道池化，两个汇总权重不同。
- [ ] 幅度RMS接近1、相邻相关接近GT、VS变好是动态分布与连贯性的证据；若配对相关低、中心MSE差，不能称准确预测了每个眉动作时刻。
- [ ] 训练loss仅用于优化诊断，不是最终动作质量指标；不用它在两臂内挑epoch或掩盖rollout退化。
- [ ] 不设置看到最终结果后才提出的百分比通过线。报告绝对数、配对差值与不利变化；若需显著性，须另说明开发集重复使用及只有三个开发身份的限制。

## 4. Chunk边界与自回归漂移

当前 `boundary_report` 将索引15→16、31→32、47→48、63→64、79→80定义为chunk边界，只统计相邻两帧都有效的pair；`inside_chunk`包含其他有效相邻pair。结果的RMS是系数/帧位移，不是系数/秒速度。换算物理速度需要除以0.04秒，不能改名后省略换算。

- [ ] 两臂三seed分别核对每个区域的边界pair数一致，与mask直接计算一致。缺检测跨gap不得加入差分；不是所有clips都贡献全部5个边界。
- [ ] 同时看边界 `pred_rms`、`gt_rms`、`displacement_mse`和chunk内对应值。不能仅因边界RMS下降就验收：静止输出也能消除跳变，但会低估GT运动。
- [ ] 对照GT的边界/内部差异，报告预测额外的边界尖峰；不要求边界和内部RMS无条件相等，因为固定切点仍可能恰好遇到真实动作。
- [ ] 从保存curves只读补算chunk序号0–5的raw MSE、相关/RMS、平均偏移、越界率以及各边界误差。首、中、末段至少分别展示；全片聚合可能掩盖后半段误差增长。
- [ ] 漂移主诊断使用raw值、相对同片GT的局部均值偏差、最后段相对首段的变化；不要先给每个chunk去均值再判断是否漂移，那会隐藏均值游走。chunk内中心相关只能作为辅助动态形状描述。
- [ ] 避免各chunk的样本组成不同导致假漂移：优先用所有比较段都有足够有效帧的同一批clip，对共同mask统计；同时报告完整样本的实际clip/frame数量。首尾差异也可能来自真实内容变化，需看误差变化和GT变化，而非只看预测幅度。
- [ ] `full-empty`、`full-oracle`差异随chunk编号的增长一并看。generated full后段恶化且oracle改善，支持生成历史误差累积；两臂都恶化可能是共同chunk模型/位置重置/训练覆盖问题，不能归因于history单一部件。
- [ ] 有效历史不足8帧或跨检测gap的片段单独标记覆盖率。历史在mask gap不更新，age仍保留原生帧间隔；不能把缺帧压缩成连续动作再比较。

本轮固定96帧，约3.84秒。即便6个chunk内稳定，也不能据此宣称长时稳定、整句任意长度或流式实时。对更长片段的后续评估应另定协议，不改变本次固定结果。

## 5. 口型、身份、情感和输出有效范围

- [ ] 对全部mode/seed，43个非上脸通道与recipe绑定的 frozen Stage3 original-local baseline逐位比较；同时核对两臂对应seed的这些通道一致。嘴部14–40只覆盖其中27维，不能用嘴部指标相同替代43维保护检查。
- [ ] 无效帧应保留传入baseline；不要把padding清零后冒充原值拷贝。主质量指标只算有效、可观测帧。
- [ ] 口部与该共同基座的全部指标应一致。它相对run12其他stage的变化来自基座选择，不能归功于history；逐位保护只说明没有额外改坏系数，不证明原口型已准确同步。
- [ ] system/audio/identity编码器冻结不意味着生成上脸的身份和全局情感自动不变。新flow允许九维均值改变，必须报告生成情感teacher读数变化，并在固定视频上检查表情形态和身份风格。
- [ ] 情感读出来自训练过的motion teacher，是非独立指标，不可冒充外部情感准确率或用户研究。参考身份检索也不是生成头像的独立身份得分。
- [ ] 原始系数越界率必须与raw误差一起报告，分别看brows、eyes、mouth。可从曲线补充越界幅度/上下界分布区分轻微负值与大幅错误，但不得clamp后再计算原始得分。
- [ ] 若渲染器对系数clamp，视频必须说明；同时保留未clamp曲线和分数，不能让视觉clamp掩盖模型越界。

## 6. 与旧state_mean比较的限制和固定锚点

旧 `state_white` 使用96帧整段零均值flow，最后将其中心动态加到固定audio4Dstate静态均值；`state_aligned`使用旧aligned renderer的中心动态。新history两臂使用16帧chunk flow、不同残差尺度、未中心化目标、可改变九维均值、新seed/训练和自回归采样。即使固定audio静态参考和其余43通道一致，变化也包含这些多个因素。

因此：`scheduled_history-full` 对 `no_history-full`是本轮history作用证据；对旧state_mean是开发候选综合结果比较。没有额外的teacher-history-only训练臂，不能把差异进一步单独归因于scheduled sampling。旧方案与新方案都使用已反复查看的405内部开发片段，绝不是未见测试集结果。

旧state_mean的三seed均值（仅用于核对报告，最终比较须验证曲线来源和mask）：

| 指标 | state_white眉 | state_white眼 | state_aligned眉 | state_aligned眼 |
|---|---:|---:|---:|---:|
| raw MSE | 0.01785991 | 0.00695785 | 0.01773277 | 0.00659319 |
| centered MSE | 0.00112264 | 0.00138538 | 0.00099550 | 0.00102071 |
| centered correlation | 0.09976 | 0.11836 | 0.13234 | 0.18752 |
| RMS/GT | 0.82888 | 0.95114 | 0.73810 | 0.70958 |
| outside fraction | 21.831% | 9.740% | 22.031% | 12.459% |
| fair raw ES | 0.09206235 | 0.05794713 | 0.09390852 | 0.06139477 |
| fair centered ES | 0.01734087 | 0.01969575 | 0.01833311 | 0.02076920 |
| fair adjacent VS | 0.00148906 | 0.00123129 | 0.00335660 | 0.00292069 |

两种旧state_mean共同嘴部raw MSE为0.00934787，越界率27.480%。该嘴部也是本轮承诺逐位保留的基座。对run12 dynamics，旧state_white虽raw与中心误差改善，嘴部ES并未全面改善；新报告同样不得只挑raw收益。

## 7. 最终交付表述

- [ ] 主表至少列旧state_white、旧state_aligned、no_history/full、scheduled_history/full；oracle独立表，不混在可部署候选名次中。
- [ ] 固定视频使用既定身份/片段/seed、同音轨和native时钟，显示chunk边界标记或配套曲线；不给不同方法调整增益、时移、相机或表情倍率。先看完整片段，不只选抬眉最明显的一帧。
- [ ] 单独写明：历史通路是否有效；generated历史是否改善边界/漂移；音频时序是否有响应；mouth/其他通道是否保护；越界和情感是否有代价；oracle/full差距多大。
- [ ] 若只有oracle成功，结论为训练条件可用但部署差距未解决。若幅度增加且跳变/越界/漂移恶化，结论为未通过联合验收。若部署连贯性和分布评分改善但相关仍低，结论限于生成动态改善，不能写成已准确重建眉动作。
- [ ] 未达到联合目标也保存两个最终checkpoint、原始指标、曲线与可复查视频，并说明下一步针对的失败环节；不以长时训练或论文创新语言替代结果证据。
