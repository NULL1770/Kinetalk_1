# Phase41：已批准的新架构实施与训练

2026-10-07。恢复时首先读本文及 CURRENT_OPTIMIZATION_STATE.md。用户最新授权：先上传当前成果，再实施 proposal_v2、清理冗余、验证、训练并安排评估和渲染；确认训练启动后结束对话并给预计耗时。旧文档待批准/Git暂停已失效。

## 已完成

- 当前成果 Git 快照 bacdd88 已推送至 codex/experiment-snapshot-20260918。
- 服务器可连接、4090 空闲；root 仅约577MiB、数据盘约310MiB空闲，须先回收有备份的冗余。

## 实际实现与启动（最高覆盖）

- 新入口 `scripts/train_expression_response.py`、模型 `kinetalk_b0/models/expression_response.py`；2,176,948个新可训练参数。历史trainer仅复用原生B0调用/类权重工具，不使用其flow目标或旧teacher。
- q：3个masked TCN＋2层4-head attention，hidden128；p：4个TCN＋2层4-head attention，hidden128。全局g32、局部u16、stride2，按固定token中心插值到25fps，不按clip长度伸缩时钟。
- style：3个TCN128，masked pooling→64D，集合聚合；两段参考分别形成独立单参考support，重复同一query各预测一次。部署时聚合两段参考。bias52与4个192D temporal residual decoder共同训练，全52D输出，无嘴部排除。
- q路径位置＋0.5真实相邻位移误差；KL从0.0025到0.01用4轮升温；0.1全局语义（类别/强度）；第5轮起0.5权重的detached p条件动作重建，只适配D/S。这是三类目标，尚未证明这些固定权重最优。
- p的772D选择发生在归一化/有限检查之前；HuBERT NaN替换结果不变。B0输出detach且模型冻结，不传h0。实际GPU q重建和p条件重建对p梯度均为零，p由KL和语义训练。
- 全本地408 passed/1 skipped；远端9个新单测通过；120次真实小样本更新（v2，归一化也仅由fit8条计算）位置误差前10步均值1.163541→末10步0.268732，ratio0.230960。仅证明可学习，非泛化成绩。
- 真实GPU两步随机训练及detached-p适配验证：loss1.013393→0.855235；恢复optimizer/RNG后第二步loss与所有权重逐位相等。B0权重校验未变。16条真实validation已跑完四probe、几何、动态干预评分，验证评估接口。
- 12个重复tar.zst传输包经本地/远端逐个SHA复核后删除远端副本，回收2,302,601,588 bytes；唯一源数据及所有历史final保留。恢复位置 `final_experiment/remote_archives/phase41_transport_backup`；verified/reclaimed.json为凭据。

## 本轮训练数据与预算

|部分|实际数据/职责|
|---|---|
|冻结B0|沿用正确neutral检查点：715条native neutral＋2583条批准emotion→neutral pair的旧训练成果；本轮不再训练B0|
|新q/p/style/decoder|从原12,536条TRAIN里固定留出2个人743条，以及其他人的句子890条；fit10,903条，20个身份|
|内部诊断|held speaker IDs23/24；句子由SHA256取模固定选择。每轮分别固定最多256条检查prior/posterior误差，完整索引落盘|
|归一化|772D均值/标准差与52D动作scale仅用fit10,903条计算，不读取内部留出或validation|
|参考|25个TRAIN/validation身份共50条独立neutral enrollment，每人2条；与query没有录音ID或句子重叠|
|最终开发评估|原1,367条validation的3个身份，完整原生帧；raw/clip_all、4个固定probe、同rig几何和动态；不能当sealed test|

单seed47、24轮、batch16原query（两个support→实际32条重建），16,368次更新；固定学习率2e-4 AdamW。每轮原子last.pt保存optimizer/随机数，final.pt为最终轮。没有按validation选择checkpoint，也没有自动推广默认。

## 唯一后台任务与恢复

- 远端root：`/root/kinetalk_phase41_expression_response_20261007`；pipeline PID2694，launch_contract/binding和所有source SHA已冻结。
- `pipeline_state.json`为阶段状态；`seed47/state.json`与`history.json`为更新/每轮曲线；`pipeline.log`为日志。
- 顺序运行训练→完整评估→导出固定八类视频输入；本地唯一collector PID13360等待完成→原SHA备份→Blender渲染八视频。
- 本地状态：`final_experiment/evaluation/diagnostics/phase41_expression_response_20261007/local_queue_state.json`。后台collector与渲染依赖本机保持开机，GPU训练/评分依赖服务器运行。
- 视频四列：GT / Neutral B0 / Audio prior mean / Posterior ORACLE。最后一列明确使用GT，只定位接收器上限，不能当部署结果；原生时钟，显示clip[0,1]，不调幅、不挑seed。
- 不重复运行launch或collector；需要恢复时先看状态与真实进程。训练异常写pipeline_state failed，collector也保存失败原因；保留现场。脚本支持显式resume，但不能直接重跑带独占lock的pipeline启动。

## 待完成与限制

训练已排入后台，本轮还没有完整开发成绩。这是新架构首轮，不是R0–R4全部完成：旧flow/四状态公平容量消融、joint vs frozen q/p、概率vs点表示、u容量、配对交换、独立音素probe、风格目标分布迁移、混合情感参考、三seed及公开benchmark仍须后续证据推进。

动态诊断有static/reverse/shuffle u、同人同情感同强度异句audio、换参考A/B/异人，以及GT static/reverse/shift/gain。错误音频负控显式长度映射仅用于干预；正常推理不改时钟。只有差异不能证明内容完全解耦或身份迁移正确。

## 错误记录

- 初次部署缺少远端tests目录，上传中止，未启动训练；创建目录后成功，未覆盖旧实验。
- preflight脚本嵌套ROOT占位替换引发Python语法错误，远端未执行；改独立RUN_PATH占位后四阶段通过。
- v1小样本使用原全TRAIN统计；v2修正为只在当前fit子集拟合，保留v1源/日志作记录，正式训练采用v2。

## 当前实施顺序

1. 核验数据、neutral B0、备份与磁盘预算；保留唯一数据和历史最终检查点。
2. 独立实现 content-conditioned response：B0冻结；p仅772D；q仅训练读动作；g32/u16 stride2；跨参考style64；4块192D decoder，全52D。
3. 三类目标：原生位置/相邻速度、归一化KL、全局类别/强度。重建梯度不进p；部署条件detach适配D/S。不默认使用未经验收h0。
4. 完成输入/梯度/掩码/时钟/参考独立/保存恢复测试、真实GPU小样本可学习性和稳定性验收。
5. 单seed固定预算首轮，阶段检查与断点；完成后自动全validation评分、因素诊断、八情感固定视频。后续三seed与消融需依据本轮曲线，不能把本轮叫全部论文验证完成。

## 边界与待验收

- sealed test不读取，不替换已发布默认，不把posterior oracle当部署成绩。
- 两段neutral参考只支持受限风格协议；mixed-emotion参考未启用。
- 高质量配对交换属于后续独立验收：先native重建与q/p可预测性，不能将有DTW的pair直接假装严格逐帧真值。
- 当前没有新模型成绩；不预称SOTA、F1≥0.7或必然可发表。
