# Phase30：冻结最新保留候选的逐帧u_a因果消融

2026-10-06，用户明确要求先做u_a作用核查。主模型为Phase26 standardized12三seed47/48/49（读取Phase29中已核SHA的同一control权重/曲线），Phase29 dropout已拒绝。不修改训练代码、模型、输出支持或loss，不拟合/读取sealed test。

全1367 validation，native-padded噪声draw42/123/2026，batch16、12Euler，原128/64及辅助四冻结TRAIN probes，raw/clip分别报。每个模型先全部三draw正常audio输出与原完整曲线逐位重放、GT/mask/time/B0精确相同，通过之后才启动任何干预。

只改u_a：audio原序列；zero全部有效帧归零；static用该clip有效帧均值；reverse反转有效帧顺序；shuffle用clip_id和固定20261006种子确定的一次置换，在所有draw/model/batch中一致。padding及缺失帧保持零，不移入有效段。reverse/shuffle保留每帧向量集合与边际分布；static保留clip均值，zero同时移除均值和动态。保留global、intensity、logits、content/h0、B0、identity和native初始noise对象与值。

不从效果改善断言因果唯一归因：reverse/shuffle分布外干预，只有固定模型依赖时间顺序的敏感性证据。若static与audio几乎相同而zero变化明显，倾向主要使用静态信息；若reverse/shuffle恶化时序，则存在有益时序利用；若破坏顺序反而改善，需要检查u_a混入的不合适动态/renderer利用。零影响也可能是冗余、弱权重或其他路径替代，不能直接宣布u_a没用。

用户追问内容泄露后、首次执行前追加固定content_static条件：实际1540输入明确包含HuBERT768内容，global/u_a共享TCN，不存在已证明的内容独立性。仅为提取替代u_a将有效帧前768维换为clip均值，保持emotion2vec/prosody和其他帧原值；替代audio global/intensity/logits全部丢弃，生成继续使用原affect里的对应对象和原content/h0/B0。它测直接HuBERT时间信息对u_a路线的敏感性；emotion2vec仍可能含音素，所以不能证明纯情感。所有6条件在任何评分结果前固定，不根据结果加/删条件。

评分：MBE/LBE、mouth/jaw/eyes/brows/cheeknose的均值偏差、centered correlation、范围、位移MSE；同632vertex rig的Lip mean/max、EVE及FDD；四probe生成F1及emotion/intensity/speaker分组；相对audio的逐区域输出响应、输入u_a动静态能量；同说话人配对bootstrap（验证仅3位，区间限制明确）。保存所有固定seed/draw结果及8类预登记片段系数，不挑结果/不部署消融输出。

结果状态、bindings、frozen-state/source/curve/probe/rig SHA均落盘。未完成前不进入门控/逐帧蒸馏方案。Phase29完整下载已通过492文件SHA核验，旧下载任务结束。
