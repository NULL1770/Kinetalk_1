# 前缀接收实验：一次有上限的继续训练

在pilot15全部结果已读后预先声明，不能当作最初预注册的同一实验。原pilot15输出/门槛不修改。它有90步/臂，FM仍下降，实际输入前缀续接已有收益，但眉接缝位移仍为参考2.02064倍，不能四舍五入当通过。本次只续训一次15小集epoch，到累计30epoch/180updates，两臂都恢复last权重、AdamW状态、CPU/GPU随机状态及批次generator。学习率、坐标、尺度、冻结local、样本、seed和solver均不改，不挑中间epoch。它是原接收pilot的延长，不伪装新stage，也不是完整fit30轮。

指标语义修正：原`receiver_gate`复用整段拼接边界，oracle的相邻两块来自不同GT重置条件，不能代表相对实际输入prefix的续接。保留该原值与failed结论，另报`conditioned_continuation`：每个有效边界当前首帧pred − 实际供应的紧邻过去一帧；oracle使用该GT帧。对照是no_prefix生成首帧相对于同GT帧的诊断性距离（no_prefix未输入GT）。原threshold数值不变：眉眼位移RMS均<=2倍参考、续接位移误差均<=对应无前缀的一半、中心MSE不劣。比较对象修正是看过结果后的评价更正，需要显式标注，不能自称原预注册门槛通过。缺检测不跨gap，前一帧invalid不计。

另外报告generated deployment整段接缝、raw/centered、三seed和逐段均值，不能只靠oracle过门槛升级。延长endpoint仍失败就终止此路线全量升级；如需新坐标化设计另建独立协议。本轮仍只8条已曝光fit，同句L1小集无泛化结论，封存test保持未读。
