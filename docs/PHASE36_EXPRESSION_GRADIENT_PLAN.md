# Phase36：情感学生的表达梯度职责

用户已再次授权继续优化。本项独立于Phase35物理单位试验；不增加loss、网络层或前向嘴mask。当前实现尚未开始正式训练。

## 唯一变化

标准化坐标、共同中性完整checkpoint、772D、TRAIN统计、源噪声、零输出头、三seed47/48/49、固定两轮1568updates，与Phase34 standardized对照一致。只将audio学生的flow和generated-semantic梯度限制为已定义的9个眉眼表达通道（41/42/43/44/45/5/6/12/13）。逐帧/实际GT保持不变，不把clip标签复制成动态真值。

renderer仍按完整已观察残差和原generated-semantic目标训练；前向预测、全部嘴通道、噪声、原损失数值、语义CE/强度CE/global蒸馏系数不变。学生在表达通道外的输出导数停止，renderer的完整梯度保留；随后沿用联合clip_grad_norm，裁剪系数可能因学生梯度改变而变化，不能宣称renderer更新逐位不变。

该范围故意先只使用自身原生眉眼GT。未将不可靠jaw pair差或所有mouth残差标为纯情感；global/强度/身份仍能前向调节嘴幅度，renderer通过h0和全嘴GT学习这种响应。这是监督梯度职责实验，**不是已经完成逐帧表达teacher重建或完全解耦**：旧global教师依然读全残差，emotion2vec也可能含音素相关信息。后续是否新增直接表达目标须依据本项干预与泛化结果，不一起堆交换/adapter/ordinal。

## 验证和接纳

1. 默认full路径须重放原Phase34真实两update全部state/实际sample、GT、time、noise。
2. 单独非表达loss对学生global/intensity/u_a的梯度精确0，而renderer非零；9表达通道对u_a非零。混合loss中renderer裁剪前梯度等于原完整loss，学生等于表达视图loss。
3. 实际数据smoke检验冻结B0/身份/teacher、772输入隔离、原生时钟、finite、初始前向/各loss数值/RNG共同。第二步验证u_a仍学习；不只detach所有flow。
4. 正式只训练三个新候选，复用Phase34三个standardized对照；全1367 validation/3draw/raw+clip/四probe/same-rig/分组/pairedCI/full80neutral gate，不择seed/epoch，不自动延长失败预算。固定八类视频和u_a static/reverse/shuffle诊断用于确认动态作用，不据t-SNE宣称达标。
5. 启动前必须实际容量足够，现root约0.515GiB，Phase35唯一collector90557尚未完成。只清理已持久本地SHA验证的精确重复大成员，不动唯一证据、baseline曲线、当前中性起点/数据，不重复评分或writer。

Git暂停、sealed未读、默认未替换；SOTA目标仍未达成。
