# 完整原生时间覆盖：30轮配对实验

本协议在正式训练开始前固定。依据一次起始历史诊断选择此路线，不新增loss或rank8适配器。用户授权提高训练预算，正式两臂各30轮，终点固定，不按开发集挑轮。

## 机制依据

同context12/chunk_teacher、历史固定128 fit、三seed，一次GT历史仅在start16供给8:16，随后全部自己生成。正常生成后半段48:96的眉/眼centered MSE为.001205746/.001400662；GT起始后为.001214519/.001389453，约+0.73%/−0.80%。raw位置误差改善较大，而同末状态的静态历史结果接近完整历史，说明初态确实影响轨迹原点，但不足以解决后续动作时机。此诊断不是可部署结果，也不证明音频能预测GT初态。

当前固定中央96帧遗漏原2315 fit约19.7%的有效native帧。本轮检验完整连续动作覆盖及下游模型上下文是否有益。原emotion2vec中心特征已经由完整wave提取，不能宣称首次给预训练音频模型整句输入。

## 两臂及预算

- `center96`：旧固定96帧输入和训练目标。
- `full_native`：同一clip的完整native时间轴、mask与动作，包含准备和收尾；按16帧chunk/8帧过去生成，尾块只padding，不压缩gap。
- 共同context12/chunk_teacher源upper与history12/no_history源local，upper与local的input/blocks/local_head均可训练；B0/身份/global audio/原state/口部基座冻结且使用原独立参考。
- 历史1707更新／608增量留句分割不变。源模型见过完整2315，608不是全模型未见泛化。405为反复使用的内部开发；封存test不读。
- 每臂30epoch、每epoch107更新，共3210；batch16、seed97、AdamW学习率1e-4/weight_decay1e-5、梯度裁剪1。每clip每epoch一次，两臂同顺序，同完整native noise及每clip flow-time；center取同绝对native位置的noise。
- 原unknown-only flow matching，训练严格GT过去、部署生成过去；12步Euler。完整臂每batch所有有效chunk累积成一次loss/一次更新。
- 两臂更新次数相同，完整臂监督帧和算力更多，明确报告有效帧曝光及耗时，不称同算力消融。

## 数据与编码

仅允许旧audio.pt绑定的2315/405成员。manifest、native、wave、emotion2vec模型和提取代码来源全部核验。不拟合新归一化或更改身份参考。native content按原FP16/FP32存储读取后转FP32计算，不伪称原文件全为FP32；原motion若历史缓存半精度，按原存储dtype验证重叠并记录数值差，不能隐瞒目标量化差异。

只保存中央覆盖外的middle768 FP16与prosody4 FP32、native索引和来源hash，避免重复整份1540缓存。原中心特征原样复用；完整wave重提在所有中心有效点校验，固定门槛middle最大误差<=.002且RMS<=.0001、prosody最大误差<=1e-6，逐clip记录真实差值及非零比例。不合格停止，不能静默放松阈值。

完整输入重新计算冻结B0/h0/global/intensity/static；不能平移motion后继续用中心encoded缓存。local按对应完整或中央窗口在线计算。这个干预同时改变动作覆盖、下游声学上下文与静态统计范围，不把全部收益归因单独的训练帧数。

## 评价和保护

主结果是raw输出，离线DC仅单列表。full输出按原native时间切回旧center96的共同评分区，两臂目标/mask/时间一致；full的实际chunk边界按native偏移计算，旧center16位置只能叫诊断网格。

固定seed42/123/2026、统一全native噪声，再切中央保证配对。seed42另做local-static/reverse，h0/global/static/身份固定。608与405在epoch30终点完整评价；记录source step0作为参考，不据中间开发曲线选epoch。报raw/centered MSE、相关、幅度/速度、边界、fair energy/variogram与逐clip结果。结果不得仅凭更大幅度或更低loss称成功。

405整脸合成始终使用旧seed对应center52基座，仅替换有效帧上脸9通道，43个其他通道与invalid逐位保护。full的上脸B0/h0可变但不改口部输出。冻结与保护不替代身份、全局情感、口型、自然度的独立质量验收。

## 启动与存储

先必要单元/回归测试，再真实两臂小跑，验证变长尾部/gap/noise、源与数据绑定、梯度路径、checkpoint恢复及终点评价。正式输出新目录，不覆盖旧实验。保存每epoch小JSON、原子last(optimizer/RNG/已完成epoch/配方hash)与固定final；曲线仅upper9并绑定已有基座。resume只从完整epoch恢复，需输入和代码hash完全一致；训练中断不会承诺无人自动重启。

用户已允许清理冗余；仅按清单删除旧smoke及更新前失败尝试的张量，正式final/数据/评估报告保留，删除SHA清单已归档。新增实验仍做磁盘预算/阈值检查，smoke可用临时内存盘，正式checkpoint必须落持久盘。

后台supervisor以独立进程运行。启动后核首epoch、日志、PID、磁盘和last checkpoint，再给实测ETA；关闭聊天不影响训练，关闭实例会中断。

