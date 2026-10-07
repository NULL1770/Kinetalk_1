# Phase35：标准化坐标与物理flow误差分开验证

当前Phase34三seed评分尚在闭合，不能将两seed临时分数当最终结果。已观察到标准化能明显提高F1，却使lip/FDD退步；下一项只检验通道重加权是否造成这种权衡。不扩大Phase34预算、不扫loss权重、不新增loss/网络层/嘴mask，也不将数值试验宣称ua纯表达已证明。

两组仍从同一个完整中性checkpoint开始，772学生、全嘴开放；共同TRAIN mean/std、diagonal source、清头、标准化输入/输出坐标，固定三seed各两轮1568updates。对照沿用已完成Phase34 standardized（原coordinate error）；候选只将flow MSE恢复共同residual scalar单位。其余semantic/global distill/generated语义及其系数不变。两个真实小smoke先证明新代码默认严格重放旧smoke、candidate仅flow误差/梯度不同、initial物理输出/state/sample/GT/mask/time/noise相同、772梯度隔离/冻结/finite/第二步ua梯度正常。

新增可选CLI `--flow-coordinate-loss-units physical`，默认coordinate保留旧计算。它仅在标准化坐标+772输入的隔离audio-stage合法；不影响推理模型/缓存/源噪声。recipe显式记录effective flow units，避免旧flow-vector-units=scalar字段掩盖实际标准化误差。

本轮用户继续优化授权覆盖此独立验证；正式启动必须等Phase34完成/已备份SHA/容量充足、真实smoke通过。源快照与唯一新driver/closure/collector单独绑定，复用Phase34固定baseline不重训。完整相同validation/3draw/raw+clip/4probe/same rig/groups/pairedCI/full80neutral timing；联合gate不放宽，fixedfinal不选seed/epoch，失败不自动扩大。先不修改D2表达目标或教师。

容量按实际baseline file sizes核预算：仅三physical新臂（旧coordinate复用），完整final/last/curve持久磁盘，加三个audit尺寸、10MiB新增metadata及384MiB临时/余量，实际预算1.77593GiB。启动要求空闲≥上述总量且≥1.8GiB，而非照搬六臂4GiB或任意2GiB阈值。smoke已占用的空间从实际free扣除。现实际free1.249GiB，四成员回收仍只有约1.794GiB，故只回收已下载完成且原始SHA再次一致的五个Phase34重复成员：47control曲线/last、47standardized last、48control/standardized last；保留全部final及三个standardized baseline曲线。完整476文件备份独立继续，不能声称整体已备份才删除这五项；每项都已持久本地核验才允许删除。

D2仍是后续必要职责修正：老师global来自全残差、ua仍间接flow监督。需要表达目标验证与具体梯度合同，不能用本试验“没有HuBERT直达”冒充完全解耦。表达/身份准确性及公平SOTA最终仍需独立证据。

## 实际启动

五成员回收661,453,720bytes、空闲1.865GiB后正式启动：driver9830、workers9831/9832/9833，closure9834；唯一localcollector90557（collect_phase35_compressed.py）等待全评分后下载SHA。binding27ad06ab237569735254c8dcbb4aebfec05c6f477b1f5eac6345248344633f0f。重复启动禁止。Phase34全部476原始成员3.227GB已完整本地SHA闭合，77005结束。尚无Phase35完整成绩。
