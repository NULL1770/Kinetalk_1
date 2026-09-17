# 直接时序声学条件与闭环眉眼监督：固定协议

2026-09-17。用户要求先把动态做出来，并要求依据 MEDTalk、SubtleTalk 等方法修正实质差距。本实验是常规效果基线，不是论文创新或对任一论文的完整复现。

## 动机与固定改动

此前 rank8 ridge、本地接口放大、TCN 小头及开放完整 renderer 均未恢复眉毛时序；最新匹配容量实验中，真实动作 oracle 条件也降低 flow loss 但未改善眉毛动态。因此本实验同时修正两个训练链差异：声学时序特征在训练中直接进入生成器；最终从噪声经过完整 Euler rollout 得到的实际生成动作接受轨迹监督。

输入使用现有训练缓存的 `content[768] + middle[768] + prosody[4]`，在 stride4 的24个时间 bin 上连接。`middle` 是已有2/4/6层聚合而不是完整 WavLM 多层，且缓存特征已经逐段中心化，故只称“缓存的中心化多源声学条件”，不声称原始音频直入或复现 SubtleTalk。所有均值和标准差只由2315个 fit clip 按真实 bin 权重计算。

新建 `1540 -> 128` 投影、三个 dilation 1/2/4 的 masked temporal blocks、`128 -> 64` 零初始化输出。64维输出绕过旧 `rank8 -> 64` local projection，使用相同固定坐标插值至25Hz，乘冻结音频分类器的非中性 activity gate 后直接送入现有 `renderer.local_emotion`。B0、identity、global audio affect、motion teacher、旧 projection/head 全部冻结；完整 renderer 与新 encoder 训练。encoder lr `1e-4`，renderer lr `1e-5`，Adam，global grad clip1，batch16，seed46，8epoch，不选最好轮。

## 两个匹配训练臂

`flow`：标准 observed flow velocity MSE。

`dynamics`：完全相同的 flow MSE，外加权重1.0的最终生成眉眼中心化轨迹 L1。每个训练 batch 使用相同 noise 作为 flow 输入与12步生成初态；生成时不读目标状态。眉眼通道固定为眼5/6/12/13和眉41--45，按每段真实有效帧去均值，再除以仅由fit集计算、下限0.02的逐通道 centered residual RMS。

首轮不同时加入 raw、velocity、std 或 mouth loss，避免无法判断起作用的步骤。完整52D renderer仍能改变嘴部，因此嘴部 raw/centered/velocity指标是验收条件而非训练项。

两臂共享严格相同的初始化、batch顺序、noise和flow time。新 encoder 末层为零，epoch0等价于来源模型的 zero-local，而不是来源 full-local；独立保存该起点，不复用旧 full 曲线。训练后以8个固定 noise seed、12步生成评估 full/zero/reverse。新64维空间没有与旧 rank8 oracle 对齐，故不报告伪 oracle 上限。

## 边界与判断

数据仍为19人fit、3人405段内部开发，句子共享且B0/global存在历史训练暴露；不读取外部280/439或封存test。结果只用于修复训练链，不作泛化或CCF-C论文证据。

必须同时检查眉/眼中心化R2、相关、energy ratio、速度误差与跨seed稳定性；检查neutral/nonneutral嘴部和jaw退化；检查full相对zero/reverse的条件净效应。仅幅度增加、均值改善或个别seed好看都不算动态成功。固定metadata样例只用于看连续动作，不参与选模型或调权重。
