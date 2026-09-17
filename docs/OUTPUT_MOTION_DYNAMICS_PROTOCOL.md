# 最终生成速度与幅度监督：匹配音频/零local对照

2026-09-17，运行前固定。上一direct实验粗汇总显示：单flow眉轨迹过强而不准，中心化逐帧L1明显减幅，仍输自身zero-local；真实审计尚在进行。此实验检验最终输出的速度与时间幅度监督能否改善该失败，不保证动态成功，也不是论文公开超参数复现。它不把“动得大”当作音频对齐正确。

## 数据、初始化与训练范围

固定既有2315段19人fit、405段3人内部开发；不读取外部280/439或封存test。共享句子、继承B0/global历史暴露等边界不变。从已执行direct实验的 **epoch000** 恢复完全相同renderer和zero-output direct encoder，不从其训练后模型继续。核验source recipe、checkpoint、原始cache及权重hash。初始化local=0，两新臂起点逐张量相同。

保留1540D stride4中心化声学→128D masked TCN→64D local，以及冻结音频activity gate。B0、identity、global encoder、motion teacher、原rank8 projection/head继续冻结。完整renderer与新encoder可训练；zero臂仍构建同encoder、消耗同初始化/RNG并保留该模块，但不让其输出影响flow或生成。encoder在zero臂没有梯度是预期，须核验其不变。

两臂 `audio` / `zero` 仅训练local条件是否置零不同；同seed46、batch16、Adam(renderer 1e-5/encoder 1e-4)、global grad clip1、每轮相同顺序/noise/flow time/choice draws、8epoch共1160步。不选最好轮、不覆盖默认权重。模块eval，禁用dropout。

## 固定目标

保留原 observed flow velocity MSE（残差scale=.25；uniform time与20% t=0）。移除此前centered逐帧L1，改成：

`L = L_flow + 1.0 * L_displacement + 1.0 * L_std`。

最终动作由与flow相同的noise经过实际12步Euler生成，生成条件不读GT。目标是**最终全动作motion**，不是单独residual。

- `L_displacement`：相邻两帧差分的逐通道归一化MSE，只有两帧均有效且该通道可观测的pair参与；不除dt、不作FPS转换。眉41--45、眼5/6/12/13、嘴14--40三个组各自对有效元素平均，再三组等权平均。尺度只由fit目标相邻帧位移RMS得到，逐通道floor=.005。
- `L_std`：逐clip逐通道在有效帧上计算时间标准差，ddof=0；比较预测/真实std的归一化MSE。眉与眼两个组各自对有效clip-channel平均，再两组等权平均。至少两帧才参与std项。尺度是fit内按有效帧计数汇总的中心化目标RMS，逐通道floor=.02。以向量范数实现std，常量轨迹的值与梯度均有限。

屏蔽帧/通道在均值、差分前清零；不让NaN padding污染统计。没有region输出头，分组仅是监督与评价。等权分组/权重1/floor是本次预先固定工程选择，不来自MEDTalk或SubtleTalk的公开超参。

速度逐GT MSE仍可能压制一对多变化；std只能约束幅度不能保证时序或合理分布，两项不可被宣传为概率生成创新。flow保留，但成功与否必须由样本分布和时间干预验证。

## 保存与审计

每epoch原子保存独立checkpoint及last，内含source/recipe/hash、训练范围、fit尺度、Adam和Python/NumPy/Torch/专用generator RNG；保存完整minibatch/noise/time/choice hash。epoch0保存seed42 full/zero/reverse并与来源zero-local严格比对；最终8固定noise seeds、12步、full/zero/reverse。zero训练臂的部署主结果是其zero模式，full/reverse仅额外检查（encoder应保持零输出）。

至少比较audio-full对matched-trained-zero、own-zero、own-reverse和来源冻结输出，并比较上一direct两臂作背景；同seed原始输出、眉眼嘴分报，raw/mean/dynamic分解、相关、能量比、相邻速度、fair ES/variogram、neutral/nonneutral嘴/jaw保护及逐人结果。仅幅度增加、raw均值改善、单seed或局部成功都不算通过。重复内部开发不能充当独立论文测试。
