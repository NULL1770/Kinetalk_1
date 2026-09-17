# 音频动态预测修正：8轮三折对照

2026-09-16，执行前记录。基尺度定标提高眉motion投影但没有提高音频预测，因此本轮固定原native rank8目标，只检验音频预测映射。无文本、VA、区域输出头、新loss、renderer训练或默认权重替换。

数据只用19人2315条fit，复用`scaled_basis_oof_v1`已审计的三个native句折及各折ridge/std/U。405/280/439开发与test目标不读取；原音频特征content+middle+prosody、4帧时钟、alpha1/rank8不变。这是已使用训练数据的描述性跨句诊断，不是独立最终测试。

三个参考：冻结ridge；ridge+pointwise修正；ridge+temporal修正。两修正网络为D→64、逐bin LayerNorm/GELU、两个共享64维残差块、64→8。块内depthwise卷积，pointwise用kernel1，temporal用kernel3及dilation1/2（额外感受野7bin，约28帧）；逐bin特征原本已有上下文，所以pointwise不是“无时序信息”。共同形状参数相同初始化；temporal初始中心核等于pointwise，侧核零，后续可学。参数相差256，不称完全控制容量。

只有最后8维输出层权重/bias为零，保证初始预测精确等于ridge；其余层正常初始化。修正逐clip按有效帧权重去时间均值，乘当前fold的fit-only真实motion投影RMS，再加到冻结ridge的8维坐标，最后经同一U解码。原位mask、缺失padding不进入卷积或统计。预测接口不读query motion/emotion/speaker。

监督是原单位真实中心化motion残差，不是ridge自身预测。只有一个loss：按真实bin有效帧计数的`mean((native_prediction-native_target)^2)/.25²`。不做通道重权、增益拟合、global gate或额外蒸馏loss。训练每fold仅fit样本，seed46+fold、Adam lr=.0002、batch32、8epoch，两臂batch顺序严格匹配，无dropout/weight decay/学习率搜索。固定epoch8评估，不读dev挑checkpoint。保存每epoch可恢复last和最终模型、Adam、RNG、来源hash。

每条fit恰好有一次句子OOF预测，所有评分回原单位。主比较temporal对ridge与pointwise，分报nonneutral/neutral/all、眉/眼/上脸/嘴/jaw、逐人；full/zero/reverse，以及真实motion投影oracle。句簇bootstrap5000（固定seed45），同时看相关性、幅度和速度/差分误差，不能以更大幅度取代正确时序。反转只反转有效输入bin而保持目标时间权重，partial-bin时不称严格物理时间反转。

进入生成实验的预先条件：非neutral眉temporal对ridge的ΔR²95%CI下界>0、眉相关提高、优于zero/reverse；temporal对pointwise若无明确优势不宣称时序结构有效。nonneutral/neutral嘴与眼的native MSE相对ridge增加单侧90%上界≤1%，mouth相关下降≤.005；眉逐人至少10/19点估计改善，并单独报失败者。这个门槛仅允许后续生成验证，不能保证最终无退化或发表。未通过则保留ridge，不追加长训或根据OOF调参。

并行监督抽检仅限旧素材中属于本轮fit的片段，按metadata固定每类2条，检查原视频/native时间戳、眉曲线与音频来源。它能发现显著错配/坏轨迹，不能替代独立高精度表情标注。结果待测。
