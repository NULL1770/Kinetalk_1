# 局部跨模态对齐与自由上脸生成：锁定实验

用户在run12失败后授权继续优化动态、时序及其他效果。旧run12及默认均不覆盖。仍用2315 fit/405反复使用的内部dev、原25fps/96帧、原身份参考与数据hash；不读test。各正式阶段12epoch，固定末轮，不按开发集选最好轮。

## 真实诊断依据

完成同405样本、seed42、12步解码的2个renderer×5种条件交换。第4阶段固定renderer，audio条件口部相关.413，换teacher local后.735，换teacher global/intensity仅.424。第3阶段固定renderer下分别.479/.842/.484。第3阶段teacher全条件口部raw MSE .003543，第4阶段变.007359，显示适配后renderer的接收行为变化。teacher输入是GT oracle，不代表部署效果。

## Phase A：音频局部接口对齐

- 恢复已训练Stage3 renderer，冻结它、B0、身份、动作teacher、原projection和Stage4 audio global/classifier。
- 新独立AlignedAudioLocal从Stage4声学trunk起步，只预测teacher同rank8、stride4、tanh分箱与有效帧加权去均值的controls，经固定teacher projection/插值得到local。
- 训练目标仅teacher controls的train-RMS标准化MSE（尺度下限.02）；AdamW lr1e-4、weight_decay1e-5、gradclip1、batch16、seed53、12epoch。无GT条件解码训练，teacher只作标签。
- 每4轮固定64dev真实解码；末轮全405/3seed，seed42额外原audio local、zero local、GT teacher local。全局保留不等于最终表情必然保持，另测生成teacher读出，明确非独立感知认证。

## Phase B：自由上脸flow配对

- 固定Phase A所有基座与global/identity，输出其他43通道直接copy。
- 单独9D flow生成相对独立neutral anchor的归一化完整上脸残差，无P/Q投影、无强制4D线性lift。
- 时间位置编码固定native帧坐标；每层同帧声学条件显式注入。音频local trunk复制PhaseA但独立可训练，64维local head置零；全局不共享可训练trunk。
- direct与soft两臂同参数shape、初始化、batch/noise/time随机流和12epoch预算，唯一use_state不同。soft使用冻结Stage4的预测4Dstate作为特征输入，训练与推理同源；direct完全不用该state。两臂都仅flow matching，不追加幅度/速度loss。
- AdamW lr1e-4、weight_decay1e-5、gradclip1、batch16、seed53、12Euler步。每轮记录batch/noise/time hash。
- 末轮405/3seed；seed42额外base、静态local/state、反转local/state、zero_state。反转不变更冻结content/global/base，所以是局部条件贡献诊断。

## 验收

同时报告raw与每段去均值误差、相关、RMS幅度比、位移误差、越界率，口部按neutral/nonneutral拆分，身份参考一致性、生成情感读出及43通道逐位保护。不能把相关低的一对多样本直接判无用，也不能单凭更大幅度判成功；需要多seed分布/固定渲染一并解释。两臂比较为探索性内部dev证据，无论文新颖性或最终测试承诺。PhaseA效果若未通过，应如实拒绝作为默认基座，PhaseB输出也不得借口嘴部旁路宣称联合成功。

## 完成后的探索性跟进

align/direct/soft均12epoch完成；同初始化与每轮随机流hash通过。align眉相关.132、眼.188、嘴.519，优于run12时序，但同Stage3的原local口部raw更好。direct/soft眉相关约.084，4D条件没有独立收益，均不替换默认。

后加residual12：冻结zero-noise aligned全轨迹，flow预测完整差量，沿用fixed aligned local；只flow loss、seed59。真实smoke后训练完成，仍失败：眉幅1.48/相关.099，嘴raw .01354。零噪声基座与差量方案不采用；它改变了两项，非单因素消融。

最后做不训练的固定分区候选：同一Stage3 renderer分别用原audio local与对齐local推理；眉眼9通道取对齐输出，其余43取原local输出。规则对全部clip/seed固定，目标是不牺牲本轮更好的口部基线来保留对齐的时序收益。此规则在查看开发集后确定，属于开发选择，不能冒充未见测试或新方法已成功。全局和身份权重继续冻结，无按样本GT择优，无参数调gain。
