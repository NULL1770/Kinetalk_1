# 当前架构与数据：Phase53 候选，2026-10-09

本文按实际 Phase53 checkpoint 的 config 和现行实现更新。Phase53 两组均未通过口型、情感、风格联合验收，不是已确定的论文最终模型。最新任务状态读 CURRENT_OPTIMIZATION_STATE.md，结果读 PHASE53_RESULTS.md。旧 Phase39/flow 说明移至 archive/CURRENT_MODEL_AND_TRAINING_before_phase53_20261009.md，不要用于画当前架构图。

## 输入与实际网络

|模块|实际输入|网络与输出|
|---|---|---|
|冻结中性 B0|查询音频的 HuBERT 第6层768D|原内容编码器：Linear→192，4个TCN，2层Transformer（6头），投影128；口型解码器3层Transformer192/6头；原局部Conv768→256→28跳接。输出52D中性动作，受原观察支持约束|
|音频 prior/student p|仅 emotion2vec768+prosody4；先切片再归一化，HuBERT不进入|Linear772→128，4个masked TCN（dilation1/2/4/8），2层Transformer（4头、FFN256）；全局g32，局部u16，均有Gaussian mean/logvar；stride2，固定原生token中心插值至25fps；u在有效帧上去clip均值|
|训练 posterior/teacher q|归一化GT−B0、B0、相邻残差位移、52D通道mask、detach后的参考style64，总272D|Linear272→128，3个TCN（1/2/4），2层4头Transformer；同样g32/u16 Gaussian。训练读取query GT，部署不调用|
|参考 style|两段与query录音和句子独立的neutral参考动作，各自B0、观察mask|统计参考编码器；posture描述156D、response描述208D，分别Linear→128→SiLU→32，拼成style64；无可学习speaker ID查表|
|响应 decoder|查询B0、g32/u16、style64|B0 52→192，4个TCN（1/2/4/8），52D残差输出；style posture32→52静态offset。g/u与reference response32分别调制192D隐藏状态，参考gain与shift有界|
|语义头|全局g32|8类情感线性头、4级强度线性头；neutral=0，MEAD表达强度1/2/3；强度是clip标签，没有伪造逐帧强度标签|

Phase53实际配置是 prior_variance=learned、center_local=true、reference_encoder=statistics、style_modulation=factorized。当前部署评分使用prior均值条件，不能把训练后验或采样噪声结果写成纯音频均值结果。实测prosody4是logF0/logRMS/periodicity/voiced；emotion2vec特征和B0特征的原始提取定义见历史归档。

最终输出为 B0 + scaled residual + scaled reference offset。情感/风格可以改变全部观察嘴部通道的幅度，没有硬嘴部排除。分支职责不会自动保证口型内容时序，仍需native jaw相关、闭口、位移和视频检查。共用同一个rig；style代表参考驱动的说话人运动倾向，不生成不同人的面部几何。

## 数据与训练职责

|阶段|数据与监督|参数更新|
|---|---|---|
|历史中性B0|715条原生neutral与2583条批准的emotion→neutral配对；安全原生时钟、局部event mask及通道观察支持|Phase53/54全部冻结；没有用情感GT替换中性目标|
|原始表达q/p训练|原TRAIN12536条中，固定fit10903条/20人；内部留出身份743、句子890。声学和动作尺度仅fit计算|q/style/decoder由原生位置+0.5相邻位移学习；q和p受KL及0.1全局情感/强度语义监督；p不接收动作重建梯度|
|prior条件接收适配|原训练后段使用detach的p条件重建查询动作|只训练接收器/参考分支；KL梯度会训练p，因此教师目标是否夹带发音误差仍要检查|
|Phase45局部均值校准|固定fit10903，对既有teacher u目标做固定ridge解析拟合|只校准p已有局部均值头；Phase53继承该u检查点，不能写成完全未做解析训练|
|Phase53参考适配|同一fit10903；620条fit neutral支持、20人；每query抽取2条同人独立支持并排除query clip/句子；部署独立enrollment每人2条|两组各固定8轮5456更新，batch16，lr1e-4，seed47；p/q及尺度冻结。style-only只更新参考相关参数；joint另外更新响应decoder；静态posture/offset只有neutral query提供梯度，动态style仍由各情感query监督|
|开发评估|原1367条、3位留出身份；2026个匹配句子/情感/强度的跨人有向风格对|不反传；raw+clip，四个冻结整段动作probe，原生时序、几何及固定视频。不是sealed test|
|Phase54当前诊断|优质配对TRAIN-fit2024、内部身份281、句子278|仅固定线性探针，所有神经模型冻结；不是新模型训练，不改变生成指标|

原始posterior重建和detach-prior重建对student的直接梯度均为零。KL会将teacher学到的分布信息传给student，所以“没有直接内容输入”不等于已证明内容完全解耦。Phase54正检查GT−alignedneutral、alignedneutral−B0和总残差的动态可预测性，不能先假定teacher污染已证实。

## 风格应怎样写进论文

定义为跨句子可复用、由参考动作驱动的说话人面部运动倾向；可包含平均姿态、不对称、开合幅度及运动变化统计。posture/response分块是当前实现选择，不是风格定义必须拆成两种。neutral支持不充分支持“每人的各情感表达风格均已学会”的强结论。

验证保持同一源音频/B0/g/u，仅换独立参考：比较同人A/B稳定性、跨人变化、是否向目标人的匹配表现统计移动，并检查内容闭合/时序。配对目标是另一段真实表演的统计，不是逐帧counterfactual GT。三位开发身份样本有限；目前结果支持部分有效与稳定性改善，不支持完全解耦或全部风格正确。

Phase53的24个完整比较视频与指标可作诊断/消融材料；大幅开口、fear时序及部分眉眼仍有偏差。论文最终架构图、概览图和t-SNE应等待最终候选确定，不能混画旧flow模型，也不能把不同检查点最好的数值拼成一个方法。
