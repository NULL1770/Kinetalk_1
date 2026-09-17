# Audio/text expression-intensity pilot v1

固定协议：2026-09-17，首轮真实训练之前定义。外部输入为音频及独立 neutral 身份参考；音频自动 ASR，冻结文本编码器提取 token。不要求情感标签，不读取参考转写，不虚构 token 时间戳。

## 模型与监督边界

原 B0、身份、motion teacher、audio-global 全部保持冻结、使用原始已验证缓存。本轮继承此前蒸馏得到的 audio-global，不进行在线 motion→audio 全局蒸馏。可学习部分为整个既有 renderer 与新增 AudioTextAffect。新增输入是原96帧时钟下1540维未作片段中心化的声学/韵律，加冻结文本 token。音频查询跨注意力读取句子文本；逐帧强度经过 learned FiLM 调制 global 的可学习投影，同时保留直接声学/文本通路；不直接缩放原 latent，不对局部条件去均值，不以 neutral 分类概率压掉动作。

强度是眉毛41–45与表情眼5/6/12/13的中性参考偏离幅度代理，不是完整心理情绪。独立 neutral 参考产生 anchor，train-only RMS尺度下限0.02；两区域各占一半。不含眨眼、视线、嘴部。监督不向主推理路径输入 GT 强度/目标动作。每次训练均用模型自己的预测强度，无 teacher mixing，无 OOF 声称。oracle 只作明确标记的诊断。

## 锁定预算与目标

固定2315 fit / 405 development，既有96帧 crop；不增加数据范围、不访问 sealed test。text/no_text 两臂各8 epoch、batch16、同seed46及同独立 minibatch/noise/time RNG；不按dev选择checkpoint。两臂构建完全相同模型并共享初始化，no_text 跳过文本路径。Adam，renderer lr1e-5，新增encoder lr1e-4，global grad clip1；无dropout、模块eval模式但允许梯度。12步Euler rollout。

总目标：标准observed flow速度MSE + 0.25 × 预测强度SmoothL1 + 0.1 × 最终raw动作重建 + 0.1 × 最终输出强度SmoothL1 + 0.1 × 合法域惩罚。raw重建按眉/表情眼/嘴三组等权，使用train-only中性参考RMS尺度、SmoothL1(beta1)。域惩罚为observed通道对[0,1]越界的绝对距离除0.02。目标动作只用于这些监督，不作生成条件；rollout从随机噪声出发。raw重建包含均值与动态，避免前轮仅velocity/std约束产生均值漂移。该逐目标重建可能减少多样性，因此不宣称解决一对多，必须另测多seed分布。

新增局部输出层从0初始化，原global不变，所以epoch0等于原zero-local生成；它不等于原full-local。首次均值强度取fit目标均值。两臂保留epoch0与每epoch恢复状态/最终epoch8，记录代码、输入、模型及随机序列哈希；不修改默认权重。

## 验证

固定8个seed(42,123,2026,7,19,73,211,997)。主full及zero用全seed；其余推理干预固定seed42：no_text、static_intensity、shuffled_text（选择不同sentence）、reverse_audio（仅新局部音频时序，原B0/global保留）、oracle_intensity。推理消融不可替代匹配训练。主比较为最终text/full与no_text/full以及epoch0/source。

报告强度误差与相关、眉/眼raw和centered误差与相关、动态幅度、口部raw/mean退化、越界/显示clamp影响、多seed公平ES/VS与个体结果；不得以std接近GT判定成功。固定旧9条锁定例子及3人连续视频，不按效果挑选。8epoch为短期可行性对照，405dev句子与fit共享，不能视作最终泛化或CCF发表证据。窗口活动条件、前序动作历史、全长随机窗口尚未实现。
