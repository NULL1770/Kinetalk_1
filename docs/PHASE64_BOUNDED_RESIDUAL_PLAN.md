# Phase64：一次受控残差幅度适配与 25 人风格检验

日期：2026-10-10。用户已授权改进、训练、评估和渲染。修改前 Git 存档 `0bc0bad` 已推送。正式基线仍为 Phase53 style-only；Phase63 不采用。

## Material Passport

Mode: experiment/run；执行者为本会话，未使用外部模型审核。证据：原始 Phase53/63 报告、当前源码、固定 TRAIN 内部划分。新实验尚无指标，不能预先承诺成功。研究数据不上传第三方服务，Git 仅保存源码/文档/汇总表。

## 假设与单一改动

Phase53 下颌范围 .124295，GT .175279；闭口 F1 .401021、jaw correlation .477797；MBE .763407、LBE .369546、lip mean 3.262565mm、整段主 probe F1 .676805。它是可比较的基线，不等同最终 SOTA。

保留原来四层 TCN receiver，新增单层 `Linear(80,52)`，仅输入 global32、local16、reference-response32：

`gain(t) = 1 + 0.5 tanh(W[g,u(t),s]+b)`

`y(t) = B0(t) + scale * [gain(t) * old_residual(t) + old_posture(s)]`

W/b 全零初始化，候选初始输出逐值等同 Phase53。仅训练 4212 个参数；旧 receiver、情感 prior/posterior、style、posture、语义头和 B0 全冻结。没有额外损失，没有嘴部通道遮蔽。原残差仍可能包含 B0 误差，不能把它命名为“纯情感真值”。增益不看 B0/内容，但缩放残差仍可能改变最终闭口和峰值；结构本身不证明内容解耦。

这不是 Phase63 的直接 B0 gain。目标是检验小容量条件增益能否纠正不同情感/人物的幅度偏差，同时避免重新训练大 decoder 的退化。它不能创造原残差中完全缺失的动态，预期能力有限。

## 固定训练协议

- Phase53 style-only SHA `80a7ca69b0f868bad6361381e1ca8e4e168ca227a0548221f8efa71b42169c41` 起点。
- 原 TRAIN-fit 10903 queries，620 独立 neutral supports；排除 query clip 和 sentence；原 mask、原生25fps、缺失帧与 channel support 全保留。
- seed47，8 epochs，batch16，AdamW lr3e-4，原 position + .5 adjacent-displacement objective；不按外部 development 调参/挑 epoch。
- 120步 GPU smoke 的技术门槛为 loss末10步/首10步 < .995、所有冻结 tensor 精确不变、HuBERT NaN 隔离、finite gradient。此门槛只验证小分支可学习，不是模型质量验收。
- 全1367 development raw/clip、四冻结probe、八情感、原2026风格配对照常报告。posterior 已非适配目标，oracle仅作诊断。

## 25 人风格验证

20 训练见过人物用 held sentences，2 内部未见人物用 speaker-dev，3 外部 development 人物；不读取 sealed/test。每人每情感按 clip SHA 固定选1条（最多200条），固定源音频/B0/g/u，切换全部25人的独立neutral参考，并比较同人 A/B。Phase53 与候选同时评估，不能只展示变化量。

报告同人稳定/跨人差异、闭口变化、centered/velocity correlation、jaw lag、四冻结probe的换风格情感标签变化、精确相同 sentence/emotion/intensity 的目标统计方向；不做跨人逐帧 GT 误差。分开报告 seen/unseen，以及逐人结果。25人不是25个完全未见人。新增8情感多人物曲线和可渲染输入；标准8情感主对比与8风格视频自动排队。

## 质量验收（预先固定）

候选仅当以下全部满足才有资格进一步验证：主clip指标 MBE/LBE/lip 不恶化；jaw range 距GT的绝对误差下降；closure F1与jaw correlation不下降；八情感中 fear/angry/sad 至少两类主probe提高且整体主probe不下降；原3人同人mouth A/B MAE不增、跨人closure disagreement不增。其他raw/四probe与25人结果完整保留。此条件严格，失败保留 Phase53，不混拼不同模型的最优数值。

不自动推广。仍缺多seed、统一预算基线与最终未读测试，不能据此宣布论文SOTA。

## 数据集决定

核查的原文：FaceDiffuser（arXiv:2309.11306，§4/附录A.2）覆盖 BIWI/VOCASET/Multiface/UUDaMM/BEAT；EmoTalk（ICCV2023，§3.4/4.1）从 RAVDESS/HDTF 构建3D-ETF；MEDTalk（arXiv:2507.06071v4，§4.1）用 EmoFace 数据训练音频生成，RAVDESS 用于多模态分支。不是所有论文都规定“两套同用途数据”，但跨域证据有价值。

本轮继续以 MEAD 作为有动作监督的主数据。CREMA-D 若只有音频/视频、没有可靠3D拟合，不可直接用于动作误差指标或B0/风格监督；若已有合格拟合，即使没有neutral-emotion配对，也可另做native表达监督/外部测试，前提是匹配rig、时间戳、mask及独立数据划分。当前路径未定位，已询问用户；不因为时间紧便把未核实数据混入主训练。audio-only外部情感泛化也需单独标注，不能当第二个3D benchmark。

原文与抽取文件已在项目研究档案中，未重新检索/下载整批文献。
