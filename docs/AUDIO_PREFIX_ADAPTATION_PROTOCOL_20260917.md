# 已有续接生成器下的音频条件共同适配

2026-09-17，context12 结果之后锁定的新实验。目标是检验接收器已经学到部分历史续接后，让眉眼专用音频条件同时适配是否能改善音频动态，而非继续只训练 flow。不是新的论文创新主张。

## 两个因素分开

1. 先不训练，使用 context12/chunk_teacher 的既有三 seed 输出，比较原输出、以旧 state_white 中可部署 audio 静态均值平移整段输出、以及独立 GT 均值 oracle。要求 metadata/hash 匹配，验证 center、位移和 RMS 不变。此项只定位整体位置误差，不能称为改善时序或因果初始化。GT 均值只用于独立评分诊断。
2. 两组均从同一个 context12/chunk_teacher epoch12 上脸权重和其冻结的 history12/no_history local 权重开始，各新增固定12轮。`frozen_local` 仅更新 upper；`adapt_local` 同时更新 upper 和专用 local 的 input、blocks、local_head。源 global audio、state、身份、B0、renderer 不变；不得使用 local 的已失配 global/state 头。音频原始1540维输入、特征归一化、h0 及有效帧保持不变。

旧 direct/history 曾联合训练 local，但尚未得到本轮 prefix 接收能力。因此此处检验的是已训 receiver 下的联合适配，不声称首次出现端到端音频训练，也不重复增加逐帧9D目标条件。

## 固定训练

- 原2315 fit /405反复使用的内部开发；原96帧25Hz；不读test或新增数据。
- context12 chunk_teacher 原目标、同9维尺度与坐标、8帧严格过去/16帧未知、clean known clamp、unknown-only flow matching；不减GT clip均值、无新loss。
- 训练始终真实严格过去；部署只用生成过去。保持与context12一致，不加入scheduled采样。
- 12 epoch ×145 updates，batch16，seed89；两组相同clip顺序、每clip [96,9] noise和一个flow time；每batch仅一步AdamW，lr1e-4、weight_decay1e-5、gradclip1。全模型eval模式dropout0，但adapt的指定参数有梯度。
- 不按开发结果选轮、挑seed、追加轮次。旧source已训12轮，新两臂是从同source各继续12轮，不伪装为从零仅12轮。
- 固定12步Euler；评估seeds42/123/2026。last保存optimizer/RNG供故障恢复核查，本入口没有自动resume选项，失败时不能覆盖原目录重启；源与历史文件不覆盖。

## 评估和合成

原始输出保留完整405×3seed，以及既有empty/reverse_history/static/reverse、独立oracle_history/oracle_reverse_history。额外seed42 local_static/local_reverse只干预专用local序列，固定h0/global/static/identity；与原来同时干预h0/local的static/reverse明确区分。

完整输出另存明确的离线DC合成模式：`static_upper + (raw_upper - valid_mean(raw_upper))`。static_upper来自独立neutral参考与冻结global audio的4D状态均值。43非上脸及无效帧严格复制同seed原基座。整段常量平移不平滑、不改幅度/速度/相位，但会改变越界率和情感读出，必须报告。此合成已有旧mean-preserving前例，本轮只将它用于prefix候选，不作创新点。

DC不进入训练known、不回灌生成历史、不读取query GT，不作为因果在线模型；raw和DC分表、分曲线。GT历史的实际接收只在raw坐标评分；不对DC后的oracle输出套用未移动GT端点。主DC报告和曲线完全排除GT oracle。

最终128fit元数据固定选样沿context12。fit只评上脸，43零占位不能用于完整视频。声学适配在fit有效而dev无效应报告为过拟合/不泛化证据；dev本身已反复使用，不当封存测试。口/身份保护不等于独立自然度验证。

## 核验与判读

相同source/初值/随机流/预算；非适配local参数无梯度且hash不变；独立system/audio所有张量和梯度保护；适配梯度确实经过input/blocks/local_head。local专用前向应等于旧local.forward()['local']，不得执行global/state输出头。

raw及DC分别报告raw/centered/correlation/RMS/越界、速度与边界、三seedES/VS及非独立情感读出。必须同时比较本轮frozen对照、source context12、旧state_white；不能把继续训练的收益全部归给解冻local。若只raw下降、时序及条件消融无净收益，结论仍是动态未解决。未预先承诺成功，不默认替换权重。

磁盘预算：新实验保存共享初值、各组last/final/raw/DC曲线与128fit，预留最低512MB余量，失败报错；不清理旧模型。真实smoke前置，各组2updates与32dev完整路径成功后后台正式启动。
