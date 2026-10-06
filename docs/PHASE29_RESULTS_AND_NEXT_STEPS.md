# Phase29完整结果与下一步

2026-10-06：三个fixed12候选与六个mesh评分已完成，补齐报告时仅修复统计文件路径等价检查和冻结代码导入路径。严格保留真实文件SHA、所有数值、数据/噪声流、冻结模型和原失败记录；不重训。

当前几何三seed均值clip：MBE .864415（对照 .870982）、LBE .420868（.421170），嘴顶点平均误差约3.55856mm（3.56219mm），最大嘴顶点误差约6.69647mm（6.69929mm）。seed49嘴误差反而变差。小收益，不能称达到目标或SOTA。三seed完整报告已闭合；下表明确区分Phase26对照与Phase29候选，完整文件下载SHA核验进行中。

后续顺序：

1. 完整关闭新候选F1/raw/clip、四probe、三seed、分组、global TRAIN/validation与几何/时序联合评估。只有整体证据支持才接纳，保持默认不变。
2. 连续表情条件：先确定dropout是否缩小TRAIN约.149与validation约2.33的差距，结合每情感/强度/说话人的眉/嘴平均姿态偏差；类别识别与连续幅度对齐分别看。不扫dropout、不重复失败的prototype/线性/reference校准、不拟合验证集。
3. 嘴部：范围已经接近GT，解决动态残差的张合时序。中性B0相关.471→最终.253，位移误差+51.8%；三固定draw间变动约占26%，均值也有干扰。先核对有效音素时序条件与残差使用，下一候选只改一个机制，保留情感/风格对嘴幅度的作用，不恢复嘴mask或部署挑draw/平均draw。
4. 同预算、固定种子验证任何候选的F1、MBE/LBE、lip mean/max、表达误差/FDD、mouth range/jaw correlation及身份冻结，不以单项改善批准。
5. 论文比较先锁定输出及指标协议、真实方法实现范围和输入预算；当前VOCA/EmoTalk等是注明的适配基线，不能冒充官方DeSTalker/MedTalk复现。开发完成后一次sealed最终评估，保留此前所有验证开发记录。当前不存在全部指标达到SOTA的证据。

进度恢复先读 CURRENT_OPTIMIZATION_STATE.md，随后查远端 postprocess_repaired_v2_state.json。当前已登记工具/helper路径及失败原因，不重复分析或重启训练。

首组完整结果（seed47，仅阶段性，不能代替三seed均值）：clip原128/64 F1 .791174/.721911，对照 .789976/.727179；raw .614920/.594793，对照 .629749/.607756。global TRAIN MSE .164824（.148714），val2.319534（2.304595），连续条件泛化未改善，不能用小幅MBE下降推断机制修复。待48/49完整闭合后作最终取舍。

## 三seed完整结论

全部1367validation×三固定draw×四冻结TRAIN probes，raw/clip、分组、精度、pairedCI、全TRAIN/val连续global与同rig几何报告已complete。三训练stream完全相同，B0/identity/teacher冻结。无sealed读取、模型推广或seed/epoch/draw选择。报告修复保留原失败证据，仅验证后忽略统计路径，冻结代码导入和启动目标绑定修复；六项契约检查通过。

| 指标 | Phase26对照 | Phase29 dropout |
|---|---:|---:|
| 原生成clip F1 128/64 | .796875/.721358 | .797647/.715031 |
| 原生成raw F1 128/64 | .619331/.591701 | .610685/.583535 |
| MBE clip | .870982 | .864415 |
| LBE clip | .421170 | .420868 |
| Lip mean mm | 3.562193 | 3.558562 |
| Lip max mm | 6.699290 | 6.696472 |
| Expression mean mm | .762788 | .751221 |
| Vertex absolute FDD mm2 | 138.879971 | 138.434368 |
| Mouth displacement MSE clip | .001816626 | .001879576 |
| Continuous global TRAIN MSE | .149291 | .165024 |
| Continuous global validation MSE | 2.327279 | 2.309306 |
| Continuous global validation meanbias MSE | .722495 | .729898 |

三seed两原clip F1都仍>.7，MBE都未到.7，三个联合gate全false。嘴动态误差+3.47%，第二套原clip F1下降约.00633，raw四probe均值下降；global验证误差仅约.77%下降，meanbias反升。此次pooled dropout不足以解决连续条件泛化和张合时序，不接受为新的最终模型，也不继续扫dropout。正则化试验保留为消融。

同协议EmoTalk-core adaptation：MBE .7450、LBE .3315、Lip mean2.973mm/max5.563mm、EVE .692mm、原F1 .5919/.6862。FaceDiffuser adaptation：LBE .3231、Lip mean3.072mm/max5.864mm、absolute FDD137.962mm2。当前情感识别较强，几何/时序尚未领先；adaptations不是官方DeSTalker/MedTalk复现。不能宣称各项SOTA。

下一项应针对表示/条件机制，而非继续p扫描、增加训练轮数或loss。先做只读的逐通道分解：按B0、残差和最终输出分别统计jawOpen、上下唇、smile及眉眼的幅度/速度/情感强度斜率，并检查content/u_a与残差时序是否错位。若确认残差速度能量持续破坏B0音素时序，再做一个单变量的内容条件残差门控候选：门控由B0内容/音频局部状态产生，初始化为全开，保留情感和风格对嘴部幅度的通路，不做固定嘴mask、不加额外监督loss。先用小规模真实流smoke和oracle/static/reverse/shuffle诊断确认接收器确实使用该条件，再固定三seed预算比较；不通过就不训练大实验。
