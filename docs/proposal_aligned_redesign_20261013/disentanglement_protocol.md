# 三重解耦怎样实现：训练样本、约束与验证

日期：2026-10-13。方案补充；未实现/训练。解释 architecture_spec.md 的三因子语义，不以加法形式或三个编码器声称已经解耦。

## 1. 因子定义与生成接口

查询 i 有音频 A_i、动作 Y_i、说话人 p_i、情感 e_i、片段级强度 k_i。参考 r 与查询 disjoint，语句/情感无需相同。

生成路径：C_i=F_c(A_i)，视觉训练阶段 (g_i,d_i,N_i)=T(Y_i; C_i,S_r)，S_r=F_s(ref_r)。E_i=(g_i,d_i)：g 是共享的情感语义/相对强度，d 是该次表演的逐帧展开；N 是受限局部事件实现变量，非第四个独立语义类别。

Y_hat_i=D_pop(C_i,E_i,N_i)+B(C_i,E_i,N_i) S_r。

视觉教师可条件化于内容/个人上下文，但没有原始 query motion 到 decoder 的额外 skip；推理 E/N 改由音频条件 flow 学生采样。编码器结构隔离只是阻断明显复制路径，不证明信息纯净。

目标是：C固定发音和时间，E固定情感意义/状态，S固定个人响应倾向；三个因子可在同一嘴部发生条件交互。不要求统计完全独立、固定通道互斥或所有人的表达曲线一致。

## 2. 内容路径的实际学习

- 768D HuBERT不能假定纯内容。内容适配器用真实音频-motion配对学习viseme/发音时间与局部跨模态对齐；B0/h0提供初始化而非新的真值。
- 真实各类情感训练一个motion→viseme observer，检查其跨说话人/跨情感的准确率后冻结。它只能读取可见发音分组，不能完全识别所有音素/文本。
- 从C的语义表示预测说话人/情感的辅助对抗probe仅作为防泄漏手段：在情绪/说话人/内容均衡或可匹配的训练batch中小权重施加，不对声速、节奏等实际发音时序要求完全去信息。
- 固定查询音频，换E/S/N时约束输出viseme证据和必要闭合/同步。保护发音时序，不将GT情感嘴部宽度强制变成B0宽度。

## 3. 风格路径：真正有GT的跨参考训练

选 query q=(人A,angry,句子1)，参考r1=(人A,happy,句子2)、r2=(人A,sad,句子3)。教师E_q/N_q和内容C_q固定。

分别生成 G(C_q,E_q,S_r1,N_q) 与 G(C_q,E_q,S_r2,N_q)，二者都以真实Y_q为重建目标。这是合法的原始数据监督：替换的是同一人的独立参考，没有创造不存在的表演GT，也无需对齐参考与查询动作。

同人跨句/跨情感对S作正样本，跨人可比情感/内容作负样本，避免模型靠情绪分布区分身份。对posterior只在两参考都有支持的响应方向约束均值/预测行为一致；缺失情绪或发音覆盖不强迫整个后验及置信度完全相同。

附加S上的情绪/内容泄漏probe，结合speaker-balanced loss；speaker分类只辅助，不等同风格。静态脸型/追踪偏置、摄影条件和数据源捷径须由实际跨source测试揭示。

人口基底以S=0定义，S中心/尺度约束及支持查询训练限制其分解漂移。但基底/响应分解一般不唯一，不能称为发现了真实物理上的纯风格变量。

## 4. 情感路径：共享语义，保留不同轨迹

- g用MEAD的8类及片段级ordinal标签监督；不同人同类/近level作为语义正样本，不同类作为负样本。只能拉近语义子空间/类别原型，不能将整条E轨迹逐帧拉成相同。
- d/N保留每次表演的实际变化，以教师驱动动作重建、可信表情状态/事件弱标签、时间平滑/局部变化约束防止退化；不把MEAD片段标签逐帧复制为强度GT。
- 用speaker-discriminator检查/小权重抑制E中的可迁移个人响应信息。它不能消灭所有自然相关性，因此输出参考交换测试与独立评估仍是核心。
- 防止E/N记住发音：受限latent容量、去掉raw-motion decoder skip、内容probe/交换输出约束、真实配对语义锚点；N的编码有有效事件/动作监督、先验及容量限制，避免成为能绕过C/E/S的自由缓存。
- 教师看到一次表演的E*/N*；音频学生拟合条件轨迹分布。teacher reconstruction可以对真实轨迹作逐帧loss，prior随机样本不能强行拟合唯一GT的每个事件时间。

## 5. 交换训练：哪些有GT，哪些没有

| 交换任务 | 动作GT | 约束 |
|---|---|---|
| 固定query，换同人异句/异情感参考 | 原query Y_q | 真实重建、参考不敏感性/覆盖-aware响应一致 |
| 固定query，换另一人参考 | 通常没有 | 内容/情感保持、个人响应变化、真实性；不能逐帧贴donor动作 |
| 固定query，换情绪 | 通常没有 | 内容/风格保持、新情感语义/强度、真实性；same text不代表时间对齐 |
| 固定C/E/S，换随机种子 | 通常没有对应多表演 | 合理多样性、条件保持、时序/事件分布；不以唯一GT的事件F1判全部样本 |

latent cycle只作补充，不能单独证成解耦。循环编码器可能协同作弊，必须有冻结的外部目标和独立测量。

## 6. 损失的职责

L = L_real_recon + L_content_anchor + L_emotion_semantic + L_style_support_query + L_factor_preserve + small_weight*L_leakage + L_regularize。

- L_real_recon只使用真实同query配对GT：52D/rig geometry、velocity/同步；可靠帧/通道mask。
- L_content_anchor监督真实配对发音/时序，L_factor_preserve约束反事实输出发音、情感语义、个人响应中的非目标因素。
- L_emotion_semantic监督共享g及ordinal聚合；L_style_support_query训练响应行为，不只预测speaker ID。
- L_leakage是均衡/条件化的辅助对抗或对比约束，不声称逼近probe随机率就证明独立。
- L_regularize控制S中心/尺度、posterior prior、N容量等，防止任选空间换名或绕过。
- 冻结训练observer/evaluator用于loss可微反传到生成器，不更新observer自身；正式报告另外使用held-out independent evaluator/实际几何和人工测评，避免优化自己的probe即高分。
- 联合解码器收到的只是受约束的codes，没有raw query/reference motion特征旁路；推理参考motion只经S。这样排除直接序列复制，但仍须测试潜空间隐含泄漏。

## 7. HDTF与未见人

HDTF无情绪GT，mask L_emotion_semantic/ordinal标签项；可做音频-mouth配对、自重建、合法同人support/query和无标签动态学习。不默认neutral，不把同片段近邻一致当跨情感证明。跨video同人先核实身份；speaker/source先split，再窗口化。

测试允许未见人一条独立enrollment参考；其query motion只用于评估，不送入生成或训练。style encoder不得把query当reference。

## 8. 成立的输出证据

三重解耦是以下联合结果：固定C换E，情感明显改变且发音/个人倾向保持；固定C/E换S，个人响应改变且内容/情绪保持；固定E/S换C，内容改变且情感/个人响应保持。

输入同人neutral/happy/angry不同句子的参考，查询保持相同，比较生成的情感、发音及个人响应；参考内容不相同时无需输出同一帧序列，但同一人的可支持响应参数应稳定。跨人同类需共享情感意义，同时保持个人形变差异。

按情绪、说话人、source和支持范围报告目标变化、非目标泄漏及置信区间。公式、接口隔离、pooling、对比/正交/cycle均不替代这些证据。当前是实现协议，不是已经完成的解耦实验。
