# TRAIN表达目标可靠性审计（2026-10-07）

只读检查，无模型/数据/监督修改，无sealed目标读取。恢复后先读CURRENT_OPTIMIZATION_STATE.md，本文不替代Phase34评分。

完成全部2,583条原训练情感→中性配对，验证source/reference均在TRAIN、同说话人、同真实文本hash、独立中性reference、native时间严格相等、shape/finite/二值mask。所有合同检查通过。报告、逐pair记录和5,166个配对文件SHA绑定已下载验证；本地`final_experiment/evaluation/diagnostics/expression_target_audit_20261007/download_verified.json`为true。

- eligible TRAIN manifest共有3,221行，其中638条是中性自配对；Stage1明确排除它们，并使用全部715条中性样本的原生动作。实际B0来源仍715+2,583=3,298，未发现扩大到低质量配对的代码证据。
- 297条通过整段mouth event gate，2,286条使用局部监督。所有2,583条都提供event mask，因此297条也仍受到实际局部mask限制，整段gate通过不代表每帧都被监督。
- event_local_mask全部为精确0/1；float32→bool不会把软权重变成全true，不存在这个假设中的数据问题。
- native可用嘴帧267,521，最终嘴帧148,140，覆盖55.375%；有效相邻帧120,902；无空嘴监督clip。不能将这些mask用于硬关闭推理嘴通道。
- source/reference supplied text hash、sync、geometry/path steps均通过已有检查；所有原artifact的`dtw_quality_verified=false`。manifest额外两路路径/event审计也存在，不能仅由这个旧flag宣布全部配对无效。缺乏独立逐音素真值；path agreement、ASR一致、嘴事件一致不等于真实逐音素精确对齐已证明。
- ASR相似度均值.9506，mouth event agreement均值.5714，两路路径p95差均值40.06ms；这些是已有质量检查的描述值，不是新淘汰阈值。
- 可信帧上的情感GT−对齐中性reference：jawOpen clip RMS均值.11755，brows RMS.20603、中心化RMS.03833。这支持“情感应允许改变嘴幅度”和大量静态表情偏差；配对差仍含发音/对齐误差，不作为纯情感逐帧真值自动投入训练。

原helper v1遗漏排除中性自配对；v2把padded valid当flat索引；v3描述性jaw索引用14(jawForward)而非17(jawOpen)。失败/旧文件保留，均未改变训练；**只引用v4、schema train_expression_target_audit_v2**，jawOpen=[17]显式绑定。最终report SHA及全部输入绑定见本地state/artifact_bindings，不引用v3的jaw读数。

## 三级强度的实际描述

仅复用已核SHA的TRAIN配对记录，按同speaker、同emotion、同一个验证过文本hash的中性reference组成469组L1/L2/L3；其中三个clip均通过整段gate的只有9组。未读取validation拟合阈值。

pair delta jaw RMS三个level均值.09245/.11813/.14967，但完整L1<L2<L3只有49.89%组；happy为33.3%、contempt22.4%、surprise76.8%。全嘴RMS完整递增57.78%，嘴部表达组48.61%、眉部61.41%。因此数据平均趋势支持强度调制，**不支持逐条统一强制jaw/全嘴RMS单调增加**。这些是带不同局部mask的对齐pair差值描述，仍包含对齐/发音误差；不能将比例当作纯表达真值质量或据9组三全gate组拟合复杂类别阈值。

逐组clip/ref/coverage、逐情感读数在`intensity_order_description.json`；当前ordinal loss仍未打开。这项审计仅排除无依据的约束，未产生模型性能提升。

## 后续D2的实施边界

当前教师只输出有效的global情感/强度，u_a没有教师逐帧表达目标。直接关掉u_a或detach其全flow梯度会丢掉已测得的眉部动态，不实施。

下一项表达职责实验必须先明确：教师读取哪些表达通道、独立参考如何去静态身份、逐帧目标来自何处、缺失/配对不可靠帧如何屏蔽监督、audio的哪个梯度更新来自表达而非B0误差。上脸9通道可使用原生时钟的自身GT与独立中性anchor；嘴角/嘴幅度只能按实际配对质量处理，不能把MEAD1/2/3复制为逐帧动作真值或强制所有通道单调增加。

教师逐帧目标需要真实重建/动态保留验证后才可蒸馏；不能仅将现有clip-global向量重复T次当作动态目标，也不能把历史未启用的四状态头说成已经生效。full residual flow继续训练renderer、h0提供内容，表达可前向影响嘴幅度；不恢复15通道嘴mask、不恢复HuBERT学生入口、不混入native-GT B0。

Phase34结果闭合后，仅选择一项证据支持的职责修改，先梯度/时钟/真实数据smoke，再固定预算匹配训练；不同时添加交换/嘴adapter/多个loss。身份需要同内容同情感不同参考的单独验证，现有source→neutral合格不能推导跨身份交换合格。
