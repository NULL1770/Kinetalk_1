# 为什么论文能做出动态，而当前音频分支还不能

## 后续路线：用户明确不采用MEDTalk文本分支（最新约束）

主方案不加入Whisper转写、RoBERTa或文本cross-attention。上一轮建议的文本消融取消。借鉴对象是音频时间信息、可靠motion监督与预测条件参与生成训练的原则；不能把更换模块名称或预训练模型称为创新。

待验证的核心假设：现有motion教师按重建能力选择动态，未确保这些动态可由音频跨句预测。比较“motion-only选动态”与“训练音频-motion配对选择共享动态空间”，既不强迫一个全脸强度，也不直接扩大rank。优先以正则低秩回归（ridge-RRR，已有统计方法，非本项目发明）实现可审计基线，再决定是否进入主模型。

具体执行顺序：
1. 锁定新增句子测试集、保持现有开发集标签；核验motion时钟/眉眼质量并扩大训练句和身份覆盖。先复用现有音频编码器核验中间层，再固定一种多层声学+F0/log-energy配置；这部分属于常规输入改进。
2. 相同监督/预算比较原音频输入与新增时序输入，直接正则预测真实中心化motion残差，检查upper收益，避免输入和目标同时变化而无法归因。
3. 训练句配对拟合共享motion基U；目标必须是 z_m=center(motion-B0-P_id)U，而不是音频模型预测值。与同rank motion-PCA/原teacher比较；原teacher若看过留句须重训严格对照。内折重拟归一化、U及正则，固定U后评原始motion单位误差、oracle/predicted/zero/reverse，不只比较不同latent的R²。接口仍最多8维，候选2/4/8仅训练内选择，不主张增维。
4. 如果可预测成分覆盖眉眼且GT条件能有效生成，冻结U、B0、身份、global，训练audio动态头、local_projection和现有ResidualDiT。统一低率z_a投影到原local_emotion接口，不加第二个音频旁路或区域decoder。使用flow及一个归一化z_a–stopgrad(z_m)对齐目标；teacher-local条件前期辅助，末期纯audio-local。固定全局输入不代表生成全局情感一定保持，须检查最终输出。
5. 控制变量依次比较旧输入/新输入、motion-only/paired基、joint阶段有无对齐。验证选好配置后多seed复核、新测试一次。检查真实眉眼、口型、global、同noise时间干预与多noise合理多样性，固定标量R²不是唯一生成验收。

若配对筛选只找到嘴部残差，按“音频可预测的内容残差”记录，不能称情感解耦；若眉眼跟踪错误先修监督，不能用更多loss补。随机、音频不可唯一决定的动作由已有flow学习条件分布，但该分工的有效性仍须验证。

执行更新：run30/31已完成真实中层音频＋韵律、RRR/PCA/direct ridge对照，以及1200训练/280验证扩展。眉眼出现小幅可预测性，但幅度弱、neutral嘴部残差退化；尚无生成成功结论。第1步仅完成metadata/hash/native时钟核验，没有完成原视频眉眼跟踪质量抽检；原teacher严格重训对照亦未完成。详情见`DYNAMIC_TRAINING_ADJUSTMENT_RESULTS.md`第12节，不能把这些待验条件写成已补齐。

run32/33已进一步完成生成小训练：共同训练破坏原能力；固定预测器/生成器，仅训练原512参数共享接口后，相对zero/reverse有小幅正确时序收益。眉毛独立收益与neutral保护仍未联合通过，不更换默认权重。后续优先训练内检验已有global的动态幅度条件化，保留原motion监督，不直接借用MEDTalk文本部分。详见实验报告第13–14节。

## 2026-09-16 论文复核与run27–29后的修正（优先于下文早期方案）

来源：MEDTalk https://arxiv.org/html/2507.06071v2 ，SubtleTalk https://arxiv.org/html/2608.06408v1 。本次核对缓存正文的MEDTalk Eq.1/9/11、SubtleTalk Sec.3.2–3.4/4.1/4.3；未声称复现官方完整代码。

- **不能把run27当作MEDTalk复现**。MEDTalk以audio加label/image/text情感指导为输入，用emotion2vec与Whisper→RoBERTa文本预测强度，在学习到的情感embedding中调norm，经fusion encoder与content-conditioned motion decoder生成动作。Eq.11约束生成动作的Int与真实动作Int一致；不只是自由latent逐帧MSE。run27固定52维动作ray直接乘scalar仅是线性proxy，几何失败不能否定latent强度调制。
- **不能把现有emotion2vec+TCN当作SubtleTalk复现**。其WavLM末层content与第3–11层声学特征、显式F0/log-energy融合后直接进入生成路径；emotion2vec→VADP是另一条VA预测路径。其逐帧视觉VA来自EmotiEffLib，窗口eye/brow/head强度另有控制，训练motion由TEASER拟合、同步/姿态筛选和时序去抖。现有final-layer emotion2vec替换audio缓存并未补齐这些条件。
- **数据和评价任务不同**。SubtleTalk训练59.76小时/2456身份，当前动态pilot224片段/4人。MEDTalk有外部情感指导及合成情感-内容配对；正文未给足以核实的总小时/句子隔离规模。不能将它们的生成展示、FDD/HDD/EIE等直接等同于我们去均值动态R²，也不能据论文推定高精度逐帧还原全部动作。
- **评估门槛需分清对象**。固定target的R²/反转是可预测性诊断，负R²不能单独否定条件生成；接入flow后还须测音频时间干预、表情变化分布/事件、固定噪声对照、视频质量和口型。随机幅度增加也不是成功。此前renderer局部/联合试验失败不等于完成了富音频条件下的整条链验证。

当前应补齐的最小链：保留B0、neutral身份及audio global；真实多层音频+韵律送入共享时序条件和现有残差flow；视觉/motion提供经质量核验的稠密表达监督，保留motion teacher作为训练端；audio预测条件从训练期就参与生成。VA可以作为注明来源的参考基线，不强制成为新主架构；替代监督是否同样可靠需实验。表达场在特征空间调制解码，不直接绑定一条固定全脸动作方向，不增加眉眼嘴独立输出头。优先锁定新测试句并扩大训练句/身份覆盖，先做忠实借鉴基线，再检验neutral身份与共享动态场的增益。

当前失败不能归结为“8D 太小”。已有 rank8/rank16 预测探针在相同句子划分、stride=4 和训练步数下，heldout R² 仍接近或低于 0；这说明目标定义和训练路径比维度更关键。

## 已确认的结构差异

1. **动态目标不同**
   - 当前 motion teacher 是从 `motion - B0 - identity` 的全脸残差中自监督学出控制点。它可以重建动作，但没有保证每个控制点都能由音频决定，可能包含眨眼时刻、跟踪噪声和个体随机动作。
   - MEDTalk 从选定表情控制器计算逐帧 L1 强度伪标签；它预测的是“这一帧表达有多强”，不是任意动作 latent。
   - SubtleTalk 使用视觉估计的逐帧 VA，并额外使用眼、眉、头部等窗口强度作为显式条件。VA 是监督接口，不是从重建 latent 中自然涌现的语义。

2. **音频条件更丰富**
   - MEDTalk 使用 emotion2vec 和 Whisper/RoBERTa 文本特征。
   - SubtleTalk 同时使用 WavLM 内容、多个中间层的多尺度声学特征、F0 和 log-energy。
   - 当前 audio student 主要输入 83D 缓存声学特征；emotion2vec 只改善了全局类别，尚未改善眼眉时序。不能把“换一个 SER 特征”当成完整复现。

3. **训练梯度路径不同**
   - 当前先用 motion teacher 条件训练 renderer，再让 audio student 拟合 teacher controls。student 的主要损失是 latent 蒸馏，最终动作误差只作为后续探查。
   - MEDTalk 在冻结 motion decoder 后，用预测的 audio content/intensity 直接重建动作，并用 embedding similarity 和 intensity consistency 约束。
   - SubtleTalk 的生成器训练时直接读取音频的多尺度特征、prosody 和动态控制。它不是先训练一个只接受 oracle 动态的 renderer，再期待 student 自动适配。

4. **数据规模和动态可识别性不同**
   - SubtleTalk-Face 约 73.83 小时、3905 个身份；当前 pilot 224 条训练片段、4 个身份。小数据很容易让 teacher 记住说话人或句子特有的动作。
   - 论文评价的是表达强度、上脸运动分布和同步，不要求音频确定每一次随机眨眼的确切时刻。

## 下一版最小改动

保留 B0 内容路径、neutral identity baseline、global audio emotion 和单一 flow renderer。只改动态接口：

1. 从 neutral-corrected residual 计算一个低频、可解释的 **expression-energy field**：对预先登记的情感相关控制器做归一化 L1/窗口统计，再去除每条说话人的 neutral 基线和句内均值。它不是 VA，也不增加用户可见的区域输出通道。
2. 音频分支用冻结的 emotion2vec/WavLM 中间特征加 F0/log-energy 预测这个场；先用原 83D 作为对照，避免一次引入不可归因的旁路。
3. renderer 训练时同时看到 teacher field 和预测 field，并对预测 field 计算一次动作 flow/reconstruction loss。这样 audio student 的梯度来自最终动作，而不只来自 latent MSE。
4. 只保留三类必要目标：动作重建、expression-energy 一致性、已有的全局情感分类/强度分类。冻结 B0 和 identity；不增加 VA 主路径、逐帧自由向量或大量 region heads。

## 判定标准

必须在新锁定句子上同时检查：audio full与自身静态均值和reverse、眼眉动态、jaw/lip-sync、global emotion accuracy。若expression-energy可预测但生成失败，应查renderer条件利用；若目标预测弱，需要区分输入、监督、容量、数据覆盖及评价口径，不能直接断言是数据/标注，也不能靠增大rank解决。条件生成还需评价合理多样性，不能只用确定性R²验收。

## 已完成的小规模对照

在同一 224 条片段和同一 sentence-disjoint split 上，使用相同的 upper-face L1 intensity 目标，分别训练 fresh audio student：

- 83D acoustic：heldout R² `-0.084`，时序相关 `-0.049`。
- 768D emotion2vec：heldout R² `-0.016`，时序相关 `-0.064`。

emotion2vec 让 R² 接近零，但没有形成稳定正向动态，说明预训练情感特征有帮助但不能单独解决问题。下一步应加入明确的 F0/log-energy 和多尺度中间声学特征，且继续使用同一个 upper-face intensity 目标，避免同时改变监督和评价标准。

随后进行的低成本上下文对照把当前 83D acoustic 替换为 `[当前帧, 帧差, 5 帧平滑]` 三分支，仍使用 upper-face L1 intensity 和相同句子划分。heldout R² 从 `-0.084` 降到 `-0.113`，因此简单拼接上下文会增加小数据过拟合，已不作为主线。
