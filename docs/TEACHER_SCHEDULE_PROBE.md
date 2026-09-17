# 训练后期动态退化：教师日程对照

2026-09-16，运行前约定。已有正式4组均在epoch2后动态下降，teacher比例同期下降；这只是相关性，不能据此认定撤除teacher是原因。本实验只改变teacher取样日程。

数据：仅原正式train 2720段/22人，按固定元数据hash留3人；余19人内按句3fold选择ridge alpha，重新拟合rank8 U、输入RMS、audio head和目标尺度。留出人的query motion不参与这些拟合；neutral登记参考继续独立。继承的B0/global可能有历史暴露，不能称全系统未见身份。已使用原280和新439开发集不用于本轮调试，512和15个封存测试目标不打开。

三组固定seed46、18epoch、batch16、Adam lr=.0002、同shuffle/noise/flow time/teacher choice原始随机数：

|组|真实motion条件的概率|推理条件|
|---|---|---|
|constant_teacher|全程.5|audio|
|audio_only|全程0|audio|
|decay_teacher|前5epoch .5→0，其后0|audio|

真实motion条件是中心化动作残差在U上的投影；audio预测同一坐标。所有条件乘同一个冻结audio gate。B0、身份、全局情感、U、audio head、renderer均冻结，仅原512参数投影学习现有flow MSE；无新loss、文本、VA或区域生成头。三组均零初始化接口。

主比较固定epoch18，epoch2是预先指定诊断，不挑best。每2epoch报告audio full和zero；epoch2/18额外评reverse/oracle，三个固定生成噪声。报告句簇区间、每人眉/眼、neutral原动作误差、口型相关/速度和冻结teacher类别读出。类别读出不是独立情感质量证据。训练内只有3留出身份，句簇区间不能代表身份总体。

解释：持续teacher若只提高oracle，不能叫audio动态成功；需audio full相对zero/reverse也改善并守住眉部和neutral/口型。如果全程teacher相对decay恢复收益，支持日程影响此条件接口；不能据此声称解决了跨数据集或新句泛化。若三组都退化，则不继续相同长训，应回查弱audio目标与flow条件传递。

## 实际结果

三组各18epoch/2610步已完成（共7830步）。实际19人2315段训练，M023/M024/M030三人405段内部开发，shared56句；U/head/scales全部重新拟合，alpha1。三组每轮batch/noise/time/choice哈希匹配，zero生成逐值相同，所有冻结参数和头不变。原280/439目标值未用于本次比较，最终测试封存。

|内部405，非中性audio full-zero ΔR²|持续teacher|纯audio|逐步撤teacher|
|---|---:|---:|---:|
|epoch2 upper（仅辅助诊断）|+.02388|+.02088|+.02422|
|epoch18 upper（主比较）|-.03894|-.25706|-.22810|
|epoch18 brows|-.06834|-.29578|-.25252|
|epoch18 eyes|-.00055|-.20648|-.19620|
|epoch18 neutral mouth原动作MSE增幅|+1.42%|+3.26%|+2.93%|

持续teacher比撤teacher的upper ΔR²高.18915，句簇95%CI[.14928,.23729]；但自身相对zero仍负，CI[-.06527,-.01423]。说明保留teacher可减轻退化，不能解决退化。三组epoch18都不接受；epoch2不替代预先规定主结果。原有默认和正式早期候选保留。

## 同样本训练误差与实际生成诊断

从19人fit按元数据hash预选64段、每情感16段，六个checkpoint使用完全相同3noise；比较t=0/.25/.5/.75/.9的flow速度MSE及同样本12步实际生成。没有使用405留人集来抽样，不能把此处fit结果称泛化。

三组epoch18的audio upper flow误差在所有t都比epoch2低，但实际生成upper更差。例如纯audio的t=0 flow MSE .59282→.56190，而同64段实际upper R² -.09907→-.22164。持续teacher对应生成R² -.09704→-.11265；其audio local RMS .01822→.04536，pure audio .01951→.07599。增长本身不是因果证据，不能据此直接选幅度阈值。

这反驳“只因撤教师”或“只因新身份”解释，并说明当前冻结生成器下的单步flow优化和实际闭环效果失配。**t=0误差也改善，故没有证据支持仅改成t=0采样。**这不表示Flow Matching公式错误。

## 单因素：直接训练推理时的生成过程

预先固定一个新arm：保持teacher概率.5、同19/3划分、18epoch/seed46/Adam/随机数/512参数，唯一把速度flow MSE替换成实际12步生成结果的observed motion MSE除以residual_scale²。不相加新loss、不改生成器、不加区域头；仍按epoch18比较，epoch2辅助。冻结模型允许梯度穿过生成计算回到原512接口。训练每batch仍消耗旧flow time随机数但不用，以保持noise和teacher choice配对。

依据是同样本flow误差下降却闭环结果恶化；新目标直接对齐实际推理轨迹。风险是音频不确定性下偏向均值、原动作MSE不一定改善动态，必须同时胜过zero/reverse并守住眉部/neutral/口型，不能仅以胜过已坏的旧epoch18宣称成功。此单因素不通过就停止本轮扩展。

实际已完成18epoch/2610步，约8分钟，随机数/teacher取样和原constant组逐epoch完全一致。epoch18 upper full-zero ΔR²=-.03642，CI[-.06288,-.01143]；brows=-.06014，eyes=-.00544。比原constant组upper只+.00252，CI[-.00003,.00516]，无可靠优势。neutral mouth误差+1.67%（单侧90%上界2.87%）和口型相关通过，但nonneutral upper速度误差+12.76%，未通过动态/速度保护。只替换为闭环原动作损失不解决问题，默认不采用。三日程加闭环共10440步；所有GPU任务已结束。

## 精确分解：总误差收益来自哪里

使用全部405内部开发、同一原帧时钟/观测mask，每clip×channel精确验证：`SSE(raw error)=SSE(error−mean(error))+n×mean(error)²`。先float64验证B0和identity同时从预测/目标中消除，再累计有效观测数、平均三noise误差；没有重投影或拟合。这是均值/时序误差分解，不等于身份/情感语义分解。

|非中性upper，epoch18|zero-local|闭环audio full|
|---|---:|---:|
|原动作MSE|.02654629|.02532607|
|时序中心化MSE|.00100574|.00103499|
|时间均值误差MSE|.02554054|.02429108|

原动作误差变化−.00122021由均值项−.00124947覆盖时序项恶化+.00002925。三种flow日程也都出现相同方向；三项变化句簇区间已保存，闭环时序恶化CI为[.00001003,.00004606]。同64fit诊断中原动作MSE也下降，只是时序部分上升，因此不能说“整体生成误差变差”或把问题仅归为闭环状态：**目前全脸原动作目标允许均值纠正换取动态退化**。这是可测误差来源，尚不能证明优化器唯一因果机制，更不能据此认定identity模块应重训。

下个研究假设（本轮未训练）：动态接口阶段改为单一去时间均值的闭环残差重建，避免用动态参数补偿平均表情；global/neutral baseline先保持冻结，继续检查均值漂移与口型保护。它是替换当前目标，不是再堆一个loss，也不拆区域。是否需要限制动态输出改变clip mean，须由该独立对照决定；现有证据不支持直接宣称可行或发布。按前述约定，本轮到此停止扩展。

已按预先固定metadata hash选择三身份×三非中性共9例、noise42绘制目标/zero/flow/rollout/oracle系数曲线，未按效果挑样或作时间对齐/幅度缩放。样本锁在曲线读取前生成，但这些身份的总体指标此前已看过，不称独立盲测。`trace_examples`保留九图和总览，仅是系数证据，未做脸视频或独立感知认证。

![教师日程和训练误差诊断](../artifacts/teacher_schedule_v1/teacher_schedule_diagnostics.png)

入口：`prepare_teacher_schedule_probe.py`、`train_projection_schedule_ablation.py`、`audit_teacher_schedule_probe.py`、`diagnose_projection_flow_time.py`。闭环单因素：`train_projection_rollout_probe.py`和`audit_projection_rollout_probe.py`。误差分解与例图：`audit_projection_mean_dynamic_tradeoff.py`和`plot_projection_schedule_examples.py`。远端结果`/root/kinetalk_runs/teacher_schedule_v1`，最终输入`data_locked`；本地`artifacts/teacher_schedule_v1`备份小权重、报告、源码、元数据、哈希，完整cache/曲线远端保留。完整251项测试通过，75条Transformer类警告；未改默认权重、未删训练数据。


后续用户已授权并完成单一去均值目标测试，见[CENTERED_DYNAMIC_PROBE.md](CENTERED_DYNAMIC_PROBE.md)：训练因服务中断仅保存epoch10，按用户新的8epoch预算分析已有同预算报告；上脸/眼部改善，眉部未通过。不是原18epoch终点完成，不续相同长训。
