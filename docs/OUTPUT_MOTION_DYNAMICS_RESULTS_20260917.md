# 最终位移＋幅度监督：真实训练与独立审计结果

2026-09-17。**本轮补齐并实际训练了该方案，但没有做出可接受的眉眼动态，默认模型不替换。** 汇总中心化眉残差RMS接近真值，同时正确性与分布评分恶化；不能把该汇总幅度当作可见动态成功，本轮也不支持只缺一项std loss的判断。

## 训练与复现

`train_output_motion_dynamics.py`，原direct epoch000初始化，rich audio TCN＋完整renderer；flow加实际12步Euler生成动作的相邻位移MSE与眉眼std MSE。audio/zero各8epoch、1160步，共2320步。同seed46、batch16、优化器、batch/noise/time/choice draws；B0/identity/global继续冻结。完整协议见`OUTPUT_MOTION_DYNAMICS_PROTOCOL.md`。

独立`audit_output_motion_dynamics.py`已完成：所有epoch及last/final、来源和fit-only scales核验；八轮随机数hash重放；audio/zero/旧direct-L1共3×8seed×3mode=72组实际生成曲线逐值一致，maxabs=0。zero臂encoder每轮不变、三种推理模式相同，两个新臂epoch0等于历史zero-local。test未加载、未挑轮次、未改默认。

## 主要结果

下表为405内部开发中331非中性片段，八seed真实生成、统一原始系数口径。

|眉部指标|冻结来源|旧direct L1|本轮audio|本轮matched zero|
|---|---:|---:|---:|---:|
|中心化残差R²|-.279269|-.046649|-.900543|-.490637|
|中心化相关|.106181|.146796|.033151|.063485|
|RMS幅度/GT|.645202|.407944|.982699|.766811|
|fair轨迹ES↓|.019025|.022625|.033865|.030181|
|fair邻帧VS↓|.004864|.001514|.001844|.001780|

audio比matched zero的眉ΔR²=-.409906，句cluster95%CI[-.593479,-.218929]；ES改善=-.003684，CI[-.005183,-.002199]；邻帧VS也变差。三个开发身份非中性眉ΔR²均负且CI排除0：M024=-.368519、M023=-.434192、M030=-.369848。

matched-zero仍保留B0/content/global音频，只关闭新增local条件。上述比较约束的是新增路径在本轮联合训练中的净收益，不是音频整体是否有用。三人RMS比约为1.192、.791、1.337；汇总.982699也不表示每个人、每条片段或每个通道幅度都正确。

audio比own-zero眉ΔR²=-.898803，CI[-1.108132,-.728463]。比reverse点估计+.118294，但CI[-.023421,.278994]跨0；ES/VS确实优于反转。这说明模型使用了时间条件，却不能证明使用得有益，更不能用反转一项掩盖matched-zero失败。

## 嘴部：动态改善被均值退化抵消

对冻结来源，非neutral mouth中心化误差从.00575187降到.00526231，neutral从.00127190降到.00089868；嘴相关及速度保护通过。但是最终raw mouth MSE分别**增加8.509%和78.868%**，单侧90%上界11.246%/85.467%。

精确分解raw误差=中心化误差+均值误差：非neutral mouth均值误差.00642795→.00795388；neutral .00084265→.00288357。raw退化来自平均口型漂移，不能因速度好或B0冻结说口型没问题。jaw17非neutral raw增加60.0%，需特别保留报告，不能只报全嘴平均。

眼/嘴raw保护失败，眉时序和生成分布两类验收均失败。对来源，上脸均值、嘴相关、速度保护通过只是部分工程指标，不构成可发布结果。

## 本轮排除了什么，尚缺什么

该固定8epoch、当前权重和数据组合没有支持“只补velocity＋std就足够”。它没有否定SubtleTalk整套方案，也不证明更多上下文、可信视觉监督、可部署区域强度条件、更合适的优化或更大覆盖训练无效。

本次汇总中心化残差RMS接近GT，但时序相关降到.033；这不是逐片段std或实际显示幅度都恢复的证明，三个人的幅度比仍有明显差异。当前证据不支持直接放大或盲扫std权重。位移/std均对常量偏置不敏感，现有flow在联合目标下没有保住口型均值；下一完整训练必须明确raw输出/均值/合法系数域的保护，而不是只管centered dynamics。

固定样例的可见域也未通过：M024新audio的左右outerUp、M030的innerUp与左右outerUp在有效帧全部为负，显示clamp后完全不动；M023右outerUp也全部为负。M023眉下压std虽增大，峰仍错位。原始std接近GT不代表最终rig真的抬眉；详见同目录连续视频和显示审计。

下一优先级是数据与条件协议：完整native时钟缓存、随机连续窗口/上下文、MEAD八类/L2及可信眉事件监督，再检验可预测活动/随机残差。在当前24bin中心化特征上再堆头或损失的证据不足。详见`BROW_DYNAMICS_RESEARCH_AND_ACTION_20260917.md`与`DYNAMIC_DATA_EXPANSION_PLAN_20260917.md`。

## 产物

- 远端完整：`/root/kinetalk_runs/teacher_schedule_v1/output_motion_dynamics_v1/`，含每epoch可恢复checkpoint与全部曲线。
- 独立审计：`paired_audit.json`；本地`artifacts/brow_review_20260917/output_compact_results.json`含全部分数、五类基线、逐人结果。
- 固定九例及三人首条连续视频：`artifacts/brow_review_20260917/visual/output_motion/`，使用旧metadata锁，不挑结果或noise。
- 本地完整测试340通过，之后新增审计相关9测试通过；真实远端训练相关5测试和评分相关5测试通过。单测不代替效果证据。
