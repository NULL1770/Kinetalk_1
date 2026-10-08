# Phase44：稳定表达教师与部署均值对齐

2026-10-08。用户授权继续优化、允许同轮多处改动，要求每轮改动前Git存档。Phase43完整成果2f2dfd8已成功推送。本轮不承诺今天达到SOTA或保证发表；目标是在现有资源下当天完成可重放对照、完整指标和八类视频。

## 新证据

Phase43 A/B/AB的full1367冻结g/u交换已经完成，并逐值重放原报告的原生几何与四probe。A正常MBE.827621/F1.695182/jawcorr.473212；audio g + teacher u为.733686/.770071/.870190；teacher g + audio u为.422999/.541066/.476343。B的audio g+teacher u为MBE.307682/F1.695520/jawcorr.849403。混合路径含GT且可能偏离训练分布，不能作为部署成绩或因果独立证明。

A的局部q/p均值差平方5.86155，p方差均值5.19638，q方差.371816；51.2%的p局部原始logvar超过上限。B分别为14.29572、7.09288、.287698和84.3%。原KL均值梯度除以p方差，方差升高可以降低均值匹配压力，但该数值也可能含真实不可预测变化，不能把所有大方差视为实现bug。Phase42unit方差并未同时改善几何与表达，不能直接恢复其失败假设。

内部held-speaker prior位置误差后半程恶化，而posterior继续改善；说明学生目标/接收空间持续漂移值得固定验证。不同参考造成的条件变化仍保留，不把身份当speaker查表。

## 固定三臂（8轮，每臂5456新updates）

| arm | 初始化 | 匹配 | 用途 |
|---|---|---|---|
| a_kl | Phase43-A final | 原归一化KL，q detach | 相同固定教师与追加预算的训练对照 |
| a_mean | Phase43-A final | 归一化decoder坐标均值+高斯标准差匹配 | 检验均值压力独立于方差的修正 |
| b_mean | Phase43-B final | 同上 | 以几何较好的B为另一候选；不冒充同起点消融 |

每臂从同一seed47的既有权重开始，新增固定8轮，同10,903 fit/固定speaker与sentence内留出。优化器重新初始化，LR1e-4，匹配系数.01、学生语义.05沿原full-beta权重（原.1乘q/p语义平均，因此p实际系数.05）。A单参考、B混合参考协议与父模型相同。原父模型24轮预算和追加8轮预算明确报告，不能与未追加模型当等预算消融。a_kl/a_mean参数、顺序、权重、更新数相同，只有匹配公式不同。

q、style、decoder全部固定且eval；p以及原emotion/intensity heads学习。没有新模块、参数、motion critic、mouth gain、GT全动作重建loss或pair数据。新的训练不计算motion重建目标；q可在no_grad下看motion生成教师latent，学生p只看772D。推理签名保持audio/B0/独立参考，无queryGT。

均值匹配定义：g原均值逐维MSE；u均值先按原native时钟插值，并采用该模型实际decoder的去均值设置，然后按有效帧/维数平均。方差部分为原token空间标准差的平方差，原有效token/维数归一化。g/u各占一半，位置/尺度各用.5，与单位方差KL的均值尺度相同。不使用p方差给均值误差降权。

它不是投影后联合分布的完整Wasserstein距离，更不是一个新的ELBO。对于未中心化token高斯，标准差平方差来自对角Gaussian transport形式；native投影后的时间协方差没有被拟合。stochastic质量未验证；当前目标是确定性的prior mean。q仍可能编码B0误差和不可由音频预测的运动；本轮不声称教师纯情感或内容完全解耦。

## 验收

默认旧代码/权重/RNG/predict保持逐值一致。均值梯度不受p方差变化影响；centered u常量平移无效；mask/NaN/时钟/有效长度归一化正确；q/style/decoder/B0严格冻结，no motion recon梯度；p/head收到有限梯度；真实GPU精确恢复；16条smoke评估。检查完整source/binding/checkpoint父SHA后才能启动。

每臂运行前校验、日志、源SHA、checkpoint、状态独立；检查资源后最多三组并行。完整1367原生raw/clip、四probe、几何/动态/每类/每身份/固定干预、同八片段视频，原SHA备份。a_kl/a_mean比较是本轮消融；b_mean只是不同父模型候选。内部留出不选外部验证checkpoint，固定8轮final。

没有联合收益则如实拒绝，不再盲目延长。后续仍需内容读出、目标风格验证、真实多seed与匹配预算benchmark，不能把t-SNE、.7 F1或单个happy视频等同可发表结论。

## 当前状态

方案记录；实现及预检待完成，未启动Phase44正式训练。Phase43因素诊断已完成，原SHA备份通过。恢复先读本文件及CURRENT_OPTIMIZATION_STATE.md最高条目。
