# 完整原生序列30轮后台训练交付

本轮检验完整动作覆盖与下游连续音频上下文能否改善眉眼动态。沿用原unknown-only flow matching，未新增loss；从同一context12模型开始，对照旧center96与full_native各固定30epoch。两组都更新upper及专用local的input/blocks/local_head，原身份、全局audio/state与口部基座冻结。正式效果尚待训练结束评审，未替换默认模型。

## 已完成验证

- 一次GT起始历史诊断：修正原点，但后段动态误差没有一致改善；因此没有继续添加初态预测头。
- 2720个旧allowlist片段的完整音频delta全部完成，共59990个额外有效帧。来源wave/model/native/中心特征重叠校验通过；不读封存test、不重拟合归一化。原生content实际FP16，按既有数据契约提升FP32计算并精确校验。
- 本地112项相关测试通过；服务器本轮100项runtime/data/evaluator/trainer测试通过。初始化诊断测试另在此前执行。
- 真实小跑两组各2epoch、4更新；center第1轮主动停止，恢复optimizer和RNG后完成后续训练与两组终点评价。恢复调用40.35秒，随机流/初始权重/更新配对通过，所有产物hash通过，32dev所有生成模式保持43其他通道与invalid基座逐位一致。
- 用户授权清理的83个旧smoke/更新前失败张量释放1760MiB，正式模型、数据和JSON报告保留。正式启动前根盘2275MiB可用。清理清单见本地artifacts/native_context_20260918/redundant_cleanup.json。

## 后台任务

2026-09-18 02:30:31（Asia/Shanghai）启动独立supervisor，PID12807。训练输出`/root/kinetalk_full_staged_20260917/native_context30`；日志为同级`native_context30.log`，进程状态为`native_context30_process.json`，启动记录为`native_context30_launch.json`。

正式训练每组1707clip、107更新/epoch、3210总更新。608为新增训练的留句集合（旧源已经见过），405为反复使用内部dev，不宣称未见源泛化。固定3seed终点评价与local-static/reverse干预；主raw、DC另列。完整输入改变了motion覆盖、下游上下文和静态统计范围，等更新次数不等算力。

关闭聊天不影响独立后台进程；关闭实例会中断。每个完整epoch在持久盘原子保存last.pt（模型、optimizer、RNG），恢复需显式`--resume`并校验相同代码/输入，不自动重启失败任务。

02:33核实center96已完成第1epoch/107更新，实测16.52秒；57MiB持久last.pt已产生，trainer PID12808存活。完整臂每batch含更多chunk，预计更慢；连同固定基线与608/405终点评价，整轮预计约30–45分钟，即北京时间03:00–03:15左右完成。这是首轮实测与序列长度推算，不是已完成时间；无需等待本对话结束。

02:33:53复核已保存第4epoch/428更新，四轮耗时16.52/15.67/14.81/15.16秒。持久checkpoint成功读取、optimizer/RNG齐全、protected local heads与源逐位相同；根盘余2202MiB。审计快照`artifacts/native_context_20260918/startup_verified.json`。

完整训练数据覆盖为1707更新集145808→184378有效帧（+26.45%），608留句52657→62881、405dev35952→47148。原生最大长度分别321/232/239。额外提取审计显示2720个clip的中心middle及prosody最大差和RMS均为0。

## 后续核验

以固定epoch30为终点，先核complete/process退出码、冻结参数、两组配对、所有文件SHA和输出保护，再比较608/405的相关、centered MSE、幅度、速度、真实native分块接缝和分布评分，最后生成固定样本视频。增大幅度或训练loss下降不足以证明音频时序学会。43通道保护亦不替代身份、口型、情感与自然度的独立感知评估。
