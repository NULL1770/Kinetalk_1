# Temporal adapter transfer12：后台交付

## 终点更新（2026-09-18）

已于00:35:04正常完成exit0，三个臂各12轮/1284updates，共1330.45秒（22分10秒）。下方启动/估时内容为历史记录，当前没有新训练运行。31个报告已下载，独立JSON来源/配对/哈希核验通过。

结论：rank8相对冻结没有实质时序收益，405开发眉/眼相关 .0501/.0735→.0484/.0723，608新增适配留出 .1818/.1644→.1795/.1616，centered误差改善不到1%。完整解冻对眼部有部分收益、开发眉部退步，整体未成功。音频均值DC在405有益却在608使raw明显变差，不能默认适用于所有片段。

当前报告位于 `artifacts/temporal_adapter_20260917/adapter_transfer12_v2/RESULTS.md`、`metrics_review.md/json`。完整曲线审计已通过：405部署分布/干预、608 source及三臂raw/DC重算一致，来源/分割/随机流/参数更新与紧凑重建核验通过；未重算教师、fit或raw oracle，不能据此声称感知质量通过。新增审计脚本两项紧凑归档回归测试通过。

固定九片曲线及三人六格视频已交付于同目录 `index.html`。每段96帧/25fps/seed42含音轨，输入/音频/视频SHA及原rig未修改验证通过；15个本地资源链接、18格非空及首末帧变化检查通过。已检查曲线与渲染单帧，未作连续自然度评价。视频为405片段DC组合，608上DC退化的结论不变。00:49核验GPU无计算进程，root剩684MiB；未追加训练、未换默认、未删旧产物。本轮交付完成，整体动态目标仍未达到。

## 启动记录

2026-09-18 00:12:51（Asia/Shanghai）已启动修正后的正式三臂实验，输出 `/root/kinetalk_full_staged_20260917/adapter_transfer12_v2`，supervisor PID11142，trainer PID11143。已核第一组完成首轮107更新、14.05秒，初值同前向逐位一致；没有正式效果结论。原默认及所有旧训练产物保留，原adapter_transfer12保留为首次启动前检查失败记录。

本轮问的是：限制音频特征适配容量，能否保留有用时序响应并减少额外错误变化。三组frozen_local、rank8_adapter、full_local使用同一context12/chunk_teacher源，各12轮。复用历史按句划分：1707片段参与新增更新，每轮107步、每臂1284步；608片段只验证新增适配迁移。源模型已见过全部2315，不能称全模型未见句泛化。405内部开发片段另行完整评价，封存test不读。

rank8只增加1024参数，在原64Dlocal输出上做两次有效帧中心化的非线性残差；均值保持针对local特征，不是保证动作均值正确。原全局情感、身份、state/口部基座及前缀续接机制保留，无新loss/标签/逐帧GT条件。原motion→audio全局学习结果保留，本轮不重新蒸馏。

启动前：83项本地/远端相关测试通过；增加同前向初值回归用例后38项受影响测试在本地/远端再次通过（相关测试总计84项）。真实三臂smoke_v3共75.43秒完整通过，覆盖三seed、local静态/反序、GT历史独立诊断、raw/DC、608接口缩小版、43与invalid保护、紧凑存储逐位重建。step0前32dev源重放max_abs0，adapter零初始化逐位一致，up首步/down第二步梯度非零；三组初始化/随机流/预算核验通过。

第一次smoke因将旧完整forward缓存与专用forward差异计入adapter均值检查失败，保留失败目录；修正为同次前向适配前后检查，不放宽2e-6容差。修正版smoke在/dev/shm临时存储以节约磁盘，小JSON已下载，正式checkpoint全部在持久root。开始前root可用1269.7MiB，采用upper9+已有base SHA的无损紧凑存储，逐位还原通过后才存盘；不删除旧实验。

首次正式尝试00:05:01启动、00:06:43在第一个optimizer更新之前被旧缓存一致性allclose断言拦下。修正为同批次原冻结local与初始各臂bitexact比较，另存旧缓存差值；初始608评价也改为与终点相同的专用前向和batch几何。没有改变目标、样本、种子或预算，未读取终点效果调参；旧失败目录不覆盖。smoke_v3复核后启动v2，启动前root1234.8MiB可用。

按正式首轮14.05秒和smoke评价耗时，三臂12轮+step0和终点608/405评价估计约25–35分钟，即00:40–00:50左右结束，实际以process/status为准。任务由独立后台supervisor运行，关闭本对话不会停止；不要关闭GPU实例，否则训练会中断。本入口保存last及optimizer/RNG，尚不承诺自动resume。

检查入口：

- `adapter_transfer12_v2_process.json`：训练PID、开始/结束时间、exit code。
- `adapter_transfer12_v2.log`：训练输出及异常。
- `adapter_transfer12_v2/status.json`：step0、训练或评价状态。
- `adapter_transfer12_v2/{frozen_local,rank8_adapter,full_local}/`：provenance、12轮日志、last/final、raw/DC完整评价、compact曲线、transfer_evaluation/curves、fit诊断、feature_drift、complete。
- `adapter_transfer12_v2/step0_transfer.json`：608固定句子留出的原始源性能，用于比较各终点增量。
- `adapter_transfer12_v2/sentence_split.json`：确切成员/索引/分割来源；`matched_audit.json`为三臂最终配对检查。

新compact dev曲线键为upper_predictions9，需使用 `scripts/train_temporal_adapter_transfer.py:restore_deployment` 和绑定的white/curves.pt重建full52，不能直接按旧predictions schema读取。transfer只有upper9，invalid为不评分的零占位，不作完整人脸/43保护声明。

完成后先核SHA、frozen、分割、随机流和重建，再比较608源→终点与三臂，405作参考。关注centered相关/MSE、眼速度、fairES/VS、静态/反序，以及固定样本。高训练相关或更大动作不判成功；不据结果追加epoch、挑seed、换rank或替换默认。
