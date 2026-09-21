# 2026-09-20 口型修复与受保护训练

## 最新终点结果（覆盖下文启动时状态）

远端实时核实：30/30 epochs正常完成，smoke/recovery均exit0，耗时1481.09秒（24.68分钟）。质量门控拒绝，GPU空闲；**identity/teacher/audio保护阶段均未启动，Stage5未启动，默认未替换、sealed test未读**。本地 `evaluation.json` 与 `independent_mouth_audit.json` 已取回。

完整446 validation、同观测支持、原始系数不截断：

| 口型指标 | 原B0 | 修复30轮 | 变化 |
|---|---:|---:|---:|
| 整体去均值相关 | 0.304501 | 0.445307 | +0.140805 |
| 整体原始MSE | 0.023918 | 0.014992 | −37.32% |
| 整体去均值MSE | 0.006210 | 0.005287 | −14.87% |
| 整体邻帧位移MSE | 0.001107 | 0.001041 | −5.91% |
| neutral原始MSE | 0.013876 | 0.014703 | **+5.96%，超3%上限** |
| nonneutral原始MSE | 0.025988 | 0.015051 | −42.08% |

唯一失败项为 `neutral/raw_mse_preserved`。neutral相关0.419837→0.481838，去均值误差−10.27%、邻帧位移误差−4.36%，所以不是neutral时间变化变差。独立NumPy逐值复算保存曲线，最大与报告差3.33e−16；raw=centered+加权均值项的分解误差≤2.43e−17。neutral均值误差从0.00918670增到0.01049528（+14.24%），超过其时序误差改善量，导致raw门槛失败。这确定了误差所在分量，但还没有证明是哪个输入/身份因素引起偏移。

真实content优于静态/反向content：逐clip centered MSE差分别−0.00088834、−0.00364976，85句簇95%CI分别[−0.00111484,−0.00066595]和[−0.00410084,−0.00321507]。三strata centered R²均正。这里验证的是**口型B0的音频时序**，不是眉眼动态。

第10/20/30轮validation整体centered MSE为0.005003/0.005060/0.005287；训练loss继续下降但此项不再改善。固定256训练诊断相关0.759819，validation0.445307，显示拟合与跨身份开发效果有明显差距。不能靠无限加轮数解决，亦不能事后把10轮改报为预设30轮成功。

后续应围绕neutral静态偏移和跨身份泛化做有对照的修复，保留本轮已有时间变化收益；不放宽3%保护、不把query真实均值用作推理校准，不在当前失败基座上继续眉眼长训。本轮状态查询没有启动新训练，也尚未渲染新视频或完成感知AV同步认证。

---

本轮接续 `MOUTH_REGRESSION_AUDIT_20260920.md`。实现和真实 CUDA smoke 已完成；正式口型修复已启动。当前仍不能声明口型或眉眼动态已经成功。

## 本轮改动

- 在 `NeutralAffectSystem` 配置中保存固定 `motion_support`，由训练集通道观测并集决定。未监督通道在 base、identity、motion teacher、flow target/noise/state/output 和每一步生成中清零。训练的 observation mask 与部署固定 support 分离；生成不读取 query GT channel mask。
- 增加固定 `residual_support`。修复实验排除嘴部 14:41，随机残差的嘴部恒为零，最终嘴部精确等于 B0 + identity。其他受支持通道仍可学习。配置必须随 checkpoint 保存/加载，这两个 support 不作为新增 state_dict key，以兼容历史模型。
- 新增口型保护工具，按 overall / neutral / nonneutral 检查相关下降 ≤0.01、邻帧位移 MSE 增加 ≤5%；neutral 原始 MSE 增加 ≤3%。缺组、无定义时序统计、观测 NaN/Inf 失败。显示或评分不截断、不重对齐、不平滑。
- identity 阶段也要相对已验收 B0 通过保护。后续 audio 相对 B0+identity 检查，不能以隔离嘴部掩盖身份偏置退化。

## 训练协议与边界

新脚本 `scripts/recover_paper_mouth.py` 从 full_v1 的、来源已知的 articulation checkpoint 继续训练，只更新 Stage1。原来为 715 neutral ×12 epochs；本轮每轮使用锁定论文训练集全部 **4,098 query / 454,566 有效帧**，固定 **30 epochs**，batch16，AdamW lr1e-4，原有 raw Huber + 0.1 displacement Huber 目标不变，线性输出头不变。

因此新 B0 的合同是 **全情感确定性音频口型**，不再把它称为情感中性口型。后续 identity residual 和 b0/h0 缓存会从新 B0 重算。mouth flow 被锁定也意味着嘴部全局表情必须由这条确定性路径与静态身份偏置承担；全局情感质量须重新检查，不能继续引用旧模型“已合格”的结论。

固定终点评估：完整 446 validation，分情感 strata；固定 metadata/RNG 选取 256 training clips 作训练拟合诊断。**256 是诊断子集，不是训练集大小**。同时保存 real/static/reverse 原生时钟预测；静态/反向操作施加在进入 B0 的 content 输入上。按句簇 2,000 次 bootstrap 的 centered MSE 差区间上界须小于0，三个 strata centered R² 须大于0，且相对旧 B0 口型保护全部通过，才允许启动后续训练。中途每10轮评估只记录、不择优挑 checkpoint。

这只是口型基座准入，不是完整感知验收，也不等于眉眼音频时序成功。现有弱 B0 validation overall 相关0.304501、centered R² −0.014854；neutral 相关0.419837，nonneutral0.284034。新基座必须接受对照，不能只以 loss 降低判好。

## 自动顺序与实际状态

`scripts/run_mouth_repair_queue.py` 执行：真实 CUDA smoke → 30轮全训练集 B0 recovery → 若门槛全通过，identity200 epochs → teacher12 epochs → audio12 epochs。identity200 epochs 是22组参考对每轮两批，不是200轮4,098条 query。后三阶段从新 B0 重新训练相应模块，保护嘴部；任何保护失败停止下游。

本次没有自动启动 upper9 Stage5 长训。先确认修复后的完整基座、身份和情感音频路径，才决定如何继续眉眼。所有结果只到 development，sealed test未读，默认模型未替换。

远端实例：`connect.nmb1.seetacloud.com:11473`。持久目录：

```
/root/autodl-tmp/kinetalk_repair_20260920/
  prepared/                       # 4,594 train/val query+enrollment分片
  code/                           # 独立源码快照
  run30/queue_status.json          # 实时队列状态
  run30/recovery/status.json       # epoch或质量门控状态
  run30/recovery/evaluation.json   # 完成后的口型/对照/验收
  run30/recovery/final.pt          # 来源明确、保存新配置的B0
  run30/recovery/protected/        # 仅B0通过后生成
```

队列 PID3870，正式 recovery PID4016；2026-09-20 14:56:56北京时间进入正式 recovery。已直接确认第1/30轮完成：4098 clips，52.56秒，loss0.207922。预计口型阶段约30分钟；若保护阶段全部启动，总计约45–60分钟，取决于终点评估与GPU负载。不是完成承诺。

实例重启清空了旧 `/dev/shm` prepared 缓存。已从同一锁定清单重新准备全部4594分片至持久数据盘，用时203.97秒；loader逐分片校验hash。重建后训练初始 B0 指标复现前轮审计。当前 prepared index SHA256 为 `27d6a4dce0d76020a48d3f8445083bc1178f760f68ed071f61f25c4aba39d674`，manifest 仍为 `6bdc33c3ee832e968b421781267dc70654e416b8e76395f2c6ce83c859e8603e`。

最初尝试上传本地2.69GB完整备份速度过慢，已停止仅该上传并删除本轮255MB不完整提取目录；完整备份、历史checkpoint和原始数据均保留。持久数据盘清理后约1.5GB剩余，队列产物预算已检查。

## 验证

- 最新一次本地联合回归：43 passed（support、mouth gate、recovery/resume、staged/resume、paper pipeline、renderer mask）。
- 远端模型/训练相关回归36 passed；新恢复流程回归3 passed（与本地重合，不加总）。
- 真实数据 CUDA smoke16 train/16 val 完整训练/评估/保存成功。smoke 被明确禁止宣称门控通过或启动保护阶段。
- 启动后的正式代码保持冻结；不在运行中覆盖。local artifacts/mouth_repair_20260920 的状态文件是下载快照，后续检查必须重新读取远端实时文件。
