# Audio prefix adaptation 12：后台交付

## 终点更新（2026-09-17）

已下载的 process/status/两臂 complete 记录确认：21:57:13 正常结束、exit 0，总计 814.67 秒（约13分35秒），两臂各12轮/1740更新。以下启动记录保留作历史，不代表仍在训练。

405个内部开发片段、三个固定推理seed的配对结果：冻结local到联合适配local，眉/眼相关 .0437/.0742 → .0796/.0818；centered MSE下降7.20%/1.46%，但眼部逐帧位移MSE上升3.42%，centered energy score变差1.06%/3.37%。改进是局部且混合的，尚未超过旧state_white的主要动态指标。离线DC只修均值，不提高时序相关；口部报告数值保持一致，不等于新完成独立口型/身份质量验证。

本地报告和三人对比视频位于 `artifacts/audio_prefix_adaptation_20260917/audio_prefix12/index.html`，数字复核见同目录 `metrics_review.md` 和 `RESULTS.md`。本轮73项本地/远端测试及真实smoke已在启动前通过；本次主审计和补充审计已从权重/曲线核验来源、配对记录、保护通道、部署分布、边界和fit指标，全部通过。原system/audio冻结记录跨链核对，但未重新加载这两者的源tensor；teacher情感读出未独立重算。

22:26左右24539端口曾拒绝连接；用户重新开启实例后已恢复，并成功执行 `scripts/audit_audio_prefix_results.py` 与 `scripts/audit_audio_prefix_supplement.py`。固定九片曲线和三人六格视频已生成；输入/音频/输出SHA、96帧/25fps/音轨、原rig未改均验证通过，页面10个本地链接有效。查看曲线和单帧，并做非空/跨帧像素变化检查；未作独立连续自然度评审。恢复后的远端GPU计算进程检查为空，没有追加训练或替换默认。

## 启动历史

2026-09-17 21:43:37（Asia/Shanghai）已正式后台启动。supervisor PID26962，训练PID26963；首组第一轮19.51秒/145updates完成。预计两组各12轮及完整评价约10–15分钟，即21:55–22:00左右；最终以process/status为准。退出当前对话不会停止服务器独立进程。

后续启动核验：第一组已完成12轮并进入评价，root剩余2.0GB；context12 complete/provenance recipe SHA额外只读核验相等。第二组和正式效果仍待完成，不能把此状态当成功结果。

不是重复全系统五阶段训练。身份、全局情感、state及口部基座保留原完整训练成果；本轮为context12 teacher接收器的两组配对继续训练，各增加12轮：frozen_local只更新upper，adapt_local同时更新眉眼专用音频input/blocks/local_head。两个初值、数据顺序、噪声、flowtime一致；只有local是否开放梯度不同，不重新训练global、不加在线motion→audio蒸馏。

另存明确离线合成：audio静态均值＋去自身均值的预测动态。不是低通/增幅/GT校正，不改变时序；不回灌前缀或用作在线因果初始化。raw与DC分别评分，GT历史仅raw独立诊断，主视频不混oracle。

先完成的无训练origin诊断：405dev×3seed，眉raw .0225456→.0186595（约−17.2%），眼 .009872→.00759054（约−23.1%）；所有动态/位移指标不变。此结果证明静态偏差可部分修正，不证明音频时序已学好。新训练结果交付时尚未知。

## 已核验

- 本地及远端73相关测试通过；DC evaluator额外独立合成验证通过。
- 真实32fit/32dev双臂smoke，共50.52秒，覆盖训练/全部raw与DC评价/三seed/8fit诊断；first-step adapt三处均有非零梯度、frozen无梯度、43+invalid逐位保护、随机流配对通过。
- 源context12前32dev seed42完整重放max_abs=0.0。
- 启动前root2.0GB剩余；新实验有空间门槛，无旧文件清理。

## 继续检查

远端根目录 `/root/kinetalk_full_staged_20260917/`：

- `audio_prefix12_process.json`：子进程、起止时间与exit code。
- `audio_prefix12.log`：训练和异常输出。
- `audio_prefix12/status.json`：当前臂/epoch或完成状态。
- `audio_prefix12/{frozen_local,adapt_local}/`：provenance、epoch001–012、last/final、raw/DC曲线与评价、fit诊断和complete。
- `audio_prefix12/matched_audit.json`：最终配对核验。
- `origin_diagnostic/report.json`：旧输出静态误差分解。

本地启动快照和origin报告：`artifacts/audio_prefix_adaptation_20260917/`。协议 `docs/AUDIO_PREFIX_ADAPTATION_PROTOCOL_20260917.md`；脚本 `scripts/train_audio_prefix_adaptation.py`、`scripts/evaluate_audio_prefix_adaptation.py`。

结束后先检查两个complete、source/curve/hash、43保护、初值/RNG、local更新范围，再比较三seed raw/DC的相关、幅度、速度、接缝、raw/centeredES/VS与旧state_white。单看训练loss不能判成功。本轮fixed epoch12，不自行追训/挑seed/改默认。未读取封存test；405dev已经反复用于开发，不能宣称最终测试泛化。

若训练失败，保留原目录；last记录optimizer/RNG供恢复核查，但本入口尚无自动resume。评估结果返回后沿历史固定9clip/3人视频，不为改善观感挑样本。
