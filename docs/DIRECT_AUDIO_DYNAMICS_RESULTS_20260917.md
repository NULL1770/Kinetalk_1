# 直接音频与中心化轨迹监督：已完成的结果

2026-09-17重新连接服务器并独立核验。两组既有实验在00:39前已完成，各8epoch/1160步；本轮未重复训练它们。固定2315 fit/405内部dev，8个noise、12步生成；全部48组最终曲线从原checkpoint逐值重建一致，训练batch/noise/flow-time逐轮hash重放一致。起点等于历史zero-local，B0/identity/global保护hash未变。未读取封存test或选择最好轮。

## 结论

**rich audio + 完整renderer + 最终中心化L1仍没有通过。** flow-only产生过大错误动态；L1显著压低时序误差和高频扰动，但眉幅度收缩、分布评分变差，正确local仍不及自身zero，口型相对冻结来源也退化。不能把本组低R²误差或更平滑动画当动态成功。

| 非中性眉，八seed汇总 | direct flow | direct + L1 |
|---|---:|---:|
| 中心化残差R² | -1.925779 | -0.046649 |
| 中心化相关 | .084234 | .146806 |
| 动态能量/GT能量 | 2.173923 | .166418 |
| RMS幅度/GT幅度（由能量比开方，近似） | 1.474 | .408 |
| fair trajectory ES，越低越好 | .020881 | .022625 |
| 邻帧fair variogram，越低越好 | .012587 | .001514 |

能量比不是幅度比。ES与R²分歧说明单GT误差不能作为概率生成唯一目标，VS也只覆盖选定相邻差分统计。

direct + L1对自身zero：眉ΔR²=-.051029，句bootstrap95%CI[-.081299,-.025371]；fair ES改善=-.000687，CI[-.001014,-.000379]。对自身reverse：眉ΔR²=+.066280，CI[.022395,.111362]。因此存在时间条件响应，但尚无优于关闭local的净效果。

对冻结来源：眉ΔR²=+.232620，CI[.188335,.288227]，但fair ES从.019025恶化到.022625；非neutral mouth原动作MSE增加3.204%（单侧90%上界5.379%），neutral mouth增加67.594%（上界72.185%）。冻结B0并未保护最终嘴部。

## 可复核证据

- 远端完整审计：`/root/kinetalk_runs/teacher_schedule_v1/direct_audio_dynamics_v1/paired_audit.json`。
- 本地精简审计：`artifacts/brow_review_20260917/direct_compact_results.json`；粗全模式汇总`quick_metrics.json`。
- 8seed、所有模式、逐人、raw/mean/dynamic与ES/VS见完整审计；本表不代表独立测试泛化。
- 固定三人首条连续六格：`artifacts/brow_review_20260917/visual/`；九例由metadata锁定，未挑seed/调gain/对齐曲线。

审计器两处问题在本轮修复：独立epoch000文件路径，以及多条件名称与旧distribution scorer要求full字段不符。仅修改评价链，训练源码/权重不变；4项来源测试、2项命名适配等价测试通过。全量340测试通过（含本轮新输出损失训练器），不把单测当效果证明。

## 本轮追加的有界验证

新协议`OUTPUT_MOTION_DYNAMICS_PROTOCOL.md`：同epoch0恢复，flow保留，逐帧L1替换为最终motion相邻位移MSE+眉眼时间std MSE；audio/zero两臂相同8epoch。加入matched zero训练才可区分新增local与训练目标的贡献。来自论文启发，不是SubtleTalk的完整复现或已知最佳超参。
