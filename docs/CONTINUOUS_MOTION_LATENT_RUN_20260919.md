# 连续动作latent自动训练交接

更新：正式流程已完整结束，退出码0，耗时417.123秒；结果及限制见`CONTINUOUS_MOTION_LATENT_RESULTS_20260919.md`。不要按下面历史运行状态判断仍在训练。AE/运动先验有进展，音频分支留句泛化未通过，无新训练挂起。

2026-09-19 02:30（Asia/Shanghai）。正式训练已在11473实例启动，supervisor PID1940、训练PID1941。AE4000更新已结束，固定16条train重构门控通过，中心化pooled R²=0.95062；已自动进入prior并核到4500/8000，GPU进程和checkpoint持续更新。重构成绩不是自由生成或音频时机成功。

## 运行与预算

- 代码：`/root/kinetalk_joint_20260918/code`
- 数据：`/root/kinetalk_joint_20260918/continuous_latent_dataset.pt`
- 输出：`/root/kinetalk_joint_20260918/continuous_latent_formal24000`
- 顺序：AE4000 → global/identity运动先验8000 → matched_global6000 → audio6000；batch24，总24000更新。
- 后台supervisor管理真实退出码，不依赖本次SSH会话。AE/prior工程门控失败则停止后续阶段；完整执行不等于实验成功。断电不会自行重启，重新开机后按下文显式恢复。
- 4090真实数据短测每更新约0.012–0.015秒；纯训练约5–6分钟，完整阶段评价和导出后估计10–20分钟。不是服务时限保证。
- 自动生成固定seed42/123/2026/77的曲线、52维NPZ、SVG/HTML、render_jobs；MP4及自然度审核稍后处理，不声称已自动渲染。

## 数据及保护

819训练+64内部留句诊断，共883段；新增627段冻结基座输出完成。训练联合观测0.972911小时（约58.4分钟），所有音频有效帧1.005888小时。global65、identity128、context202，九个生成通道为`[41,42,43,44,45,5,6,12,13]`。

严格排除预定8个留句的新增训练片，原64只覆盖其中部分。历史上游已曝光，不能称独立最终测试；本轮未读取405开发目标或封存测试。训练统计只来自train；生成不使用query动作/动作mask/目标均值。其余43通道及音频无效帧复制冻结基座。upper变化可能影响感知身份和情感，整体质量仍须视频验证。全局/身份教师成果保留，但冻结期间没有在线蒸馏。

数据SHA256：`6f876fd7596eba204185d4a646c0886a15c7d1984a8550c19d4f6e20ddceeb32c`。合并2649个文件全部hardlink，未删除历史资产；正式启动前根盘剩约3.7GiB。

## 已验证

远端81项针对性测试通过，包括tail/gap、fit统计、生成无目标泄漏、43保护、AE/flow中断恢复逐值一致、错配只换音频、配对条件与四阶段完整CPU小流程。真实数据CUDA smoke每阶段100更新、共400，完整执行；两适配支起点、batch序列、更新数和noise/time配对校验通过。

短测AE容量门控通过（仅4条train小诊断）；短测先验幅度门控失败。smoke显式绕过质量门控只验证执行能力，不能作为自由生成质量证据。正式运行未绕过任何门控，使用固定16条train检查，64holdout独立评价。

## 查状态与恢复

读取输出下`status.json`和`pipeline_status.json`。训练阶段每250步更新`*_last.pt`；各阶段终点`*_final.pt`。`training.log`保存异常，`paired_training.json`保存两支配对校验。`*_gate.json`记录继续依据；末期适配gate只记录并完成对照。最终状态`complete`仅表示执行完成；`stopped`表示门控停止；`failed`表示程序错误。

恢复前核验没有该输出的活动进程；launcher也会检查。必须保留当前源码及数据，不能改预算后冒充原实验恢复：

```bash
cd /root/kinetalk_joint_20260918/code
/root/miniconda3/bin/python3.12 -m scripts.launch_continuous_motion_latent \
  --dataset /root/kinetalk_joint_20260918/continuous_latent_dataset.pt \
  --output /root/kinetalk_joint_20260918/continuous_latent_formal24000 \
  --code-root /root/kinetalk_joint_20260918/code --device cuda \
  --ae-steps 4000 --prior-steps 8000 --adapt-steps 6000 --batch-size 24 --resume
```

正式协议中的五个源码SHA256已与本地逐项一致。固定终点评价包含AE、prior、matched_global、audio以及audio同权重static/reverse/mismatch；不按holdout挑epoch或seed。完整自然有效段保留原长度；专门长合成压力测试、插值对照和全脸视频仍待后续审核。
