# 连续先验保持与有界音频适配

2026-09-19。用户授权继续优化及测试。上一轮完整训练后AE重构成立、global先验运动量级改善，但额外逐帧音频适配在64内部留句上退化、多样性收缩。本轮检验冻结先验上的小幅音频修正是否改善这一问题，不预先保证成功。

## 固定方案

- 保持旧运行源码不变；新增`prior_audio_adapter.py`、`train_prior_audio_adapter.py`、`audit_prior_audio_adapter.py`和独立launcher。
- 新adapter包裹不可更新的全局Flow，`v=v_prior + 0.35*tanh(delta)`，3层时序块，输入有序逐帧音频、noisy latent、全局条件和时间；输出层零初始化。仅adapter更新，AE和prior冻结。
- 每坐标latent速度限幅不等于输出轨迹限幅，不保证多样性或自然度；仍以四seed自由生成验收。关闭adapter与原先验逐值相同。
- 仅flow matching一个训练目标，无额外VA、幅度或逐样本目标轨迹回归损失。本轮先检验结构约束，不一次叠加多个正则项。
- 独立matched_static adapter从同一初始状态训练，相同batch/noise/time/更新数；其音频在完整native音频有效run内求均值，然后切监督segment。保留全局条件/身份/b9。

## 内层隔离与固定预算

从历史819训练片的26句按固定hash取5句为inner_validation，其余21句训练。旧64保持outer诊断角色。重算统计并从头训练新AE/prior，避免旧AE/prior已经看过inner验证动作；冻结基座及音频特征上游历史曝光仍披露，不能称完全独立数据。整份数据会载入内存，但outer目标不用于训练、选点或调参。

固定AE4000、prior14000、audio3000、matched_static3000，共24000更新，batch24。先验训练预算14000对应上一轮较稳的8000+6000全局先验。adapter仅评1000/3000两个预定节点，取最早通过内层规则者，不在多个任意epoch中找最好。若都失败，不推广；仍对固定最终点做一次outer诊断并清楚标记失败。

内层接受条件（启动前锁定，不按结果放松）：音频centered ES胜prior且句子cluster95%CI上界<0；raw ES不高于prior1.02倍；居中seed间half-pair距离至少保留prior的80%；速度/参考及四组动态幅度/参考都在[0.5,1.5]；平均centered ES胜同模型static和matched_static。原始越界单列、43保护精确核验。只有5句，区间用于内部研发决策，不当作论文显著性或自然度认证。

通过只允许进一步测试，不自动替换基座。outer只在decision落盘后评分；static/reverse/mismatch用于解释，不用其挑种子、权重或阈值。原64多轮暴露，仍不是封存测试。本轮不访问405目标或封存test。

## 验证与恢复

初步新增73项测试通过：adapter冻结/梯度/尾mask/顺序/随机性；内层统计独立、static全音频run定义；端到端两节点+恢复；真实evaluator档案接受；恒定/种子塌缩/CI跨零拒绝；后台真实退出码/PID归属检查。末审修正恢复到recovery目录时audit路径，以及窄窗口缺失artifact_directory的恢复。

每250步保存last，各固定节点保存可复核checkpoint。source/data/stat/AE/prior/protocol绑定哈希；原先验state逐值哈希验未变，matched臂初始化/采样链/步数核对。后台进程独立于SSH会话；断电后显式resume。训练终止、质量门控停止、全部执行结束分别记录，不把complete写作实验成功。

服务器73项测试通过；真实GPU每阶段100步的两节点完整短流程跑完，配对校验通过，短流程不作效果结论。正式输出：`/root/kinetalk_joint_20260918/bounded_audio_formal24000`，supervisor PID2083、训练PID2084。613片训练/206片内层验证/64片外层诊断；新AE4000完成、train动态重构R²=0.975247并通过继续检查。

正式24000更新和全部评估已完整结束，退出码0，总耗时977.396秒，约16.3分钟。1000/3000两个内层节点均未通过，不推广；按预定规则只对3000做外层诊断。四段全长视频、分数独立复算、下载校验完成。详细结果见[BOUNDED_AUDIO_ADAPTER_RESULTS_20260919.md](BOUNDED_AUDIO_ADAPTER_RESULTS_20260919.md)。当前无本轮训练进程，默认权重未替换。

恢复命令（必须确认没有该输出活动进程；保持代码及预算不变）：

```bash
cd /root/kinetalk_joint_20260918/code
/root/miniconda3/bin/python3.12 -m scripts.launch_prior_audio_adapter \
  --dataset /root/kinetalk_joint_20260918/continuous_latent_dataset.pt \
  --output /root/kinetalk_joint_20260918/bounded_audio_formal24000 \
  --code-root /root/kinetalk_joint_20260918/code --device cuda \
  --ae-steps 4000 --prior-steps 14000 --adapter-steps 3000 \
  --milestones 1000 3000 --batch-size 24 --valid-sentences 5 --max-delta .35 --resume
```
