# 已启动：五阶段各12 epoch

## 训练后复核更新

本轮已于2026-09-17 13:59完成，训练exit0，647.946秒，5592步，五阶段均12epoch。原始自动导出exit1（跨阶段浮点输出逐位比较过严）；随后核验system/audio权重逐位一致、同次43非上脸通道逐位保持，记录跨阶段max差3.257e-4、RMS1.705e-5后修复导出，未改变训练权重或曲线。远端 `/root/kinetalk_full_staged_20260917/visual/` 已重新成功导出。

本地[结果页](../artifacts/full_staged_review_20260917/index.html)含三人固定六格视频和九片曲线；[指标审查](../artifacts/full_staged_review_20260917/metrics_audit.md)详述结果。当前动态未成功：眉毛动幅1.65×GT、相关.058，动态MSE相对Stage4+89%；真实慢状态oracle相关.511。全局输入情感分类96.8%；身份参考基线改善，但只3名dev。口部raw误差描述性相对旧full-local+22.8%，不能说口型已通过，亦非lip-sync退步百分比。默认模型没有替换，不建议直接追加epoch。

以下保留启动时交接记录，状态请以上述更新为准。

启动：2026-09-17 13:48:09 Asia/Shanghai。实例 connect.nmb1.seetacloud.com:24539，4090。实验根目录 `/root/kinetalk_full_staged_20260917/`，训练结果 `run12/`，隔离源码 `code/`。

交付前核验：B0 articulation、identity、teacher均完成12轮，audio已完成第2轮并进入第3轮，训练状态running；完整2315条teacher每epoch约12.6–12.9秒，audio约15.2秒，B0 422条每epoch2.7秒，身份49参考对每epoch0.11秒。含后续动态、各阶段多seed评价与导出，总体估计10–20分钟，约13:58–14:08完成（非完成承诺）。以status/summary实际状态为准。

主训练PID6017，后台wrapper6016。退出SSH/对话不影响nohup训练；关闭实例会停止，需要恢复。无需定时通知，用户训练后返回查看。

## 状态与日志

```bash
cat /root/kinetalk_full_staged_20260917/run12/status.json
tail -n 25 /root/kinetalk_full_staged_20260917/run12.log
cat /root/kinetalk_full_staged_20260917/run12_process.json
```

`run12/summary.json` 且status=complete表示训练完成；wrapper进程文件还记录导出退出码。异常保存 `failure.json`（若异常发生在初始化前，查看进程退出码与log）。

每阶段：`articulation/identity/teacher/audio/dynamics` 下有12个epoch记录、`final.pt`、`evaluation.json`、`identity.json`、`curves.pt`、`complete.json`。主目录 `last.pt` 含optimizer/RNG，可按完成epoch恢复。固定405内部dev，3个noise seeds；dynamic另外seed42的base/static/oracle/reverse对照。训练完成后wrapper自动导出原固定九片曲线、三人待渲染NPZ与review.html；不冒称已经生成视频。

恢复前先确认原PID已退出，防重复训练：

```bash
cd /root/kinetalk_full_staged_20260917/code
nohup env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /root/miniconda3/bin/python -u scripts/run_full_staged_background.py --run /root/kinetalk_full_staged_20260917/run12 --resume > /root/kinetalk_full_staged_20260917/run12_resume.log 2>&1 < /dev/null &
```

启动前完成全部5阶段真实smoke与恢复路径；模块/数据/runner本地32项测试通过，远端相同32项为启动前置。追加2项optimizer/RNG逐位恢复和冻结边界测试、4项固定导出测试在本地和远端均通过。导出器已上传供后台训练结束自动调用；最后核验root仍有5.9GB可用。旧模型/默认权重未覆盖。

## 回来后如何判断效果

按 [完整协议](FULL_STAGED_SPLINE_PROTOCOL_20260917.md) 检查：身份参考检索是系数风格身份而非mesh身份；teacher阶段是GT motion oracle，audio/dynamics才是audio-only；内部teacher的生成情感分类不能替代独立感知评价。第五阶段嘴部应与第四阶段逐位一致，但第四阶段是否优于旧模型仍需比较。

当前新方案为独立neutral基线＋四维有符号连续慢状态＋不能抵消该状态的随机上脸残差，保留逐帧音频；无文本、VA、窗口std输入、历史context。与SubtleTalk有明确结构区别，效果和新颖性尚待实验证据。12epoch是用户选择的短期完整续训，不保证充分收敛。
