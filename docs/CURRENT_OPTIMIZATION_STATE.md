# 当前优化状态：已回退 v5 matched control，2026-10-12

用户要求“先回退上一个版本”。本地与服务器当前模型选择已切换为 **v5 matched control**，checkpoint SHA256 `372dbc2a23913743cb67452bf6f5756e3be195f81c468f2a1f972afbb79c9add`。当前无新训练。模型的联合生成路径与恢复入口详见 [当前架构](CURRENT_MODEL_AND_TRAINING.md)，实际回执见 [rollback_receipt.json](model_rollback_20261012/rollback_receipt.json)。

## 本次完成

- 本地/远端 checkpoint 与 binding SHA、一致的模型源码已核验；严格加载为 `KineTalkV4Model`，joint output 权重形状 52×192。
- 当前入口为本地 `configs/current_model.json` + `scripts/current_model.py`，远端 `/root/kinetalk_current_selection/current_model.json` + `current_model.py`。未覆盖旧实验的 binding、checkpoint 或显式 runner 参数。
- 两个真实固定开发片段 angry/happy 跑通单参考纯音频推理，无查询 GT 输入、无教师调用，参数保持不变。未重跑已有完整评估。
- 确认 v6 queue status=stopped 且没有活跃 v6 worker。保留 direct final 与 posterior resume384、全部结果；不重启原队列。

## 回退依据

| sample / clip_all，1367 条开发集 | Phase53 | v5 control（恢复版） | v6 direct |
|---|---:|---:|---:|
| mean lip distance mm↓ | 3.262565 | 3.048290 | 3.305541 |
| 最大顶点 LVE mm↓ | 6.132446 | 5.853746 | 6.221172 |
| 上半脸 EVE mm↓ | 2.237557 | 2.435403 | 2.615629 |
| closure proxy F1↑ | .401017 | .553729 | .382824 |
| 生成动作情感 probe F1↑ | .676805 | .758772 | .403719 |
| angry probe F1↑ | .648276 | .862974 | .503937 |

当前 v6 direct 在口型、情感与动态诊断上退步，不继续推广。它更换了多处设计且训练预算与旧解码器预训练不匹配，不能据此断言所有分支式模型必然失败。恢复 v5 的联合残差生成也不等于三个创新已被证明。

## 停止的探索与可恢复记录

v6 direct seed47 完成 768 updates 与五项评估，23 个原始文件 SHA 验证、六张 CSV、angry/happy 视频及固定时点检查完成。posterior seed47 中断于可恢复 checkpoint384/768；seed91 两臂未开始。[停止回执](v6_factor_paths_20261012/manual_interruption.json) 与 [复盘](v6_factor_paths_20261012/three_contribution_reconsideration.md) 保留，不将旧 run/state.json 的 training400 视为活跃训练。

用户三个贡献方向保持为：残差改造；教师—学生的情感及动态学习；口型/情感/风格联合生成与解耦。所恢复 control 的教师蒸馏配置为 none，动态/风格及教师优势仍需针对性验证；HDTF 混合训练的其他结果不能并入本 control 的指标。当前只回退，未实施或启动下一版改造。

旧进度中的 active process 声明均为历史，已原样归档至 [回退前优化记录](archive/CURRENT_OPTIMIZATION_STATE_before_v5_rollback_20261012.md)。当前状态以本文与 selection/rollback receipt 为准。
