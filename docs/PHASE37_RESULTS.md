# Phase37 明确表达／韵律学生结果

固定三seed47/48/49、每臂2轮1568update、全部1367 validation、draw42/123/2026。全部新输出已SHA验证；无sealed/default推广/Git上传。

| 指标（clip_all三seed/三draw平均） | Latent772 | ExpressionProsody772 |
|---|---:|---:|
| MBE | 0.904603 | 0.906694 |
| LBE | 0.446154 | 0.451459 |
| 生成F1 原128 | 0.683872 | 0.617731 |
| 生成F1 原64 | 0.581147 | 0.549179 |
| Lip mean mm | 4.008668 | 4.021237 |
| Lip max mm | 7.598816 | 7.640006 |
| Expression mean mm | 0.769161 | 0.762216 |
| Expression max mm | 2.669011 | 2.635881 |
| 绝对mesh FDD mm² | 153.447688 | 162.015833 |
| 嘴部位移MSE | 0.003709 | 0.006340 |
| jaw centered相关 | 0.255015 | 0.226260 |
| jaw q90-q10 | 0.229451 | 0.229957 |

raw两原F1与MBE：
- full: F1 0.527434/0.469797，MBE 0.930567。
- named: F1 0.487035/0.431355，MBE 0.936852。

三seed pilot门槛全部通过：False。各项gate和配对speakerCI见本地report/readout；只有3位validation speaker，不能宣称SOTA。

输入隔离/冻结/实际sample-noise stream已验证。该结果比较命名表达／实测韵律及TRAIN情感强度原型学生职责；不能证明teacher/u_a纯情感，也不能将native Phase26旧成绩算作新772性能。

恢复入口：CURRENT_OPTIMIZATION_STATE.md；完整协议：PHASE37_NAMED_EXPRESSION_PROSODY_PLAN.md；阶段数据/结构：CURRENT_MODEL_AND_TRAINING.md。
