# Phase36 表达学生梯度职责结果

固定三seed47/48/49、每臂2轮1568update、全部1367 validation、draw42/123/2026。全部新输出已SHA验证；无sealed/default推广/Git上传。

| 指标（clip_all三seed/三draw平均） | FullGradient772 | ExpressionGradient772 |
|---|---:|---:|
| MBE | 0.904603 | 0.911180 |
| LBE | 0.446154 | 0.452668 |
| 生成F1 原128 | 0.683872 | 0.637154 |
| 生成F1 原64 | 0.581147 | 0.545352 |
| Lip mean mm | 4.008668 | 4.055325 |
| Lip max mm | 7.598816 | 7.680107 |
| Expression mean mm | 0.769161 | 0.770594 |
| Expression max mm | 2.669011 | 2.679868 |
| 绝对mesh FDD mm² | 153.447688 | 154.460142 |
| 嘴部位移MSE | 0.003709 | 0.004778 |
| jaw centered相关 | 0.255015 | 0.244812 |
| jaw q90-q10 | 0.229451 | 0.228588 |

raw两原F1与MBE：
- full: F1 0.527434/0.469797，MBE 0.930567。
- expression: F1 0.493897/0.438090，MBE 0.938615。

三seed pilot门槛全部通过：False。各项gate和配对speakerCI见本地report/readout；只有3位validation speaker，不能宣称SOTA。

输入隔离/冻结/实际sample-noise stream已验证。该结果只比较情感学生的表达梯度职责；不能证明teacher/u_a纯情感，也不能将native Phase26旧成绩算作新772性能。

恢复入口：CURRENT_OPTIMIZATION_STATE.md；完整协议：PHASE36_EXPRESSION_GRADIENT_PLAN.md；阶段数据/结构：CURRENT_MODEL_AND_TRAINING.md。
