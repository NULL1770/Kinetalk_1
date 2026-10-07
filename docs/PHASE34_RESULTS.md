# Phase34 中性772D坐标优化结果

固定三seed47/48/49、每臂2轮1568update、全部1367 validation、draw42/123/2026。全部新输出已SHA验证；无sealed/default推广/Git上传。

| 指标（clip_all三seed/三draw平均） | Centered772 | Standardized772 |
|---|---:|---:|
| MBE | 0.910817 | 0.904603 |
| LBE | 0.440420 | 0.446154 |
| 生成F1 原128 | 0.203660 | 0.683872 |
| 生成F1 原64 | 0.251196 | 0.581147 |
| Lip mean mm | 3.926834 | 4.008668 |
| Lip max mm | 7.390061 | 7.598816 |
| Expression mean mm | 0.808946 | 0.769161 |
| Expression max mm | 2.828810 | 2.669011 |
| 绝对mesh FDD mm² | 144.884468 | 153.447688 |
| 嘴部位移MSE | 0.003854 | 0.003709 |
| jaw centered相关 | 0.268314 | 0.255015 |
| jaw q90-q10 | 0.212523 | 0.229451 |

raw两原F1与MBE：
- control: F1 0.134476/0.120974，MBE 0.935267。
- standardized: F1 0.527434/0.469797，MBE 0.930567。

三seed pilot门槛全部通过：False。各项gate和配对speakerCI见本地report/readout；只有3位validation speaker，不能宣称SOTA。

输入隔离/冻结/实际sample-noise stream已验证。该结果只比较中性772D的坐标与误差单位；不能证明teacher/u_a纯情感，也不能将native Phase26旧成绩算作新772性能。

恢复入口：CURRENT_OPTIMIZATION_STATE.md；完整协议：PHASE34_NEUTRAL772_COORDINATES.md；阶段数据/结构：CURRENT_MODEL_AND_TRAINING.md。

固定八类视频已完成：GT／Neutral B0／Centered772／Standardized772，seed47/draw42/M025/005，所有系数/audio/rig/frame/videoSHA通过。t-SNE/原probe混淆矩阵/同embedding身份着色已完成；full source SHA及两生成view特征逐位闭合。Windows重算GT特征最大差7.75e−7，GT逐位guard失败如实记录，1367条预测零差、F1仍严格.660344；原导出GT未改，未放宽任何模型/传输零误差guard。

逐类原128/64F1：neutral .358/.348、angry .610/.517、contempt .786/.720、disgust .591/.453、fear .741/.591、happy .901/.899、sad .744/.431、surprise .740/.690。Happy不是主要失败类；原64的disgust/sad多被判angry，neutral召回约28%。t-SNE聚类改善不能认证生成逼真/内容解耦。

Phase35独立physical-flow三组正式训练已启动9830/9831–9833，closure9834；不扩大N1预算/不推广。实质ua表达监督仍未改，需要后续D2目标/梯度合同。
