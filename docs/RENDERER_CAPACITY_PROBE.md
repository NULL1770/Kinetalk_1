# 冻结学生、扩大生成器训练：预先协议

2026-09-16。用户授权“继续优化”，随后明确“先把动态做出来，别创新”。本轮是常规效果基线，不开发创新模块、不作论文成功主张。

## 固定对象与唯一主干预

来源 `/root/kinetalk_runs/teacher_schedule_v1/scaled_centered_v1/uniform/final_epoch008.pt` 及其原始权重/缓存/基。固定2315段19人fit、405段3人内部dev，均四情绪。留身份dev与fit共享56句且已多次使用，不称独立测试。B0/global历史暴露不重新声称消除。外部280/439和test512/15不读取目标。

保持B0、neutral identity、global audio encoder、motion teacher、rank8基/特征尺度/目标尺度/部署ridge head全部冻结。开放**原renderer全部参数与local_projection**，不重初始化，所有模块eval以禁用dropout。参数增加是优化容量对照，不是创新。flow直接使用原cached h0/B0，不重算量化content。

三个匹配arm：oracle_local用真实训练motion投影；audio_local用预先固定的训练句cross-fit预测；zero_local置零local、保留content/global/identity。三臂同初值、seed46、batch16、每轮同顺序/noise/flow time/未使用choice draw、Adam lr1e-5、clip_grad_norm1、8epoch/1160步。只用原observed flow velocity MSE，uniform t并20%t=0。不给任何臂额外label/GT global。oracle是诊断上限，不是部署结果。

与旧out_proj的直接差值同时包含容量与训练学生条件改变，不把这次训练说成纯单变量复现旧试验。三新臂相互匹配，以audio对matched zero的对照判断条件净收益。

## 训练条件与交叉拟合边界

使用既有三句折。**固定整个19人fit已拟合的U、feature_std、target_scale**，每折仅在其fit句上重拟alpha1线性系数，并对未参与系数拟合的句子预测，合并为2315条训练controls。目标为原motion投影，预测回相同8维坐标并按原target_scale归一化，不旋转基、不用405调整。

这是共享基下student系数cross-fit，不是整条监督路径严格OOF：U/尺度已见全fit且alpha以前选过。目的是让renderer在训练时接触预测误差，不以这些controls报告无偏泛化。部署仍使用既有全fit head；与cross-fit学生训练量不同的分布差异须报告。保存每折系数/句清单/原始预测和来源hash，核验all-fit重拟能还原原head，并做holdout target poison不影响本折拟合的测试。

## 评价、保存与判断

固定epoch8，不挑best轮。epoch0八seed基线复用已核验的audio_flow_v1/audio_local的epoch000，仅当原始权重／投影／head／cache及曲线hash绑定一致。每臂最终保存8noise（42/123/2026/7/19/73/211/997）×full/zero/reverse/oracle，12步生成；原始输出不校时、不调gain、不挑seed。

独立从checkpoint重建曲线、验证RNG一致、冻结权重不变；分眉/眼/嘴、neutral/nonneutral、逐人和mean/dynamic误差。报告full对source-full、own-zero、reverse、matched-trained-zero，fair ES/variogram，句cluster置信区间；noise不是独立样本。沿用既有audio_flow动态/分布/保护门槛，保留全部失败。训练teacher情感准确率只作诊断。

结果分类：oracle可表达不等于audio成功；audio只增加std不等于合理动态；raw MSE下降不等于时序改善。audio有净眉收益且口型保护成立再考虑下一验证。oracle明显改善而audio弱可继续查条件信息；两者都差先查监督、优化和生成链。8epoch阴性不当理论不可能。

同步检查已有rig能否渲染GT/B0/source/audio/oracle并导出连续视频；渲染若缺资产，明确记录，不以系数图冒充已看视频。原始系数评价和显示用clamp/fill分开记录。预先固定metadata样本，不选好例。

每epoch原子保存last，最终完整epoch8、Adam/RNG/来源hash和源代码快照；产物写root盘，不写接近满的autodl盘。不覆盖默认、不删除现有资产。保持操作日志，必要文件下载校验。
