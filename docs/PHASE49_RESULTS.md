# Phase49 独立neutral参考坐标结果

2026-10-08，冻结诊断完成，25人物=20TRAIN-fit+2原internal-held身份+3external dev。只执行enrollment原B0，两个query缓存均0，不使用query动作拟合参数、不读sealed、不更新模型或默认。Source16f36d1已推送。6已有native/context测试通过，原Phase48三个人物style A/B code统计逐项重现，B0及parent状态冻结。

## 关键证据

以下RMS同人/跨人比例是同类特征自身的距离比例，不能当目标身份正确率；只有2/3held/dev身份，置信度有限。A->B检索三组都100%，并不等于同人编码不变。

|描述量同人/跨人RMS比|TRAIN-fit20|internal-held2|external-dev3|
|---|---:|---:|---:|
|原64D编码|.010170|.367054|.465366|
|真实neutral嘴部均值|.321787|.366295|.280882|
|GT-B0嘴部均值|.817906|1.122915|.391860|
|真实neutral嘴部动态|.475501|.413560|.481717|

原code在反复使用相同20人固定参考的训练集极度一致，在未见人明显变动。支持参考编码对训练enrollment过拟合的诊断，但不是单独因果证明。实际neutral GT自身也有跨参考差异，不能强制所有真实姿态完全相同。

直接用GT-B0残差不能当纯身份/风格：TRAIN/internal-held嘴部同人/跨人比例反而变高，held为1.123；其中混有B0发音误差及参考内容分布。B0本身的均值也携带人物/音频关联（TRAIN A->B参考检索95%），所以不能仅靠输入张量名字声称因果内容-身份解耦。

M025 A/B在GT中browDownLeft均值.0451/.1574、browInnerUp .0916/.00795；即使限制B0预测闭口(<.05)帧仍为.0444/.1561及.1014/.00787。说明参考自身的眉部姿态就不同，不能只责怪编码器。jawOpen GT均值.00175/.00266，B0为.05758/.04137；上唇通道GT约.50–.57，B0约.07–.16，存在明显channel分配/B0误差。这里只是native系数事实，尚不能断言ref拟合坏或物理发音错误，需要对应参考音频/视频与数据质量进一步核验。

## 后续设计约束

- 人物风格宜分成稳定neutral基础姿态与内容条件下动作响应/幅度习惯；独立参考中的瞬态表达及B0误差作为不确定性处理，不能直接混入人物常量。
- 参考编码须检验未见人/不同语句和reference A/B可靠性。优先做内容归一的可观测统计/轻量reference encoder与现有encoder固定预算内部对照；单独新增一致性loss不足以避免训练enrollment记忆。此处尚未更改或宣布实现这些结构。
- 与高质量对齐TRAIN-pair teacher目标坐标诊断一起决定表达teacher输出：区分neutral风格、B0误差、可预测表达及随机表演成分。学生772D情感/韵律边界保持。
- 不用dev/展示视频拟合gain，不把100%reference retrieval或80%target-stat改善写成身份成功率，不宣称SOTA。

## 存储/复现

原诊断完成后2MiB输出断言超限，实际2172526bytes，仅多约75KB的JSON。原worker failed及binding保留，storage_acceptance_receipt记录诊断/冻结门通过、无重执行并接受<8MiB；8原始成员下载并重新核验SHA。后续诊断应测算真实序列化体积，避免重复不合理的小预算预测。

远端/root/kinetalk_phase49_reference_coordinates_20261008；本地final_experiment/evaluation/diagnostics/phase49_reference_coordinates_20261008。不要重跑已完成worker。

为下一次训练准备空间：两份旧Phase34 smoke last.pt（含optimizer恢复状态）均在本地原download_manifest下验证SHA且远端再验证、无活跃进程后才清理远端副本。保留本地可恢复原件与逐文件归档收据、旧final/code/curve/report及当前B0/parent/data/probe/rig。释放152853632bytes，rootfree410513408bytes；data仍约93MB，正式大训练仍需更多空间。

当前模型指标仍是Phase47-latent F1.719825/MBE.825160/LBE.383860/Lip3.276868mm，未因本轮诊断改善。Phase48八情感动作风格视频已完整验证；论文四类图仍按用户要求暂不制作。
