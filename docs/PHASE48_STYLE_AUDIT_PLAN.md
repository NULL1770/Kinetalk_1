# Phase48：固定音频的身份/动作风格核验

2026-10-08。用户要求继续优化，并实际渲染不同身份参考的区别、验证用途；若定义/效果不对再调整。Phase47已关闭，2135eaf已推送并独立核验。此次先做冻结诊断，不更新任何模型，不训练probe、不读sealed、不自动采用新默认。

## 当前实际定义

ReferenceStyle输入两段独立neutral motion、该参考的冻结B0和通道mask，输出64维style；输入含residual和B0。训练时按同人物A/B单参考重建同一query，并通过decoder静态bias和四层条件调制影响52D动作。没有独立身份证明损失、跨身份真实GT重建或shape身份输出。当前共用rig只能验证动作风格，不可宣称长相身份生成。Phase47-latent还含由style预测的常量修正，须同时比较父Phase45-u与latent，分开判断是否只增均值。

## 固定协议

- 原开发集1367、3人物M025/M037/M039，全部固定音频/B0/g/u；交换每人独立neutral参考AB，另看原人物A和B。部署不读取query GT；GT仅离线计算核验指标。
- 按sentence_id+emotion_id+intensity_id匹配真实跨人物query；初步metadata：387匹配组/1087clips，其中313组三人物，280clips无跨人匹配。单一句子ID不证明精准音素/帧对齐，不做跨人逐帧GT误差、不重定时。精确统计以正式清单为准。
- 普通同人物输出须复现Phase45/47已保存的部署曲线；冻结parent/B0/correction状态核验，全部参考与query clip disjoint、neutral且对齐自身native clock。B0、g/u逐值不变；不存在query GT进入部署。
- 描述性指标：跨人输出变化/同人A-B变化的均值和动态区域差异；风格code稳定性；jaw/mouth中心化与相邻位移相关及固定native lag敏感性；clip/raw均保留。它们不是独立音素读出或身份分类正确率。
- 目标方向验证：对匹配相同内容/情感/强度的donor真实native motion，比较换donor参考前后生成clip均值、centered RMS和q90-q10的距离、predicted-vs-real跨人差值方向；另外看以独立neutral参考均值为坐标的表达均值。保留分人物/情感和未匹配数量，不将特定GT姿态变化误称精确跨身份动作监督。音频和表演差异仍混杂，仅3开发人物。
- 固定8个原M025展示clip，不按结果挑最好身份。视频展示GT/sourceAB/M037AB/M039AB/sourceA/sourceB；原生25fps、同音频、同rig/镜头，第三方方法图与论文四类图暂不制作。每个交换参考和checkpoint/correction SHA、native时钟/通道和渲染收据保存。

## 执行与决策

1. 预变更存档并推送；实现独立审计/有意义的输入与目标比较测试，不改变旧模型/evaluator。
2. 真实GPU smoke后正式1367冻结审计，产物预计<20MiB，当前root仅约291MiB仍足以有界诊断；不启动大训练、不删历史证据。
3. 取回原SHA核验、渲染8情感身份交换及同人A/B视频、检查完整decode/rig/native时钟。结果和目标方向统计先判断风格是否只是敏感而非正确可迁移。
4. 有足够证据再选最小正确设计调整：如明确基础姿态与表达响应的风格坐标、独立参考一致性或manifest优质跨重建；不为让差异明显任意扩大style gain，也不把换人后对原人GT误差上升当迁移成功。新训练必须先解决存储并重新存档。

持续边界：neutral B0冻结，student772D情感+韵律，无内容/queryGT部署/动作重建student梯度；口型通道开放；原raw/clip、4probe、clock/rig不改。每轮修改前推Git；不创建子agent、不重复旧collector。结论必须区分风格路径活动、参考稳定性、目标风格方向及长相身份。
