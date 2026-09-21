# 2026-09-20 完整 MEAD 训练终点：动态未通过静态对照

> 后续口型排查：本报告的训练终点数字保留，但原六格视频有显示错误——未监督 tongueOut 仍被驱动，已另行重渲染。全446条口型分解确认新B0较弱，Stage4的嘴部速度误差为B0的2.91倍；口型也尚未达标。identity阶段旧生成报告还含未训练renderer，不可据其判断身份质量。修复及证据见 `MOUTH_REGRESSION_AUDIT_20260920.md`。显示修复未改变checkpoint和原始评分。

完整队列已结束，所有子进程 exit 0。2026-09-20 12:23:53 至约13:02:27（北京时间），总耗时2313.62秒，约38.6分钟。完成audio五阶段、独立static动态臂、446条validation的三采样和全部干预、句簇bootstrap。默认模型没有替换，sealed test没有打开。

**结论：本轮解决了完整数据覆盖和旧Q投影限制，仍未实现可靠的局部音频动态收益。** 音频臂的raw ES更好，但centered ES显著差于独立static。不能把训练完成、幅度恢复或反转音频变差写作“音频动态成功”。

## 协议与可比范围

- 4098条train query、44条独立neutral reference；446条validation query、6条reference。22/3身份互斥，完整native帧，454566个train有效帧，约5.05小时。
- 这是本轮筛选并排除历史封存冲突后的完整MEAD四类训练划分：neutral715、angry1110、happy1144、sad1129；validation分别80/123/120/123。不是全部原始MEAD或八类全量，也未包括CREMA-D。537条test query和6条test reference仅核元数据。
- 所有KineTalk模块重新初始化；content/emotion2vec是冻结预训练特征。content只用全部715条neutral train query，其余动作阶段每epoch遍历全部4098条。identity每epoch22个互补reference pair。
- content/teacher/audio/dynamics各12epochs；identity200epochs=400steps。独立static仅从同一stage4重新训练dynamics12epochs；相同upper初始权重、batch/noise/time种子，局部audio与h0都时间池化，global audio和identity保留。
- 单次训练种子47，三次推理噪声42/123/2026。三次噪声不是三次独立训练。85个canonical sentence clusters的bootstrap区间不代表跨身份不确定性，validation仅3身份且有历史开发暴露。
- 与旧405×96帧、613/206开发实验的数值不可直接当成同数据提升。

Manifest SHA256：`6bdc33c3ee832e968b421781267dc70654e416b8e76395f2c6ce83c859e8603e`。

## 终点数值

以下是446条完整validation、三采样、每clip等权；系数不clamp。FaceDiffuser BEAT语义区域按ARKit通道名映射，含本数据mask适配，不是复现其官方数据集榜单。

| 条件 | MBE | LBE | ABS-FDD | Upper9 ABS-FDD | raw ES | centered ES |
|---|---:|---:|---:|---:|---:|---:|
| Stage4 audio base | 0.819653 | 0.433960 | 0.099974 | 0.074487 | 0.516770 | 0.128605 |
| Audio dynamics | 0.805965 | 0.433960 | 0.086424 | 0.052103 | 0.432878 | 0.121442 |
| Independently trained static | 0.812318 | 0.433960 | 0.085434 | 0.049533 | 0.456319 | 0.116166 |
| Audio model, static state | 0.804682 | 0.433960 | 0.086892 | 0.055929 | 0.430440 | 0.116520 |
| Audio model, reversed local audio | 0.805467 | 0.433960 | 0.085716 | 0.051255 | 0.433171 | 0.125598 |
| Audio model, oracle target state | 0.735931 | 0.433960 | 0.085941 | 0.050859 | 0.239091 | 0.128761 |

Audio − independent static（负值利于audio），2000次句簇bootstrap：

- raw ES：−0.023440，95%区间[−0.034788, −0.011375]。
- centered ES：+0.005276，95%区间[+0.003063, +0.007669]，audio更差。
- variogram：audio0.043809，static0.048180；此项有改善，但不抵消centered ES失败，且不能单独证明时间对齐。

Oracle使用GT慢状态，是目标条件诊断。reverse只反转local acoustics、保留h0，不是完整音频置换；主要归因依赖独立static训练臂。主FDD不含眉且对帧排列不敏感，不能证明动态时机。

## 瓶颈证据与解释边界

1. **音频慢状态没学出验证时序。** seed42记录的state centered correlation为0.008174，R²为−1.043945；同模型将state固定为片内均值反而改善centered ES（0.121442→0.116520）。这直接否定“慢状态预测已可靠”的说法。
2. **放开随机残差恢复了动作幅度，也暴露速度问题。** Audio上脸标准化速度RMS是GT的3.46倍（static3.96倍）；up/down/squint/wide幅度比分别1.466/0.763/1.057/0.495。音频臂各组原始系数越界比例25.14%/6.63%/0.083%/10.95%。这些是系数诊断，视觉自然度另查，不能用显示clamp掩盖评分。
3. **全局分类高准确率不等于生成质量。** audio全局情感分类94.39%；生成动作经本轮训练teacher读出约77.13%（seed42），后者不是独立感知评价。motion-teacher阶段LBE0.223566使用了目标动作条件，不能与部署audio的0.433960混作同任务结果。
4. **局部教师到音频的转换仍需定位。** Teacher renderer接收motion-derived local；audio阶段有global蒸馏、语义和flow/rollout，但没有对teacher local的显式逐帧蒸馏。缺此loss本身不证明是唯一原因；此前多个local蒸馏/直接音频路线已经失败，不能再次把它当成未经尝试的确定修复。
5. **全量覆盖没有自动解决动态。** 当前只证明这套结构、损失和固定12epochs预算未通过，不能宣称音频物理上无信息，也不能归因于某个唯一bug或保证延长训练有效。

后续有依据的定位应先做同终点train-vs-validation、多步solver及条件替换诊断，分清拟合/跨身份泛化/随机高频残差，再决定新训练。不得直接因本轮失败追加全量长训或修改当前终点重算成成功。

## 独立复算与产物

新增`audit_paper_full_run.py`只读hash绑定的final和保存曲线，不读dataset、不重采样。全446×3×5模式×2臂的MBE/LBE/signed及ABS-FDD/Upper9-FDD、raw及centered ES由独立NumPy公式复算，最大差2.220446049250313e−16。两臂system/audio权重与stage4逐tensor完全相同；每个动态干预的非upper43通道与同次base逐值完全相同。两臂GT、mask、clock、clip顺序一致。

完整特征本地tar2694072320字节，SHA256 `608d9f1c1c8f11d9c01fc2347dc5645cf39f0c61d9fa65fa1ef932d975d3aaa3`已验证。Audio final SHA256 `001ef31ee516cc810ee9a0bd2d6cf9a6251f119db8e2a872bc48870fb09ea6f2`；audio dynamics曲线SHA256 `342bd04390a85d30c202ad01c9819d7d4abcd170dd24372b5f9a474fbbcd4e41`，已下载核验。全部332文件（六个final、三份大曲线及报告）保存于本地`artifacts/paper_training_20260920/run_backup`，另有275660800字节恢复tar，SHA256 `31c26ac2ab12e20b064688c0b1fb36b7b519c09686df854812a7cf51cea2c3e0`已验证；包含两臂last/optimizer/RNG、source、日志、时序诊断与固定视频输入。

固定视频选择为每个validation speaker按clip_id排序第一条：M025/M037/M039的angry_L1_001，seed42，全native时长，GT/stage4/audio/static/reverse/oracle六格，不做幅度或lag拟合。三段已全部渲染完成93/154/75帧，分别3.72/6.16/3.00秒，视频和音轨时长一致，rig映射全帧最大误差0，原blend未修改，音频hash与native一致。每片两个无效边界帧仅为显示用最近有效帧填充，评分排除；仅显示做[0,1]截断，原始评分不截断。已查看六个固定接触帧，显示对象/相机/六格正常；不能用静帧或封装时长替代完整自然度及AV感知验收。视频在本地`artifacts/paper_training_20260920/videos`。

FaceFormer/CodeTalker/FaceDiffuser同协议重训、AV offset/confidence、learned Multimodality/FD/WInD、独立情感/身份感知及sealed test仍未完成，不填代理数字。当前不适合声称“音频驱动细腻动态已成功”。
