# 2026-09-20 完整训练划分与动态路径修正

用户要求先上传现有源码，再修改和训练。修改前快照为 GitHub 分支 `codex/experiment-snapshot-20260918` 的 `66712ea`。

## 数据版本

此前的 4118 条 identity_disjoint 候选存在历史封存冲突：test15 的 14 条进入 query，1 条进入 enrollment。本轮按历史封存的三个句子排除全部 20 条 train 录制，并重新选择中性参考。不能使用原候选直接训练。

| role | query | enrollment | query valid frames |
|---|---:|---:|---:|
| train | 4098 | 44 | 454566 |
| validation | 446 | 6 | 47349 |
| sealed test，仅元数据 | 537 | 6 | 61171 |

Manifest canonical SHA256: `6bdc33c3ee832e968b421781267dc70654e416b8e76395f2c6ce83c859e8603e`。

训练身份22人、validation3人、test3人；跨角色身份和clip互斥，句子允许跨角色重用。这是新身份泛化协议，不宣称新句泛化。每人的query和neutral enrollment的speaker-sentence对互斥。原生train/val/test source metadata SHA与本地副本一致。旧开发暴露不消失，validation明确作开发，不能称untouched test。

完整4594个train/val query和reference分片已物化，约200秒、2.6GB。每条保留所有原生帧和mask，最大321帧；454566个训练有效帧约5.05小时。没有打开native test或旧test15的动作/音频目标。CREMA-D仍为单独候选域，不并入本次MEAD协议。

统一输入为native content768、emotion2vec中层2/4/6的逐帧LN均值768、韵律4，全部对齐原始25fps clock。波形和原生artifact均核SHA；仅train query有效帧拟合输入mean/std和运动尺度。独立中性参考用于该身份anchor与identity，不从query求部署anchor。预训练content/emotion2vec来源单列，不加载历史KineTalk checkpoint。

## 实现改动

五阶段对应content、identity、motion-global teacher、audio-global、dynamics。各KineTalk模块从随机初始化开始，前四阶段完成后冻结非动态模块。

旧v3将随机innovation投影到Q=I-P，导致最终慢速group state严格等于音频state预测，随机生成器无法纠正低频误差。旧output-state loss因此几乎就是重复state loss。新`AudioResidualFlow`保留音频慢状态作为条件均值，随机残差不再Q投影，允许表达慢速和快速条件不确定性。

新动态loss为FM + 0.5×确定性state Huber；每四batch使用两次独立噪声实际12步rollout，增加0.1×raw fair trajectory ES、0.1×centered fair ES及0.1×越界项。FM residual target为normalized GT减去detach的audio mean。单个随机endpoint不施加逐GT MSE；fair ES含样本间spread项。这是研究假设，不能由公式或会动推断自然度、音频时机已经成功。

额外训练匹配的static动态臂：从完全相同stage4 system/audio和相同upper初始化出发，使用相同训练batch/noise/time种子，池化局部audio_features和h0的时间信息。全局音频情感与独立identity保持。推理full/static_state/oracle_state/reverse使用同三噪声种子42/123/2026。Oracle是目标条件诊断，不能放进audio-only主表。

预算：content/teacher/audio/dynamics各12epochs；identity200epochs。身份只有22个互补参考pair，一epoch仅2步，故200epochs=400步。动态audio/static各遍历全部4098query，固定终点，不按validation挑epoch。该预算不代表模型已收敛或论文方法锁定。

## 自动执行和评估

入口`run_paper_full_queue.py`顺序执行完整audio五阶段→匹配static动态训练→配对validation报告。每epoch保存optimizer、所有RNG、源码/输入hash和last checkpoint，可用原参数加`--resume`恢复。每阶段保存final；异常有明确退出码和状态，不将失败写成complete。

保存FaceDiffuser官方BEAT语义区域的ARKit-MBE/LBE/signed FDD/ABS-FDD，另报Lip23/Upper9。全部原始系数不clamp，按真实mask评分。FDD不含眉且对时间排列不敏感；另外保存Upper9 fair ES、variogram、raw OOB、速度和区域轨迹指标，audio相对独立static的句簇bootstrap区间。身份及global读出单独记录；训练teacher读出不是独立感知认证。

五阶段GPU smoke通过；另外6clips×3seeds×5干预的完整自动报告路径通过。此前的34项指标/新路径测试和30项旧数据/训练/resume回归通过。首次启动默认Python缺funasr，改用已有emotion2vec专用venv；首次smoke报告输出目录缺mkdir，修复后重新完整通过。

FaceFormer/CodeTalker/FaceDiffuser尚未同协议重训。AV offset/confidence、learned-feature Multimodality、FD/WInD仍pending，不填代理值。动态成功、视觉自然度、口型同步与外部论文比较仍待实际训练结果及验证，sealed test暂不执行。

## 资源与恢复

远端独立目录`/root/kinetalk_paper_20260920`。完整特征及大曲线使用`/dev/shm/kinetalk_paper_full_v1`与单独artifact目录；shm重启会丢失，必须本地备份校验。compact checkpoint和日志在root持久目录，不能依赖shm作为唯一恢复来源。原始数据、默认权重和历史实验均保留。

2026-09-20 12:23:53北京时间，完整队列已启动：队列PID2648、训练PID2712，run目录`/root/kinetalk_paper_20260920/full_v1`，日志`full_v1/audio.log`、`full_v1/static.log`，状态`full_v1/queue_status.json`。实际加载4098/446、454566train有效帧；口型和identity阶段已完成并写出checkpoint，teacher阶段每epoch约27秒。整体预算估计30–50分钟，包含独立static和三采样完整验证，结果尚未据此宣告成功。

修改后实现提交`85fb20a`已上传同一分支，已核对远端五个执行源码文件哈希。首个完成阶段checkpoint已下载并核SHA。新增封存冲突、全帧读取和配对bootstrap合同测试后，共67项相关测试通过（其中新pipeline8项），上述数字不代表质量验收。
