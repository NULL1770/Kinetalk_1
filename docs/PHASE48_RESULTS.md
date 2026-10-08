# Phase48 固定音频的身份/动作风格核验

2026-10-08。完整开发1367条，固定原音频/B0/g/u，只交换独立neutral动作参考。无模型训练、无默认替换、无sealed读取。比较冻结Phase45-u与Phase47-latent。387个相同sentence_id/emotion/intensity组，1087匹配clips，313个三人组、74个两人组，280条未匹配；2026个有向跨人对。不同人真实音频和表演仍混杂，比较native clip统计，不作逐帧跨人GT误差/重定时，也不能作因果身份正确率。

## 实际结果

clip[0,1]目标人物统计（越接近越好），各pair等权，数据来自audit/report.json：

|模型|整体均值误差改善对比例|嘴部均值改善对比例|眉部均值改善对比例|整体均值变化方向余弦|嘴部均值方向余弦|
|---|---:|---:|---:|---:|---:|
|Phase45-u|79.8124%|71.0267%|59.7236%|.397240|.308975|
|Phase47-latent|80.3554%|72.3100%|60.2665%|.407957|.323506|

latent嘴部标准化统计误差平均改善：centeredRMS .014494、q90-q10 .088862、相邻位移RMS .001633。它们只证明描述性幅度/均值朝目标人物统计改善，并非精准目标表演或音素保持。六个speaker方向与八个情感完整保留，不能用总均值覆盖失败方向：M025->M037及M025->M039眉部均值误差反而上升；M025->M037嘴部相邻位移误差上升。sad嘴部均值改善比例49.3%，仍弱。

**风格路径有可迁移的部分证据，参考稳定性不足，尚不能写成全面正确的身份解耦。** 当前相同rig不生成脸形，只是动作风格。跨人变化的latent/all51平均MAE .032516；同人A/B仍为.024839，约跨人差异76.4%。同人A/B嘴部MAE .018927，闭口(<.05)判定分歧22.313%；跨人相应分歧23.189%。同人变化太接近跨人变化，不能简单放大style gain。

|人物|独立参考A/B编码余弦|A/B编码RMS|
|---|---:|---:|
|M025|.224025|.868523|
|M037|.957833|.342058|
|M039|.855787|.571309|

M025尤其不稳定；目前不能区分参考自身的neutral姿态/动作差异、参考语句内容依赖、编码不稳定各占多少。先诊断参考坐标和可靠性，再选结构修改。jaw平均lag接近0不能证明每条都无偏移；jaw/mouth相关也不是独立音素读出。

## 8情感视频与检验

[八情感风格交换视频目录](PHASE48_VIDEO_GALLERY.md)。6格为GT/M025AB/M037AB/M039AB/OwnA/OwnB，同一源音频与情感条件，原生25fps、无填帧、无增益/lag拟合，共同rig/镜头。每个视频完整ffmpeg解码、ffprobe帧数/音轨、rig逐帧值、GT与父baseline时钟、source/audio/rig/videoSHA核验通过。固定原8clip，不按效果挑选。已人工查看全部固定中间帧及happy/neutral较大单帧：可见嘴部幅度、眉眼基础姿态差异；angry/disgust/fear/contempt与GT仍有表达差异。检查只支持这些可见现象，不证明完整动态准确。论文图仍未制作。

## 正确性及预算失败的处置

6个有意义测试，本地/远端均通过。24真实GPU smoke及完整1367复现：B0逐值一致、latent差异0，父最大1.2278557e-5，原rtol1e-5/atol2e-5未改变；parent/B0/correction状态精确冻结，无queryGT部署或新student内容通路。原代码和数据/probe/rig未改。

首个smoke因24条重组的padding上下文导致2.6166e-5数值超界，在任何完整审计前停止；以原canonical32缓存B0及原canonical16推理上下文修复，新v2执行。旧失败保存，未放宽数值容差。

完整计算通过后worker的20MiB存储预算断言失败，实际32765298bytes，主要为25MB无损统计。原binding及失败保留；所有原manifest成员重哈希后写storage_acceptance_receipt.json，确认仅存储超限、低于40MiB且root仍有200MiB以上空间，不重新计算/修改数值或模型。勿把原pipeline_state的failed误认为推理失败，也勿伪改它为complete。

实施档案：48005a6/c01d984均推送。远端/root/kinetalk_phase48_style_audit_v2_20261008；本地final_experiment/evaluation/diagnostics/phase48_style_audit_v2_20261008。original SHA下载、全部pair无损per_pair.json.gz、native_statistics.npz、16个parent/latent renderinputs、8视频收据保留。

## 后续优先级

1. 在TRAIN/原internal-held及disjoint neutral enrollment上，比较真实参考A/B的姿态与B0残差统计、内容相关幅度和编码变化；dev只作描述性复核，不拟合参数。先判断M025不稳定来源，避免误惩罚真实风格差异。
2. 风格定义明确为人物稳定neutral姿态/发音幅度习惯，以及条件于情感的表达响应，不能把单条参考表演的瞬态当人物常量。优先检验内容归一后的参考统计和跨参考共同分量，再决定编码/decoder是否须分开基础姿态和响应；不复制失败Phase46/reference，不以差异更明显为目标。
3. 与approved aligned TRAIN-pair的teacher目标坐标诊断连起来：仅使用原teacher_mask AND event_local_mask AND source_observation，区分B0发音误差、表达动态和人物因素。学生继续772D情感/韵律，无motion-reconstruction梯度或内容输入。
4. 新训练前按原SHA核验本地历史备份后释放仅冗余远端artifact空间；不删当前B0/parent/data/probe/rig依赖。定型改动先推Git、独立smoke再训练。SOTA、未见测试、独立内容读出和人评仍待补，主指标未因本次冻结审计变好。
