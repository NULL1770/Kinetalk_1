# Direct audio final epoch8：连续画面核验

2026-09-17。固定 metadata 九例沿用 `trace_selection_seed20260923.json`，只取原锁中每名 speaker 的第一条制作连续视频。未按结果换样例、种子或训练轮次。导出脚本：`scripts/export_direct_audio_examples.py`；产物：`artifacts/brow_review_20260917/visual/`。这次检查包括真实渲染的连续 MP4、等间隔画面序列、源视频，不把系数曲线当作视频证据。

## 画面结论

`direct flow` 的上半脸确实活动更多，但错峰和快速变化更多；`direct dynamics` 的连续运动较平滑，眉下压仍偏弱或方向不符。它相对本臂 `zero` 有可见的条件作用，但三条样例并不支持“眉动态已经恢复”。静态表情姿态也有偏差：M030 原片/GT 是较轻的眉动作，生成结果长期保持更强眉下压。增加逐帧动态 loss 没有自动校正片段均值。

视频六格统一为 GT、B0、source、direct flow、direct dynamics、zero。`source` 是来源模型在 `audio_flow_v1/audio_local/epoch000_curves.pt` 的 seed42/full；`zero` 是 direct dynamics epoch8 同种子的 zero-local，不是另训练的 zero 臂。B0 是缓存的 B0+identity baseline。

三条完整视频都为96帧、25fps、3.84秒，同一 `arkit2.blend/face.001`、360像素格、16samples。52通道按名字绑定，六个对象的 shape keys 独立，逐帧实际 shape key 与待显示数值最大误差0，原blend没有被保存或修改。M023/M024全部96帧有效；M030只有81帧有效，原索引0和82–95按最近有效帧补显示，尾部静止不作为模型表现计分。

|固定样例|GT左眉下压 std|direct flow std|direct dynamics std|本臂zero std|direct dynamics 与GT相关|
|---|---:|---:|---:|---:|---:|
|M023 angry L3 001|0.0990|0.0955|0.0313|0.00725|0.522|
|M024 angry L3 025|0.0508|0.1006|0.0302|0.0137|0.231|
|M030 angry L1 016|0.0184|0.0608|0.0226|0.00355|-0.490|

以上为已观察有效帧的原始系数 std/相关，仅用于解释画面，不取代完整405段多seed审计。M023 的GT左眉下压峰在第39帧，direct flow在第87帧，direct dynamics在第3帧，说明幅度接近也不等于峰位正确。M024 GT第63帧，dynamics第75帧。M030 weak-GT样例中dynamics幅度接近却负相关，不能单以std通过。

## Clamp影响

原始系数没有修改；显示统一 clamp 到[0,1]。三个样例的左右眉下压在所有模式下均无 clamp，故主要眉下压缺峰/幅度不足不由 clamp 造成。局部抬眉通道有重要影响：M023 dynamics 的 `browOuterUpLeft` 88.5%帧小于0，std由0.00192压至0.000431；M030 的 source/flow/dynamics/zero 的 `browInnerUp` 全部有效帧小于0，显示后完全不动。所有通道总 clamp 比例在 direct dynamics 中分别24.64%、15.46%、21.75%。这暴露生成系数域偏移，不能将显示裁剪前的抬眉 std 当可见运动。

## 原视频核对

三个原视频经远端 `mead_media_v1/provenance` 定位到本机 `E:/mead/...`；本地文件size和mtime均与旧记录吻合，新算源文件SHA256保存在 `source_video_bindings.json`。按缓存 native start 0.08/1.00/0.00秒截取原视频，未拟合lag或拉伸时间；原视频约30fps，GT rig为25fps。源视频片段和 `source_vs_gt_contact.png` 均已保存。

原片显示：M023主要持续皱眉并叠加小变化，GT rig有相似下压姿态但眉毛毛发、细皱纹和内侧眉形没有忠实重现；M024原片持续强皱眉、挤眼和鼻梁皱缩，GT rig有明显下压但柔化了肌肤褶皱与挤眼强度；M030原片确实较轻微，GT rig较弱变化与其总体趋势一致。M023源视频约第49帧的闭眼能在GT眼区看到对应变化，但这种局部对应不构成整个tracker时序/幅度准确的证明。

本轮只完成三段人工画面对照，没有重新跑tracker、几何投影重建误差或AU人工标注。因此不能用“系数→rig自洽”宣称raw监督准确，更不能从某一个speaker恰好对应推断普遍成功。固定rig也不能检验身份保持；本轮没有唇同步独立打分和情感人评，不能据此说身份、口型、全局情感已无问题。

## 文件与复核

- `mead_M023/comparison.mp4`、`mead_M024/comparison.mp4`、`mead_M030/comparison.mp4`：真实连续六格，附原时钟音频。
- 各目录的 `source_native_crop.mp4`、`source_vs_gt_contact.png`：源视频与GT对照；M030原片只有3.30秒，GT补齐至3.84秒的后段不属于观测。
- `raw_seed42.png`、`centered_seed42.png`：固定九例原始系数与去B0后居中的曲线；不作视频代替品。
- `provenance.json`、各目录的 `display_report.json`/`rig_audit.json`、`display_compact.json`：曲线/模型/cache/split哈希绑定、clamp、mask与真实渲染检查。
- `review.html`：本地播放选择器，含正常速度和0.25倍速。

仍属内部validation405诊断；B0/global有历史训练暴露，句子共享。本结果不作封存test/新身份泛化或论文录用证据。

## 同日追加：output displacement/std 分支

新增 `scripts/export_output_motion_examples.py`，同样只在CPU读取保存的final epoch8 curves；分别验证两臂curve/checkpoint/recipe/cache sidecar、output inventory，以及相同初始化、batch/noise/time/choice序列。新的六格固定为 GT/source/direct L1/output audio/output own zero/trained zero。own zero是同一个audio-trained模型去掉local条件；trained zero是匹配的独立zero训练臂。三条视频和固定九例raw/centered曲线位于 `visual/output_motion/`。

仍采用同一模型、光照、360像素格、16samples和96帧25fps。三条逐帧shape key误差均0；进行了连续播放和等间隔画面检查。output audio在M023下压起伏更明显且比direct flow平滑；M024变化仍弱；M030的走势转为与GT正相关，但trained zero也产生几乎同程度的趋势，而且持续下压姿态远强于原视频/GT。因此不能把M030的单例改善归因于成功学习了音频。

|样例|GT左眉std|direct L1 std|output audio std|own zero std|trained zero std|audio相关|trained zero相关|
|---|---:|---:|---:|---:|---:|---:|---:|
|M023|0.0990|0.0313|0.0721|0.00903|0.0555|0.441|0.507|
|M024|0.0508|0.0302|0.0300|0.0110|0.0443|0.229|-0.144|
|M030|0.0184|0.0226|0.0280|0.00588|0.0284|0.567|0.559|

M023 output audio的最大左眉下压在第11帧、GT第39帧；M024仍是第75帧、GT第63帧。M030 output audio左眉均值0.441、GT仅0.088，说明动态方差匹配不约束静态偏置。以上仍为seed42固定例，最终成败需看独立405段8seed审计。

显示域退化更加突出：output audio中，M023右outerUp全部负值、左outerUp86.5%负；M024双侧outerUp全部负值，innerUp86.5%负；M030的innerUp和双侧outerUp全部负值。100%负值的通道在[0,1]显示时整段变成0，即使raw std不为0也没有可见抬眉。三个样例全通道clamp比例27.0%、25.2%、30.5%。眉下压仍0% clamp，所以其可见偏置/错峰是模型输出本身，而抬眉缺失有明确的系数值域因素。

本追加不能支持“通用眉毛动态已解决”。它说明最终运动损失能改平滑性和幅度，同时仍缺真实域的均值/范围校准、正确峰位及相对匹配zero训练的普遍增益。raw/centered/std必须与实际clamp后画面共同审查。

独立最终审计亦判失败（详见 `OUTPUT_MOTION_DYNAMICS_RESULTS_20260917.md`）：audio相对matched-zero的非neutral眉ΔR²为-0.409906，句子cluster bootstrap 95%区间[-0.59348,-0.21893]，三名speaker均负；嘴部raw误差相对来源在非neutral增加8.509%、neutral增加78.868%。这些全样本结果优先于此处单段视觉印象。本分支只保留为失败诊断，不替换默认模型。
