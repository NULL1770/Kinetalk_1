# Phase62：V3最终接收结果，否决幅度校准

2026-10-09。完整1367开发条目，原生25fps，raw/clip及四个冻结整段统计probe。相同Phase53 style-only检查点，仅接收V3 TRAIN-neutral校准；没有神经参数更新，没有sealed/test。

|方法|MBE↓|LBE↓|Lip均值mm↓|主F1↑|jaw范围|jaw相关↑|闭口F1↑|
|---|---:|---:|---:|---:|---:|---:|---:|
|Phase53|.763407|.369546|3.262565|.676805|.124295|.477797|.401021|
|Phase62 V3|.762805|.368069|3.243318|.678671|.126920|.476134|.388260|

GT jaw范围.175279。范围仅增加2.11%，几何只有微小收益，闭口F1降低.012761、超过既有.01容差；相关性也未改善。因此不采用，也不能说解决了用户看到的口型差距。四clip F1=.678671/.648264/.739853/.689024，没有只挑辅助probe。

19原始文件/115710802bytes及293份源码SHA已核验；336个最终摘要字段从per_clip重放完全一致。检查点SHA80a7ca69b0f868bad6361381e1ca8e4e168ca227a0548221f8efa71b42169c41；校准SHA8a2faec88f87b3178c69dedc40ed5d200c55c30757517af2ffa7af22398133dc；报告SHAb4764bb8964dde037f74ba209b92519b65c331efe9f7e8d69bf4274766d167bc。所有参考干预现在使用同一query scaffold，区别于不能用于身份推断的Phase60对照。

八情感固定片段均已渲染，全帧解码/音频/rig/原生时钟核验通过。查看了八个固定中间帧的联系图；不是完整实时播放：候选与Phase53视觉差异很小，angry/contempt嘴形及fear/眉眼仍不匹配。GT依然保留，没有重新挑帧。视频目录为 `final_experiment/evaluation/rendered/phase62_compare_<emotion>_20261009/`。完整基线+候选raw/clip与八情感指标位于 `final_experiment/paper_tables/phase62_development_20261009/`。这些是诊断材料，不是论文成功结果。

接下来仅运行PHASE63_NATIVE_AFFINE_PLAN中预先规定的新接收器。正式模型仍是Phase53；停止继续只做单调幅度图的诊断循环。
