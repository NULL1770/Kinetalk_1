# 当前恢复入口：2026-10-07 Phase36

**最新覆盖：Phase36已完全闭合，三seed全1568updates/全评分/387文件1,782,721,197bytesSHA备份/八视频COMPLETE，三gatefalse，拒绝。** F1 .683872/.581147→.637154/.545352，MBE .904603→.911180，LBE .446154→.452668，lip4.00867→4.05532mm，mouth速度MSE .00370947→.00477843。11565、9275、75621都已exit0，不再wait/重启。因素16514已结束，16文件全SHA备份；两臂fixed47/66cells的3draw重放严格0，state不变，static/reverse仍改善jaw。Phase36后继是Phase37明确表达/韵律目标，不是扩大Phase36预算。

**Phase37本地实现/40相关测试通过；尚未真实smoke或正式训练。** 先读PHASE37_NAMED_EXPRESSION_PROSODY_PLAN.md。新`--audio-supervision expression-prosody`：ua64=4有符号表达状态+4标准化原生韵律+56zeros；已有state_head4启用，旧local_head64保留兼容不作候选条件；renderer完整flow/generated监督对audio条件detach，学生改学native4stateMSE+原CE+TRAIN情感×强度原型globalMSE（label查表目标不随query变化）。只改trainer/slow_state_affect两source，没有新神经层/嘴mask/critic。namedlayout在config保存，新增SlowStateAffect.from_checkpoint恢复模式；命名checkpoint不能静默按旧flow训练。全suite尚待更新/真实默认重放未做，不能报新性能或默认推广。所有原控制路径仍默认flow/latent。

只以本页和各phase结果页为准。旧ACTIVE/pending approval语句属于历史；用户再次明确授权继续优化，不重复询问。Git暂停、sealed test未读、默认模型未替换。保持中性B0，HuBERT只直接进B0/h0，情感学生772D=emotion2vec768+prosody4，全部已观察嘴通道前向开放。

## 唯一当前活动

Phase36 `/root/kinetalk_phase36_expression_gradients_20261007`：driver15949、workers15950/15951/15952、closure15953，唯一local compressedcollector session11565。只读`.codex-finalizer/phase36_status.py`。三seed47/48/49各2轮1568updates；新expression候选复用Phase34三个standardized已完成对照（标为full，服务器symlink）。不要再启动训练、评分或同目标writer。

冻结因素诊断已排队：唯一driver16514、独立root`/root/kinetalk_phase36_factors_20261007`，等main closure complete才运行固定seed47两臂/66metadata cells，3draw精确重放后draw42的ua/identity干预。不是全量性能或异人配对真值；不影响main collector目录，source只复用已验证helper。helperSHA45da3c.../driverSHAd72e6f...，不重复启动。local readout/固定八视频自动收尾helper已准备，唯一session待记录。

唯一local finish **session9275** 已等待11565备份，通过后自动summarize_phase36_local.py→render_phase36_fixed.py；读取local_finish_state.json。不要再启动readout/render writer。最新三个worker均已完成1568参数updates，47日志arm_complete，48/49仍最终rollout/save；launch running在此阶段不代表需要重训。没有新性能结论。

唯一改动是audio学生的flow/generated-semantic梯度限制为9个原生眉眼表达通道。renderer继续完整残差/原generated-semantic梯度，前向和所有嘴通道不变；共同联合梯度裁剪系数可变。语义CE、强度CE、global蒸馏及系数保留，没有新增loss或网络。**不是完整D2表达teacher，也未证明完全解耦**：旧global teacher仍看全残差，emotion2vec可含音素相关信息。具体设计/验证/接纳见PHASE36_EXPRESSION_GRADIENT_PLAN.md。

150源码只trainer SHA变化；binding `121248f768d139735d6d43fef085585a4ff115d21d7f94600d60dc00c42ece88`。28相关检查、全389passed/1skip；两个实际32TRAIN/24val、各2update smoke通过：默认所有state/实际流精确复现Phase34，候选初始state/输出/全部loss及sample/GT/mask/time/noise一致；HuBERT梯度0/NaN输入隔离、冻结、非表达mouth→student梯度0、第二步u_a非零。候选最终25个audio tensors已变化；这些是正确性证明，不是性能成绩。

预算沿实际三候选final/last/curves和三audit尺寸：1.77593GiB包含384MiB余量，启动floor1.8GiB。实际回收后free1.884GiB才启动，全部新final/last/curves持久/root。不能把已预算输出占用造成free下降当作启动失败。Phase36 baseline本地从Phase34读取，不假定collector会复制目录symlink。

## 已闭合，不重跑

**Phase35完整训练/评分/备份COMPLETE，拒绝physical单位。** 三seed三draw clip原生成F1 .683872/.581147→.229124/.266488，MBE .904603→.915915，LBE .446154→.457559，lipmean4.00867→4.02053mm，vertexFDD153.448→135.782mm²；嘴速度误差也退步，三gate均false。不延长。唯一collector90557已exit0，383文件1,781,631,489bytes原memberSHA核验通过。read_phase35_local_results.py已把fullbackup标为true；看PHASE35_RESULTS.md。

**Phase34完整训练/评分/备份/八视频/t-SNE COMPLETE；N1拒绝。** clip三seed原生成F1 .203660/.251196→.683872/.581147，MBE .910817→.904603，LBE .440420→.446154，lipmean3.92683→4.00867mm，FDD144.884→153.448。仅seed49联合gate通过，不推广/不扩大预算。476文件3.227GB SHA核验通过，collector77005已exit0，旧84935停止exit1；都不再wait。manifest60be9696e9fdf6abe9fdd9cf70dea5b0d1fe4bb950a071fc4a7ae570f3cb9887。

固定八视频路径：`final_experiment/evaluation/rendered/phase34_neutral772_coordinates_{emotion}_s47_20261007/comparison.mp4`，列为GT/Neutral B0/Centered772/Standardized772；第三列不是Phase35。共同冻结original128 probe t-SNE/confusions在Phase34 `descriptive/`。Happy两原probe F1 .901/.899，较弱类neutral/angry/disgust/sad。GT跨环境feature重算maxabs7.75e-7、34416值不逐位相同，但1367预测全相同/GT F1严格.66034433；两生成view严格一致，原GT文件未改。任何模型重放/传输SHA零容差未放宽。

Phase32/33均已闭合：772输入修正本身未提升性能；static u_a改善jaw时序却抑制眉部/smile动态，不能简单清零；身份code改变动态，bias只静态，但正确身份/完全解耦未证明。阅读PHASE33_FROZEN_RESULTS_AND_NEXT_PLAN.md，勿重跑旧提取或因素诊断。

## 已删除的服务器重复成员

全部保留原本地memberSHA恢复路径和receipt，数据/中性起点/全部final/三个Phase34 standardized baseline curves均保留。历史重放须先恢复具体缺失成员，不宣称服务器原路径仍完整。

- Phase34首5：47control曲线/last、47std last、48control/std last；回收661,453,720bytes；selected_reclaim_state.json。
- Phase34后4：48/49control曲线、49control/std last；回收864,346,416bytes；remaining_reclaim_state.json。
- Phase35确切5：三physical last、47/48physical曲线；每个member先本地持久SHA再服务器SHA才unlink；回收940,773,296bytes；selected_reclaim_state.json。49physical曲线和全部final仍在。383全量备份随后闭合，无需补下载或再清理这5项。

## 表达/强度目标边界

见EXPRESSION_TARGET_AUDIT_20261007.md，v4/schema v2/jawOpen index17。B0实际715原生中性+2583批准情感→中性，其他低质量配对不扩大。全部2583 input/hash/speaker/reference/native clock/shape/finite合同通过，mouth有效55.375%；297full gate也仍受局部mask限制；event mask精确0/1，无软权重bool bug。旧artifact dtw_quality_verified=false不能否定额外manifest/path/event检查；但配对差仍有发音/对齐误差，不能直接当纯情感逐帧GT。

469同人同情感同reference三级强度组，仅9组三full gate。jaw RMS平均上升，但严格L1<L2<L3仅49.89%（happy33.3%）；不加统一jaw/全嘴ordinal loss。MEAD三级是clip强度条件，不是逐帧情感真值。

## 接下来的实际顺序

1. 等唯一Phase36训练/closure，若具体report失败保留失败证据，只修报告，不重训/放宽合同。
2. 等唯一collector11565完整SHA闭合；按全三seed联合gate判断，不择seed/延长失败pilot。
3. 从固定M025/005/seed47/draw42系数渲染八情感，GT/B0/fullgradient/expressiongradient，无gain/retiming。如需因素诊断优先复用已有精确helper/固定66cells，不重复历史提取。
4. 再据表达和口型结果决定直接逐帧表达teacher目标或可靠内容载体/幅度结构；不把本轮梯度范围称完整D2，不一起堆交换/嘴adapter/多loss，也不退回1540/native路线。尚未達到SOTA。

SSH由remote_ops从私有旧helper AST读取连接，不输出/上传凭据；远端python为`/root/miniconda3/bin/python`。PowerShell不要猜文件通配路径，先rg --files；native teacher代码在neutral_affect.py，学生在slow_state_affect.py；collector state叫collection_state.json。

旧恢复全文逐字节保存在`archive/CURRENT_OPTIMIZATION_STATE_through_phase36_launch_20261007.md`，SHA 90da40b7ccb9b5d1d06c630a19746c07fa678cdca547fa2b73e06b2bad11575a。除需要历史具体细节外，不读这份长历史来重复分析。
