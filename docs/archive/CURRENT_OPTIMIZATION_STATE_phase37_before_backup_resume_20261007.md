# 当前恢复入口：2026-10-07 Phase37

## 最高优先级事实

用户再次授权继续正确架构优化。保持中性B0；学生772D=emotion2vec768+prosody4，HuBERT只直接进B0/h0；全部嘴通道前向开放。不堆critic/交换/ordinal，不用Phase26 native-GT/1540成绩。Git暂停，sealed test未读，默认未替换，SOTA尚未达成。不创建子agent。

Phase35/36训练、统一评分、完整SHA备份和Phase36八视频/因素诊断均已闭合，全部三seed联合gate失败；不再wait/重跑。Phase35 collector90557、Phase36 collector11565/localfinish9275/factorscollector75621全部exit0。

## 唯一当前工作：Phase37明确表达／韵律学生

先读PHASE37_NAMED_EXPRESSION_PROSODY_PLAN.md。本地实现完成，全suite393passed/1skipped；最后元数据修正后17相关回归passed。两真实2update smoke已通过：默认state/actualstream精确重放Phase34，新学生flow/generated梯度0、statehead有梯度/localhead不用、fullmouth renderer有梯度、HuBERT NaN隔离、模式恢复逐位一致。正式训练ACTIVE，还没有新性能或默认推广。

- 新`--audio-supervision expression-prosody`；默认仍flow/latent。
- ua64=4有符号上脸表达状态+4标准化原生韵律+56zeros；使用已有state_head4，旧local_head64保留state keys但候选不作为条件。
- 4状态由自身原生眉眼GT、独立中性anchors、既有TRAIN scales和stride16固定spline监督。不读嘴GT，不广播clip情感标签。
- 韵律真实列1536:1540：logF0无声0、logRMS、periodicity、voiced，TRAIN声学mean/std标准化。
- global64改为TRAIN情感×强度8×4原型，缺失cell count0；同标签条件目标严格同一，不再逐clip全残差蒸馏。原型教师仍看全残差，不能声称完全解耦。
- renderer条件detach，使full flow/generated-semantic不训练学生；学生表达stateMSE1+原semanticCE.1+globalMSE.5；renderer仍原完整目标/generated.2、全嘴开放。
- 只有trainer/slow_state_affect两个source变，无新神经层。config记录布局/stride，恢复必须SlowStateAffect.from_checkpoint；named checkpoint不能静默warm默认flow。

唯一远端root `/root/kinetalk_phase37_named_expression_prosody_20261007`，本地diagnostics同名（不含kinetalk_）。正式driver22623/worker22624–22626/closure22627，三seed47/48/49×2轮1568update；基线复用Phase34 standardized full。唯一collector session50441，唯一local readout/eightvideo finish session42620；别再启动writer。binding SHA589ba6010e374f9f503cebbcf0d1323e199557476a008b82e1787da48875b367。状态只读phase37_status.py/latest_status.json。auditor验证两个source差异/audio_supervision差异，gate公式未变。

独立named因素诊断也已排队：唯一driver23079，root `/root/kinetalk_phase37_factors_20261007`，唯一collector35609。只等main完整评分后才用seed47/66cells；3draw精确原重放后，whole ua及分别expression4/prosody4的static/reverse/shuffle，独立身份参考/code/bias干预。config-aware from_checkpoint恢复，权重/固定条件检查，不做训练/小样本F1/异人GT重建评分。新helperSHA5ea0dce…，不重跑Phase36full因素对照。

真实smoke要求：默认全state和actualstreams精确重放Phase34；named初始system/audio/sharedsource相同、physical初始velocity=TRAINmean（不要求named全部loss相同）；实际flow/generated→学生grad0，statehead非零/localhead未用，嘴renderergrad非零，HuBERTNaN隔离，padding/frozen/finite，保存named配置后逐位重放。

容量先闭合：服务器原free462045184bytes/GPU空闲，无活动旧实验。9个已完整备份成员逐个本地+远端SHA复核后unlink，共1805119840bytes：Phase36三seed expression last/curves+两smoke last；Phase35 seed49physical curve。保留全部final及Phase34三baselinecurves。receipt见Phase37 diagnostics/reclaim_receipt.json及远端Phase36/phase37_reclaim_receipt.json。清理后free2267176960bytes，扣新smoke后约1.885GiB；实际正式预算1907078782bytes（含三prototype、384MiBreserve）/floor1.8GiB通过才launch。训练后下降是已预算输出，不误报启动失败。

## 已闭合结果

| clip三seed/三draw | Phase34 standardized/full | Phase35 physical | Phase36 expression9 |
|---|---:|---:|---:|
| 原生成F1 128/64 | .683872/.581147 | .229124/.266488 | .637154/.545352 |
| MBE/LBE | .904603/.446154 | .915915/.457559 | .911180/.452668 |
| Lip mean mm | 4.008668 | 4.020528 | 4.055325 |
| FDD mm² | 153.447688 | 135.782384 | 154.460142 |
| mouth velocity MSE | .003709471 | 退步 | .004778430 |

Phase34只有seed49gate通过，整体未接纳；35/36全三gatefalse，拒绝不扩预算。Phase36嘴速度误差+28.8%。完整备份：34=476文件/manifest60be9696…；35=383文件1781631489bytes/46a51bfb…；36=387文件1782721197bytes/701d04c6…。

固定八视频：`final_experiment/evaluation/rendered/phase36_expression_gradients_{emotion}_s47_20261007/comparison.mp4`；M025/005、s47/draw42，GT/B0/FullGradient772/ExpressionGradient772，无gain/retiming。尚未在app展示新批。

Phase36因素root `/root/kinetalk_phase36_factors_20261007`，66cells两臂三draw原重放严格0、权重不变，16文件已SHA备份。static/reverse仍改善jaw，但static压眉/smile动态；不作为泄漏因果证明或全量成绩。

## 数据与历史边界

B0用715原生neutral+2583批准emotion→neutral；实际嘴监督55.375%，eventmask精确0/1，297full gate也受局部mask。见EXPRESSION_TARGET_AUDIT_20261007.md，仅v4/schema2/jawOpen17有效。469强度三元组jaw严格递增49.89%，不加统一ordinal。

12536TRAIN/22speaker、1367validation/3speaker；身份各2独立neutral参考。B0四TCN+2encoder/3decoderTransformer；teacher三maskedTCN；student四maskedTCN；renderer四192D/6head DiT。身份code128/bias52调运动风格，固定rig不改脸形。阶段职责见CURRENT_MODEL_AND_TRAINING.md。

GT原128probeF1=.66034433；统计motion F1不是逐帧分类器，也非唯一论文指标。MEDTalk/EmoTalk原评估已核；DESTalker全文未核，勿猜。Phase34/36的t-SNE只描述，不能证明解耦或SOTA。默认发布权重仍phase2_fullmouth_timing000_20261004，不用历史native表当新成果。

已删历史重复成员均有本地原SHA/receipt：Phase34首5+后4；Phase35前5+本次49curve；Phase36本次8。历史精确重放须先恢复对应原member，不能假定服务器全路径仍完整。

## 下一步

1. 已完成freeze/smoke/capacity，等唯一三seed训练；不重复launch或改绑定源码。
2. 等唯一closure，同1367×3draw/raw+clip/4probe/same-rig/pairedCI/80neutral gate；失败只修报告具体原因，保留原证据，不重训。
3. 等collector50441全SHA备份/localfinish42620固定八视频；已经安排，不启动第二个writer。
4. 按全部三seed联合gate判断；失败不自动延长。用户目标仍全指标SOTA及正确口型/表达，不承诺未实现改善。

SSH由remote_ops从私有helper AST读连接，不输出/上传凭据。远端Python /root/miniconda3/bin/python；SFTP不会创建父目录。文件先rg --files，Windows文本显式UTF8。所有训练/helper状态唯一，避免重复writer。

历史完整恢复页见archive/CURRENT_OPTIMIZATION_STATE_phase37_before_real_smoke_20261007.md，以及更早through_phase36_launch归档；除具体缺失历史证据外不重读。
