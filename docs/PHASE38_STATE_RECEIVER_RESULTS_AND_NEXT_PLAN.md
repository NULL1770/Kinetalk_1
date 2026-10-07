# 冻结表达预测／接收诊断与下一步（2026-10-07）

Phase37三个seed全部不通过，不推广。Phase38只诊断seed47、66个metadata预选cell，没有训练，不计算小集合F1、不使用sealed。draw42原始生成严格重放，模型前后状态不变。完整8文件按远端SHA下载（精确数以download_verified为准）；报告SHA bd38f1100283cc9e5f293e0c08d683a7a2fcc35f76eb95ca996ebd8fe907b06d。

## 结果

| 四状态平均的逐clip描述 | raise | down | squint | wide |
|---|---:|---:|---:|---:|
| 学生／GT中心化RMS | .1296/.0404 | .1208/.1001 | .1303/.1721 | .1024/.2038 |
| 学生与GT中心化相关 | −.0192 | .2388 | −.0031 | .0633 |
| state总MSE／centered MSE | .5949/.0251 | 1.2870/.0369 | .6351/.0671 | 2.4937/.2288 |

表达预测既有平均偏差又有动态误差。上脸状态MSE主要由片段平均偏差贡献，不能仅看训练loss下降就说动态正确。

只把ua前4维替换成原生GT的表达目标；content/B0/h0/global/intensity/identity/prosody/noise全固定，剩56维零占位未变：

| 66cell clip描述 | 预测表达 | GT表达oracle | static GT表达oracle |
|---|---:|---:|---:|
| brow总MSE | .053922 | .039127 | .038956 |
| brow平均偏差MSE | .049822 | .035082 | .035019 |
| brow中心化相关 | −.01500 | −.02033 | −.01553 |
| jaw中心化相关 | .28626 | .28883 | .28863 |
| jaw速度MSE | .010520 | .010494 | .010426 |

真实状态可改善眉部平均偏差，动态状态与static oracle几乎同样；这支持接收器动态使用不足，而不是学生单一瓶颈。不证明ua包含音素泄漏。oracle绝不计作可部署改进。

Phase37身份诊断：异人code引起jaw/smile/brow响应RMS约 .0518/.0649/.0612，同人独立A/B参考的相应变化显著更小（jaw约.00785），bias只产生静态变化（raw验证）。有身份响应，不代表正确身份风格已证明；未用原人GT给异人打重建分。

## 下一项：Phase39表达状态逐帧调制接收器

单一结构变量：为renderer输出归一化token增加零初始化Linear4→384，预测每帧shift/scale（192维各一组）；仅读取named ua前4个表达状态。它直接调制已有完整52D速度头的隐藏表示；原prosody/content cross attention与global/identity机制保留。没有额外critic/ordinal/交换/loss，没有嘴部输出mask或上脸接管。

此方案的依据是oracle真实动态仍难以进入输出，不能声称它已解决预测误差或所有嘴时序。学生仍仅772D、原stateMSE/CE/prototype训练，flow/generated条件detach继续成立。动态表达能前向影响所有观察嘴通道；身份仍调已有完整renderer，内容仍由B0/h0进入。旧默认关闭，默认模型不更换。

必需检查：关闭时state keys/RNG/前向及原真实默认两update精确不变；开启零初值输出与关闭完全一致，共享参数相同；初始化不改变训练noise/sample/dropout流；启用后表达调制参数可接收renderer梯度，改变状态时动作有响应且嘴梯度有效；HuBERT不进入学生，flow/generated仍不训练学生，padding/frozen/finite和checkpoint config恢复准确。拒绝在latent ua上静默启用，必须明确expression-prosody模式。

pilot按47/48/49、各2轮、原全TRAIN/1367val×3draw、raw/clip/四probe/same-rig/配对CI/neutral与弱类gate。基线复用Phase37 named，另外保留Phase34 Full的总体质量门槛，不拿恢复37退步的相对改善冒充超过34或SOTA。先闭合Phase37备份并回收已SHA验证重复成员，确认实际预算再launch。失败不扫调制强度或延长失败预算。

更新：Phase39已实现，399passed/1skip与两真实2update正确性检查通过；固定三seed正式调度已启动。seed47完整评分/53成员SHA备份闭合，原128/64 F1 .635258/.572974、MBE .893344、嘴速度MSE .006432124；同seed Phase37 .638484/.572148、MBE .892900、嘴速度MSE .006441897，暂无明显收益，联合gate和Phase34门槛均失败。seed48正在运行，49待执行。保持固定预算闭合，不扩大失败项；全部完成后汇总和固定视频再决定下一单变量修正。最新实时入口仍为CURRENT_OPTIMIZATION_STATE.md。
