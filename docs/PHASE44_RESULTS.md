# Phase44 完整结果：固定教师后的均值匹配仍未解决幅度收缩

2026-10-08。三组各追加8轮、5,456次更新，均已完成全1,367条开发集评估；24个视频与12方法对比表已完成。训练与因素诊断状态均为complete，收尾时SSH GPU为1 MiB、0%利用率。三组均不替换默认模型：a_mean优于同预算a_kl，但没有联合超过父模型Phase43-A；SOTA目标尚未达到。

## 本轮改动与实际训练

冻结中性B0、motion posterior教师、参考style模块和decoder，只训练audio prior及原emotion/intensity heads。学生仍仅接收emotion2vec768+prosody4，没有HuBERT/口型内容入口、query GT输入或motion reconstruction梯度；全部嘴部输出通道保持开放。教师仍从动作获得latent，不能据此宣称教师目标已经纯表达、没有B0发音误差。

- a_kl：Phase43-A final起点，原KL匹配，固定教师与接收器。
- a_mean：相同起点和追加预算，改用decoder实际坐标中的均值匹配及token高斯标准差匹配。
- b_mean：Phase43-B final起点，使用相同新匹配，保留B的混合参考训练。

同10,903 fit clips、seed47、LR1e-4、固定8轮final。a_kl/a_mean为同父模型、同预算的匹配方式消融；b_mean为不同父模型候选。父模型本来已训练24轮/16,368更新，所以和父模型的比较不是等训练预算消融。没有增加网络模块、参数、pair数据、分类器或口型gain。

## 部署结果

统一audio prior mean、clip[0,1]，不包含query GT。jaw范围是每片段q90−q10后平均，GT=.175279；范围接近GT仍需结合相关和位移误差判断。

| 模型 | MBE↓ | LBE↓ | Lip mean mm↓ | Expression mean mm↓ | 主F1↑ | jaw范围 | jaw相关↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase43-A父模型 | .827621 | .388066 | 3.319447 | .736281 | .695182 | .142617 | .473212 |
| Phase44-a_kl | .830979 | .390528 | 3.288994 | .709988 | .649915 | .109889 | .483639 |
| Phase44-a_mean | .825540 | .393767 | 3.268096 | .694858 | .688865 | .111908 | .499695 |
| Phase43-B父模型 | .787836 | .358676 | 3.033152 | .718263 | .638624 | .111435 | .439372 |
| Phase44-b_mean | .796286 | .359408 | 3.046861 | .702664 | .599134 | .091355 | .468744 |

a_mean相对a_kl的主F1提高.03895，唇/表情顶点误差和jaw相关也改善，但LBE略差。相对父A，主F1下降.00632，张口范围下降约21.5%，GT范围短缺从18.6%扩大至36.2%。b_mean也未超父B。几何或相关的局部改善不能掩盖表达与幅度退化，因此不延长这三组训练。

| 模型 | 原128主probe | 原64 | 辅助128 | 辅助64 |
|---|---:|---:|---:|---:|
| Phase43-A | .695182 | .645552 | .722729 | .647927 |
| a_kl | .649915 | .573093 | .674771 | .568687 |
| a_mean | .688865 | .587781 | .708668 | .592489 |
| b_mean | .599134 | .554874 | .629714 | .567706 |

四个F1均为冻结的整段动作统计probe，不是逐帧情感动态准确率。a_mean辅助128的.708668不能替代主probe宣布F1达标；音频分类头.877547也不能当生成动作F1。GT四probe本身约.660344/.641175/.663951/.645885，说明分类器有误差，生成超过GT并不代表动作比GT正确。raw结果同样保留：a_mean MBE .826557、LBE .394441、Lip mean 3.276034mm、四F1 .604039/.542651/.634800/.547860。

a_mean主probe每类F1：neutral .3333、angry .4237、contempt .8187、disgust .7989、fear .6268、happy .9053、sad .8794、surprise .7248。happy表现较好不能代表八类都正确；neutral/angry明显拉低macro-F1。

## 定位到了什么，尚不能断言什么

在同一全1,367条开发集、固定权重下，交换全局g和局部u得到：

| 模型/条件 | MBE↓ | 主F1↑ | jaw相关↑ |
|---|---:|---:|---:|
| a_mean：audio g + audio u（部署） | .825540 | .688865 | .499695 |
| a_mean：teacher g + audio u | .417349 | .568802 | .500059 |
| a_mean：audio g + teacher u | .737836 | .746556 | .881652 |
| b_mean：audio g + teacher u | .244505 | .692071 | .873945 |

交换中的teacher读取query GT，且混合条件可能偏离训练分布；它们只是敏感性诊断，不作为部署成绩、可实现上限或因果独立证明。结果支持继续分别检查全局表达偏置和局部动态预测：a_mean眉部MSE约97.2%来自均值偏差，而教师u显著改善jaw相关。更强的音频情感分类并不自动解决这两个问题。

a_mean局部prior方差降至.384070，接近teacher .371816，触上限比例为0；父A prior方差5.19638、51.2%触上限。但输出幅度仍收缩，说明去掉大方差并未充分解决问题。A的decoder会去除u的时间均值，raw token u MSE包含decoder不使用的DC自由度，不能将6.887对父5.862直接解释成有效动态匹配退化；后续应在实际native/centered坐标分析。

固定96条干预中，a_mean jaw相关：正常.477747、static .422714、reverse .393871、shuffle .400926、同情感/强度/身份错语句音频.392390。时变u确实被使用，但不证明音素内容完全解耦。内部held-speaker prior位置误差从父A .471754改善为.441167，held-sentence却从.438263恶化为.473472；不能只看一条留出曲线宣布泛化改善。

## 外部方法与论文表格

| 方法 | MBE↓ | LBE↓ | Lip mean mm↓ | 主F1↑ |
|---|---:|---:|---:|---:|
| VOCA-core | .746703 | .332384 | 3.014416 | .497716 |
| FaceFormer | 1.296804 | .544334 | 4.537841 | .123846 |
| EmoTalk-core | .745003 | .331531 | 2.973232 | .591884 |
| FaceDiffuser | .792727 | .323110 | 3.071679 | .154482 |
| Phase43-A | .827621 | .388066 | 3.319447 | .695182 |
| Phase43-B | .787836 | .358676 | 3.033152 | .638624 |
| Phase44-a_mean | .825540 | .393767 | 3.268096 | .688865 |

上述基线是同rig/同开发集/相同指标实现的共享音频ARKit改编，不是原论文官方数字。基线12,536 fit、80或100轮、seed42，预算与我们不同；FaceDiffuser历史预测缺checkpoint哈希绑定的限制仍保留。MEDTalk/DESTalker没有已核验训练结果，不补造表格。主F1领先部分基线不足以宣称SOTA，几何和其他probe仍有差距。

[完整比较与消融表](PHASE44_EXPERIMENT_TABLES.md)包括raw/clip几何mean/max、MBE/LBE/FDD、动态/幅度/四probe、真实重训练消融、冻结干预、每情感/每身份、训练预算。CSV和JSON位于`final_experiment/paper_tables/phase44_development_20261008`。单seed、多轮开发探索，未读取sealed；这些不是最终独立测试结论。

## 视频与正确性归档

[三组共24个视频](PHASE44_VIDEO_GALLERY.md)，每组完整八情感，四列依次为GT / Neutral B0 / Audio prior mean / Posterior ORACLE。第三列才是部署输出，第四列使用GT。视频SHA、原生帧数/时钟、音轨、渲染值及全帧解码均通过。A固定1/4及3/4帧、B固定中间帧拼图已目视检查：图像正常，happy仍有笑容，angry/contempt等与GT有明显表情差异；静帧不证明整个视频时序正确。

每臂196个成员原SHA验证，另15个因素诊断文件备份。119个冻结state tensors（含归一化buffer）与父模型逐位相同，全1367 B0预测/指标/probe完全相同，oracle四probe及confusion完全相同。

初始collector对oracle浮点报告要求逐字相等而失败；已保留原failed状态、日志、锁和helper于`collection_recovery_v1`。只修本地收集验证器：oracle aggregate metric容差rtol=1e-6/atol=1e-8，全1367系数最大绝对容差5e-5，实测A两臂1.466274e-5、B 2.515316e-5。冻结权重和B0仍要求精确相等。没有修改模型、评分公式、原报告或checkpoint。恢复collector已complete，不要重启历史锁或launch脚本。

训练代码8f74658、收集修复前存档12dee08已推送。`closure_verified.json`记录完整成员复核、24视频SHA、报告来源与表格SHA；另有download_verified、frozen_output_verified、preflight_verified及factors_verified凭据。

## 下一步顺序

1. 先定位教师目标：在模型实际使用的native/centered坐标，分开全局均值、表达动态和B0残差，检查哪些教师变化能由772D音频在内部held-speaker/held-sentence上预测。配对诊断仅用manifest批准且对齐的数据，不将错误对齐当监督。
2. 根据证据决定是否修改教师表达空间与decoder职责。若教师u主要补发音误差，应修目标分工；若目标可预测而学生容量/接收失配，再比较对应结构。不能因oracle好就默认学生应复现全部GT运动。
3. 用独立内容读出检查情感变化是否改变音素时序，用同内容/情感、不同独立参考验证目标身份风格方向；参考有响应不等于身份迁移正确。通过后再做多seed及匹配预算基线。

上述是后续工作顺序，本轮没有实现或启动下一轮训练。当前保留Phase43-A作为表达参考、Phase43-B作为几何参考；不直接增加分类loss、不盲目延长epoch、不用统一口型gain追数值。
