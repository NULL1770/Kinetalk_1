# Phase47：简单均值接收器小幅改善，参考交互不泛化

2026-10-08。两候选均完成 TRAIN10903 解析监督拟合、原内部身份743/句子890留出检查、完整1367开发集、八情感视频。共16视频及19方法/15监督训练候选表已生成。没有读取sealed或替换默认模型。

## 同一原协议的部署结果

|模型|MBE↓|LBE↓|Lip mean mm↓|主F1↑|jaw范围|jaw相关↑|
|---|---:|---:|---:|---:|---:|---:|
|Phase45-u|.827171|.388674|3.320219|.709448|.153116|.490531|
|Phase47-latent|**.825160**|**.383860**|**3.276868**|**.719825**|.151821|.489966|
|Phase47-reference|.857411|.407056|3.476989|.554851|.151904|.489830|

主表为audio prior mean / 原clip[0,1]，不使用query GT。GT jaw范围.175279。latent的主F1增加.01038，MBE约下降.24%、LBE1.24%、Lip1.31%；jaw范围减小.85%、相关基本不变，固定阈值闭嘴F1从.394769升至.401578。属于有限的小幅收益，没有解决整体动态或达到联合SOTA。

四probe：latent .719825/.664336/.744721/.677342，对应父模型.709448/.663190/.729681/.668836。raw则为.659494/.603158/.676337/.628080，对应父.661798/.613819/.670570/.628775，并非全部改善。眉部mean-bias MSE .047571→.048268，略退化。原评分、clip/raw、四probe、GT均保留，不按probe改目标。

latent主probe分情感：neutral 0.460177、angry 0.680272、contempt 0.815190、disgust 0.821516、fear 0.511111、happy 0.934473、sad 0.870886、surprise 0.664975（精确值以原report为准）。弱neutral/fear仍未解决；总F1的提高不是每类都提高。GT原主probe本身.660344，音频情感头.880874，均不能替代生成动作/动态真实性评价。

reference虽在内部留出部分指标改善，外部三人物明显退化，拒绝采用。不由此宣称“身份模块无用”或确定因果原因；该分支只加入每人两段neutral的均值及预测情感交互，独立身份信息和有效泛化覆盖有限，额外容量未获得可靠收益。

## 本轮实际改动

原Phase45-u全部参数、B0、772D prior、motion posterior、style、decoder与语义heads冻结。只用TRAIN真实动作拟合原预测的逐片段均值误差，各clip等权、仅观察通道、原TRAIN scales。固定ridge=.001，不扫超参数，不训练情感probe。

- latent：已有g32+style64的仿射接收器，5044系数。
- reference：追加独立neutral残差均值52、预测情感概率8×均值52，共29380系数。是不同容量对照，不是等参数实验。

输出只加52D常量，无新逐帧gain/时间重映射/嘴部遮蔽；raw centered轨迹、位移和范围不变，clip后边界效果照原协议实测。student没有动作重建梯度、HuBERT/content或queryGT入口；身份不查表。g已含类别/强度监督，本轮未新添强度标签入口。

继承父24epochs/16368updates+一次10903的u解析拟合，本轮两个固定接收器各一次10903解析监督拟合、追加SGD0。不能称“未训练”。parent checkpoint与correction分别SHA绑定；原模型权重文件没有被冒充改动。

## 核验与视频

本地/远端31项相关检查以及两组16clip评价smoke完成。真实数据诊断中parent/B0逐位不变，HuBERT NaN隔离，raw位移最大数值差1.1920929e-7。原 evaluator 没有修改；新增适配器与真实部署API在no_grad下精确一致。全1367 B0预测/指标/probe精确相同，父oracle四probe精确相同；其系数因GPU布局存在最大1.4662743e-5浮点差，使用预先检查的rtol1e-5/atol2e-5，不将oracle纳入部署结果。

[16视频](PHASE47_VIDEO_GALLERY.md)，[完整表格](PHASE47_EXPERIMENT_TABLES.md)。第三列才是部署；第四列为父posterior oracle读取GT。16视频全帧解码、原生25fps、音轨、GT/B0时钟和rig值检查通过。固定中间帧视觉检查：latent没有渲染破损，但angry眉部张力不足、contempt不对称表达较弱、fear例子的张口仍偏离GT；静帧不能验证整个视频动态，也不因happy个例宣称全部成功。论文图尚未制作。

第一版连续TRAIN16 smoke覆盖不足，参考映射严重外推并触发位移数值检查；正式拟合没启动。失败root保留，改用覆盖八类的128smoke/两留出各32，未放宽检查。评估启动模板OUT替换误伤STDOUT造成SyntaxError，发生在worker启动前；已改独特占位符并compile验证，不影响训练/评价值。

## 下一步重点

1. 不再延长Phase46或给reference分支盲增容量。保留latent作为新开发候选，默认不自动换；对原始时序没有宣称新提升。
2. 优先检查教师表达目标是否把人物基础姿态、可预测表达和不可预测/B0误差混在g/u中。改全局/局部目标前，用manifest批准且对齐的neutral/emotional对，分开neutral参考相对表达和B0发音误差；学生仍只表达/韵律。
3. 身份应以独立参考人物的表达方向验证，不能仅以换参考有变化为结论。先测参考目标稳定性/内容关联，再决定是否采用显式姿态坐标或交叉重建；未验证的设计不画成已完成论文模块。
4. 主指标差距仍在：同rig EmoTalk-core适配的MBE .745003/LBE .331531/Lip2.973232，latent仍落后，且预算/数据不同。MEDTalk/DESTalker无已核验数值；多seed、匹配预算、独立内容读出、真实目标身份迁移、未见测试和人评仍缺。不能用当前F1、t-SNE或小幅均值改进承诺SOTA/发表。

## 绑定与恢复

pre-change474307e；实现2df2af5；smoke修复225119d；评估实现754b702均已推送。最新归档见CURRENT_OPTIMIZATION_STATE.md。

远端诊断 `/root/kinetalk_phase47_static_response_v2_20261008`、完整评价 `/root/autodl-tmp/kinetalk_phase47_static_eval_20261008` 均complete，GPU空闲。评价结束root/data空闲约291MiB/89MiB；下一轮必须先做存储安排，不删历史证据。

本地 `final_experiment/evaluation/diagnostics/phase47_static_response_20261008` 保存诊断、两组评价、源代码/独立correction、原SHA下载清单和视频收据。collector已complete，不重新dispatch/collector。31相关检查为30已有+1评估适配器；四个新静态模块测试包含于30中，不能把31与30重复相加。

最终closure_verified.json已写入诊断与论文表目录：204诊断下载成员、5评价root成员、两组各14下载成员及原始manifest均重哈希通过（各清单有重叠，不作为独立样本相加）；16视频与24表格来源校验通过。旧Phase46表中的全部历史指标/消融/预算/分组行精确不变，新两行的parent/correction独立SHA及继承24epoch/16368updates、一次父解析拟合/一次新解析拟合记录通过。表生成器语法与git diff --check通过；没有重复评价或渲染。
