# Phase52 完整结果与论文材料状态

Material Passport: MODE=validate; STATUS=ANALYZED。全部1367条开发集、原生时钟、固定rig、raw/clip与四冻结probe。sealed未读；没有新训练或模型默认替换。

固定8epoch/5456updates训练与完整评估、2026组风格统计匹配均完成。56本地+56远端检查通过；原始53成员166814939bytes及197preflight成员均本地SHA核验。额外直接比较149个受保护权重张量/切片与Phase45-u父模型逐元素相等。

## 主指标

|模型|MBE↓|LBE↓|Lip mean mm↓|EVE-max mm↓|主clip F1↑|Jaw range|Jaw corr|
|---|---:|---:|---:|---:|---:|---:|---:|
|Phase47 latent|.825160|.383860|3.276868|2.518575|.719825|.151821|.489966|
|Phase51 temporal|.859197|.416775|3.451083|2.581025|.734619|.145985|.488077|
|Phase51 statistics|.809449|.393823|3.386572|2.357694|.712325|.126797|.468724|
|Phase52 style only|0.762598|0.355341|3.081770|2.351000|0.620855|0.121521|0.495885|
|EmoTalk-core适配版|.745003|.331531|2.973232|2.416746|.591884|.154294|.565796|

Phase52相对Phase47的MBE/LBE/Lip mean改善约7.58%/7.43%/5.95%。相对Phase51 statistics也改善几何与jaw相关性，但张口范围再缩小，四个probe全部下滑。当前不是联合SOTA；EmoTalk-core适配版和VOCA-core仍有更低的部分几何误差，数据/训练预算不完全匹配。EVE-max2.351低于本表适配基线，但单项开发结果不能写整体SOTA。MEDTalk/DESTalker没有核验训练行，不能填官方跨rig裸数值作对比。

Raw MBE/LBE/Lip mean=.763333/.355887/3.087444mm；raw四F1=.588789/.621786/.668169/.602628。Clip四F1=.620855/.626592/.690247/.629989。主GT probe F1=.660344；该probe为整段动作统计分类，不是逐帧情感真值。音频头F1=.880874且冻结不变。不能由F1单项宣布真实情感失败或成功，但多probe下滑和视频问题不能忽略。

Happy主F1=.900，disgust=.786；fear=.171，sad=.700，angry=.588。Fear GT主probe=.593，说明fear低分不能全部归因于GT判别器不可靠。动作生成端发生变化，audio prior/semantic heads和neutral B0保持相同。究竟静态style bias还是动态modulation改变情感线索，尚需隔离干预，不能由权重冻结推导已知因果。

GT平均jaw range=.175279，Phase52=.121521，为GT约69.33%。所有八类分组平均范围均偏小，sad约56%、happy约60%，fear约75%；固定样例某些时刻又过度张口。这是幅度及时间变化的误差，不适合统一乘gain。眉部centered correlation约.068；GT动作包含不可预测成分，相关低本身不等于所有表情不正确。

## 风格及局部动态

保持源音频/B0/g/u/rig，用独立neutral参考替换人物。2026有向sentence/emotion/intensity匹配只比较独立表演的native统计，绝非逐帧目标GT。嘴部mean朝donor改善87.36%，range78.97%，displacement81.93%；眉部mean约59.87%，仍弱。

Own A/B整体MAE为cross的37.15%，嘴部31.83%；比Phase48约76.4%更稳定，但整体不如Phase51 statistics29.21%。同人A/B闭口阈值分歧15.89%，跨人41.55%；这些是rig阈值诊断，不是音素错误率，但不能据此声称换风格严格保留全部闭口。平均lag接近零也不能排除个别偏移。人物style code cosine高不能替代输出稳定性验证。

固定96clip干预：normal jaw correlation=.469676，static_u=.425489，reverse_u=.397767，shuffle_u=.400556。支持u对原生变化有用；不证明它仅含情感动态、不拟合B0误差或具有因果内容解耦。wrong-matched-audio同时替换g/u且映射长度，仅负控，不能当主要native评分或u单变量证据。

## 论文材料可用范围

16视频完成并全帧解码、25fps、音频/输入/rig哈希核验。已检查全部16中帧以及happy/fear/angry/sad各5均匀固定时间点；视觉review收据单独保存，不改原collector closure。Happy笑容与换参考的动作差异可作定性材料；fear/sad/angry仍须保留失败及时间序列，不能只挑happy声称全面正确。

1. 八情感/风格视频、22方法/18候选开发表可用于当前实验记录与受控消融；正式论文主结果须统一一个最终checkpoint，不能用各阶段最佳指标拼成不存在的模型。
2. 风格可称reference-driven speaker motion style，不能声称脸形身份生成、完全身份/内容解耦或neutral参考能恢复全部未见情感习惯。
3. t-SNE/架构/首页图仍按用户要求暂缓，尚未输出论文文件。prior冻结，所以自身g空间的t-SNE不变，不能拿它证明Phase52生成质量；需要共享冻结GT/生成特征空间的诚实对照。
4. 外部方法×情感×同词的高清独立白底/透明图及核验词时间戳尚未导出。当前384px/tile视频是核验素材，不能称最终高清截图交付。
5. 多seed、最终未见测试、严格预算对齐及人评还没有本轮证据；当前3dev身份不能证明普遍泛化。

## 后续优先级

继续优化，首先解决情感表达与风格调制的兼容，而不是重训分类器或改变neutral B0。下一步对TRAIN/internal-held分别隔离参考静态bias、style modulation、global/local条件，定位geometry改善和fear/sad情感线索弱化来自哪里；不在dev拟合gain。之后只针对被证实的接收器/表达目标问题做受控修正，保持772D输入与学生梯度边界。此阶段未定义新架构、未启动新训练。

需要同时恢复强度/表情差异与嘴部动态幅度，保留风格收益并核验闭口/时序；先检查approved优质配对的表达目标是否混入neutral-B0误差，再决定修正目标或接收器，避免再次只靠限制训练范围/堆loss取得单指标改善。Phase52作为有意义的scope对照保留，未自动替代Phase47/51。

文档：[八情感及风格视频](PHASE52_VIDEO_GALLERY.md)、[完整表格](PHASE52_EXPERIMENT_TABLES.md)。原始本地路径：final_experiment/evaluation/diagnostics/phase52_style_scope_20261009。
