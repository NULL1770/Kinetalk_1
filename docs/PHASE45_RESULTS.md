# Phase45：动态均值校准完成，主F1首次达到.7

2026-10-08。完整1,367开发集；中性冻结B0，学生只emotion2vec768+prosody4、无HuBERT入口；没有query GT部署、无全动作重建梯度进入学生、未读sealed、未替换默认。

## 结果

|模型|MBE↓|LBE↓|Lip mean mm↓|主F1↑|jaw范围|jaw相关↑|
|---|---:|---:|---:|---:|---:|---:|
|Phase43-A父模型|.827621|.388066|3.319447|.695182|.142617|.473212|
|45-g|.812183|.384545|3.236552|.652318|.102990|.473340|
|45-u|.827171|.388674|3.320219|**.709448**|**.153116**|**.490531**|
|45-gu|.811533|.384389|3.231630|.670008|.112470|.491220|

GT平均范围.175279。u范围距GT的差从18.6%缩至12.6%，四动作probe为.709448/.663190/.729681/.668836，全部相对父A提高；主probe始终原128，没有换判定器。F1是生成动作整段统计，而不是音频分类头，也不是逐帧情感标签。

u改善表达和幅度、几何几乎持平；g/gu虽稍改善几何，但降低表达和幅度，不采用。MBE仍约.827，未达用户期望.7；只是单seed开发候选，不宣称全面SOTA/论文验证完成。默认未自动替换，Phase45-u作为下轮表达参考。

主probe分情感：neutral0.468468、angry0.689189、contempt0.773270、disgust0.832918、fear0.477612、happy0.934097、sad0.844687、surprise0.655340；以原report精确值为准。fear/neutral仍弱，不能只凭happy宣称八类都好。

固定96条动态诊断：normal jawcorr.470822，static.423373，reverse.398554，shuffle.398073；u对时序有可测影响，但不是独立音素或因果解耦证据。

## 实际改动与训练

只在Phase43-A已有head中替换local mean的128×16=2048个权重（u）；g候选替换32×129=4128个全局mean系数，gu共6176。没有加层/网络/损失，保留local bias以及所有方差行、prior骨干、语义head、教师、style、decoder和B0。D/S/Q均未重新训练。

来自一次固定ridge=.001 TRAIN10903clip等权解析监督fit；身份留出743、句子留出890用于诊断，外部1367只评价。不是训练免费或SGD等预算；父模型24epochs/16368updates，追加SGD0，另报解析fit10903×1。方差未拟合，采样质量未验证；部署仍prior mean。见PHASE45_TARGET_DIAGNOSIS.md和PHASE45_PLAN.md。

## 核验与视频

28项针对性测试本地及服务器通过；最长真实TRAIN与短clip混合GPU batch的native均值等价、variance逐值相等、HuBERT NaN隔离、B0 digest和保存恢复均通过。三组16clip评价通过后完整1367，约4分钟完成。下载后只允许mean指定行变化，其余张量/行逐位一致；B0全1367预测/指标/probe精确相同，oracle采用既定浮点容差且probe精确相同，未改原评分。

三组各192原SHA成员；audit182、pipeline5；24视频全部原生25fps、全帧解码、有音轨、GT/B0/时钟逐值一致、rig误差0，closure重新哈希。每候选八情感均给出；第三列Audio才是部署，第四列Posterior_ORACLE使用query GT。

已查看u八类视频的25%/50%/75%截图：happy笑容可见，neutral相对克制；angry张口/嘴部张力仍比GT弱，fear/surprise例子存在口型时间差。截图仅定性，不能证明所有动态正确或身份迁移正确。

[24视频入口](PHASE45_VIDEO_GALLERY.md)，[15方法表/11个真实监督训练行](PHASE45_EXPERIMENT_TABLES.md)。外部四方法仍为同rig共享音频ARKit改编，不是原论文官方成绩；单seed/预算差异保留。MEDTalk/DESTalker无核验训练数值，不补造。

## 后续优先级

先保留较好的u，定位g目标与嘴部平均幅度的错配及fear/neutral混淆。可优先考虑固定prior/teacher，只让已有decoder在真实部署mean条件上适配，用原目标而非增加loss；必须有冻结对照和幅度/语义保持验收，尚未实施/启动。不能直接堆loss或再延长Phase44。

独立内容读出、目标身份迁移而非仅“换参考会变”、多训练seed、匹配预算外部基线、公开benchmark/未见测试和人评仍为论文缺口。未做因果解耦或统计显著性声明。

## 原始绑定

- g: checkpoint `959a81ac84dae93603147b45bfb980912808d20765628eb1b221463a41f206e5`; report `ccaf60154f41e332711e3d8abeece933ef689f0256264f79701b94262e63c664`.
- u: checkpoint `dfaca4c0dce60cf8380c08788dabf547db7433c098a599522f8ce2286851e4eb`; report `73d195ad4e45d0f06340376782cea497bb58c06101877bb05a4f7067ac6bf1d6`.
- gu: checkpoint `2923782dbd7d54857190ec2e2d14b142166aa9c0bc675f6b813dd4a93ba51fc6`; report `5edb0c5b338ba9fa8732484845cfa5045746372e1a61819e74087979a44ac253`.

closure_verified.json记录成员与视频/表格源复核。远端/root/kinetalk_phase45_target_audit_20261008/head_candidates完整关闭；本地同名diagnostics/head_candidates完整关闭，不重新启动pipeline/collector。
