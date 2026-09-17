# 音频上下文／特征实测（run06–09）

结论：本轮完成4组新增训练和固定噪声解码，**没有通过动态＋口型联合验收的新方案**。保留全部对照；不把更好分类或更低全脸MSE称作动态成功，不替换默认配置，不启动生成器联合适配。

## 对照设置

- run06：从run03继续800步，与run04同初始权重、minibatch、尺度，仅audio dilation从[1,2,4]改[1,4,16]。参数量不变，局部TCN跨度15→43帧；不是密集覆盖43帧。
- run07/08/09：固定同一motion teacher、B0、identity、DiT及场投影；audio从seed44重新初始化，各224条训练、1600步。输入分别为83D声学、768D缓存HuBERT、768D emotion2vec。所有同形状权重一致，后两组audio初始权重完全一致；minibatch/teacher尺度一致。83→768输入层增加65760参数，不能把全部差异只归于表征。
- 三噪声42/123/2026，从纯噪声Euler12步；full、真实local均值广播、reverse及global/local交换使用相同噪声。推理无query motion；teacher和交换条件仍是oracle。
- emotion2vec冻结官方`iic/emotion2vec_plus_base`，93,178,133参数；185个有效状态张量逐项匹配固定权重。FunASR构造但推理不用的预训练decoder被移除。16k完整音频→真实50Hz/768D帧特征→按卷积中心插值到native时间。252条训练/开发提取完成，18125个训练有效帧拟合统计；本批无需边界延拓。隔离venv复用原torch2.5.1，不修改原环境。

外部emotion2vec预训练／微调语料与学术情感语料的重叠尚未排除；小集100%类别准确率不代表无泄漏的跨数据集分类结论。

## 原28条开发集

| 模型 | full MSE↓ | mean MSE↓ | 眼时序相关↑ | 眉时序相关↑ | jaw相关↑ |
|---|---:|---:|---:|---:|---:|
| run04原上下文 | .010811 | .010884 | .07395 | .07275 | .38645 |
| run06长上下文 | .010963 | .010978 | .07703 | .07442 | .40682 |
| run07声学 | .010491 | .010686 | .06896 | .05690 | .40191 |
| run08 HuBERT | .011511 | .010857 | .04877 | .08750 | .35651 |
| run09 emotion2vec | .009853 | .009843 | .03380 | .02948 | .36077 |

长上下文与原上下文的逐片段MSE改善为−.000152，描述区间跨0。HuBERT相对声学为−.001038，clip/speaker/sentence重采样区间均低于0；幅度更大不等于时序更准。emotion2vec类别准确率96.4%（声学78.6%），但只有12/28条full优于自身mean；眉眼相关降低，不能按全脸MSE选择它。

## 锁定新句子：26条，只有6个不同句子编号

元数据按seed20260917选择，排除所有224训练、28开发、16 enrollment的句子编号。原28个单元中M003 angry L1/L3无剩余句子，明确保留26个可用单元；不是完整平衡测试。四人仍为已登记身份，B0预训练可能见过这些材料。预先指定run07与run09进行一次比较；此后这批数据不再叫未触碰测试集。

| 指标 | 声学run07 | emotion2vec run09 |
|---|---:|---:|
| clip类别准确率↑ | 84.6% | 100% |
| 强度等级准确率↑ | 80.8% | 69.2% |
| full MSE↓ | .010348 | .010435 |
| 静态mean MSE↓ | .010664 | .010430 |
| 眼时序相关↑ | .09319 | .05936 |
| 眉时序相关↑ | .03065 | .03054 |
| 眼动态幅度/GT | .53272 | .41579 |
| jaw相关↑ | .39077 | .37359 |
| 嘴部速度幅度相关↑ | .18576 | .15010 |

新集B0 jaw相关.50452、嘴部速度幅度相关.25322，均高于两组。emotion2vec相对声学逐片段MSE改善−.000115，clip/speaker/sentence区间均跨0；12/26条改善。其full相对自身mean为−.0000142，clip区间[−.0001269,.0000974]。声学full相对reverse有时序信号（19/26条改善），但full相对mean仅13/26、区间跨0，动态仍弱。三噪声先在片段内平均；四人、六句子的区间只能作为描述性敏感度，不能证明总体显著性。

## 含义与下一步

更强的语音情感分类不自动变成可预测动作轨迹。本轮反对“只换预训练特征即可解决动态”；尚不能区分teacher目标不可预测、容量/数据不足和teacher-only生成训练偏差各占多少。

下一个独立实验应先测**目标可预测性**：在训练集内跨句验证，比较原controls、时间平滑后的controls及可观测动作残差轨迹能否被声学预测，并分别扣除clip均值报告增益。若只是高频不可预测，再缩小动态监督带宽；若低频也失败，再改变teacher的训练目标。保持同一低率场、单DiT和既有flow/差分目标，不新增VA、逐帧自由向量、区域输出头或一串loss。emotion2vec更适合先作为global候选；把声学保留给local也只是待测假设，不能因本轮结果直接定版。renderer适配继续以可靠学生收益为前提。

本地77项测试通过。真实远程校验涵盖非audio张量不变、转换train缓存字节一致、audit source/hash链和冻结归一化。只看曲线，未完成渲染人脸的感知评估；本轮不能保证可发表或已完成解耦。

复现入口：`train_neutral_affect_task_ablation.py --audio-dilations 1 4 16`用于run06；`train_neutral_affect_feature_probe.py --audio-source acoustic|content --steps 1600 --seed 44`用于特征对照。emotion2vec先运行`extract_emotion2vec_pilot.py`，训练时`acoustic`表示读取转换缓存的`query.audio`，不是再读83D。评估必须使用对应run的`effective_config.yaml`，因为dilation不在state_dict里。新audit用`prepare_neutral_affect_audit.py --allow-incomplete-cells`；提取时`--reuse-feature-data`固定原train；评估显式传`--audit-selection`及emotion2vec的`--feature-source-data`，否则hash检查拒绝新数据。

产物：`artifacts/neutral_affect_pilot_20260916/context_feature_paired.json`、`emotion2vec_feature_comparison.json`、`run09_emotion2vec_probe_eval/`、`run07_acoustic_probe_audit26/`、`run09_emotion2vec_probe_audit26/`。官方权重/音频哈希、时钟和脚本版本在`data224_emotion2vec/emotion2vec_provenance.json`。训练checkpoint保留远程同名目录，未覆盖旧权重或删除数据。
