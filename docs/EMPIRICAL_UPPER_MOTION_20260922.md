# 真实动作库增动基线：运动分布改善，局部音频时序仍待解决

## Material Passport

- Origin Skill / Mode：academic-research-suite / experiment-agent，run + descriptive validation。
- Date：2026-09-22。
- Verification Status：ANALYZED；真实 TRAIN 拟合/校准与 446 development 评分完成，保存供体可重建结果。不是独立 test 或主观自然度结论。
- Status：可运行的非参数生成候选，不替换默认模型，不作为已经成立的新贡献；MEDTalk parity 未建立。

## 1. 本轮实际改变

[参考强度解码器](REFERENCE_INTENSITY_DECODER_20260922.md)完成正式训练后，音频预测的眉眼变化仍明显收缩。继续以逐帧 MSE 训练同类局部预测器缺少支持。本轮冻结该确定性输出，在其 upper9 上叠加 TRAIN 真实动作的连续变化，测试运动分布是否可以恢复。

动作库对每个连续有效 run 做固定五帧三角平滑 `[1,2,3,2,1]/9`，边界按有效支持归一化，随后逐通道减 run 均值。采样保持原生 25 fps、九通道共同裁切，不改变极性、不变速、不拼接。供体必须足够长，并排除与 query 同 speaker **或**同 sentence 的记录。裁切后不重新去均值。

Conditional 以冻结音频 global64 的标准化均方距离加 0.25×独立中性参考 identity128 距离检索 top32 runs，再均匀抽样；Unconditional 在相同长度/排除池内均匀抽样。二者都保留 query 的确定性中心，因此“无条件”仅指新增动作的供体选择。每个 seed 选择一个供体及原生裁切位置，部署不读取 query motion 或情感标签。

有界组合为 `upper = center + room * tanh(temperature * donor_delta / max(room, eps))`，正向 room 为 `1-center`，负向为 `center`。温度 0 精确保留中心，其他 43 通道与无效帧逐 bit 复制冻结 prior。此变换可引起上脸均值变化，必须单独报告。真实中心化 motion **不是**统计上独立的预测残差；叠加到已时变的 center 可能重复计入方差，本轮是启发式增动基线。

这条路径延续历史真实 medoid/dictionary 探索，与本轮新确定性中心组合；不能声称首次提出运动先验、多 seed 或 ES/variogram。

## 2. 固定预算与可复现协议

- 来源固定为 run12/protected/audio/final.pt，SHA256 `be975208b978309e7eff328fb07418e413e49a8fe7f76b9bac67c341083e0fbb`。
- 参考 final SHA256 `90472e2de36bf60ba815c1052a4325eb6b731d7cb422b617717b7f43e6cf1607`。
- TRAIN fit2287 建库，cal255 比较温度 `{0,0.5,1}`；cal 已参与确定性模型选轮，所以这是探索性校准。冻结上游也已经见过完整 TRAIN。
- 选择规则事先固定：raw fair ES 不超过零温度的 1.005 倍，fair variogram 优于零温度；候选中取 centered fair ES 最低者，否则回退 0。选定温度为 **1**。自然度、情感正确性不由该规则认证。
- 全4098 TRAIN重新建库，然后对446 development固定评分。开发集已在历史实验使用，不能作为独立测试。没有读封存 test。
- 固定8 seeds：`42,123,2026,47,53,79,101,211`，全部评分，无best-of-K；视频固定第一颗seed42。所有供体/offset已保存，选轮bank与center保留在远端selection_replay.pt。
- ES使用完整九维轨迹的fair energy score，分别评价raw和run-centered轨迹；VS使用lag1/4/16/32、幂0.5并扣除有限样本均值方差项。评分尺度继承参考模型的TRAIN统计。

真实运行耗时 **100.695秒**。远端：`/root/autodl-tmp/kinetalk_active_20260921/empirical_motion_20260922`；本地：`artifacts/empirical_motion_20260922`。没有额外神经网络训练。

## 3. Development 结果

| 指标（ES/VS越低越好） | 确定性中心 | Conditional bank | Unconditional bank |
|---|---:|---:|---:|
| raw fair ES | 0.629341 | 0.551558 | 0.542911 |
| centered fair ES | 0.159092 | 0.120327 | 0.119180 |
| fair variogram | 0.050758 | 0.025273 | 0.026623 |
| raw upper MSE | 0.022826 | 0.023728 | 0.023797 |
| brow temporal std | 0.004674 | 0.018328 | 0.020022 |
| eye temporal std | 0.002310 | 0.020698 | 0.020897 |
| 相对中心的平均绝对片均值变化 | 0 | 0.003111 | 0.003991 |
| 无足够长供体的fallback clips | 0 | 0 | 0 |

GT同口径brow/eye temporal std为 **0.018998/0.019971**。Conditional的眉/眼幅度约为GT的96.5%/103.6%，消除了确定性输出明显欠动态的问题；这不是自然度等价证明。按抬眉/压眉/squint/wide四组的pooled归一化RMS比为1.099/0.897/1.094/**0.365**，wide仍不足，不能只用两区域均值掩盖通道差异。帧位移pooled归一化RMS为0.04544，GT为0.05804；模型并未因幅度恢复变成逐帧白噪声。

Conditional相对确定性中心的raw ES、centered ES、fair VS分别降低约12.4%、24.4%、50.2%。按speaker与sentence分别重采样的cluster CI均支持改善，但development只有**3个speaker**，speaker不确定性估计很有限。raw MSE增加约3.95%，中心化协方差距离也增加约6.58%；分布采样与唯一真值逐点误差、共变匹配存在取舍，不能只报告改善项。

条件检索没有体现综合优势：相对Unconditional，raw ES更差且两轴CI为正，centered ES稍差（speaker CI跨0）；fair VS略好且两轴CI为负。因此本轮支持“真实动作分布补充有效”，不支持“全局条件检索稳定更好”，更不支持局部音频与每次眉眼动作已对齐。冻结中心在M025上的均值偏差也没有因此得到解决。

每个query使用同一固定seed面板，top32查询大多取相同rank组合，8次draw可能重复供体。CI条件于该面板，不涵盖另选生成随机面板的全部不确定性。top-k按run而非unique clip选择；实际Conditional使用1127个供体clip/18个speaker，Unconditional使用1822/22。最终bank含4098 TRAIN clips/22个speaker/96句，无query重叠和同speaker/sentence排除违规。最长243帧query仅18候选/4个speaker，长度支持不均衡需保留。Conditional供体情感同类约91.82%（无条件25.62%），这是按clip元数据做的事后描述，标签未进入检索，也不能替代生成脸的情感评估。没有根据dev结果调整距离权重、top_k或温度。

## 4. 推理与视频

```powershell
python scripts/evaluate_empirical_upper_motion.py --data <prepared-dir> --source <frozen-stage4.pt> --reference-run <reference-run-dir> --output <fresh-dir> --device cuda
python scripts/infer_empirical_upper_motion.py --bank <empirical-dir/bank.pt> --checkpoint <reference-run/final.pt> --input <deployment-input.npz> --output <fresh-animation.npz> --speaker mead_M025 --sentence text_a6ba33f343d77c4eab734e7e --seed 42 --device cpu
```

输入沿用参考解码器的白名单NPZ：预计算prior52、audio1540、音频global64、中性identity128/anchor52、valid/times。这仍不是raw WAV到完整系统入口。CLI校验bank协议与参考checkpoint哈希，不读query GT/标签；speaker和sentence必须为精确元数据标识。缺失标识会报告无法执行对应排除；供体不足直接报错，而评分runner会明确回退确定性输出并计数。

CLI输出original_prior/smooth_prior/deterministic/empirical/unconditional及供体、seed、哈希和保护报告。正式五格视频则为GT/smooth_prior/deterministic/empirical/unconditional，固定neutral/angry/happy/sad各clip_id首条（均M025），位于 `artifacts/empirical_motion_20260922/render_<emotion>/comparison.mp4`。显示范围裁剪独立于raw统计，不改变已评分曲线。

本轮模型16项、CLI6项针对性测试通过，加上参考路径79项，共101项已执行。独立审计通过bank和全部供体offset重建四例×两模式×8 draws，ES/VS误差小于1.8e−10，seed42曲线误差小于1.5e−8，nonupper43逐bit保持。CPU完整CLI四例首seed供体全部与正式评分相同，empirical最大输出误差4.98891e−5，unconditional为4.96507e−5；保留预设1e−5数值门未通过。保护检查通过，不称严格逐值一致。最初使用M025短标识的复放也保留，精确mead_M025复放逐bit相同（两个标识均不在TRAIN供体中）；正式命令采用精确标识。

四类视频均已完成，ffprobe核实原生25fps、完整92/93/111/124帧且含音轨。已检查angry时间接触图的渲染状态，未将静帧检查冒充全视频主观自然度评测。

## 5. 当前可以与不可以得出的结论

现在有可复现的运动分布候选和可渲染的眉眼动态。它保留原嘴部输出，并显著改善相对静态倾向中心的运动分布评分。它仍依赖训练动作库，不是已经学成的局部音频动态模型，也没有完成生成情感/身份主观验证、独立测试或MEDTalk共同协议对比。继续研究应围绕这些缺口，不把检索或随机动态换名写成已成立的贡献。
