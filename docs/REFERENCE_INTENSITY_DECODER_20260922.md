# 独立参考强度解码器：正式训练完成，音频动态仍偏弱

## Material Passport

- Origin Skill / Mode：academic-research-suite / experiment-agent，run + descriptive validation。
- Date：2026-09-22。
- Verification Status：ANALYZED。已核对真实 TRAIN 诊断、代码、本地测试与正式训练产物；CUDA smoke 和 `formal20_30_10` 均已完成。CPU/GPU独立复放与四类视频已生成，保留数值差异，感知验收仍待完成。
- Scope：替换 upper9 生成路径的实验候选。未替换默认模型，未读取封存 test，未建立 MEDTalk 性能对等或论文成功结论。

## 1. 为什么修改路径

前一轮冻结先验 gain 路径不能稳定改善音频时序，且只能缩放已有动作。保留原 upper 均值、要求新分支严格中心化，还会让持续表情偏移无法由该分支表达。本轮保留已验证的音频口型和其他 43 通道，**释放 upper9 的均值与动作形状**，以独立中性参考作为身份相关坐标原点，由学习式解码器生成完整 upper9。

这个决定是对输出容量和监督定义的修正，不是已证明失败只有一个原因，也不是换一种名称重报旧成功。

真实诊断见 `artifacts/reference_decoder_20260922/target_diagnostic/evaluation.json` 和同目录 `summary.json`。共 4098 TRAIN query、44 独立中性 enrollment；没有载入 development/test 张量或渲染器。固定 ridge α=10、PCA24、12000 帧预算，只在 fit 格拟合特征统计、PCA 和回归器。

| TRAIN 内分格 | query / speaker | 当前 RMS 裁剪相对 RMSE | 韵律 ridge 中心化 R² | PCA24+韵律中心化 R² | PCA 预测/目标时变 std |
|---|---:|---:|---:|---:|---:|
| seen speaker / seen sentence（fit） | 2287 / 17 | 21.74% | −0.00634 | 0.00527 | 19.92% |
| seen speaker / new sentence | 954 / 17 | 22.32% | −0.00656 | 0.00053 | 20.40% |
| new speaker / seen sentence | 602 / 5 | 23.86% | −0.00760 | 0.00348 | 19.30% |
| new speaker / new sentence | 255 / 5 | 24.48% | −0.00340 | 0.00678 | 23.43% |

当前片内中心化 RMS 对同一段动作的标注随片长改变；独立 neutral-relative RMS 和 velocity RMS 在相同内区间精确不变。这只验证**裁剪一致性**，后两种目标没有训练预测器，不能称为更可预测。固定前四个元数据样本的图在 `target_crop_first4.png/pdf`，全部为 M003，不能代表所有身份或视觉质量。

韵律 ridge 四格都输给静态基线；PCA 有弱有序信号，但只能解释约 0.05%–0.68% 中心化时序方差。new/new 的 full 相关为 0.12431，mismatch 为 0.12303；相对静态的时序收益远小于部分原始 MSE 收益。未见 speaker 格没有一致变差，**本诊断不支持将失败主要归因于陌生身份**。不同格的数据组成不同，不能反推陌生身份有益；也不能用固定线性探针证明音频信息不存在。

## 2. 新路径与条件边界

```text
query audio → content768 + emotion2vec middle768 + prosody4（25 fps）
           → TRAIN-fit 标准化 / PCA24 + prosody4 → native-rate student → intensity[t,2]
query audio → 冻结音频网络 → static global emotion code
独立 neutral enrollment → 冻结身份编码 + raw-motion anchor9
intensity[t,2] + static global / identity / anchor9
           → 学习式 temporal decoder → bounded absolute upper9
冻结源输出的其他43通道 + 新 upper9 → ARKit52
```

“audio 条件”指 query 情感和时序条件来自音频；系统仍需独立中性参考，不能称为无参考生成。Query motion 只提供训练监督和离线评分，不参与部署条件构造；query 情感标签不进入 student/decoder。Global 是冻结音频网络输出，不是 GT 情感标签，也不使用 query motion teacher 提供的全局向量。

新强度定义是 brow5 / eye4 的**中性参考相对平均绝对偏差（MAD）**：先以 TRAIN-fit 通道尺度标准化 `upper9 − anchor9`，区域内取绝对值均值，再在有效连续段内做五帧滑动平均。它保留持续表情偏移，没有片内去均值；是系数幅度代理，不等价于感知情感强度或 MEDTalk EIE。诊断比较的是 RMS，本轮训练换成 MAD，数值不可直接横比。Oracle、student 监督和最终输出强度损失使用同一个 anchor、尺度、窗口与 MAD 定义。

Student 是原帧率轻量时序网络（dilation 1/2/4），以 softplus 输出两维非负控制；decoder 使用两层局部时间卷积（kernel 5/3）生成九维系数。Decoder 输出为 `sigmoid(logit(clamp(anchor,.01,.99)) + residual)`；这是学习路径内的边界映射，不是事后截断。没有 stride16，没有 prior upper 轨迹、query 动作、随机微动作或自回归生成历史输入。静态 intensity 在有效段边界也应生成静态 upper；常数输入测试防止将 padding 边界伪运动当成动态。

均值不再受旧 Stage4 upper 均值限制，持续抬眉等行为可通过新路径表达。代价是原上脸均值保护不再成立，必须单独检验均值误差、情感和身份感知。其他 43 通道与无效帧由原基线逐 bit 复制；这保护既有口型，不能证明原口型已经达到绝对质量标准。对称卷积与全片静态上下文使当前实现属于离线模型。

## 3. 三阶段预算与数据协议

冻结源：`/root/autodl-tmp/kinetalk_calibrated_20260920/run12/protected/audio/final.pt`。代码目录为 `/root/autodl-tmp/kinetalk_active_20260921/code`；正式运行输出为 `/root/autodl-tmp/kinetalk_active_20260921/reference_decoder_20260922/formal20_30_10`。

| 阶段 | 选择预算 | 更新参数 | 输入和目的 |
|---|---:|---|---|
| Oracle receiver | 20 epochs | decoder | GT 参考相对强度；检验最终九维输出能否执行控制 |
| Audio student | 30 epochs | student，冻结 decoder | 纯音频局部特征 + 固定全局/参考条件；学习同一定义的强度 |
| Joint adaptation | 10 epochs | student + decoder | 小学习率适应预测强度分布；允许选择 0 轮保留前阶段 |

三阶段沿 TRAIN 内同一 speaker×sentence 分折，fit=2287、calibration=255；两个混合格共 1556 条不参与选轮。所有新特征、PCA、通道、global/identity 标准化统计只在 fit 估计。校准指标为 normalized upper MSE + 0.25×output-intensity MSE + 0.1×normalized velocity MSE。Student/joint 训练另加 0.5×预测强度 MSE；按 clip 等权，mask 不跨越无效间隙。

选出三阶段轮次后，从新初始化在完整 4098 TRAIN 重训，重新计算全部 TRAIN 统计，使用已选轮次，不再以 development 选轮。完成后才载入 446 development query 及其独立 enrollment 做一次本轮全对照。新 loader 逐角色读取 shard：选择阶段不打开 development 文件，test role 被拒绝。Manifest、index、源 checkpoint 和实际载入 shard 的哈希均记入协议。

**留出只适用于新 student/decoder；冻结 Stage4 已训练过完整 TRAIN，不能宣称整个系统实现严格未见 speaker/sentence 评估。** Development 在项目历史中已反复使用，仍是开发集。一次新评估不会把它变成独立测试集。

## 4. 对照、验收与证据范围

Oracle 比较 real / static / reverse / zero，评分必须使用**解码后的 upper9 和重新提取的输出强度**。真实目标输入自相关为 1 不构成接收器成功，更不是音频成功。接收诊断报告 real 相对 static/reverse 的中心化 upper 和强度误差，以及 speaker/sentence 两种 cluster CI。本轮允许按固定预算完成 student 可行性实验，不因继续训练便宣布 oracle 通过。

最终音频对照为 full、static hidden、static predicted intensity、reverse、shuffle、mismatch，并保留 original prior 和 window5 smooth prior。两个 static 是同一模型的推理干预，**不是独立训练的 matched-static 消融**。Reverse/shuffle 操作预计算局部特征，不反转波形重新提特征。Mismatch 只从当前评分群体找非自身 donor，替换局部音频特征；接收样本的冻结 global/identity/reference 保持不变，因此不能当作完整音频错配。

必须分别报告原始、片均值和中心化 upper 误差、预测强度与最终输出强度、时变幅度/二阶差分、非 upper43 bit-exact，以及 full 对 static/reverse/shuffle/mismatch 的成对比较。上脸均值拟合改善不等于动态改善。学习式 sigmoid 可以避免 upper 输出越界，但无法替代时序证据、视频自然度、生成情感或身份评价。当前没有 MEDTalk 可比实验或主观自然度结论。

## 5. 工程验证与版本来源

本地新增相关测试 **79 passed（55 + 24）**；覆盖参数冻结和梯度、独立参考和角色隔离、fit-only 统计、NaN padding/间隙、静态边界、upper 组合保护、推理一致性等。真实 CUDA smoke 在 TRAIN fit/calibration 各 32 条样本上完成 2/2/1 轮，耗时 67.084 秒，输出在 `/root/autodl-tmp/kinetalk_active_20260921/reference_decoder_20260922/smoke`。它确认数据、接口和训练流程可运行，不能据此判断正式效果。Test 与 smoke 记录来自本轮执行回执；本文件不将其冒充正式训练复现。

| 来源 | SHA256 |
|---|---|
| frozen source checkpoint | `be975208b978309e7eff328fb07418e413e49a8fe7f76b9bac67c341083e0fbb` |
| data manifest | `6bdc33c3ee832e968b421781267dc70654e416b8e76395f2c6ce83c859e8603e` |
| real TRAIN diagnostic evaluation | `7fb89bba2b09e6dc993cbf35a0b8c0bbc32c636db1a60fe9d6b90c94787139d6` |
| executed diagnostic script | `80b4c034593712535c378859ff73b3501e245d2b0084fe7f59366f9c034c076c` |
| current diagnostic script | `c3b40ce587ba5a1c9822a9a21d7fdb91be8ac302c5d5b6a8acac7e3905f4648c` |
| local `scripts/train_reference_intensity_decoder.py` | `d0feb5a81e194282f4c18005d7bdb0b09d3ffaf9e3328055ae6bf7975320daab` |
| local `scripts/reference_decoder_data.py` | `fbf6abceab1f8fcf61b254262bca48b356451a097053c17dcb0dc5edb7a59fcc` |
| local `kinetalk_b0/models/reference_intensity_decoder.py` | `50b3b5bf1a906df1c3a1975010dd6abdcf36f633551a419fa39440dbcdb04616` |
| local `scripts/infer_reference_intensity_decoder.py` | `04af3fe8874b49ebebc992a719ee0fbe996f44771bd33a43dcfb716be79e654f` |

已逐字节核对正式 `protocol.json.source_files`：训练 runner、角色 loader 和 decoder 三份源码均与本地上述哈希一致。推理脚本哈希为本地读盘值。诊断运行脚本已下载为 `executed_diagnose_regional_targets.py`，字节哈希与其记录精确一致。它与当前诊断脚本只有元数据选例方式、相应 protocol 文本、增加 index 哈希字段三处差异；目标、分折、PCA/ridge 与评分代码一致。原 evaluation 未改写；旧 16 例仅覆盖四个 speaker，不能描述为最终多 speaker 轮询版本。

## 6. 正式结果：低幅音频候选，静态对照更优

`formal20_30_10` 已完成，耗时 **784.664 秒**。TRAIN 校准选择 oracle/student/joint 为 **13/28/0 轮**；joint 的 10 轮预算未改善选择指标，按预先允许的 0 轮回退。之后重新初始化，在全 TRAIN 用 13/28/0 轮 refit，完成全部 446 development 对照。没有替换默认模型，没有读取 test。以下均来自本地已下载真实产物，未修改原 JSON。

| 产物 | SHA256 |
|---|---|
| `formal20_30_10/final.pt` | `90472e2de36bf60ba815c1052a4325eb6b731d7cb422b617717b7f43e6cf1607` |
| `formal20_30_10/evaluation.json` | `c1531a8b55c362c2c69d3ad2c5b2f50c2bc6ae68aea1c84c2d059f1db0389d4d` |
| `formal20_30_10/protocol.json` | `455b4a0fa311f65d15ac39b98dd966a89d4e30df4c8454a73395cd7e8fa20b7c` |

255 条 TRAIN calibration 的 oracle 最终输出强度相关 **0.77487**、中心化 upper MSE **0.02822**；oracle-static 为 0.04157，oracle-reverse 为 0.06873。输出强度误差和中心化 upper 误差均优于 static/reverse，两种 cluster CI 支持接收门通过。这里检验的是 GT 强度能否被 decoder 执行；音频 student 必须另外通过真实音频对照。

以下 development 均按 clip 等权。Upper 列均用本轮 TRAIN 通道尺度归一化；强度为本轮 neutral-relative MAD，不能与旧 RMS 结果直接比较。

| 条件 | upper MSE | mean upper MSE | centered upper MSE | 输出强度 MSE | 输出强度相关 |
|---|---:|---:|---:|---:|---:|
| full | 0.547519 | 0.510292 | 0.037227 | 0.242930 | 0.06343 |
| static hidden | 0.548488 | 0.512594 | **0.035894** | 0.244033 | 未定义（常量） |
| static intensity | **0.546792** | 0.510898 | **0.035894** | 0.242869 | 未定义（常量） |
| reverse | 0.547240 | 0.509517 | 0.037723 | 0.243041 | −0.01515 |
| shuffle | 0.547574 | 0.511110 | 0.036464 | 0.243371 | −0.00296 |
| mismatch local audio | 0.551800 | 0.513691 | 0.038109 | 0.241193 | 0.00725 |
| original prior | 0.651656 | 0.557574 | 0.094083 | 0.157488 | 0.03622 |
| smooth prior | 0.609351 | 0.557574 | 0.051777 | 0.157740 | 0.04990 |

Full 原始 upper 系数 MSE 为 0.022826，original prior 为 0.024252；但这不构成动态成功。Full 相对 static hidden 的 mean error 下降 0.002302，centered error 反而增加 **0.001333**。后者 speaker cluster 95% CI=[0.000024, 0.001979]、sentence CI=[0.000754, 0.002128]，均支持静态更优。Full 相对 shuffle 的 centered error 也更差；相对 mismatch 的微弱 centered 改善（−0.000882，两类 CI 均为负）尚不足以抵消静态失败。Development 仅 3 个 speaker cluster，bootstrap 不构成广泛跨身份结论。

| 原始系数运动统计 | GT | original prior | smooth prior | full | static |
|---|---:|---:|---:|---:|---:|
| brow temporal std | 0.018998 | 0.023981 | 0.013653 | **0.004674** | 0 |
| eye temporal std | 0.019971 | 0.022962 | 0.012695 | **0.002310** | 0 |
| 二阶帧差分能量 | 0.000066857 | 0.004460306 | 0.000055222 | 0.000000996 | 0 |

Full 眉部/眼部幅度只有 GT 的约 **24.6% / 11.6%**。输出强度 std 为 0.015734，GT 为 0.083937；预测端强度相关也只有 0.04717。显著降抖同时伴随过小运动，不能将低加速度称为更自然。Full 互相关峰值绝对位移约 8.98 帧，只是弱相关下的峰值诊断，不是实测系统延迟，也不授权统一平移音频。

所有对照的 non-upper43 均逐 bit 沿用冻结 prior，口型通道保护成立；upper 均值按设计变化。Full upper 无 [0,1] 越界，不含事后 clamp 或平滑。冻结全局音频分类准确率 0.96413 只检验音频分类，不能证明生成脸的情感正确。

这轮真实证据表明：学习式 receiver 已能接收真实强度，但音频 student 仍输出弱变化，最终表现接近静态预测，**没有实现可靠音频时序动态**。暂不能据此断言瓶颈只在 student、数据或目标中的任一项。

## 7. 推理产物与复放

训练入口 `scripts/train_reference_intensity_decoder.py`，部署入口 `scripts/infer_reference_intensity_decoder.py`。本地正式产物位于 `artifacts/reference_decoder_20260922/formal20_30_10/`，包括 checkpoint、协议/评分、选轮历史、每种情感按 clip_id 排序第一条的 comparison/input NPZ。固定四例均来自 M025：neutral 111 帧、angry 92 帧、happy 93 帧、sad 124 帧，不是按指标选出的最佳片段。

```bash
python scripts/train_reference_intensity_decoder.py --data <prepared-dir> --source <frozen-stage4-final.pt> --output <fresh-dir> --device cuda --oracle-epochs 20 --student-epochs 30 --joint-epochs 10 --batch-size 32 --seed 53
python scripts/infer_reference_intensity_decoder.py --checkpoint <formal-dir/final.pt> --input <formal-dir/mead_M025_angry_L1_001_input.npz> --output <fresh-inference.npz> --device cpu
```

推理需要预计算 prior52、audio1540、音频 global64、独立 neutral identity128/anchor52、native times 和 valid；不是 raw WAV 到完整系统的入口。输入不得以 query motion/label 代替参考条件。输出包括 original_prior/smooth_prior/full/static 及哈希和保护报告。

CPU 独立推理已完成四例：原始及平滑 prior 逐 bit 一致；full 最大绝对差为 5.01275e−5，static 为 3.73125e−5，超过预先采用的 1e−5 数值阈值，因此 `replay/summary.json` 保留 `passed=false`，不事后放宽阈值。远端同 GPU 的单样本重放相对训练批量输出，full 最大差为 1.16676e−5、static 最大差为 2.98e−8。该隔离支持设备/批量数值差异的可能性，但尚未将每项误差归因，不能称为严格逐值复现。所有重放的 mouth27、nonupper43、无效帧均逐 bit 保护。

四例有声视频已生成于 `artifacts/reference_decoder_20260922/render_{angry,happy,neutral,sad}/comparison.mp4`。ffprobe 核实均为原生 25 fps、完整 92/93/111/124 帧并带音轨。五格顺序为 GT / original prior / smooth prior / full / static；显示 clamp 与原始曲线评分分开。视频生成与帧检查不构成情感、身份或主观自然度验收。完整 development `curves.pt` 保留在远端，本地保存报告、checkpoint、固定四例 NPZ 及视频。

本轮训练成功退出、可见上脸运动、音频时序有效、情感自然和论文贡献成立是不同结论，需由各自产物支持。
