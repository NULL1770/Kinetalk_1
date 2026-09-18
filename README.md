# KineTalk：中性身份与时序情感场实验

> **2026-09-18 当前状态**：固定时钟先验之后，四组活动条件的三种子验证也已完成。将目标简化成活动概率后，353开发集Brier仍轻微恶化约0.09%，三个种子均未胜过静态基线，未接生成器。默认模型未替换；保留既有身份、情感与口型基座不等于完整质量已经验收。
>
> 从[当前实验总览](docs/CURRENT_EXPERIMENT_STATUS_20260918.md)、[最新活动条件实测](docs/ACTIVITY_CONDITION_RESULTS_20260918.md)和[前序运动先验实测](docs/CLOCKED_PRIOR_RESULTS_20260918.md)开始阅读。最新验证入口为 `scripts/launch_clocked_motion_prior.py --activity`，见[活动条件协议](docs/ACTIVITY_CONDITION_PROTOCOL_20260918.md)；完整五阶段入口为 `scripts/train_full_staged.py`。这些是不同阶段的研究实验，不是已经验收的统一部署入口。
>
> Git 保存源代码、配置、测试与文字实验文档。数据、权重、曲线数组和对比视频留在本地/实验服务器，不随仓库发布；文档中的 `artifacts/` 路径指外部实验产物，单独克隆仓库不会包含它们。复现实验还需按对应协议准备指定数据和源 checkpoint。

## 历史实验记录（下文“当前／最新”均为各次记录时的状态）

> **2026-09-17 最新核验**：完整renderer容量三臂、直接声学flow/L1两臂，以及本轮新增最终位移+std的audio/zero两臂均已完成；后者各8epoch/1160步、72组曲线独立重建一致。眉幅度恢复但时序/分布与口型保护仍失败，默认不替换。参见[本轮综合结论与后续路线](docs/BROW_DYNAMICS_RESEARCH_AND_ACTION_20260917.md)、[新实验实测](docs/OUTPUT_MOTION_DYNAMICS_RESULTS_20260917.md)、[论文原文复核](docs/LITERATURE_DYNAMICS_RECHECK_20260917.md)。当前实际训练覆盖为2.205小时/19人/四类固定crop，不应等同完整native库存。

> **当前状态：正式2720段/22人训练与3训练seed＋PCA对照已完成（各18epoch），开发集保留弱动态收益；新身份眼部改善、眉部未通过，默认权重不替换。详见[正式结果](docs/FORMAL_TRAINING_RESULTS.md)。**
> 后续19人训练／3人内部验证的三种教师日程及闭环目标对照均已完成，仍未通过。误差分解证实平均表情纠正掩盖动态退化；本轮停止追加相同路线长训，见[教师日程与目标诊断](docs/TEACHER_SCHEDULE_PROBE.md)。
> 当前唯一主方案见 [中性身份基线与时序情感场架构](docs/ARCHITECTURE_NEUTRAL_AFFECT.md)：neutral 锚定身份执行基线、低率时序情感场、冻结 B0 与残差 Flow Matching。
> 初始化小试验入口 `scripts/train_neutral_affect_pilot.py`；当前正式接口训练入口 `scripts/train_formal_predictable_projection.py`，结果位于远程 `/root/kinetalk_runs/formal_predictable_v1`。原 B0 未改动。
> 下文仅描述已有 **v9 VA 历史代码**。`train.py` 仍是 v9，不能作为新架构训练入口；新试验不使用 VA/q。

初始小样本结果见 [小规模训练报告](docs/NEUTRAL_AFFECT_PILOT_RESULTS.md)；当前准入与正式结果见下方最新记录。

最新完成1200训练/280开发/12人的无文本音频与共享动态基对照，以及5组各600步生成适配。冻结audio预测器和renderer、只训练原共享接口；共享audio概率门控使run35两seed通过工程准入。正式入口为 `scripts/train_formal_predictable_projection.py`，协议见[正式训练准入](docs/FORMAL_TRAINING_READINESS.md)。默认模型不替换。见[实测记录第12–15节](docs/DYNAMIC_TRAINING_ADJUSTMENT_RESULTS.md)；此前上下文和音频对照在[音频特征优化结果](docs/AUDIO_DYNAMIC_CONTEXT_FEATURE_RESULTS.md)。

最新短对照：[去均值动态实测](docs/CENTERED_DYNAMIC_PROBE.md)，同预算8epoch上脸/眼部改善、眉毛仍未通过；默认未替换。后续小对照预算8epoch。论文指标核对见[原文比较](docs/DYNAMIC_PAPER_METRICS_COMPARISON.md)。

后续两个8epoch的[损失尺度对照](docs/SCALED_DYNAMIC_PROBE.md)已结束：上脸略增益，但嘴部退化且眉毛仍失败，未采用。[训练内动态基量纲诊断](docs/SCALED_MOTION_BASIS_PROBE.md)也已完成：眉motion投影变强但音频预测未改善，嘴部下降，未接入生成器。270tests通过；下一优先级是固定目标改善音频时序预测，无默认权重变更。

最新[音频时序修正](docs/TEMPORAL_AUDIO_REFINER_RESULTS.md)三折两臂各8epoch已完成：训练拟合改善但跨句眉/眼/嘴预测变差，未采用。固定8条源视频检查未发现整批对齐错误；284tests通过。保留原预测器及生成权重，不继续同路线长训。

## 现有 v9 VA 对照

原生音频内容驱动冻结 B0，**类别＋真实片段强度等级＋逐帧视觉 VA** 控制情感，同人跨句跨情感参考控制个人表达风格。生成器只接收明确语义的投影，不接收可自由复制动作的情感隐变量。BS 幅度不再充当情感强度标签。

```text
BS = frozen B0(target content)
   + ResidualDiT(semantic conditions, multi-reference style, noise)
reference residual = reference BS - frozen B0(reference content)
```

现有配置为 `configs/train.yaml`。第一轮实验使用 MEAD，先验证 neutral identity 与时序情感场，再加入 CREMA-D 跨库验证。

## 数据与环境

安装 `requirements.txt`。数据与 checkpoint 不进入 Git。

远程实验隔离在 `/root/autodl-tmp/kinetalk_semantic_v9_20260915`，原项目不覆盖。主配置使用：

| 输入／输出 | 路径 |
| --- | --- |
| Train / val 清单 | `pilot_data/train.jsonl`、`pilot_data/val.jsonl` |
| Native motion/content/audio | `/root/autodl-tmp/kinetalk_data/processed/native_affect_style_v4_refmask/clips/*.npz` |
| Visual VA | `pilot_data/va/*.npz` |
| Frozen B0 | `/root/autodl-tmp/kinetalk_b0_residual_train/outputs_protocol_v1/stage1_neutral.pt` |
| v9 训练产物 | `outputs/semantic_pilot/` |

原生资产包含 `motion[T,52]`、`content[T,768]`、`audio[T,83]`、`times[T]`、`mask[T]` 和 `channel_mask[52]`；音频为 mel80＋prosody3。视觉 sidecar 保存真实时间戳、VA、有效性、置信度和视觉来源。缺失标签、时钟错误、未知身份关系直接报错，不按长度重采样或补造情感。

## v9 历史运行方式（暂停）

以下仅记录旧 VA 接口，当前不执行。只有后续明确恢复 VA 对照实验，并核验源码一致性、完整 VA 标签与质量后，才可从隔离目录使用：

```bash
python train.py --config configs/train.yaml --stage preflight
python train.py --config configs/train.yaml --stage critics
python train.py --config configs/train.yaml --stage generator --checkpoint outputs/semantic_pilot/critics.pt
python train.py --config configs/train.yaml --stage evaluate --checkpoint outputs/semantic_pilot/generator.pt
python train.py --config configs/train.yaml --stage audio --checkpoint outputs/semantic_pilot/generator.pt
```

`pilot` 串行执行小规模流程；各阶段质量门槛仍需满足，不能把探索性运行当作正式验收：

```bash
python train.py --config configs/train.yaml --stage pilot
```

默认预算为 critics 400／generator 300／audio 300 steps，batch 8，cross_every 2，decode 4。这是小规模验证预算，不是完整收敛训练。

单独保存固定目标音频、完整情感和噪声的风格干预报告：

```bash
python -m scripts.diagnose_semantic --config configs/train.yaml --checkpoint outputs/semantic_pilot/generator.pt --split val --max-batches 2 --output outputs/semantic_pilot/fixed_condition.json
python -m pytest tests -q
```

诊断区分风格码变化、原始 BS、clamp、51 通道按名称导出及 CSV 回读，并记录同人参考变化与跨人变化。还须检查情感保持、donor 风格、发音时序和自然度。

## 文件分工

- `kinetalk_b0/models/semantic.py`：语义投影、动作／音频语义读出器、多参考风格与残差生成器。
- `kinetalk_b0/semantic_data.py`：原生时钟、身份、标签和独立参考契约。
- `kinetalk_b0/semantic_losses.py`：语义、风格对比与跨人输出约束。
- 训练入口只使用 `train.py`；视觉教师提取脚本已移出主方案。
- `scripts/diagnose_semantic.py`：v9 条件干预；`diagnose_legacy.py`：通过 `--legacy-root` 隔离审计旧 checkpoint。

旧 Stage2–4 的模型、导出和训练脚本只用于历史审计，不是 v9 主入口。请勿用旧 CSV 导出命令加载 v9 checkpoint。
