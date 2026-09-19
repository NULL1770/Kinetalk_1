# 口部残差候选：独立内部分割试验

实现文件：`scripts/train_mouth_residual_candidate.py`。这是修复 MBE/LBE 口部误差的候选试验，不自动替换任何默认模型，也不把系数指标通过写成口型同步或身份质量通过。

## 固定协议

- 读取 `continuous_latent_dataset.pt`；SHA256 必须匹配已有 `bounded_audio_experiment_v1` 的 `protocol.json`。
- 沿用该协议的 613 fit / 206 inner-validation 精确成员和顺序。外层历史 64 条数据的目标、特征和基座不会被索引。继承的数据集统计量完全丢弃。冻结基座/特征提取器本身仍继承既往训练暴露，因此这不是独立最终测试。
- 输入为原生 1540 维音频特征、冻结全局情感/身份上下文、冻结基座的 27 维口部系数。输入均值/方差只由 fit 片段计算；片段等权，片内原生有效帧等权，不读取目标或目标观测 mask 来拟合输入统计。
- 64 宽度、膨胀 1/2/4/8 的四层 TCN；输出零初始化，预测 `.15 * tanh(residual)`，不 clamp，保留原始值便于检查越界。只更新 ARKit 项目顺序的 `14:41` 共 27 个口部通道。其余 **25** 个通道精确保留；口部有效域外帧也精确保留。不能将这个分支描述成“保留 43 个通道”。
- 单一 raw coefficient MSE，先在每片的真实观测口部系数上平均，再等权平均片段。原始 `channel_mask & valid` 进入 loss；缺失值绝不填成 GT 零。训练抽样按片段等概率；输入 TCN 的 run 仅由原生 valid 和 25 FPS 连续时间决定，不由目标 mask 决定。
- 正序音频 / 同容量静态对照使用相同初始化、片段抽样、预算。静态对照将每个完整原生连续 run 的音频换成 run 均值，两个模型都保留动态冻结口部基座。因此这里检验的是**相对现有音频基座的额外有序声学收益**，不是音频与完全无音频对照。
- 默认种子 42、123、2026，每种子每臂 1500 steps，batch size 4，AdamW 3e-4。只报告固定最终步，不选择最好 seed/最佳 epoch；音频模型还用反序音频检查时序敏感性。原基座单独报告。

## 输出和验收边界

每个候选输出共享原生 `arkit_benchmark.json`，使用已审计的 FaceDiffuser ARKit 代码公式和区域，包含 MBE/LBE/FDD、补充 Lip23/Upper9。AV offset/confidence、Multimodality、FD/WInD 保持真实依赖未完成的 pending 状态，不用代理值冒充。

同时输出 `mouth_diagnostics.json`：Lip23/Mouth27 MAE、系数/秒速度误差、预测与 GT 的通道时间标准差及差值、原始越界比例。速度只在两帧均观测、原生均有效且时间相邻时计算，不跨时钟/有效域缺口。时间标准差是动态幅度退化诊断，不能证明动作时序正确。

该口部支路是确定性的；三种训练随机种子是模型重复，不是同一生成模型的随机样本。不能将跨训练种子的差别叫 Multimodality。需继续通过统一渲染/AV 同步检查和视觉检验；不能单凭 LBE 降低验收，不能把更平的嘴当作成功。

每 250 步原子写 checkpoint，定期原子写 `status.json`，最终保存全部预测和报告；当前 CLI 不自动续跑或覆盖已有输出目录。若程序停止可读取 `latest.pt`、日志判断状态，重试用新的输出路径。不会因为已有部分输出就误报 complete。

## 执行

```bash
python -m scripts.train_mouth_residual_candidate \
  --dataset /root/kinetalk_joint_20260918/continuous_latent_dataset.pt \
  --reference-protocol /path/to/bounded_audio_formal24000/protocol.json \
  --output /path/to/mouth_residual_v1 --device cuda
```

先用独立路径做 `--steps 2 --seeds 42 --width 16 --batch-size 2 --save-every 1` smoke。该 smoke 仍严格使用完整固定分割，只降低训练预算。正式结果仍需独立执行默认三种子预算。

测试文件：`tests/test_mouth_residual_candidate.py`。覆盖原始观测 mask、无 GT 统计/推理、固定分割排除 outer、时间缺口、25 通道精确保留、batch padding 独立性、静态对照、单目标等片段权重、原子输出与小规模完整 CPU 流程。
