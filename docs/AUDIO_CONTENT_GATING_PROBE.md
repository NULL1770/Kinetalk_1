# 音频与内容门控探针

目的：检验当前冻结 emotion2vec 帧特征是否需要同步 content 帧特征来预测动态；这是待验证假设，之前负 R² 不能直接证明缺少语义。

脚本：`scripts/probe_audio_content_gating.py`。保持 B0、identity、global emotion、motion teacher、renderer 的权重不变，仅训练独立小头：

`emotion2vec 768 → 64 → FiLM → 两个 depthwise 时序块 → scalar energy`

`content 768 → 16 → (gain, shift)`，通过 `h*(1+0.5*tanh(gain))+0.5*tanh(shift)` 调制同一个表示。FiLM 最后一层零初始化，因此与同 seed 的 audio-only 对照初始输出严格相同；没有新增 motion 区域通道。

- `--mode audio_only|film|content_only`：同 backbone 的三组对照。
- `--target upper_l1|energy_velocity`：默认上脸 L1，stride=4，每段去时间均值。energy_velocity 仅作可选诊断。
- 唯一 loss：动态目标 MSE；输入统计量、目标 RMS 只在训练句拟合。
- split 复用 `sentence_split(seed=45)`；固定最后一步报告，不按 heldout 挑 checkpoint。
- heldout 同时测 content 清零、有效帧反序、跨片交换；输出幅度变大不能算成功。

调用示例（与旧 probe 使用同一 config/data/checkpoint）：

```bash
python scripts/probe_audio_content_gating.py --config EFFECTIVE_CONFIG --data EMOTION2VEC_DATA --checkpoint AUDIO_PT --output NEW_OUTPUT --mode film --target upper_l1 --steps 1200 --seed 45
```

以相同参数分别跑 audio_only 和 content_only。输出保存 config、provenance、summary、预测曲线与小头权重。主模型训练前后做完整 state hash 检查。

通过条件：heldout R² 超过零和 matched audio-only，反序或交换 content 后正确性下降；再跨 seed/句子分组复验。通过前不接入主 renderer，不声称动态生成已解决或保证论文录用。

边界：data224 有 51 个句子标识；seed45 划分为 41 个训练句（190 段）和 10 个保留句（34 段）。反复使用同一 split 已成为开发集；需新句子/更多身份做最终检验。content 是已有音频内容表示，不等价于经过验证的文本语义标签。

数值审查后新增独立实验开关 `--readout linear`（去掉去均值之前的 tanh 饱和）与 `--audio-input clip_centered`（仅用该片段音频计算 DC 偏置）。二者不增加参数或 loss。`--split-seed 45` 可以固定句子划分同时改变模型训练 seed。旧默认保留以复现实验；完整日志还记录饱和率和 audio 干预，不能仅凭相关上升判成功。
