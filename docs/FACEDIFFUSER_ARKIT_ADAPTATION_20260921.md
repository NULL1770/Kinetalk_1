# FaceDiffuser ARKit 适配准备记录

更新：2026-09-21。状态：**模型、独立全量训练器与测试实现；此记录不代表GPU正式训练已启动或完成。**

## 来源与保留的模型

官方仓库：[uuembodiedsocialai/FaceDiffuser](https://github.com/uuembodiedsocialai/FaceDiffuser)，固定 commit `e15f3500fdae0eda962f5d018488dfa0a1a9d552`。本地 `third_party/facediffuser/provenance.json` 保存文件URL/SHA256，官方原始文件及 CC BY-NC 4.0 LICENSE 保留。适配文件标注来源、修改和许可，不表示官方背书。

`kinetalk_b0/models/facediffuser_arkit.py` 对齐官方 `FaceDiffBeat`：

- diffusion timestep 的 one-hot 经过 Linear + Mish；
- `[audio, noisy motion, timestep]` 拼接后 LayerNorm；
- 单向多层 GRU，官方默认两层，层间 dropout 0.3；Linear 预测原始系数 x0；
- cosine beta schedule，默认1000步，x0 MSE，FIXED_SMALL后验方差；
- 完整 ancestral DDPM，无跳步，不裁剪预测x0或输出。官方测试同样 `clip_denoised=False`。

单测直接从固定官方源码提取 `FaceDiffBeat`，仅替换HuBERT和帧率对齐步骤为缓存接口，相同权重/密集输入下核心输出逐位一致。cosine schedule也直接与官方函数比较。

## 必须在论文中声明的适配

这不是官方BEAT复现，应标为 **FaceDiffuser-ARKit (cached audio)**：

1. 使用MEAD批准的训练/开发身份及句子划分，25fps完整原生时钟和mask；不读取sealed test。
2. 模型模块支持通用缓存维度；正式runner已在启动前改为匹配全部音频信息：`raw content768 + normalized audio_features1540 = 2308`。后者按原数据准备定义为HuBERT768 + emotion2vec768 + prosody4，使用与KineTalk相同的完整TRAIN统计进行归一化，绝不在开发集重估。HuBERT内容以raw和normalized两种表示出现，因为KineTalk基础口型和状态分支原本分别接收这两路；没有新增目标信息。官方使用50Hz HuBERT、相邻两帧拼成1536D，还更新HuBERT第2层以后的参数。此匹配缓存版冻结编码器、特征维度/对齐方式和声学特征来源均不同，不能省略这些差异，不能称官方BEAT复现。
3. 外部接口ARKit52，默认内部固定51通道支持，tongueOut(index51)缺失通道不训练、不生成，置零仅是接口占位。任何额外缺失通道或帧均按真实Boolean mask清除，不根据finite值制造监督。
4. 内部缺失帧保留原始位置，零输入token进入GRU，递归仍按原生clock步进；不拼接缺失前后片段。尾部padding不进入有效输出/损失。
5. 损失先按每片有效系数取均值，再对片段等权平均。变长批处理不让长片段或padding改变片段权重。
6. 默认 `clip_condition_dim=0` 为官方BEAT风格音频输入。可显式设置正维数，将相同可推理的全局音频情感、身份参考码/中性参考向量拼入条件，模块对额外条件显式detach，防止更新冻结条件编码器；必须作为 **+ matched frozen reference/global conditions** 单独报告。模块没有查询GT、GT均值、标签或文本推理参数。
7. 此基线直接生成整脸，不复制KineTalk已修复口型、不使用KineTalk独有的眉眼后处理；否则会混淆模型比较。

`FaceDiffBeat`构造函数默认latent/GRU为256，官方main CLI默认512。适配构造函数保留类默认256；正式实验必须在运行配置中锁定尺寸、总参数量、预算，不能含糊称其为官方配置。

## 调用接口

```python
model = FaceDiffuserARKit(content_dim=768, latent_dim=256, gru_hidden=256,
                          num_layers=2, diffusion_steps=1000)
losses = model.training_losses(target52, timesteps, content, valid,
                              noise=noise52, channel_mask=channel_mask)
losses['loss'].mean().backward()
model.eval()
sample52 = model.sample(content, valid, initial_noise=noise52,
                        generator=generator, channel_mask=channel_mask)
```

训练时间索引为long `[B]`，建议统一均匀采样。噪声由调用者显式提供，生成全链使用调用者Generator。相同shape/模型/种子可复现；增加padding会改变随机生成器抽样布局，因此跨padding比较应显式对齐每步噪声，不声称整条随机链自动逐位相同。

## 已实现的独立训练器

`scripts/train_facediffuser_arkit.py` 不修改主队列。随机初始化整脸解码器，正式参数为512 latent / 512 GRU / 2层，Adam lr1e-4，不进行梯度裁剪，100轮、batch16、seed47。每片随机均匀采样diffusion step，前向1000-step cosine schedule；最后完整1000步生成，seeds42/123/2026。

clip condition按固定顺序拼接 `global64 + identity128 + independent-neutral-anchor52 + audio-intensity1 = 245` 维，均来自同一冻结条件源，全部detach。逐帧输入已匹配KineTalk可用的全部缓存音频：raw HuBERT content768及TRAIN归一化audio_features1540，合计2308维。显式保留所有输入信息不表示两模型处理架构相同；本版应标为 **FaceDiffuser-ARKit-cached + matched-all-audio/frozen-clip**。源checkpoint的生成器不参与基线预测，不复制口型。

正式模式拒绝4098 TRAIN / 446 development以外的覆盖；`--smoke`固定前4条训练/2条开发、1轮，仍用512维和1000步×3seeds检验真实开销。每个采样batch写开始/完成状态及耗时。`--resume`严格比对数据/配置/源代码/源checkpoint hash和last checkpoint，恢复CPU/CUDA/数据采样RNG；不存在静默减少DDPM步数。

输出 `protocol.json`、每轮记录、`last.pt`、`final.pt`、`native_curves.pt`、统一ARKit/动态报告及`evaluation.json`。输入布局和TRAIN统计hash写入protocol，统计张量随last/final保存，恢复时严格匹配。不修改模型类核心API，runner负责native时钟下拼接与mask；无效音频NaN先清除再归一化。没有KineTalk效果保护门槛，不把基线变好或变坏作为“训练成功”判断；正式开发结果只表示可供比较，视觉/独立情感/AV仍待验收。

CPU测试包括6项runner检查：完整1000-step×3seed流程（仅测试将hidden缩小）、模拟训练后中断恢复、禁止部分数据冒充全量、冻结条件不读查询GT、口型不复制B0、完整音频输入严格使用提供的TRAIN统计并忽略无效payload。模型的8项独立测试保留，合计14项通过。

```powershell
python -u scripts/train_facediffuser_arkit.py --data PREPARED --source FROZEN_AUDIO_FINAL --output FRESH_SMOKE --smoke
python -u scripts/train_facediffuser_arkit.py --data PREPARED --source FROZEN_AUDIO_FINAL --output FRESH_FORMAL --epochs 100
```

## 正式比较尚需完成

- 在真实GPU执行小规模完整优化与采样检查，再排入独立100轮预算；目前runner使用固定末轮，不按结果更换最佳种子。
- 静态/反序音频对照保持初始噪声及所有后验噪声一致。若使用额外逐帧条件，必须一起干预。
- runner已接入共同 `paper_generation_report.py` 和原始mask输出MBE/LBE/FDD、三种种子及完整本地动态诊断；实际结果仍须训练完成。AV/FD/WInD等没有合规评价器的条目保持pending。
- 不把此适配的结果与FaceDiffuser论文BEAT表直接排名；必须等同数据重训完成。
