# 动态训练数据扩展：覆盖审计与执行方案

2026-09-17。目的：结束“四类、固定中央96帧、小范围重复优化”与实际native资产量之间的脱节。更多数据、八类或长上下文本身都不保证眉动态成功；先用可核验训练覆盖和独立保留集判断哪些变化有用。本文不改现有锁、不启动数据重分配、不读封存test目标、不改正在执行的训练协议hash。

## 1. 目前真正训练了多少

主审从当前2315-fit cache统计到 **198,465有效帧，25Hz下约2.205小时独特观测**；8epoch重复这些数据，不会变成17.6小时新数据。`native_clip`默认固定中央96帧，名义窗口3.84秒；padding和invalid帧不计有效时长。前序记录251,584帧若来自完整native metadata，不能与当前实际crop曝光混为一谈。

本地留存native **train metadata**（不读NPZ）按`valid_frames>=32`初审：

|来源|可用train段|说话人|不同句子ID|有效观测小时（25Hz）|类别覆盖|
|---|---:|---:|---:|---:|---|
|MEAD|12,959|22|164|15.763|完整8类|
|CREMA-D|5,797|72|12|4.026|6类，无contempt/surprise|

这些是**全native train库存**，包括当前开发身份/开发句等已有保留角色，不能直接全部变成fit。来源是`artifacts/formal_readiness/native_metadata/train.jsonl`；最终以本轮远端审计的输入hash和结果为准。

**远端实查补充**：本轮`artifacts/brow_review_20260917/data_coverage.json`确认上表数值。当前2315条完整native共有247,259有效帧，中央裁剪实际只用198,465，遗漏48,794有效帧，覆盖80.266%。仅排当前405的身份/句及现有enrollment metadata后，另有5,232条MEAD候选，其中fear892、surprise904、contempt932、disgust914；这仍未完整排除其他历史开发/预留，不能直接启动这些候选训练。CREMA-D同条件剩5797条，但还需域/未知强度契约。

MEAD库存里当前四类方案完全缺失的类别：contempt 1,707、disgust 1,776、fear 1,744、surprise 1,728，合计6,955段；各有22个train身份，L1/L2/L3均有样本。不能仅凭标签推断每段必有可靠眉峰，但fear/surprise等明显更值得纳入连续视觉真值抽检。现有四类方案还漏过中等级L2，不应把current high/low限定误作全数据条件。

CREMA-D有大量`intensity=-1`未知强度（例fear 779/986）；不得把-1映射成0或伪造等级。其6类任务与MEAD8类分开报分，录制方式、tracker噪声与系数幅度先审计。两库不同sentence ID格式不证明文本不同。

## 2. 可复现只读审计工具

`scripts/audit_dynamic_training_coverage.py`仅接受显式native-train JSONL、当前renderer cache及可选显式排除JSONL。cache以mmap加载，只读取train valid mask以及train/validation clip字符串；不索引motion/content/audio，不遍历native目录，不打开NPZ，不读取test目标。

示例（native路径由服务器实址确认）：

```bash
python scripts/audit_dynamic_training_coverage.py \
  --native-train /root/autodl-tmp/kinetalk_data/processed/native_affect_style_v4_refmask/train.jsonl \
  --cache /root/kinetalk_runs/teacher_schedule_v1/data_locked/renderer_cache.pt \
  --output /root/kinetalk_runs/dynamic_training_coverage_20260917.json
```

可重复追加`--exclude-metadata <enrollment或reserved或历史dev.jsonl>`；显式`--development-metadata`还会保守排除其中的身份。上述默认命令只排当前cache的开发clip/身份/句，**不认证候选可直接训练**。必须补齐现有enrollment、旧280/439/audit26等历史开发和reserved metadata后，才能做完整角色审查。

输出包括每库/每类/每等级/人/句/有效小时、当前fit真实crop小时、完整fit metadata与crop遗漏帧、新clip、相同fit身份/句的新录制、相对fit的新身份/句候选、neutral参考句数。hash记录输入与脚本，写后核验输入未变化；不写新manifest、不改lock。跨库新句、B0/global未见仍标未认证。工具已通过合成metadata的train-only拒绝、排除与crop计数检查。

## 3. 先扩大“可见时间”，再扩大任务覆盖

**固定人物/句/类的窗口对照**可先只用现有fit，避免把目标、人物和容量一起改变：固定中央96帧 vs 训练每epoch确定seed的随机连续窗口。预先生成或记录`epoch/clip/start/end`，每步对齐motion/content/多源audio和times，mask维持原生钟。参考identity窗口独立固定，不能从query的随机crop取参考。比较相同优化步预算，并额外报告独特有效帧覆盖；不要把更多观测偷当成纯“随机增强”效果。

缓存需要先改为完整native-clock sequence或按完整时间索引取slice；现有仅24 bin的中心化缓存不能移动到新窗口使用。原先按clip中心化和整段声学统计语义须明确：训练目标用当前窗口还是完整片段中心化，验证与推理须一致；不得把padding长度用于插值。窗口边界卷积不足可保留context halo再只计中心loss，并在训练集固定短/长样本检验边界不跳。

**完整连续上下文对照**：按长度bucket训练完整片段，或更长连续窗口；用真实mask和原生timestamps，保留B0按长度计算契约。推理全序列与切窗重叠拼接必须做同段对照，检查眉峰衔接、mouth时序以及切窗边界伪动。先在fit内验证速度/内存可行性，不以扩大帧数改变未声明的epoch定义。

## 4. 八类MEAD与跨库扩充的次序

先在现有19个fit身份内，按现有排除句/参考句规则加入MEAD8类及L2，保持3个开发身份的所有情绪都不入fit。这能增加表情事件覆盖而不破坏当前留身份开发。明确新的fit-only特征尺度/运动尺度需要重估，旧数据接口不能简单复用默认四类allowlist。global原有8类ID要保持0--7，不能把happy/sad重映射为2/3。

按人×类×强度报告数目与有效时长；采样是否平衡属于固定训练设计，避免大样本happy/mouth掩盖少数fear/surprise/brow。对每类按metadata预选连续样本核验GT眉动作、tracker/rig、动作峰与音频时间；质量失败的排除必须有独立、可复现规则，不按模型预测“挑好数据”。八类的实际测试范围也相应扩展，不能训练八类却仅报四类。

CREMA-D先作为单独域：核对52维channel可观测性、取值、FPS、原始视频与neutral参考充分性，保留dataset标识与未知强度mask。当前`neutral_data.native_clip`对所有intensity设valid=True不适用于未知-1；进入新数据路径前需明确支持unknown，而不修改历史已跑数据。建议先MEAD训练→CREMA独立验证；再固定协议比较合并训练，分别报域内/跨域，不能把混合训练后的CREMA成绩叫未见域泛化。

## 5. 独立验证与发表边界

当前405、旧280/439及被多次查看的audit集继续叫开发，不能改名为最终测试。数据扩展前按**完整历史曝光清单**区分新录制、动态模块未见身份、新句、身份+句联合未见；MEAD标注句ID跨情绪/身份需全局排除，同脚本的新录制不叫新文本。

封存test目标保持封存，先只用既有锁中metadata审查是否支持预定任务。不够形成平衡八类/新句联合测试时，应直接记录不足，另取新数据/公开独立集或重新采集；不能从已观察pool筛一个“新测试”。B0和global历史训练来源仍需核对manifest/checkpoint hash，最多声称动态模块未见，不能冒充整系统未见。

最后效果需同一rig、同一音频、固定样本展示GT/B0/来源/新模型，以及匹配zero训练、时序反转和多seed分布；眉/眼/嘴、类别/个人分开报。至少验证新增数据是否提高正确眉事件和合理分布，而不只是增幅或追平训练集。若当前输出目标的匹配试验仍不能利用正确audio，先解析该失败，再把八类/随机窗口当作**数据覆盖对照**，不要预言扩大十倍就会解决。
