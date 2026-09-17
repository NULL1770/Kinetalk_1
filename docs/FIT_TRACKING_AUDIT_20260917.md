# 固定 fit 原视频、眉眼提取与平滑核验

日期：2026-09-17。**这8条样本没有发现列映射错误或把30fps系数直接按25fps解释的证据；现有平滑也没有抹掉大部分眉眼幅度。** 这些结论是有限的上游排查，不等于 MediaPipe 已准确提取了真实、独立的眉眼运动。

## 范围与可复核材料

当前远程 renderer cache 的2315条 fit allowlist 已保存到 `artifacts/history_context_20260917/fit_allowlist.json`。先按元数据固定字典序最前两个 fit 身份 M003、M005，每人 neutral/angry/happy/sad 取 clip_id 最小一条，得到8条 `L1_001`，在读取目标和视频前保存 `tracking/selection.json`。未读取内部开发集、外部开发集或 test motion；没有改变数据、训练代码和提取器。

这8条共享同一句台词、均为 level 1，所以不代表整个训练集分布，不能据此估计其他人物、强度和句子的提取质量。

- [原视频眉眼裁剪与逐通道曲线](../artifacts/history_context_20260917/tracking/review.html)
- [64张固定原帧接触表](../artifacts/history_context_20260917/tracking/all_source_contact_sheet.jpg)
- [全部数值、时钟、来源哈希](../artifacts/history_context_20260917/tracking/summary.json)
- [审计脚本](../scripts/audit_fit_tracking_visuals.py)

每条视频固定抽8个均匀时刻，位于当前训练crop覆盖范围；用实际 fps-filter 源帧对应关系取得原始像素，不拟合 lag、不按峰值选帧、不改变动作幅度。原视频裁剪使用统一归一化矩形 `(x0=.30,y0=.31,x1=.67,y1=.64)`。每张图同时显示 raw 虚线与 final 实线，蓝色背景为实际训练crop；训练mask不支持的抽样时刻有额外文字标记。

## 当前训练数据与本地上游确实一致

当前实例下载的8个完整 native NPZ 均通过 `artifacts/formal_readiness/native_metadata/train.jsonl` 的 SHA256 核验。每个 native 的 `motion` 与本地 `D:/kinetalk_data/processed/coeffs_final/mead/` 对应 `coeffs` **逐位相同**。每个 native provenance 中 `raw_sha256` 与 `bs_sha256` 也与本次读取的 raw/final 文件一致。smooth NPY 与 final NPZ 的系数逐位相同。

native 的观测mask比原始检测mask更严格，还含声学/内容特征时间支持。因此下表同时给完整片段raw检测有效帧与当前96帧训练crop观测mask的结果，不把被mask的边缘帧算入训练证据。

## 30fps到25Hz的实际路径

原提取源码在 `D:/实验室项目/新实验/数据集/`。默认配置 `keep_normalized_video:false`；8条 `02_media.jsonl` 记录均为 `needs_resample:true,fps:25`。`03_extract_blendshapes.py` 使用原视频的真实时间戳经 FFmpeg `fps=25` 筛选/复制帧，然后以 `idx*40ms` 传给 MediaPipe VIDEO 模式。不是先提取30Hz系数，再给系数数组改一个25Hz标签。

本次使用历史配置仍存在的 FFmpeg 6.1.1，在 fps filter 前后插入 `showinfo`，用每帧checksum匹配源帧，保存全部输入/输出索引、PTS及日志。8条输出帧数均与 raw/native 系数长度相同，输出PTS严格为 `arange(T)/25`。原视频约30Hz，选中原帧的PTS相对25Hz样本时间范围为 **−6.72ms至+20.00ms**，是帧选择量化，不是随片段增长的累计漂移。不能把所有时刻默认成简单的最近帧取整；接触表已使用实际 fps filter 对应关系。

原提取没有保存逐帧PTS，本次是以现有同路径二进制复现采样算法，并不是恢复原进程的逐帧运行日志。此检查不证明 MediaPipe 在被丢帧、压缩、姿态和光照变化下仍能准确追踪。

## 平滑到底衰减了多少

历史日志和实际源码一致：检测失败帧先线性插值，然后 Savitzky–Golay `window=5,polyorder=2,axis=0,mode=interp`，最后裁剪到 `[0,1]`，不做逐说话人静息归零。本次对8条逐条复现，**与保存的final系数最大绝对差全部为0**。

下表先在每条有效观测内按通道减去均值，再合并平方能量计算 RMS 保留率。帧位移只采用相邻两帧均有效的差分；并不跨越缺失帧连接。

| 区域 | 全片动态RMS保留 | 全片帧位移RMS保留 | 实际训练crop动态RMS保留 | 实际训练crop帧位移RMS保留 |
|---|---:|---:|---:|---:|
| 眉，41–45 | 98.45% | 78.22% | 98.50% | 79.62% |
| 表情眼，5/6/12/13 | 98.21% | 80.22% | 98.15% | 80.03% |
| 眨眼，0/7 | 97.69% | 81.97% | 97.66% | 82.05% |

单片眉动态RMS保留率为97.05%–99.70%。因此平滑确实抑制了快速变化，但这8条中没有把眉毛变化压成静态的证据。该操作可能同时去掉噪声和快速真实动作，不能把位移下降全叫“去噪收益”；也没有据此建议直接改成未平滑监督。

## 通道映射核验

提取器按 MediaPipe 返回的 `category_name` 建立 gather 索引，不默认两种52维顺序相同。全部8条提取记录 `missing_channels` 都只有 `TongueOut`，未使用 positional fallback。我们用打乱顺序的类别名和已知索引值验证了实际映射函数，并核对当前 `render_dynamic_rig_comparison.py` 的全部52个通道名与上游一致。

| 训练ARKit索引 | 含义 | MediaPipe索引 |
|---|---|---|
| 41,42 | 左/右压眉 | 1,2 |
| 43,44,45 | 内眉/左外眉/右外眉上抬 | 3,4,5 |
| 5,12 | 左/右眯眼 | 19,20 |
| 6,13 | 左/右睁大 | 21,22 |

没有发现这一链路的列错位。映射正确只证明某个tracker输出写到了声明的通道，不能证明这个数值是准确的真实运动。原始数据也没有保存每帧的完整 MediaPipe 类别列表，不能仅靠本次源码检查排除所有历史版本问题。

## 视觉观察与仍未解决的问题

已查看全部64张接触表，并细看 M003 angry 与 M005 happy 的原帧/曲线图。原视频存在局部眉眼和头部变化；最终系数没有因 SG5 滤波变成完全静态。M005光照较暗，低对比度可能影响人眼观察和tracker，但本次没有测量其误差。

M003 angry 在约 **1.84秒**原帧闭眼，同时左右 `browDown` 出现明显下降；同片三个抬眉通道接近零。这是**同现观察**：可能包含真实面部协同运动，也可能含眨眼/头姿导致的提取耦合，不能把该系数峰谷直接当作独立眉毛动作的正确GT，也不能据此断言tracker出错。需要独立人工眉眼关键点/动作标注或不同提取器，在同一帧对照后才能判断。

此外，提取源码实际使用 MediaPipe VIDEO 跟踪模式，worker跨clip复用 landmarker，仅递增时间戳而不在每个clip重置实例。源码注释“逐帧独立推理”并不准确。跨片段状态的实际影响未测，本次没有重提取，也没有把它认定为失败原因。

**建议：保留当前锁定训练数据继续验证模型，不因猜测先整体去平滑或重映射。**如果最终音频动态仍弱，把独立人工/第二提取器的局部帧核验作为后续数据质量证据；现有“tracker数值有动态、映射和时钟自洽”不能替代“监督表达了真实、可从音频学习的眉眼动态”。
