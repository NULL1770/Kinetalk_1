# Phase32：清理冗余并迁移中性B0上的772D学生（2026-10-07）

用户授权：暂停Git上传，先删除冗余，再继续任务。保持B0中性，不新增loss或D2/D3/D4结构。

## 清理完成

- 已完成旧`/root/kinetalk_temporal_source_20261005`的619文件归档；原始4,729,137,497字节（4.404GiB），压缩约2.235GiB。
- 压缩包及每个原始成员在服务器和本地均核SHA，本地持久备份`final_experiment/remote_archives/20261007/kinetalk_temporal_source_20261005.tar.zst`，manifest及verified收据同目录。
- 本地验证完成后才删除原服务器run，并留下`ARCHIVED.json`恢复指针；当前数据、默认中性起点和唯一备份未动。root空闲从.523到4.926GiB。
- 本地4个传输range临时副本已删除，保留唯一持久archive；递归删除被自动审核阻止后，改为4个显式文件逐个删除并重新核主archiveSHA，成功。
- 失败保留：首次pack解析ps表头失败，未写/删原文件；v2使用headerless ps。原单连接SFTP慢传保留prefix后由4个独立range合并成功，没有重下已传prefix。

## 输入与预算锁定

远端根`/root/kinetalk_phase32_neutral772_20261007`。冻结源码150文件，bindingSHA`ed67312394fcba26570bd845f46f674ae85f48c61ca5ae64f3af9accf1f6b123`。

- 两臂同一完整中性起点`phase2_fullmouth_timing000_20261004/audio/final.pt`，SHA`e17659536a6fcaaaec2d9c22f99403fe4d9f1c8690c3cd182583894545f5ab45`。
- B0 lineage与Phase31中性监督模型一致；冻结其配套identity_encoder/bias和motion_teacher，均保留原残差坐标；只更新audio student与renderer。不开mouth mask、不清renderer头，不混入native模型或其TRAIN统计。
- legacy输入1540；候选emotion2vec768+prosody4＝772，从同一TRAIN moments截取最后772维、input.weight删除前768列；其余初始state逐位相同。
- 原标准高斯source、legacy scalar flow坐标；全部loss不变。3seed47/48/49，每臂2轮/1568update、12536TRAIN/轮、batch16、12Euler，final固定；full1367validation、draw42/123/2026，raw/clip都报，4个既有冻结TRAIN probes。
- 三seed并行、每seed按legacy→affect772顺序，各2CPUthreads；persistent root输出保留last.pt/optimizer/privateRNG及全部final/曲线。无sealed、无默认推广、无Git上传。

## 真实smoke已通过

首次调用canonical --smoke只加载8TRAIN导致1batch，helper期望2update而失败；梯度与冻结检查已过，失败记录保留，不改预算规避。

fresh v2 helper仅为smoke选择32真实TRAIN/24validation metadata cells，保留完整TRAIN统计，1轮2update；正式训练仍全12536。legacy和772两update都通过：

- 样本、GT/mask/time、noise/flow-time实际stream SHA完全相同。
- 中性B0/identity/teacher state及其grad均不变；audio/local_head/renderer梯度非零且finite；native时钟/mask正确；last含optimizer和privateRNG。
- 772直接HuBERT梯度max=0；将HuBERT改NaN或仅输入772时，所有情感输出逐位一致，表达输入梯度非零。
- 20项输入/runner/resume本地测试通过。尚无772新性能结论，不能使用旧native F1冒充。

## 当前运行与接纳

最新：六臂1568update全部exit0/三pair合同通过；训练driver终态training_complete。closure15199真实24cellpreflight passed后已开始全验证，原二轮训练不重启；rootfree2.228GiB。当前只有评分/等待下载，新成绩尚未闭合。

driver14577，canonical状态`launch.json`。先读状态再继续，不重复launch。两个arm的真实stream必须完全匹配，warm第一步验证完整system逐位等于中性checkpoint、audio只有申明输入迁移；训练结束核冻结因素及1568update，随后fullvalidation审计。

## 报告收尾已补齐（不修改训练）

- 本地已下载并SHA验证10个不可变binding/smoke/training helper文件，收据`initial_evidence_verified.json`；正式训练原argv由远端launch保留。评分与训练不重叠。
- 新report-only helpers：`audit_phase32_pair.py`、`phase32_statistics.py`、`finish_phase32.py`，由独立`closure_launch.json`记录SHA/argv/PID，`postprocess_state.json`为评分canonical。等待六final完成，再seed47真实24cell评分preflight，通过后最多两个CPUworker评分全部三seed。
- 固定全1367 native validation、全部三draw、raw/clip、原四probe顺序；核actualpaircontract、非renderer系统state、TRAIN mean/std及仅声明输入迁移、GT/mask/native times/clip顺序。与旧验证表同rig/同公式，输出MBE/LBE/coefficientFDD/lipmean+max/expressionmean+max/vertexFDD及逐类/强度/身份、jaw范围/相关/速度、眉部meanbias。保存GT及B0参照、固定M025八情感四列渲染输入；不把head或t-SNE当生成F1。
- 原D1门槛在看到新结果前明确统计实现：配对speaker95%CI上界<0表示MBE或mouthdisplacement改善；两原F1及neutral/fear的CI上界<0为明确退步；LBE/lipmean+max/expressionmean/FDD的CI下界>0为明确退步。逐seed报告，三seed均通过才标pilot通过；CI只有3位speaker，不能当SOTA认证，不自动扩大训练/推广。
- 8项评分检查通过：原classification_metrics逐类及macroF1完全一致；优化后的重复speaker-bootstrap与显式展开2000次结果一致；门槛挡住几何/两原F1/neutral/fear退步。没改模型、数据或loss。

下一步：检查closure preflight/评分状态，失败保留后只修具体report问题；评分完成下载全部新权重/曲线/证据并核SHA，再报告772D性能及是否需要D2。Git仍暂停。

唯一localcollector：exec session9685，`.codex-finalizer/collect_phase32.py`，本地`collection_state.json`。它每45秒只读等待原训练及评分全部complete，再增量下载六臂final/last/curves和完整新证据，支持partial续传及逐SHA；不要同时对同local目标手动下载。`.codex-finalizer/phase32_compact_status.py`只写本地latest_status并读取远端状态，不重启任何任务。

本轮不启动新的六模型因素诊断：旧Phase31的identity/u_a证据完整保留；772训练后的同条件u_a/身份重验在D1成绩闭合后再接续。当前评分的jaw范围/相关/速度与B0参照不等于这一因果干预已经完成。

接纳沿用D1计划：至少MBE或mouth动态取得配对改善，两原clip F1、LBE/几何无明确退步，neutral/fear弱类无新崩溃；不扫seed/epoch/权重，不因结果修改门槛。772输入纠偏本身的功能验证即使通过，也不意味着u_a纯情感或已达SOTA。若时序与姿态仍差，再以具体证据提出D2表达目标和梯度职责修正，不直接堆loss。
