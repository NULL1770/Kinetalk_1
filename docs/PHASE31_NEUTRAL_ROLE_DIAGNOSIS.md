# Phase31：中性 B0 与因素职责诊断（2026-10-06）

用户已选择按原设计保持B0中性。Phase26 native-GT模型仅作历史对照；原D1从native起点迁移方案作废。当前D0冻结干预及全validation已有曲线核查已完成，没有新训练或772D成绩。

## 已完成：输入/监督审计 v2

真实远端报告：`/root/kinetalk_phase31_neutral_role_20261006/input_audit_v2.json`。本地证据目录：`final_experiment/evaluation/diagnostics/phase31_neutral_role_20261006`；argv、helper/权重/manifest源码SHA均存于请求和报告。

- TRAIN12536段/22人；715原生中性anchor，2583非中性→中性安全配对，有效3298段。
- 297配对允许完整嘴部监督，2286使用局部嘴部事件mask。mask仅约束监督可信度，不关闭前向残差通道。
- 默认中性B0与原safe-DTW监督模型stage1 hash一致；native-GT候选与它有101个state tensor不同。
- 所有2583配对的boundary_status/viseme_status均缺失，不能声称音素边界已被严格验证，也不能自动复用于任意情感或跨身份交换。
- 890行source不在TRAIN，审计未打开它们的artifact；638中性配对source被原生anchor覆盖，避免重复计数。

第一版报告错误地将数值emotion标签当作名字，覆盖计数无效。v1报告/请求保留；v2用metadata类表解析并严格校验原recipe715/2583。5个相关测试通过。模型、数据、loss均未改。

## 已完成：冻结输出诊断

首次冻结诊断在B0严格重放前失败，未执行干预：辅助脚本误用renderer的2/16批次计算B0，原流程为32批次的B0缓存，TF32在不同真实长度分组形状下有约1e-4差异。修复诊断缓存调度，保存v1失败证据；不修改模型或降低零容差。完整本地测试371通过/1跳过。

中性默认与native历史模型各用自己的冻结源码/权重/身份坐标。每个validation说话人×情感×强度cell选字典序首clip，共66段；先复现原32批次B0缓存及原2/16批次renderer。三draw42/123/2026全部逐位重放通过，之后在draw42固定B0/h0、global/intensity、音频、native时钟和noise，改变u_a或独立参考code/bias。模型state和原文件均不变、无optimizer或梯度，不读sealed。完整35个远端文件（约20MB）已下载逐文件SHA核验；原失败源码/状态保留，远端driver13040已结束。

### 口型：不能再用“统一放大”解决

全1367 validation已有曲线只读核查，表中仅80段中性、clip_all、final为三draw平均指标：

| 模型/输出 | jaw Q90−Q10范围 | GT范围 | centered相关 | 相邻帧位移MSE |
|---|---:|---:|---:|---:|
| 原中性 B0 | .104261 | .111884 | .441482 | .001746206 |
| 原中性最终输出 | .135220 | .111884 | .279330 | .003156498 |
| native历史 B0 | .090971 | .111884 | .464513 | .001243351 |
| native历史最终输出 | .106433 | .111884 | .255299 | .001903238 |

原中性B0范围约GT93%，残差后约121%；最终范围变大但相关下降。中性定位能与同步中性GT比较；情感query的中性B0不以情感原生GT为目标，不把该误差当情感B0训练失败。

### u_a：有用动态与不良嘴部扰动同时存在

66cell、clip_all、draw42，以下为jaw：

| 历史模型 | 条件 | centered相关 | 位移MSE | 范围 |
|---|---|---:|---:|---:|
| 中性 | 原u_a | .379395 | .004095 | .193783 |
| 中性 | static u_a | .425967 | .003400 | .139394 |
| 中性 | zero u_a | .429527 | .003263 | .139015 |
| native | 原u_a | .316063 | .002738 | .187916 |
| native | static u_a | .440271 | .001929 | .141969 |
| native | zero u_a | .438841 | .001859 | .140379 |

同一66cell GT范围.176906。静态化/置零改善jaw相关及位移误差，却压低真实表达动态：中性模型brow范围.046691→static .014979（GT.049491），smile范围.170560→.080525；native也类似。shuffle会大幅恶化速度误差。不能据此部署zero/static，也不能从这种分布外干预单独证明音素泄露；它支持u_a当前职责混杂的具体诊断，仍需772D迁移及明确表达目标验证。

### 身份：renderer确实读取code，但正确性尚需验证

在同query/global/intensity/u_a/B0/noise下，只换异人code，raw jaw中心化响应RMS：中性.010563、native.017377；只换bias中心化响应约1e-8（符合常量bias）。同人参考A/B分别相对完整参考的动态响应约.0016–.0017 / .0028–.0029，异人响应明显更大。中性validation参考same-code距离.005593，异人mean .017941；native .006543 / .019785。

这证明身份code不只参与最后静态加法，而能影响生成动态；不能证明对应身份风格已经正确，也不能把异人输出与原query GT的重建误差当身份成败。所有异人/零code/bias输出仅报响应，不计算其原身份GT目标得分。当前固定rig任务是系数/运动风格身份，不是外貌身份。

### 表情/F1：静态姿态问题仍是主项

66cell原输出，眉部clip MSE的97.10%（中性）/96.76%（native）来自平均姿态偏差；static u_a不能修好它，甚至可能变差。这与此前全validation native眉部96.59%及连续global跨speaker域偏移证据一致。不能只加强u_a或把一个happy样本看起来正确当作所有情感成功。

本轮没有新分类器训练或新F1。历史native候选原clip两probe F1 .796875/.721358，raw .619331/.591701；不属于772D中性模型，不把teacher准确率或t-SNE当生成成绩。

## 明确的下一步

1. 使用旧默认完整中性坐标起点，对比1540继续训练和772 affect+prosody迁移；B0、配套identity/teacher固定，既有loss和noise不变。原Phase26 native起点、native统计/renderer混拼作废。固定三seed2轮pilot方案见下一轮计划；当前未启动。
2. D0已证明u_a影响嘴时序，但还没有条件化内容泄露probe或精确音素事件证据；这些保持为待办，不宣称解耦。
3. 若输入单变量不能解决动态，优先提出D2明确表达目标/teacher逐帧读出及梯度职责修正，选一种机制替换有污染风险的通路，不堆叠loss；新增结构仍需用户审批。
4. 持续检查global跨身份泛化及类别姿态，保持全嘴可调。现manifest仅认证source→neutral监督；情感/强度/身份交换仍需新的直接配对质量审计，不从间接neutral映射推定。

证据：`readout.json`（66cell）与`full_jaw_report.json`（1367已有曲线），`download_verified.json`核35文件。`fixed_factor_curves.png`固定展示M025的neutral_L1_001/happy_L3_001，没有按表现选片或调整gain/lag；仅display clip_all。所有报告均development-only，未推广默认模型，未达SOTA。

GitHub保留诊断源码、精简SHA绑定结果、重放记录和固定曲线图；完整约20MB原始报告/导出及失败证据存于本地上述目录，不把`download_verified`误读为仓库包含所有权重/数据。完整本地测试371通过/1跳过。Git默认空白检查把Windows原始CRLF报作trailing whitespace；保留原字节以维持SHA，用cr-at-eol检查通过，未格式化冻结证据。.gitattributes限制该证据目录禁用换行转换。

后续D1必须使用同一中性B0对应的身份bias、teacher残差及flow坐标；不能拼接native权重/统计。D2/D3/D4新增训练目标或结构仍是待审批方案。
