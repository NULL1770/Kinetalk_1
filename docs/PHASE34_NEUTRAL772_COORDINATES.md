# Phase34：中性772坐标对照（2026-10-07）

用户已阅读Phase33诊断和N1具体计划后要求继续优化，N1获授权；Git继续暂停。中性B0与772输入保持，后续表达职责修正由具体证据决定，不把多个变化混进本对照。

- 远端：`/root/kinetalk_phase34_neutral772_coordinates_20261007`；canonical `launch.json`与`postprocess_state.json`。只读状态`.codex-finalizer/phase34_status.py`。
- binding SHA`8b9944280ccf018824b6999e9106a1399331daee6035912c7ef4df58c71e822f`。原Phase32的150模型/训练源码逐SHA复制，没改训练公式代码；只启用已经测试的坐标选项。
- 起点完整中性checkpoint SHA`e17659536a6fcaaaec2d9c22f99403fe4d9f1c8690c3cd182583894545f5ab45`；统计SHA`eed104a6ec78ecdb77a5a834890895e3436c6a6718216faf6e37427ed70b7ca5`，只拟合12536TRAIN/1372104帧。
- 两臂均学生772，保留全51观察通道，包括jaw/lip；冻结B0/identity/teacher，只训练audio+renderer。共同diagonal source与TRAIN mean，共同清零output head；对照centered scalar，候选standardized坐标及相应flow误差单位。后者改变相对通道误差权重，没有新增loss项，不声称loss数值完全不变。
- 固定seed47/48/49；每臂2轮1568updates，batch16，12Euler；final固定，full1367validation×3draw，raw/clip及4冻结TRAIN probes、同rigmesh/区域动态/分组/pairedCI。新gate增加全80中性jaw相关/速度无明确退步，不能拿漂亮F1掩盖口型。
- 真实两臂32TRAIN/24validation smoke各2update已经pass；初始system/audio state和source完全相同、初始物理速度等于TRAIN mean，实际sample/GT/mask/time/白噪声流一致；772 HuBERT梯度0/NaN不影响输出、表达梯度非零、冻结state保持、loss/grad有限，local head在清零头首步后第二步获得非零梯度。36已有坐标/source/input/runner回归通过。
- 正式driver3138，三个control worker3139/3140/3141；每seed之后自动跑standardized。仅这一driver，禁止重复launch。每臂last持久保存optimizer/privateRNG，不仅存RAM。
- Phase32本地419文件已完整核SHA，再逐个核服务器6curve/6last的12个精确SHA后删除重复服务器文件，回收2,596,596,880字节。原final、评分、源码、数据保留；`duplicate_reclaim_state.json`与本地准确成员可恢复。Phase32旧replay如需要曲线须先恢复，不误称文件仍在服务器。
- 新实验持久存储预算约3.4GB；清理后空闲4.83GB，smoke后4.58GB。初始错误测试文件名导致pytest未运行，改用rg找到的真实测试集36pass，不将其当实验失败。未发生正式模型失败。

尚无Phase34最终性能，不能将smoke loss或Phase26高分报告为本实验达标。评分/下载/渲染完成后追加固定结果；根据结果再处理ua表达职责、连续global泛化和身份验证。新的字段与评价helper另SHA绑定，评价公式保持原定义。

## 本轮继续（2026-10-07）

三control已各1568updates/exit0，候选standardized worker4100/3952/4185继续，唯一driver3138。唯一closure3403与collector84935等待；禁止重跑。TRAIN-only配对audit v4完成并下载SHA核验，嘴mask全二值、55.375%覆盖，详见EXPRESSION_TARGET_AUDIT_20261007.md。

已适配本地summarize_phase34_local.py及render_phase34_fixed.py。下载全部SHA闭合后才能读出；渲染固定seed47/draw42/M025/005八类，四列GT/Neutral B0/Centered772/Standardized772，native时钟、原display clip_all、无gain/retiming/挑seed。音频已核全部存在，沿用此前固定audio binding，不通过择视频掩盖弱类。
