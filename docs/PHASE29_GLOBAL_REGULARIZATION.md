# Phase29：连续 global 泛化的单变量试验

2026-10-06。先读 CURRENT_OPTIMIZATION_STATE.md；Phase26/27/28 已完成，禁止重跑。

依据：Phase28 TRAIN global MSE约.149，validation约2.33；类别内连续相关TRAIN约.93、validation约.21。Phase27仅换GT-global时MBE .871→.633，但这是定位结果。嘴部范围已接近GT，不扩大嘴、不加loss、不遮嘴。

唯一候选：audio pooled128进入global_head前，训练期独立逐样本/逐特征 inverted dropout p=.1。无新增参数、层或loss；u_a仍读取原hidden序列。emotion/intensity头共享global，因此其训练输出也受影响，共同TCN更新后u_a也可能间接受影响，不能声称强度或时序训练完全不变。部署/evaluate调用默认p=0。

训练器原本所有模块保持eval；新包装仅在明确的训练调用中临时设置audio根模块training标记，子模块模式不变，用后恢复。模型eval下dropout关闭。default p=0保留原调用及RNG行为。独立CPU Generator(seed+290000)，last checkpoint保存/恢复其状态；不修改原sample/flow noise/ft RNG。既有coordinate refit禁止resume的规则保留，不承诺该组合可resume。

固定seed47/48/49、12epochs、batch16、9408updates，每臂12536 TRAIN×12。与Phase26 standardized12相同初始Phase21权重、TRAIN moments、全嘴support、head mean-field reset、source及stable runtime。不扫描p、不挑epoch/seed/draw；只有完整性错误可停止。先old/new-default/candidate真实2-update smoke、默认state/loss、私有RNG与实际流验证，方可训练。完整actual sample/GT/mask/time/noise/ft流必须等对应Phase26 standardized12，避免再训练已完成control。其它snapshot文件逐SHA相同，两个patch必须是已登记精确文本变换。

全1367 validation×draw42/123/2026×四冻结TRAIN probes；原128/64与辅助分别报，raw/clip均保留。评估MBE/LBE/LVE、眉/嘴静态bias与运动误差、mouth/jaw范围、中性jaw correlation、精度/越界、情感/强度/speaker分组及paired speaker CI。不自动推广默认，不打开sealed test；SOTA尚未建立。

验证：CPU/GPU默认train/eval逐位等旧模型，非零local head下u_a完全不变、私有RNG及checkpoint重放、mask/NaN梯度/padding/mode恢复均通过；28项本地回归通过。三臂真实2update（batch4、8个TRAIN smoke inputs）old/new默认system/audio/loss逐位等，实际sample/GT/mask/noise/ft及原全部RNG一致，私有RNG两次抽样数闭合，B0/identity/teacher冻结。候选audio和renderer确实不同于默认。smoke不代表收益。

两次验证脚本失败保留：首次batch16只能1update，fresh smoke2改batch4；smoke2完成GPU后，通用RNG比较遗漏numpy.ndarray，报告失败。修复比较后只读取已完成state，无重复GPU训练；smoke_closure_state complete及smoke2_verification passed才允许launch。

存储：不删历史。新训练完整工作输出在RAM，完成后在数据盘`/root/autodl-tmp/kinetalk_phase29_archives_20261006`全文件/目录无损压缩、逐SHA验证；archive锁串行保存。只有持久包完整验证才标记arm complete，root另存final权重及metadata，curves使用SHA一致RAM审计副本。root初始约1.44GiB、data盘约1.06GiB。RAM工作副本不能冒充完成证据。

canonical `/root/kinetalk_phase29_global_dropout_20261006`，launch发起后只能查状态，不重复启动。全局泛化另做冻结TRAIN/validation学生提取，对照已bound相同teacher codes，不拟合校准。曲线审计MC只有MBE/LBE等系数指标；mesh LVE需要后续真实rig同协议计算，未补算前不能称各项达标。

最新canonical改为`/root/kinetalk_phase29_global_dropout_v2_20261006`：首次容器重启丢RAM，旧日志明确无stage_start/更新。旧失败保留；三个control曲线从持久archive全包/member SHA恢复。v2最初serial，容量核查后仅替换dispatcher并启动未开始48/49，seed47optimizer从未重启；source/预算相同，parallel_adoption记录。CPUmesh BLAS资源单独限制，GPU99%。每epoch完整last另存root/durable_last，watcher仅复制，原RNG不变。

补算同既有validationgap的固定632vertex rig，vertex及BSFDD evaluator SHA精确等原方法；各seed/control/candidate/raw+clip完整几何。比较表保留baseline adaptations/FaceDiffuser conditions的限制及baseline单checkpoint与候选三seed区别，不报官方SOTA。Finalizer会等全rig结果及比较表完成才标complete，本地collector等待全部complete自动下载逐SHA。训练期间不根据中间得分改模型或选择seed。
## Latest closure addendum
All three fixed12 candidates and six fixed-rig mesh audits complete. Original report failed on statistics.path despite byte-identical copies. Repaired contract verifies actual bytes against recorded SHA and retains strict equality of every other source field (six tests pass); no formula/weights changed. Failure evidence preserved. Fresh report driver4838, report_repair_v3.log / postprocess_repaired_v2_state.json; use download_phase29_repaired_v3.py only after closure complete. Original collector stopped failed; never relaunch model training. Candidate MBE preliminarily .864415 vs .870982, LBE .420868 vs .421170, insufficient for .7/SOTA. Full F1/global/CIs pending report closure.
