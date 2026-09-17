# 正式共享动态接口训练：结果与下一步

2026-09-16。用户授权继续优化，在稳定后正式训练。本轮已完成正式训练和复核，所有GPU训练进程已结束；**没有替换默认模型**。

## 做了什么

- run34/35：在全局音频概率上用固定 `g=1-p(neutral)` 乘整条动态控制，teacher/audio一致，无新头或loss。两训练seed各600步、各3noise，通过预设动态、neutral原动作、口型、速度/global保护。
- 正式数据2720片段/22人/67句，原1200全部保留；79条独立neutral参考每人2–4条。原280段/25句继续开发，逐句与训练隔离。
- 固定content+emotion2vec中间2/4/6层+韵律，rank8；只用训练句3fold CV重拟alpha=1、U、输入RMS。原280低率upper R²=.02201，CI[.01505,.03265]，reverse=-.01300。没有调gate或新增架构搜索。
- 冻结B0、identity、global、motion teacher、audio head/U及整个renderer，只训练原512参数local_projection，已有flow MSE。前5epoch teacher概率.5→0，后全audio；每2epoch按保护约束挑checkpoint，至少10/最多40epoch，patience5。

## 正式训练真实结果

RRR seed46/47/48和PCA seed46均完成**18epoch、3060步、每epoch完整2720样本**，按预设规则早停，各选中epoch2/340步。四组共12240正式优化步；不能把选中早期检查点说成后期训练仍在改善。

| 原280开发，非中性 | RRR46 | RRR47 | RRR48 | PCA46 |
|---|---:|---:|---:|---:|
| upper full-zero ΔR² | .012925 | .012656 | .012491 | .011355 |
| brows full-zero ΔR² | .005843 | .005757 | .005682 | .004249 |
| selected epoch | 2 | 2 | 2 | 2 |

RRR46 upper95%句簇CI[.006855,.021346]，brows[.002072,.011657]，neutral mouth原动作误差比zero +.872%、单侧90%上界1.465%。三RRR seed保护检查全部通过。生成upper绝对动态R²仍约-.073，属于弱收益。**这280已用于选epoch，上述区间是选后的描述性复核，不能作为独立测试显著性。**

RRR46−PCA46 upper差值仅.001570，CI[-.000523,.003765]；眉和眼对PCA的差值也跨零。正式数据下没有证实RRR比同维PCA更好，不能将既有RRR方法包装为已验证创新。

![训练饱和与跨人差异](../artifacts/formal_training_v1/formal_training_diagnostics.png)

## 新身份复核：阻止默认替换

固定正式方法和选中checkpoint后，评估native val的439段/3人（其中非中性363段/62句），每人独立neutral参考。未重拟U/尺度、未改gate或回选epoch。它与训练共享57句，所以只称动态模块**跨身份同句开发评估**；原B0历史暴露未知。native test512及原reserved15目标仍封存。

三训练seed都出现同一问题。以预先指定主seed46为例：

| full相对zero的动态ΔR² | M025 | M037 | M039 | 合并 |
|---|---:|---:|---:|---:|
| upper | +.05123 | -.01197 | -.01048 | +.02463 |
| brows | +.01125 | -.03889 | -.02803 | -.00666 |
| eyes | +.13784 | +.02737 | +.00139 | +.07732 |

合并upperCI为正，但删掉M025后增益为-.01162；仅合并报告会掩盖两个身份退化。眼部三人方向一致；眉部未通过保护，三seed跨身份完整检查均失败。neutral/嘴部/global保护仍通过。只有3个身份，不足认证广泛跨人泛化。

## 在同一时间尺度定位问题

对真实目标和生成残差都用同4帧bins、同mask/权重、同真实目标。生成残差只分箱中心化，不再乘U，防止隐藏生成器引入的误差。均为非中性，主seed46：

| 同clock upper R² | 原280开发 | 新身份439开发 |
|---|---:|---:|
| frozen audio直接解码的动态 | .02205 | .03018 |
| 真实motion投影oracle，经同audio gate | .52014 | .46262 |
| 生成zero-local | -.02665 | -.11199 |
| 生成audio full | -.01143 | -.08012 |
| 生成motion oracle | .40560 | .22772 |

新身份的motion投影目标仍有表示能力，经过冻结生成器/共享条件路径损失更大；音频本身也远未接近oracle，M037的眉部直接预测已负。不能只说“音频信息不够”，也不能直接断言identity是唯一原因；条件分布、随机生成、监督质量可能共同影响。表内分箱R²不可与前述原帧R²混用。

## 下一步决策

停止给这套512接口增加相同训练轮数，保留正式epoch2候选和旧默认。开发收益从epoch2开始连续下降，说明现有flow目标与弱audio条件下的时序指标仍不一致；其与teacher比例衰减同时发生，尚不能将其单独归因为衰减日程或过拟合。下一次改动先在**训练身份内部留人验证**，分别检验持续真实motion条件、纯audio和当前衰减训练的同预算接口表现，定位训练分布问题；再检验neutral参考能否提供动态响应尺度，而不只是静态偏置。若前者通过而跨人仍失败，才用同一共享全脸接口做有界校准，先用motion oracle确认传递损失减少，再检验audio full。校准不能读取query motion，不能靠新身份开发结果调尺度，不能拆眉眼嘴生成头或堆新loss。

此为由诊断提出的下一实验，不是本轮已实现/验证的结论。当前保留baseline、同维PCA和zero/reverse/oracle，若校准无收益就回到音频目标/监督质量；不再把rank8本身当作确定瓶颈。

后续更新：训练内留人教师日程对照已实际完成，持续teacher比撤teacher退化小，但三组epoch18仍失败；同64个fit样本的flow各时刻误差下降与实际生成恶化同时存在。见[新实验记录](TEACHER_SCHEDULE_PROBE.md)。共享身份校准尚未实施。

## 复现与资产

- 远程根：`/root/kinetalk_runs/formal_predictable_v1`。数据缓存和结果约2.5GB，root盘剩22GB；autodl原盘约410MB，未删除训练数据。
- 正式入口：`scripts/train_formal_predictable_projection.py`。`launch_seed46.json`保存精确启动和resume argv；`rrr_seed46/last.pt`是epoch事务恢复点，含Adam、Python/NumPy/torch CPU/CUDA/训练生成器RNG及来源hash；best快照内嵌last。改配置/代码/数据会拒绝恢复。
- 每run保存best/last、source快照、epoch/开发JSON和最终3noise曲线；曲线sidecar绑定checkpoint/recipe/cache。现有best/last、审计和比较已备份本地`artifacts/formal_training_v1/`，大cache和完整曲线留远端并有hash inventory，可由锁定manifest重建。
- `audit_formal_projection.py`、`compare_formal_projection_runs.py`、`audit_cross_identity_projection.py`、`audit_projection_delivery.py`只读复核，无重新选模。
- 16条训练源视频按固定hash抽样，时间对应及动作范围检查，无整段错配；原嵌入音频与native wav长度全同、最低相关.99999983。慢变/近饱和眉毛和眨眼干扰仍存在；没有做完整生成脸视频或独立口型同步认证。
- 测试：最终完整230项通过。测试验证合同/恢复/配对，不保证情感效果或CCF录用。
