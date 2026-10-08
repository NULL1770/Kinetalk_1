# 当前恢复入口：Phase45 教师目标诊断进行中（2026-10-08）

先读 [PHASE45_PLAN.md](PHASE45_PLAN.md) 和 [目标诊断](PHASE45_TARGET_DIAGNOSIS.md)。TRAIN10903fit/743身份留出/890句子留出的冻结诊断已完成，182原SHA成员本地核验；g误差下降12.46%/16.47%，u仅1.85%/1.18%，不是最终动作指标。源码94e2ea3及结果0a97b9d通过SSH转发推送，94e2ea3远端ref独立相等。

固定g/u/gu三mean-head解析监督候选已实现，28项测试与历史六CSV/rows兼容核验通过；B0/q/style/decoder/学生骨干/方差/语义head保持，仅现有均值切片训练集拟合。一个固定ridge solve，非SGD等预算、非无需训练。即将部署独立head_candidates/code做真实GPU门槛及16clip smoke，然后1367完整评估/24视频/表格。尚无候选动作成绩，不自动推广。具体执行状态以Phase45远端state/receipt为准，不重跑已完整关闭的目标诊断。

Phase44已完整关闭，不重跑。以下保留其事实作为参考。

恢复先读 [PHASE44_RESULTS.md](PHASE44_RESULTS.md)，再按需看 [完整表格](PHASE44_EXPERIMENT_TABLES.md)、[24视频](PHASE44_VIDEO_GALLERY.md)。不要重跑已结束的训练、评估、collector或因素诊断。旧进度完整保留在 [历史状态](archive/CURRENT_OPTIMIZATION_STATE_before_phase44_closure_20261008.md)，其旧ACTIVE/待审批/Git暂停不是当前状态。

## 当前事实

- Phase44 a_kl/a_mean/b_mean 各8追加epochs/5456updates，seed47、10903fit；full1367开发评估全部完成。父模型各24epochs/16368updates。
- 三臂各196成员、root预检208成员、因素15成员全部原SHA复核；24视频全帧解码/音轨/时钟/rig通过且SHA再核，9份表格来源SHA通过。closure_verified.json已写入diagnostics和paper_tables。
- 远端 /root/kinetalk_phase44_prior_refine_20261008；本地 final_experiment/evaluation/diagnostics/phase44_prior_refine_20261008。root/per-arm local_queue_state均complete；远端训练与factors_v1状态均complete，收尾SSH GPU 1MiB/0%。不要重跑phase44_launch.py。
- 部署prior_mean/clip：a_kl MBE.830979/LBE.390528/Lip3.288994/主F1.649915；a_mean .825540/.393767/3.268096/.688865；b_mean .796286/.359408/3.046861/.599134。
- a_mean优于同父同预算a_kl，但未联合胜过父A（.827621/.388066/3.319447/.695182）；jaw范围.142617降至.111908，GT.175279。主F1未达.7，辅助.708668和音频分类头.877547不能替代主F1。
- 三组均不推广，不延长训练，未达SOTA。Phase43-A仅作表达参考、B仅作几何参考；默认未替换、sealed未读取。

## 本轮做了什么

仅audio prior与原emotion/intensity heads追加训练；B0、posterior教师、style和decoder冻结。a_kl用KL，a_mean/b_mean用实际decoder坐标均值+token标准差匹配。没有新增模块、pair、motion重建梯度或内容入口。模型源8f74658和收集修复前12dee08已推送；本轮结果归档提交见git log。

冻结119 state tensors和全1367 B0输出逐位相同。oracle浮点重算引起初始收集断言失败，原失败证据在collection_recovery_v1；只对oracle采用metric rtol1e-6/atol1e-8、系数atol5e-5，四probe/confusion仍精确相等。恢复collector142288已结束；不启动旧collector6196。没有改原始报告、指标公式、权重。

## 已得诊断与下一步

- a_mean音频g+教师u：MBE.737836/主F1.746556/jawcorr.881652，正常为.825540/.688865/.499695；教师g+音频u MBE.417349而F1.568802。全部teacher交换用了query GT，仅是敏感性诊断。
- prior u方差.384070接近teacher.371816且触上限0，输出仍收缩。不能把方差大当唯一原因。
- A decoder去u时间均值；raw token u MSE含无效DC坐标，不据此直接断言有效匹配变差。下一步先检查native/centered教师目标、B0误差和可预测性，再按证据设计训练。音频头分类不是目前唯一瓶颈。
- 同内容/情感不同参考的目标风格、独立内容读出、多seed和匹配预算基线仍未完成。不是当前已启动的后台任务。

## 持续边界

用户允许同轮多改，但每轮代码修改前先Git上传存档。B0中性且冻结，情感学生只emotion2vec768+prosody4；嘴部全开放，情感/身份允许调幅度。无query GT部署、无oracle排名、无sealed开发调参、无自动默认替换。不盲目堆loss，不以t-SNE或单happy证明解耦/SOTA。

每次正式结果给八类视频、外部改编基线和真实训练消融表。MEDTalk/DESTalker无核验训练行，不补造。四冻结F1是整段统计probe而非逐帧判别。凭据仅由私有helper读取，不输出/提交。third_party/voca_reference/是无关未跟踪目录，保持不动。不要创建子agent。

最新架构和实验职责见PHASE44_PLAN.md及PHASE41_IMPLEMENTATION_AND_TRAINING.md；早期CURRENT_MODEL_AND_TRAINING中的Phase37结构是历史，不能替代Phase41以后模型。
