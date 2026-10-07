# Phase37：将情感学生改为明确的表达／韵律目标

最新覆盖：三seed训练与全部评分已完成，联合gate全部失败，拒绝此候选，不延长训练。clip生成F1原128/64由 .683872/.581147 降到 .617731/.549179；MBE .904603→.906694，嘴速度MSE .003709471→.006339550（+70.9%）。下面的启动叙述为历史。SSH重启已恢复，原manifest备份待续传；固定视频与独立因素诊断随后闭合。

Phase36三seed全部失败，不能推广。全量clip F1 .684/.581→.637/.545，MBE .9046→.9112，嘴位移误差+28.8%。固定seed47/66cells两臂的3draw精确重放、权重不变均通过；static/reverse ua仍改善jaw，说明只筛反向通道无法修正其语义和时序。

下一项采用一个完整的学生职责定义，复用已有模块，不增加多个critic/交换/ordinal：

- global64只蒸馏TRAIN的情感×强度原型；同标签条件的目标严格相同，不再逐clip拟合完整残差教师潜变量。原型由现有冻结motion teacher生成，teacher自身仍未重建为纯表达模型，不能宣称完全解耦。
- ua64明确为4个有符号上脸表达状态＋4个原生韵律量＋56个零占位。4个状态使用已有state_head和readout_slow_state/masked_slow_state，以自身原生眉眼GT、独立中性anchor和已验证TRAIN scales监督；不是复制clip情感标签。韵律是原缓存logF0、logRMS、periodicity、voiced，按原TRAIN声学统计标准化。
- renderer继续全残差flow与原generated-semantic监督，条件在这两项反传前detach；学生改由表达状态MSE、原情感/强度CE、原global蒸馏训练。状态项替换原不受约束的flow学习路径，未同时加嘴loss/critic/交换。已有state_head4启用；旧local_head64保留兼容但不作本候选条件。
- global/强度/4韵律/identity仍能前向影响所有嘴通道；HuBERT只在B0/h0，保持中性B0及全部嘴开放。不是把上脸4状态直接限制成上脸动作输出。

默认flow/latent路径保持不变。新模式需要在checkpoint config记录并由相同配置恢复；所有新恢复器必须读取该模式，不能只按772维猜测。先默认真实重放＋命名模式真实两update smoke，证明状态头非零梯度、嘴GT扰动不改变学生表达目标、韵律原生时钟和HuBERT隔离、renderer全嘴有效／冻结／finite。正式三seed2轮 matched基线复用Phase34 standardized；新source/binding、完整同协议评分和容量预算必须先闭合。失败不自动扩大，不拿音频头或原型oracle当生成F1。

用户已授权继续正确架构优化。实现/393全suite+1skip/17最新相关检查完成；两实际2update smoke全部通过，默认重放Phase34精确，新fullflow/generated→audio梯度0、statehead有梯度、localhead不用、全嘴renderer有梯度、HuBERT NaN隔离、named配置恢复逐位一致。已启动唯一三seed2轮driver22623/worker22624–22626/closure22627；collector50441/localfinish42620。binding589ba601…；启动容量按实际预算1907078782bytes通过。正式成绩尚未产生；SOTA尚未达成。
