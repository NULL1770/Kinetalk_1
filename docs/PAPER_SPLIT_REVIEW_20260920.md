# KineTalk 论文 split 审查记录

本文件只审查 metadata candidate，不批准训练，也不读取 sealed-test 动作目标。

## 结论

`identity_disjoint` 是当前推荐的主轨道：训练/验证/测试身份严格分离，query 与 neutral enrollment 不共享 speaker-sentence 对；它直接检验新身份泛化，但句子可跨 split 复用，因此不能命名为新句子泛化。

`joint_disjoint` 的第一版 hash 候选保留为 rejected candidate。它的句子集合完全不重叠，但 test 只有 45 个 query clips，主要因为全局 hash 把稀疏或只在部分源 split 出现的句子分给了 test。它不能作为稳定主表。

`joint_balanced_disjoint` 是新的候选，不是已批准协议。它只在三个 MEAD source role 都有至少 3 个非中性 clips 的 32 个共同句子中选择，稳定 hash 分配 12 个 validation 句和 8 个 test 句，其余句子（包括稀疏/中性-only 句）进入 train。当前 metadata 计数为：

| 轨道 | train query | val query | test query | query 句子交集 |
|---|---:|---:|---:|---|
| identity_disjoint | 4118 | 446 | 537 | 允许跨 split |
| joint_disjoint（rejected） | 2997 | 88 | 45 | 无 |
| joint_balanced_disjoint（candidate） | 2516 | 115 | 105 | 无 |

balanced candidate 的 manifest SHA256 为 `074043d12c88c8fba0c529e45fe421bfed1eafb73800197583c83cac710ee437`。生成脚本是 [`scripts/lock_balanced_joint_manifest_20260920.py`](../scripts/lock_balanced_joint_manifest_20260920.py)。

## 仍需在训练前锁定的项目

1. 明确论文主任务是新身份，还是同时新身份与新句子；两个任务应分表报告。
2. 给出历史 exposure ledger：所有曾用于训练、统计拟合、参考 enrollment、调参或可视化选择的 clip ID；sealed test 目标继续保持不可读。
3. 以最终 manifest、enrollment 规则、排除表、source metadata hash、音画时钟和 channel mask 生成不可变 protocol hash。
4. 训练阶段只使用 train query 与允许的 train enrollment；validation 只用于预算/模型选择；test 只在版本锁定后一次评估。
5. 四模块（content/B0、identity、global emotion、dynamics）都要绑定同一个正式 train partition；只重训 dynamics adapter 不能称全量训练。

在上述项目完成前，不启动全量训练。
