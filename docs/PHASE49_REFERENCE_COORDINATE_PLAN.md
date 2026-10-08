# Phase49 独立neutral参考坐标诊断

Phase48已完成并存档2010572。用户继续优化动作风格；当前M025两段参考不稳定，但还不能确定是参考真实姿态、内容/B0误差还是编码导致。先做小型冻结诊断，不改变模型/默认/划分，不训练、不读sealed。

只读取原independent neutral enrollment的A/B与自身native frozenB0。分别计算TRAIN-fit、原internal held身份和external dev的A/B稳定性；不用query motion训练任何映射。描述量：GT native均值/动态统计、B0残差均值/动态统计、原64D编码。同人A/B与跨人物AB均值距离比例，以及A->B参考检索仅为参考稳定性的诊断，不称身份迁移正确率。参考自身B0 jaw闭口(<.05)帧上的GT姿态单独记录；若没有闭口帧，标为不可估计，禁止凭空补值。

分离检查：M025参考A/B的jaw、嘴角、眉眼GT均值、residual均值、动态、闭口数量，按原native mask/clock。差异不能直接称内容泄露。GT均值仍受语句分布影响，residual也包含B0误差；两段neutral不足以证明情感条件下的个人风格。

产物预计<2MiB，独立fresh目录、源/checkpoint/data SHA和冻结状态记录。若参考自己也有明显不一致，先修正风格坐标/参考鲁棒性定义；若可观测统计较稳定而编码不稳定，再考虑内容归一的参考编码和有明确角色的跨参考一致性，避免增加一串loss。新训练仍需解决远端存储，所有模型改动前推Git。
