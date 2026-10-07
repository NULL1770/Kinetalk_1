# Phase35 物理flow误差结果

三个seed47/48/49、固定2轮1568updates，全1367validation×3draw/raw+clip/四冻结probe/同rig均完成。源summary/SHA已核；这不是sealed结果。

| clip_all三seed/三draw平均 | Coordinate772 | Physical772 |
|---|---:|---:|
| 生成F1 原128 | 0.683872 | 0.229124 |
| 生成F1 原64 | 0.581147 | 0.266488 |
| MBE | 0.904603 | 0.915915 |
| LBE | 0.446154 | 0.457559 |
| Lip mean mm | 4.008668 | 4.020528 |
| Lip max mm | 7.598816 | 7.575538 |
| Expression mean mm | 0.769161 | 0.783224 |
| Vertex FDD mm² | 153.447688 | 135.782384 |
| 嘴部位移MSE | 0.003709 | 0.004770 |
| jaw相关 | 0.255015 | 0.248878 |

三个联合gate均失败，拒绝physical单位；FDD改善不能抵消情感、MBE、LBE和嘴时序退步，不延长本项预算。

完整新产物本地SHA备份完成：True。唯一collector90557；评分/训练已结束，不重复启动。

下一项独立使用Phase34 standardized对照，只改audio学生的表达梯度职责，见PHASE36_EXPRESSION_GRADIENT_PLAN.md。仍保持中性B0/772/全嘴前向；当前尚未证明完全解耦，Git暂停、默认未替换。
