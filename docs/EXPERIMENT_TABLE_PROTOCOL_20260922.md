# KineTalk 实验表协议（2026-09-22）

本文档锁定论文实验表的表示、输入轨道、数据范围和结果门槛。所有最终表格必须由 `scripts/aggregate_paper_tables.py --require-test` 生成；任何来源报告没有显式 `test_loaded=true` 时，聚合器拒绝进入 final-test 表。

## 数据和输入轨道

主协议使用 MEAD 的八类情感：neutral、angry、contempt、disgust、fear、happy、sad、surprise。CREMA-D 的标签集合不完整，因此只作为独立的跨数据集附录，不与 MEAD 八类宏平均混合。训练和验证使用固定身份与句子划分；封存测试集只在最终评估阶段读取。

外部方法按输入条件分轨：

- **Audio-only**：音频、同一中性身份参考和部署时可用条件；不提供真实情感标签、真实上脸强度、文本或目标动作。
- **Emotion/style-conditioned**：允许方法所需的情感标签或风格参考，但所有接受该接口的方法必须使用同一来源；该轨道不与 audio-only 排名。

MEDTalk、DEITalk 和 DiffPoseTalk 的公开协议包含额外情感或风格条件，因此其 MetaHuman/FLAME 原始数字只放文献背景表，不直接填入本项目 ARKit52 主表。

## Table 1：ARKit52 主定量表

统一 MEAD 八类、同一 split、25 fps、相同有效帧和通道 mask。方法行只允许填写在同一 ARKit52 协议下真实重训并完成测试导出的结果。

| Method | MBE ↓ | LBE ↓ | Upper9 intensity error ↓ | |std gap| ↓ | Lip23 velocity error ↓ | OOB ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Static reference | pending | pending | pending | pending | pending | pending |
| FaceFormer-ARKit | pending | pending | pending | pending | pending | pending |
| CodeTalker-ARKit | pending | pending | pending | pending | pending | pending |
| FaceDiffuser-ARKit | pending | pending | pending | pending | pending | pending |
| KineTalk | pending | pending | pending | pending | pending | pending |

MBE/LBE 使用 ARKit 系数空间定义；`Upper9 intensity error` 是本项目上脸系数强度代理，不命名为 MEDTalk EIE；`|std gap|` 是 ARKit 时间标准差差的绝对值，不能单独证明音频事件时刻对齐。

## Table 2：结构消融表

训练消融与推理干预分开解释。`emotion2vec` 只作为冻结音频特征来源，不作为方法贡献或结构消融项。

| Variant | Residual branch | Static mouth calibration | Emotion teacher–student | Bounded center | Time-varying audio branch | MBE ↓ | LBE ↓ | Upper9 intensity error ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full KineTalk | ✓ | ✓ | ✓ | ✓/pending | ✓ | pending | pending | pending |
| w/o residual branch |  | ✓ | ✓ | ✓/pending | ✓ | pending | pending | pending |
| w/o mouth calibration | ✓ |  | ✓ | ✓/pending | ✓ | pending | pending | pending |
| w/o emotion teacher–student | ✓ | ✓ |  | ✓/pending | ✓ | pending | pending | pending |
| w/o bounded center | ✓ | ✓ | ✓ |  | ✓ | pending | pending | pending |
| w/o time-varying branch | ✓ | ✓ | ✓ | ✓/pending |  | pending | pending | pending |

若 bounded center 在封存测试集没有稳定收益，Full 行应移除该模块，bounded center 只保留为失败消融或附录诊断，不为其补写正向贡献。

## Table 3：情感表

情感表分为两个 panel。Panel (a) 复述文献的原生指标，仅作协议背景；Panel (b) 报告统一 MEAD-ARKit52 的真实结果。独立 emotion probe 只在 TRAIN 的真实动作上拟合，在 VAL/TEST 冻结后同时评估 GT 和生成动作，并报告生成结果相对 GT probe 的差值。

Panel (b) 建议列：Upper9/Brow5/Eye4 MAE、intensity proxy MAE、independent probe balanced accuracy、macro-F1、八类 recall、GT-versus-generated Δ，以及按句子簇 bootstrap 95% CI。纯音频和带情感/风格条件的方法分轨。

## Table 4：动态控制表

动态表使用同一测试片段的 matched-static、full audio、reverse audio、shuffle audio、mismatch audio 和 oracle slow-state。oracle 仅验证输出接收器，不作为最终方法结果。

| Condition | Upper9 envelope/intensity MAE ↓ | Pearson/Spearman ↑ | Peak lag (frames/ms) ↓ | |std gap| ↓ | Velocity/smoothness |
|---|---:|---:|---:|---:|---:|
| Matched-static | pending | pending | pending | pending | pending |
| Full audio | pending | pending | pending | pending | pending |
| Reverse audio | pending | pending | pending | pending | pending |
| Shuffle audio | pending | pending | pending | pending | pending |
| Mismatch audio | pending | pending | pending | pending | pending |
| Oracle slow-state | pending | pending | pending | pending | pending |

主结论门槛是 full audio 相对 matched-static 的配对句簇差异、相对 reverse/shuffle/mismatch 的显著优势，以及口型 LBE 不恶化。若只有 oracle 通过，结论只能是接收器可用，不能声称音频到眉眼动态已经解决。

## 顶点补充表

共享 `arkit2.blend` 只支持 avatar-space 顶点评估，不是 FLAME LVE。渲染器导出 neutral mesh 和 52 个 blendshape delta 后，使用 `scripts/evaluate_vertex_lve.py` 计算：

- `Avatar-MVE`：全脸顶点平均欧氏误差；
- `FaceDiffuser-LVE²`：每帧唇部顶点最大平方误差的时间平均；
- `LVE`：每帧唇部顶点最大欧氏误差的时间平均；
- `EVE`：眼睛和额头区域最大欧氏误差的时间平均。

该表只能比较在相同 ARKit52、相同 blendshape rig 和相同测试片段上重新导出的结果。FaceFormer、CodeTalker、EmoTalk、DiffPoseTalk 的 BIWI/VOCASET/FLAME 原始顶点数值不能直接拼入本表。

## 生成和审计

最终表格命令示例：

```powershell
python scripts/aggregate_paper_tables.py `
  --input main=path/to/main_test.json `
  --input ablation=path/to/ablation_test.json `
  --input emotion=path/to/emotion_test.json `
  --input dynamic=path/to/dynamic_test.json `
  --output artifacts/paper_tables_final `
  --require-test
```

不满足测试集标记时只能生成 development/source-report 表，不能作为论文最终数值。
