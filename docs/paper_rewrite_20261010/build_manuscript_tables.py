"""Regenerate manuscript tables from existing reports; never train or rescore."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "final_experiment/paper_tables/phase53_development_20261009"
OUTPUT = ROOT / "paper_draft/作者记录_20261010/tables"
OUTPUT.mkdir(parents=True, exist_ok=True)
FILES = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(name):
    path = SOURCE / name
    FILES[str(path.relative_to(ROOT))] = digest(path)
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def indexed(name):
    return {r["method"]: r for r in read_csv(name)}


def read_json(relative):
    path = ROOT / relative
    FILES[relative] = digest(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


GEOM = indexed("table1_geometry_clip_all.csv")
COEF = indexed("table2_coefficients_clip_all.csv")
DYN = indexed("table3_dynamics_semantics_clip_all.csv")
PROTOCOL = read_csv("table8_training_protocol.csv")
ABLATION = read_csv("table4_trained_ablation.csv")
INTERVENTION = read_csv("table5_inference_interventions.csv")
EMOTION = read_csv("table6_emotion_breakdown.csv")
GAP = read_json("final_experiment/evaluation/diagnostics/validation_gap_20261005/report.json")
STYLE = read_json("final_experiment/evaluation/diagnostics/phase53_receiver_20261009/style_only/style_audit/report.json")
ALL = read_json("final_experiment/paper_tables/phase53_development_20261009/tables.json")
assert not STYLE["test_loaded"] and STYLE["matched_directed_pairs"] == 2026
assert ALL["validation_only"] and not ALL["test_loaded"]

NAMES = {
    "VOCA-core (shared audio/ARKit)": "VOCA-core（适配）",
    "FaceFormer (shared audio/ARKit)": "FaceFormer（适配）",
    "EmoTalk-core (shared audio/ARKit)": "EmoTalk-core（适配）",
    "FaceDiffuser (frozen KineTalk conditions/ARKit)": "FaceDiffuser（KineTalk 条件适配）",
    "Phase53-style_only": "KineTalk",
}
METHODS = list(NAMES)
OURS = METHODS[-1]
TABLES = {}
RECORDS = {}


def values(source, method, fields):
    return [float(source[method][key]) for key in fields]


def table(key, headers, rows, decimals=None, directions=None, reference_rows=()):
    decimals = decimals or [4] * (len(headers)-1)
    directions = directions or [None] * (len(headers)-1)
    assert all(len(r) == len(headers) for r in rows)
    best = {}
    for col, direction in enumerate(directions, 1):
        pool = [r[col] for i, r in enumerate(rows)
                if i not in reference_rows and isinstance(r[col], (float, int))]
        if direction and pool:
            best[col] = (min if direction == "min" else max)(pool)
    display = []
    for i, row in enumerate(rows):
        cells = [row[0]]
        for j, value in enumerate(row[1:], 1):
            if not isinstance(value, (float, int)):
                cells.append(value)
                continue
            cell = f"{value:.{decimals[j-1]}f}"
            if i not in reference_rows and best.get(j) == value:
                cell = "**" + cell + "**"
            cells.append(cell)
        display.append("| " + " | ".join(cells) + " |")
    result = "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "---|" + "---:|"*(len(headers)-1),
        *display,
    ])
    TABLES[key] = result
    RECORDS[key] = dict(headers=headers, rows=rows, decimals=decimals,
                        directions=directions, reference_rows=list(reference_rows))
    with (OUTPUT / (key+".csv")).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    return result


geom_fields = ["vertex_lve_sqrt_mean", "eye_forehead_eve_sqrt_mean"]
table("geometry", ["方法", "MBE↓", "LBE↓", "LVE/mm↓", "EVE/mm↓"],
      [[NAMES[m], *values(COEF, m, ["arkit_mbe", "arkit_lbe"]),
        *values(GEOM, m, geom_fields)] for m in METHODS], directions=["min"]*4)

gt_range = next(r["jawOpen/gt_q90_q10"] for r in ALL["rows"]["clip_all"]
                if r["method"] == OURS)
table("dynamics", ["方法", "口部位移 MSE↓", "下颌相关↑", "下颌范围"],
      [[NAMES[m], *values(DYN, m, ["mouth_displacement_mse", "jaw_centered_correlation",
                                  "jaw_q90_q10"])] for m in METHODS]
      + [["真实动作", "—", "—", gt_range]],
      decimals=[6, 4, 4], directions=["min", "max", None], reference_rows=[5])

f1_fields = [f"F1_{i}" for i in range(1, 5)]
table("semantics", ["方法", "原始-128↑", "原始-64↑", "稳定-128↑", "稳定-64↑"],
      [[NAMES[m], *values(DYN, m, f1_fields)] for m in METHODS]
      + [["真实动作（读出参照）", *[r["macro_f1"] for r in GAP["gt_probes"]]]],
      directions=["max"]*4, reference_rows=[5])

for key, methods, labels in [
    ("reference_ablation", ["Phase51-temporal", "Phase51-statistics"],
     ["时序参考编码", "统计参考编码"]),
    ("scope_ablation", ["Phase53-joint", OURS],
     ["参考编码器与完整解码器", "仅参考相关参数（KineTalk）"]),
]:
    table(key, ["设置", "MBE↓", "LBE↓", "LVE/mm↓", "EVE/mm↓"],
          [[label, *values(COEF, m, ["arkit_mbe", "arkit_lbe"]), *values(GEOM, m, geom_fields)]
           for m, label in zip(methods, labels)], directions=["min"]*4)

ints = {r["intervention"]: r for r in INTERVENTION if r["method"] == OURS}
labels = {"normal": "原生局部条件", "static_u": "静态局部条件",
          "reverse_u": "时间反转", "shuffle_u": "时间打乱"}
assert all(int(ints[k]["n"]) == 96 for k in labels)
table("interventions", ["设置", "MBE↓", "LBE↓", "下颌相关↑", "下颌范围"],
      [[label, *[float(ints[k][v]) for v in ["arkit_mbe", "arkit_lbe",
         "jaw_centered_correlation", "jaw_q90_q10"]]] for k, label in labels.items()],
      directions=["min", "min", "max", None])

table("style_stability",
      ["参考替换", "查询对数", "全脸 MAE", "全脸相关", "口部相关", "闭口分歧"],
      [[label, STYLE["response"][key]["n"],
        *[STYLE["response"][key]["metrics"][v] for v in ["all51/mae",
          "all51/centered_correlation", "mouth/centered_correlation", "jaw_closure_disagreement"]]]
       for label, key in [("同人 A/B", "candidate/raw/own_A_B"),
                          ("跨人聚合参考", "candidate/raw/cross_AB")]],
      decimals=[0, 5, 4, 4, 4])

target = STYLE["target_direction"]["candidate/raw"]["metrics"]
table("style_target", ["统计量", "源参考误差↓", "目标参考误差↓", "改善比例/%↑"],
      [[label, target["all51/"+k+"_before"], target["all51/"+k+"_after"],
        100*target["all51/"+k+"_improved"]]
       for label, k in [("均值", "mean"), ("中心化 RMS", "centered_rms"),
                         ("分位幅度", "q90_q10"), ("位移 RMS", "displacement_rms")]],
      decimals=[4, 4, 2])

table("emotion_breakdown", ["情感", "原始-128", "原始-64", "稳定-128", "稳定-64"],
      [[r["emotion"], *[float(r[k]) for k in f1_fields]]
       for r in EMOTION if r["method"] == OURS])

supplement_rows = []
for policy in ("raw", "clip_all"):
    for r in ALL["rows"][policy]:
        if r["method"] not in NAMES:
            continue
        supplement_rows.append([NAMES[r["method"]]+"/"+policy,
          *[r[k] for k in ["arkit_mbe", "arkit_lbe", *geom_fields,
                          "lve_mean_mm_mean", "eve_mean_mm_mean",
                          "arkit_fdd_absolute", "vertex_fdd_absolute_mm2_mean"]]])
table("supplementary_geometry",
      ["方法/输出策略", "MBE", "LBE", "LVE/mm", "EVE/mm", "唇部平均距离/mm",
       "表达平均距离/mm", "系数FDD绝对值", "顶点FDD绝对值/mm²"], supplement_rows)

template = (Path(__file__).parent / "实验正文模板.md").read_text(encoding="utf-8")
for key, value in TABLES.items():
    template = template.replace("{{"+key+"}}", value)
assert "{{" not in template
(ROOT/"paper_draft/04_实验_中文版.md").write_text(template, encoding="utf-8", newline="\n")

appendix = """# 附录 A 补充结果与实现

## A.1 输出策略与补充几何指标

表 A1 同时给出未裁剪（raw）和裁剪（clip_all）结果。LVE/EVE 采用逐帧区域最大顶点距离；“平均距离”对区域内顶点取均值。系数 FDD 为上脸区域平方系数能量的时间标准差之差的绝对值，顶点 FDD 在对应顶点能量上计算，单位为 mm²。FDD 描述运动能量变化，不衡量事件发生的时间位置。

**表 A1　两种输出策略下的完整几何及系数统计。**

"""+TABLES["supplementary_geometry"]+"""

## A.2 逐情感读出

**表 A2　KineTalk 的逐情感 F1。所有类别使用同一生成检查点及固定读出器。**

"""+TABLES["emotion_breakdown"]+"""

## A.3 参考编码与训练实现

表达预训练阶段的时序参考编码器输入为归一化参考残差、参考基座及通道掩码，共 156 维。三层 TCN 编码后进行有效帧池化与 64 维投影，再跨参考聚合。最终参考适配将其替换为正文的统计编码器，保留先验、后验、表达分类头及解码器中可复用的响应参数。新参考调制权重从零初始化，姿态头和统计编码器重新学习。编码器替换对照沿用各自结构对应的初始化，因此同时反映参考表示及其接入方式。

训练尺度由拟合分区动作相对于中性参考锚点的逐通道 RMS 计算，下限为 0.02。音频特征按拟合分区均值与标准差归一化，标准差下限为 0.001。高斯编码器的对数方差限制为 [-6, 2]。原生索引的位置编码及有效帧掩码用于时间注意力，局部条件以固定 token 中心插值。

参考适配对统计编码器和姿态头使用 0.01 的权重衰减；同时包含冻结表达列和可学习参考列的调制矩阵不进行整矩阵衰减，并屏蔽表达列梯度。没有中性查询的批次跳过姿态参数更新；各情感查询均可更新参考响应分支。

情感读出器原始输入为 306 维，稳定版本仅保留训练标准差大于 0.00010001 的 279 个统计特征。两个版本分别使用 128 或 64 维隐藏层、ReLU 和 0.1 Dropout。标准化只由真实训练动作估计，权重轮次按真实开发动作 macro-F1 选定后冻结。macro-F1 衡量统计表征中的类别可读性；生成动作的分数高于真实动作参照，并不等价于更高的感知质量。
"""
(ROOT/"paper_draft/06_附录_中文版.md").write_text(appendix, encoding="utf-8", newline="\n")
sections = [
    (ROOT/"paper_draft"/name).read_text(encoding="utf-8").strip()
    for name in ["01_摘要_中文版.md", "01_引言_中文版.md", "02_相关工作_中文版.md",
                 "03_方法_中文版.md", "04_实验_中文版.md", "05_讨论与结论_中文版.md"]
]
related, references = sections[2].split("## 参考文献", 1)
sections[2] = related.strip()
combined = ("# KineTalk：基于中性发音基座与参考运动风格的情感三维面部动画\n\n"
            + "\n\n".join(sections) + "\n\n# 参考文献\n\n" + references.strip()
            + "\n\n" + appendix.strip() + "\n")
(ROOT/"paper_draft/论文_当前中文版.md").write_text(combined, encoding="utf-8", newline="\n")
provenance = dict(
    schema="manuscript_tables_20261010_v1", scope="development; no new experiment",
    model="Phase53-style_only", checkpoint_sha256=STYLE["candidate_checkpoint_sha256"],
    main_policy="clip_all", style_policy="raw", test_loaded=False,
    sources=FILES, tables=RECORDS,
    output_sha256={p.name:digest(p) for p in OUTPUT.glob("*.csv")},
)
(OUTPUT/"provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+"\n",
                                     encoding="utf-8", newline="\n")
print(json.dumps({"tables":len(TABLES), "source_files":len(FILES),
                  "model":provenance["model"], "test_loaded":False}, ensure_ascii=False))
