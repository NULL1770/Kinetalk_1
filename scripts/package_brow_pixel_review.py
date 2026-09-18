"""Plot locked pixel/BS evidence without turning it into motion ground truth."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.diagnose_brow_state_nuisance import sha, write_json
from scripts.package_sparse_brow_teacher import checked, href


def package(pixel, states, nuisance, analysis, state_review, output):
    pixel, states, nuisance, analysis, state_review, output = map(
        Path, (pixel, states, nuisance, analysis, state_review, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("fresh output required")
    read = lambda p: json.loads(p.read_text(encoding="utf8"))
    roots = {"pixel": pixel, "states": states, "nuisance": nuisance}
    for root in (*roots.values(), analysis):
        for name, rec in read(root / "manifest.json").items():
            checked(root / name, rec)
    result = read(analysis / "analysis.json")
    # The association report already validates clock/selection and all arrays;
    # bind the plot to exactly those audited input manifests.
    recorded = result["manifest_inputs_verified"]
    actual = {k: sha(v / "manifest.json") for k, v in roots.items()}
    if recorded != actual:
        raise ValueError("association input manifest mismatch")
    summary = read(pixel / "summary.json")
    rows = read(states / "clips.json")
    ids = {r["source_id"] for r in rows}
    if ids != {r["source_id"] for r in summary["clips"]}:
        raise ValueError("pixel/state clip mismatch")
    output.mkdir(parents=True, exist_ok=True)
    (output / "curves").mkdir()
    cards = []
    for row in rows:
        cid = row["source_id"]
        arrays = []
        for root in (pixel, states, nuisance):
            with np.load(root / "arrays" / (cid + ".npz"), allow_pickle=False) as z:
                arrays.append({k: z[k].copy() for k in z.files})
        p, s, n = arrays
        times = p["times"]
        fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True)
        for group, inds, color in (("raise", [2, 3, 4], "#a04120"), ("down", [0, 1], "#224b9a")):
            y = s["smooth5"][:, inds].mean(1)
            y = np.where(s["valid"], y, np.nan)
            axes[0].plot(times, y, label=group, color=color)
            delta = np.diff(y)
            delta[~p["pair_valid"][1:]] = np.nan
            axes[3].plot(times[1:], delta, label="delta " + group, color=color, lw=.8)
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("BS [0,1]")
        for j, side in enumerate(("right", "left")):
            ax = axes[1+j]
            good = p["corrected_observed"][:, j]
            raw = np.where(good, p["raw_normalized"][:, j, 1], np.nan)
            ax.plot(times, raw, color="#b0b0b0", label="raw on shared support", lw=.8)
            ax.plot(times, p["corrected_normalized"][:, j, 1], color="#147568", label="2D rigid-corrected", lw=.8)
            ax.axhline(0, color="#333", lw=.4)
            ax.set_ylim(-.04, .04)
            ax.set_ylabel(side + " dy / eye dist")
            ax.text(.01, .92, f"{good.sum()} / {p['pair_valid'].sum()} observed pairs", transform=ax.transAxes, fontsize=8)
        axes[3].set_ylim(-.15, .15)
        axes[3].set_ylabel("BS delta / frame")
        for j, label in enumerate(("blink L", "blink R")):
            axes[4].plot(times, n["nuisance_values"][:, j], label=label, lw=.8)
        axes[4].set_ylim(0, 1)
        axes[4].set_xlabel("native seconds; image y positive downward; NaN remains missing")
        axes[4].set_ylabel("same-tracker blink")
        for ax in axes:
            ax.legend(fontsize=7, loc="upper right")
        fig.suptitle(cid + " | diagnostic only; fixed axes may crop extreme motion", fontsize=10)
        fig.tight_layout()
        dest = output / "curves" / (cid + ".png")
        fig.savefig(dest, dpi=110)
        plt.close(fig)
        overlay = pixel / "visual" / (cid + "_initial_roi.jpg")
        cards.append(f'<section id="{cid}"><h2>{cid}</h2><img loading="lazy" src="{href(dest,output)}">'
                     f'<details><summary>初始ROI：绿为稳定区，红/蓝为眉区</summary><img class="roi" src="{href(overlay,output)}"></details>'
                     f'<a href="{href(state_review,output)}#{cid}">原视频、阶段与端点重构</a></section>')
    table = ""
    for group in ("raise", "down"):
        values = result["aggregate"][group]["bilateral_mean"]["smooth"]
        table += "<tr><td>" + group + "</td>" + "".join(
            f'<td>{values[arm]["pearson_expected_sign"]:.3f}</td>' for arm in ("all", "blink_near", "nonblink")) + "</tr>"
    intro = ('<h1>眉区像素运动与系数方向核验</h1><p><b>这是原视频监督诊断，不是新模型生成结果。</b>'
             '32片3505帧逐帧RGB哈希匹配。双眉各3365/3473对可测（96.89%）；M003 happy L3 028仅前21/129对可测，后108对明确缺失，其余31片全部可测。</p>'
             '<p>仅每个有效run首帧用landmark定位区域，后续用像素LK跟踪，扣除稳定脸区的二维相似变换。'
             '剩余图像位移不是解剖眉高或独立真值；三维头姿、皮肤、光照与漂移仍可能影响它。'
             '表中按抬眉对应−dy、压眉对应+dy校正方向。未搜索时间偏移，未用该相关删除标签。</p>'
             '<table><tr><th>平滑系数差分 vs 双眉平均位移</th><th>全部</th><th>眨眼邻域</th><th>其余</th></tr>' + table + '</table>'
             '<p>机械阶段覆盖不足只说明完整起落教师的适用范围有限，不能推出所有运动先验都无法训练。'
             '整片相关不能保证局部事件阶段正确，当前没有新的神经训练任务。</p>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>眉部像素证据</title><style>body{max-width:1350px;margin:auto;padding:28px;font:16px/1.6 system-ui;background:#eef2f7;color:#172235}section{background:white;padding:18px;margin:24px 0;border-radius:10px}img{width:100%}.roi{max-width:720px}td,th{padding:8px;border:1px solid #ccd}table{border-collapse:collapse}h2{font-size:18px}</style>' + intro + ''.join(cards) + '</html>'
    (output / "index.html").write_text(page, encoding="utf8")
    write_json(output / "provenance.json", {"input_manifests": actual,
               "analysis_manifest_sha256": sha(analysis / "manifest.json"),
               "code_sha256": sha(__file__), "clips": len(rows), "training_result": False})
    return {"clips": len(rows), "path": str(output / "index.html")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pixel", "states", "nuisance", "analysis", "state-review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(package(a.pixel, a.states, a.nuisance, a.analysis, a.state_review, a.output)))
