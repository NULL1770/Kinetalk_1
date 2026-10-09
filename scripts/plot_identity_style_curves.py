"""Plot reference-conditioned identity/style evidence from a frozen audit.

The plot reports behavioral style transfer on the shared rig.  It does not
call the model or fit a speaker classifier, and it does not label the curves
as facial-geometry identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SPEAKER_NAMES = {9: "M025", 16: "M037", 17: "M039"}


def load(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["test_loaded"] is False
    assert data["training_performed"] is False
    assert data["default_replaced"] is False
    key = "candidate/clip_all"
    assert key in data["target_direction"]
    return data, data["target_direction"][key], {
        "own": data["response"]["candidate/clip_all/own_A_B"],
        "cross": data["response"]["candidate/clip_all/cross_AB"],
    }


def finite(values):
    return np.asarray(values, dtype=float)


def build(report: Path, output: Path):
    data, target, response = load(report)
    output.mkdir(parents=True, exist_ok=True)
    directions = target["by_speaker_direction"]
    speakers = sorted({int(k.split("->")[1]) for k in directions})
    regions = ("mouth", "brows", "all51")
    stats = ("mean", "q90_q10", "displacement_rms")

    # Each target point averages the two incoming source directions.  The
    # source report already contains clip-level aggregation and provenance;
    # this is only a deterministic presentation transform.
    rows = []
    for target_sid in speakers:
        incoming = [v["metrics"] for k, v in directions.items()
                    if int(k.split("->")[1]) == target_sid]
        row = {"target_speaker": target_sid, "target": SPEAKER_NAMES.get(target_sid, str(target_sid))}
        for region in regions:
            for stat in stats:
                for suffix in ("before", "after"):
                    vals = [x[f"{region}/{stat}_{suffix}"] for x in incoming]
                    row[f"{region}/{stat}_{suffix}"] = float(np.mean(vals))
        rows.append(row)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), constrained_layout=True)
    x = np.arange(len(rows))
    labels = [r["target"] for r in rows]
    colors = {"before": "#6b7280", "after": "#2563eb"}
    for suffix, label in (("before", "source reference"), ("after", "swapped reference")):
        axes[0].plot(x, [r[f"mouth/mean_{suffix}"] for r in rows], marker="o",
                     linewidth=2, color=colors[suffix], label=label)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("normalized mouth mean error")
    axes[0].set_title("Reference target alignment")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=.22)

    own = response["own"]["metrics"]
    cross = response["cross"]["metrics"]
    bars = [own["mouth/mae"], cross["mouth/mae"]]
    labels2 = ["same speaker\n(A↔B)", "cross speaker\n(reference swap)"]
    axes[1].plot(labels2, bars, marker="o", linewidth=2.3, color="#d97706")
    axes[1].set_ylabel("mouth MAE")
    axes[1].set_title("Style sensitivity and stability")
    axes[1].grid(axis="y", alpha=.22)
    for i, value in enumerate(bars):
        axes[1].annotate(f"{value:.3f}", (i, value), xytext=(0, 8),
                         textcoords="offset points", ha="center", fontsize=8)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"05_identity_style_curves.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=.1)
    plt.close(fig)
    protocol = {
        "schema": "identity_style_curve_v1",
        "source_report": str(report),
        "source_key": "candidate/clip_all",
        "training_performed": False,
        "test_loaded": False,
        "speaker_labels": SPEAKER_NAMES,
        "target_points": rows,
        "same_vs_cross_mouth_mae": {"same_speaker_A_B": bars[0], "cross_speaker": bars[1]},
        "interpretation": "Behavioral reference-conditioned style on a shared rig; not facial geometry identity.",
        "limits": [
            "Three development identities only.",
            "Cross-person targets are native clip statistics, not framewise counterfactual ground truth.",
            "The curves do not prove speaker recognition or geometry transfer.",
        ],
    }
    (output / "05_identity_style_curves.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.report, args.output)
    print(json.dumps({"output": str(args.output), "targets": len(result["target_points"])}))
