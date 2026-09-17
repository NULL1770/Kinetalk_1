"""Plot existing temporal-refiner audit summaries; no fitting or new scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARMS = (("ridge", "Frozen ridge", "#476B91"),
        ("pointwise", "+ Pointwise", "#C58A43"),
        ("temporal", "+ Temporal", "#B45764"))
GROUPS = (("brows", "Brows"), ("eyes_expression", "Eyes"), ("mouth", "Mouth"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_values(report):
    rows = {}
    for arm, _, _ in ARMS:
        comparison = report["comparisons"][arm + "__vs__zero"]
        if list(comparison["candidate"]) != [arm, "full"] or list(comparison["baseline"]) != ["ridge", "zero"]:
            raise ValueError("Expected full prediction compared with shared ridge zero")
        rows[arm] = {}
        for group, _ in GROUPS:
            row = comparison["populations"]["nonneutral"][group]
            baseline = row["baseline"]
            if baseline["r2_against_zero"] != 0 or baseline["prediction_rms"] != 0:
                raise ValueError("Expected a true zero baseline")
            candidate = row["candidate"]
            if not np.isclose(row["r2_improvement"], candidate["r2_against_zero"], atol=1e-12, rtol=0):
                raise ValueError("R2 differs from the paired improvement against zero")
            values = {"r2": candidate["r2_against_zero"], "r2_ci95": row["r2_improvement_ci95"],
                      "amplitude_ratio": candidate["prediction_rms_amplitude_ratio"],
                      "correlation": candidate["pooled_centered_correlation"]}
            if not np.isfinite([values["r2"], *values["r2_ci95"], values["amplitude_ratio"], values["correlation"]]).all():
                raise ValueError("Missing or nonfinite report scores")
            rows[arm][group] = values
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    study = args.study.resolve()
    report_path = study / "paired_audit.json"
    report = json.loads(report_path.read_text(encoding="utf8"))
    rows = read_values(report)
    outputs = [study / "summary.png", study / "summary.svg", study / "plot_provenance.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Fresh summary.png/svg and plot_provenance.json required")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.5))
    x = np.arange(len(GROUPS))
    for number, (arm, label, color) in enumerate(ARMS):
        offset = (number - 1) * .2
        data = [rows[arm][group] for group, _ in GROUPS]
        r2 = np.array([r["r2"] for r in data])
        ci = np.array([r["r2_ci95"] for r in data])
        axes[0].errorbar(x + offset, r2, yerr=np.stack([r2 - ci[:, 0], ci[:, 1] - r2]),
                         fmt="o", markersize=6, capsize=3, color=color, label=label,
                         linewidth=1.5, linestyle="none")
        for ax, key in zip(axes[1:], ("amplitude_ratio", "correlation")):
            ax.bar(x + offset, [r[key] for r in data], width=.18, color=color, label=label)
    axes[0].set_title("Prediction accuracy", loc="left", fontweight="bold")
    axes[0].set_ylabel("R² against zero; 95% sentence-bootstrap CI")
    axes[0].set_ylim(-.02, .12)
    axes[0].axhline(0, color="#77818A", linewidth=.9)
    axes[1].set_title("Motion amplitude", loc="left", fontweight="bold")
    axes[1].set_ylabel("Prediction RMS / target RMS")
    axes[1].set_ylim(0, 1)
    axes[1].axhline(1, color="#77818A", linewidth=.9, linestyle="--")
    axes[1].text(1.02, 1.01, "Target amplitude = 1", transform=axes[1].get_yaxis_transform(),
                 ha="right", va="bottom", color="#5C6872", fontsize=9)
    axes[2].set_title("Temporal correspondence", loc="left", fontweight="bold")
    axes[2].set_ylabel("Pooled centered correlation")
    axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.set_xticks(x, [label for _, label in GROUPS])
        ax.set_xlim(-.55, 2.55)
        ax.grid(axis="y", color="#DDE2E7", linewidth=.7)
        ax.set_axisbelow(True)
    fig.suptitle("Audio-only OOF diagnostic: larger motion did not improve prediction",
                 fontsize=16, fontweight="bold", x=.06, ha="left", y=.98)
    fig.text(.06, .89, "19 existing training identities · 2,315 clips · three sentence folds · fixed epoch 8 · nonneutral scores",
             color="#5C6872", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .86), ncol=3, frameon=False)
    fig.text(.06, .11, "R² panel uses a shared zoomed range across all regions. Amplitude and correlation use 0–1 scales.",
             fontsize=9, color="#5C6872")
    fig.text(.06, .065, "Descriptive training-internal result; intervals condition on saved OOF predictions. No renderer trained. Entry checks failed.",
             fontsize=9, color="#5C6872")
    fig.subplots_adjust(left=.065, right=.975, bottom=.24, top=.75, wspace=.32)
    fig.savefig(outputs[0], dpi=180, facecolor="white")
    fig.savefig(outputs[1], facecolor="white")
    plt.close(fig)
    provenance = {"schema": "temporal_refiner_report_plot_v1", "source_report": str(report_path),
        "source_report_sha256": sha(report_path), "script_sha256": sha(__file__),
        "source_training_input_sha256": report["provenance"]["input_sha256"],
        "evaluation_added": False, "training_performed": False,
        "population": "nonneutral", "values": rows,
        "ci_definition": "Existing paired 95% sentence-bootstrap R2 improvements over exact zero; equivalent to R2 intervals.",
        "axes": {"r2": [-.02, .12], "amplitude_ratio": [0, 1], "correlation": [0, 1]},
        "output_sha256": {path.name: sha(path) for path in outputs[:2]}}
    outputs[2].write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    print(json.dumps({"output": [str(path) for path in outputs]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
