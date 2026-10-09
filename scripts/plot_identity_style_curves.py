"""Plot reference-conditioned identity/style evidence from a frozen audit.

The plot reports behavioral style transfer on the shared rig.  It does not
call the model or fit a speaker classifier, and it does not label the curves
as facial-geometry identity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def time_curves(report, data, output):
    """Plot all eight fixed audit utterances without smoothing or retiming."""
    artifacts={}; features=('Jaw opening','Smile mean','Inner brow raise','Smile L - R')
    for key, receipt in data['render_inputs'].items():
        emotion=key.split('/')[-1]
        path=report.parent/'render_inputs'/('candidate_'+emotion+'.npz')
        assert sha(path)==receipt['sha256']
        with np.load(path,allow_pickle=False) as z:
            motions=z['motions'].copy();times=z['times'].copy();valid=z['valid'].copy()
            names=z['mode_names'].tolist();channels=z['channels'].tolist()
            support=z['channel_mask'].copy();clip=str(z['clip_id'].item())
        if (motions.shape!=(6,len(times),52) or not valid.all()
                or not np.isclose(np.diff(times),.04,rtol=1e-5,atol=1e-7).all()):
            raise ValueError('Native contiguous display span required')
        ix=[channels.index(n) for n in ('jawOpen','mouthSmileLeft','mouthSmileRight','browInnerUp')]
        if not support[ix].all(): raise ValueError('Selected channels must be observed')
        # Same clipping convention as the aggregate panel. Targets remain raw.
        rendered=motions.copy();rendered[1:]=rendered[1:].clip(0.,1.)
        jaw,left,right,brow=[rendered[:,:,j] for j in ix]
        traces=np.stack([jaw,(left+right)/2.,brow,left-right],axis=-1)
        fig,axes=plt.subplots(2,4,figsize=(12,5.1),sharex=True,sharey='col')
        colors=['#2563eb','#d97706','#059669']
        for j,feature in enumerate(features):
            for k,color in zip((1,2,3),colors):
                axes[0,j].plot(times,traces[k,:,j],color=color,lw=1.3,label=names[k])
            axes[0,j].plot(times,traces[0,:,j],color='#6b7280',lw=.9,ls='--',alpha=.75,label='Source GT')
            for k,color in zip((4,5),('#7c3aed','#e11d48')):
                axes[1,j].plot(times,traces[k,:,j],color=color,lw=1.3,label=names[k])
            axes[0,j].set_title(feature);axes[1,j].set_xlabel('Native audio time (s)')
            for a in axes[:,j]: a.grid(axis='y',alpha=.2)
        axes[0,0].set_ylabel('Reference swap\nCoefficient');axes[1,0].set_ylabel('Same-person A / B\nCoefficient')
        axes[0,0].legend(loc='upper left',fontsize=7,frameon=False,ncol=2)
        axes[1,0].legend(loc='upper left',fontsize=7,frameon=False)
        fig.suptitle('Fixed audio and expression; reference-conditioned motion style — '+emotion,fontsize=11)
        fig.tight_layout(rect=(0,0,1,.94))
        stem='05b_style_native_'+emotion
        for ext in ('png','pdf','svg'):
            fig.savefig(output/(stem+'.'+ext),dpi=300,bbox_inches='tight',pad_inches=.1)
        plt.close(fig)
        with (output/(stem+'.csv')).open('w',encoding='utf8',newline='') as f:
            writer=csv.writer(f);writer.writerow(['clip_id','mode','time_s',*features])
            for k,name in enumerate(names):
                for i,t in enumerate(times):writer.writerow([clip,name,float(t),*traces[k,i].tolist()])
        artifacts[emotion]=dict(source=str(path.resolve()),sha256=sha(path),clip_id=clip,
            frames=len(times),modes=names,plot_policy='clip_generated_0_1_GT_raw',
            no_smoothing=True,no_retiming=True,features=list(features))
    return artifacts


def build(report: Path, output: Path):
    data, target, response = load(report)
    output.mkdir(parents=True, exist_ok=True)
    directions = target["by_speaker_direction"]
    speakers = sorted({int(k.split("->")[1]) for k in directions})
    regions = ("mouth", "brows", "all51")
    stats = ("mean", "q90_q10", "displacement_rms")

    # Weight incoming directions by their pair counts, preserving the
    # original clip/pair aggregation rather than averaging unequal groups.
    rows = []
    for target_sid in speakers:
        incoming = [v for k, v in directions.items()
                    if int(k.split("->")[1]) == target_sid]
        row = {"target_speaker": target_sid, "target": SPEAKER_NAMES.get(target_sid, str(target_sid))}
        for region in regions:
            for stat in stats:
                for suffix in ("before", "after"):
                    vals = [x['metrics'][f"{region}/{stat}_{suffix}"] for x in incoming]
                    row[f"{region}/{stat}_{suffix}"] = float(np.average(vals,weights=[x['n'] for x in incoming]))
        rows.append(row)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), constrained_layout=True)
    x = np.arange(len(rows))
    labels = [r["target"] for r in rows]
    colors = {"before": "#6b7280", "after": "#2563eb"}
    for suffix, label in (("before", "source reference"), ("after", "swapped reference")):
        offset=-.055 if suffix=='before' else .055
        axes[0].plot(x+offset, [r[f"mouth/mean_{suffix}"] for r in rows], marker="o",
                     linestyle='none', color=colors[suffix], label=label)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("normalized mouth mean error")
    axes[0].set_title("Reference target alignment")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=.22)

    own = response["own"]["metrics"]
    cross = response["cross"]["metrics"]
    bars = [own["mouth/mae"], cross["mouth/mae"]]
    labels2 = ["same speaker\n(A↔B)", "cross speaker\n(reference swap)"]
    axes[1].bar(labels2, bars, width=.5,color=['#7c3aed','#d97706'])
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
        "source_report_sha256":sha(report),
        "source_key": "candidate/clip_all",
        "training_performed": False,
        "test_loaded": False,
        "speaker_labels": SPEAKER_NAMES,
        "target_points": rows,
        "aggregation":"pair-count-weighted incoming source directions",
        "same_vs_cross_mouth_mae": {"same_speaker_A_B": bars[0], "cross_speaker": bars[1]},
        "interpretation": "Behavioral reference-conditioned style on a shared rig; not facial geometry identity.",
        "limits": [
            "Three development identities only.",
            "Cross-person targets are native clip statistics, not framewise counterfactual ground truth.",
            "The curves do not prove speaker recognition or geometry transfer.",
            "Source GT in time plots is context for the source performance, not a target-person counterfactual.",
        ],
    }
    protocol['native_time_curves']=time_curves(report,data,output)
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
