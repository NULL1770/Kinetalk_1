"""Create an exploratory t-SNE plot from train/validation emotion features.

The plot is descriptive only.  It never reads the sealed test manifest and it
does not fit or select any model parameter.  Audio points use the valid-frame
mean of the cached emotion2vec intermediate representation; motion points use
observed upper-face mean, standard deviation, quantiles and adjacent-frame
velocity statistics. Both views are exported so that class separation is not
confused with a claim about generated motion quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def upper_motion_features(motion, valid, channel):
    """Keep affect posture AND dynamics; the mean of centered motion is zero."""
    from kinetalk_b0.emotion_probe import motion_features
    if motion.ndim != 2 or valid.shape != motion.shape[:1] or channel.shape != motion.shape[1:]:
        raise ValueError('Expected motion[T,C], valid[T], channel[C]')
    if valid.dtype != torch.bool or channel.dtype != torch.bool:
        raise ValueError('Observation masks must be boolean')
    upper = (41, 42, 43, 44, 45, 5, 6, 12, 13)
    x = motion[:, upper]
    observed = valid[:, None] & channel[list(upper)][None, :]
    if not channel[list(upper)].any():
        raise ValueError('No observed upper-face channels')
    if not torch.isfinite(x[observed]).all():
        raise ValueError('Nonfinite observed upper-face motion')
    # Missing channels are fixed zero dimensions, excluded from fitted variance;
    # arbitrary unobserved values must never create apparent separation.
    x = torch.where(observed, x, 0.)
    return motion_features(x, valid)


def _load_rows(root: Path, role: str):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    if role not in ("train", "val"):
        raise ValueError("Only train and val may be read")
    rows = manifest["roles"][role]["query"]
    return manifest, rows


def _feature(row, root: Path, kind: str):
    saved = torch.load(root / "clips" / (row["clip_id"] + ".pt"), map_location="cpu",
                       weights_only=False)
    valid = saved["valid"].bool()
    if not valid.any():
        raise ValueError("No valid frames: " + row["clip_id"])
    if kind == "audio":
        if "middle" not in saved:
            raise ValueError("Prepared shard lacks emotion2vec middle features: " + row["clip_id"])
        x = saved["middle"].float()
        return x[valid].mean(0).numpy()
    return upper_motion_features(saved["motion"].float(), valid,
                                 saved["channel_mask"].bool()).numpy()


def build(root: Path, output: Path, max_per_class: int | None, seed: int):
    root = root.resolve()
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(root / "manifest.json")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    if manifest.get("sealed_test_targets_loaded") is not False:
        raise ValueError("Prepared data must explicitly seal test targets")
    rows = []
    for role in ("train", "val"):
        rows.extend((role, row) for row in manifest["roles"][role]["query"])
    if max_per_class is not None:
        rng = np.random.default_rng(seed)
        selected = []
        for emotion in sorted({row["emotion"] for _, row in rows}):
            candidates = [(role, row) for role, row in rows if row["emotion"] == emotion]
            if len(candidates) > max_per_class:
                candidates = [candidates[i] for i in rng.choice(len(candidates), max_per_class, replace=False)]
            selected.extend(candidates)
        rows = selected
    audio, motion, meta = [], [], []
    for role, row in rows:
        audio.append(_feature(row, root, "audio"))
        motion.append(_feature(row, root, "motion"))
        meta.append({"clip_id": row["clip_id"], "role": role, "emotion": int(row["emotion"]),
                     "intensity": int(row["intensity"]), "speaker": row["speaker"]})
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "features.npz", audio=np.asarray(audio, np.float32),
                        motion=np.asarray(motion, np.float32))
    (output / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf8")
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("t-SNE plotting requires scikit-learn and matplotlib") from exc
    labels = np.asarray([x["emotion"] for x in meta])
    names = manifest.get("emotion_names", [str(i) for i in sorted(set(labels))])
    embeddings = {}
    train_idx = np.asarray([i for i, item in enumerate(meta) if item["role"] == "train"], dtype=np.int64)
    if len(train_idx) < 2 or len(meta) < 3:
        raise ValueError('t-SNE requires at least two TRAIN and three total clips')
    for name, values in (("audio", np.asarray(audio)), ("motion", np.asarray(motion))):
        # Fit normalization and linear reduction on train only. Validation is
        # included only as an unlabelled visualization set.
        scaler = StandardScaler().fit(values[train_idx])
        z = scaler.transform(values)
        if z.shape[1] > 50:
            reducer = PCA(n_components=min(50, len(train_idx), z.shape[1]), random_state=seed).fit(z[train_idx])
            z = reducer.transform(z)
        embeddings[name] = TSNE(n_components=2, init="pca", learning_rate="auto",
                                perplexity=min(30, max(1, (len(z) - 1) // 3)),
                                random_state=seed).fit_transform(z)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    palette = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4",
               "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    for ax, (name, xy) in zip(axes, embeddings.items()):
        for color_index, emotion in enumerate(sorted(set(labels))):
            idx = labels == emotion
            label = names[int(emotion)] if int(emotion) < len(names) else str(int(emotion))
            ax.scatter(xy[idx, 0], xy[idx, 1], s=10, alpha=.62,
                       color=palette[color_index % len(palette)],
                       label=label, edgecolors="none")
        ax.set_title(name + " t-SNE")
        ax.set_xlabel("dimension 1"); ax.set_ylabel("dimension 2")
        ax.legend(markerscale=2, fontsize=8, ncol=2)
    fig.savefig(output / "emotion_tsne.png", dpi=220)
    plt.close(fig)
    report = {"schema": "emotion_tsne_v3", "source": str(root),
              "roles_read": ["train", "val"], "test_loaded": False,
              "clips": len(meta), "emotion_names": names, "seed": seed,
              "max_per_class": max_per_class, "fit_roles": ["train"], "palette": palette,
              "features": "emotion2vec middle mean and upper9 mean/std/q10/q90/adjacent mean-abs-velocity/std-velocity",
              "motion_feature_dimension": 54,
              "scope": "Descriptive real TRAIN/validation features; not generated-motion F1 or quality certification",
              "plot": str((output / "emotion_tsne.png").resolve())}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--seed", type=int, default=47)
    a = p.parse_args()
    print(json.dumps(build(a.prepared, a.output, a.max_per_class, a.seed), ensure_ascii=False))


if __name__ == "__main__":
    main()
