"""Lock metadata-selected validation405 examples, then plot stored coefficient traces.

Selection never deserializes a tensor or opens a prediction. Plotting reads only
the internal heldout cache and already generated final-epoch curves. These are
coefficient traces, not rendered faces or evidence of perceived emotion quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


SCHEMA = "projection_schedule_trace_selection_v1"
SELECTION_SEED = 20260923
NOISE_SEED = 42
EMOTIONS = {1: "angry", 5: "happy", 6: "sad"}
CHANNELS = [(41, "browDownLeft"), (43, "browInnerUp"), (5, "eyeSquintLeft"), (17, "jawOpen")]
SPEAKERS = ["mead_M023", "mead_M024", "mead_M030"]
SCOPE = "Coefficient traces only; not face videos, tracking validation or evidence of perceived emotion quality. Internal heldout identities; frozen B0/global may have seen them."


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def selection_hash(row):
    metadata = {key: row[key] for key in ("clip_id", "speaker", "speaker_id", "emotion_id", "sentence")}
    return canonical_hash({"seed": SELECTION_SEED, "metadata": metadata})


def select_examples(rows, split_lock):
    """Choose one clip per fixed speaker/emotion using metadata only."""
    if split_lock.get("schema") != "projection_schedule_internal_split_v1" or split_lock.get("source_role") != "formal_training_queries_only":
        raise ValueError("Require the training-internal data_locked split")
    for flag in ("outer_development_loaded", "new_identity_development_loaded", "outer_development_target_values_accessed", "test_target_values_accessed"):
        if split_lock.get(flag) is not False:
            raise ValueError(f"Forbidden outer development/test exposure: {flag}")
    ids = [row["clip_id"] for row in rows]
    if len(rows) != 405 or len(set(ids)) != 405 or ids != split_lock.get("validation_clip_ids"):
        raise ValueError("Require exactly the locked, ordered validation405 metadata")
    if sorted(split_lock.get("heldout_speakers", [])) != SPEAKERS or sorted({row["speaker"] for row in rows}) != SPEAKERS:
        raise ValueError("Unexpected heldout speakers")
    if set(ids) & set(split_lock.get("fit_clip_ids", [])) or set(SPEAKERS) & set(split_lock.get("fit_speakers", [])):
        raise ValueError("Fit/heldout overlap")
    for row in rows:
        if row.get("dataset") != "mead" or row.get("split") != "train" or row["emotion_id"] not in (0, 1, 5, 6):
            raise ValueError("Expected native training MEAD rows with original emotion IDs")
    selected = []
    for speaker in SPEAKERS:
        for emotion, label in EMOTIONS.items():
            candidates = [(i, row) for i, row in enumerate(rows) if row["speaker"] == speaker and row["emotion_id"] == emotion]
            if not candidates:
                raise ValueError(f"No metadata candidate for {speaker}/{emotion}")
            index, row = min(candidates, key=lambda pair: (selection_hash(pair[1]), pair[1]["clip_id"]))
            selected.append({"index": index, "clip_id": row["clip_id"], "speaker": speaker,
                "speaker_id": int(row["speaker_id"]), "emotion_id": emotion, "emotion": label,
                "sentence": row["sentence"], "metadata_hash": selection_hash(row), "eligible_count": len(candidates)})
    return selected


def make_lock(metadata_dir):
    metadata_dir = Path(metadata_dir).resolve()
    if metadata_dir.name != "data_locked":
        raise ValueError("Only the refitted data_locked source may be used")
    manifest_path, split_path = metadata_dir / "validation.jsonl", metadata_dir / "split_lock.json"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf8").splitlines() if line.strip()]
    split_lock = read_json(split_path)
    picks = select_examples(rows, split_lock)
    return {"schema": SCHEMA, "metadata_dir": str(metadata_dir), "selection_seed": SELECTION_SEED,
        "noise_seed": NOISE_SEED, "channels": [{"index": index, "name": name} for index, name in CHANNELS],
        "selection_rule": "Lowest SHA256 of canonical JSON {seed, metadata:{clip_id,speaker,speaker_id,emotion_id,sentence}} per heldout speaker and nonneutral emotion; clip_id breaks hash ties.",
        "selection_uses_metadata_only": True, "predictions_opened_during_selection": False,
        "tensor_values_deserialized_during_selection": False,
        "source_sha256": {"validation_manifest": sha(manifest_path), "split_lock": sha(split_path),
            "cache": sha(metadata_dir / "renderer_cache.pt")},
        "internal_split_lock_sha256": canonical_hash(split_lock),
        "validation_clip_ids": [row["clip_id"] for row in rows], "clips": picks, "scope": SCOPE}


def validate_query_metadata(query, lock):
    if list(query["clip_id"]) != lock["validation_clip_ids"]:
        raise ValueError("Cache clip order differs from metadata lock")
    for pick in lock["clips"]:
        index = pick["index"]
        speaker = query["speaker"][index] if "speaker" in query else "_".join(query["clip_id"][index].split("_")[:2])
        if (speaker != pick["speaker"] or int(query["speaker_id"][index]) != pick["speaker_id"]
                or int(query["emotion_id"][index]) != pick["emotion_id"] or query["sentence_id"][index] != pick["sentence"]):
            raise ValueError("Selected cache identity/emotion/sentence differs from metadata")


def load_final_curves(run, lock, *, rollout):
    import torch
    run = Path(run)
    provenance, summary = read_json(run / "provenance.json"), read_json(run / "summary.json")
    recipe = provenance["recipe"]
    digest = canonical_hash(recipe)
    schema = "projection_rollout_probe_v1" if rollout else "projection_schedule_ablation_v1"
    arm = "constant_teacher_rollout" if rollout else "constant_teacher"
    if recipe.get("schema") != schema or summary.get("schema") != schema or recipe["args"].get("arm") != arm or summary.get("arm") != arm:
        raise ValueError("Wrong final curve arm")
    if provenance.get("recipe_sha256") != digest or summary.get("recipe_sha256") != digest:
        raise ValueError("Recipe hash mismatch")
    if recipe["input_sha256"].get("cache") != lock["source_sha256"]["cache"] or recipe["input_sha256"].get("split_lock") != lock["source_sha256"]["split_lock"]:
        raise ValueError("Run does not use the metadata-locked cache/split")
    if summary.get("completed_epochs") != 18 or recipe["args"].get("epochs") != 18 or recipe["args"].get("seed") != 46:
        raise ValueError("Only fixed epoch18 seed46 is permitted")
    for value in (recipe, summary):
        for flag in ("outer280_loaded", "new_identity439_loaded", "test_loaded"):
            if value.get(flag) is not False:
                raise ValueError("Run reports forbidden data exposure")
    if summary.get("checkpoint_selection_performed") is not False or summary.get("frozen_unchanged") is not True or summary.get("head_unchanged") is not True:
        raise ValueError("Final run frozen/selection contract failed")
    curves_path = run / "final_epoch018_curves.pt"
    checkpoint_path = run / "final_epoch018.pt"
    sidecar_path = curves_path.with_name(curves_path.stem + ".provenance.json")
    sidecar = read_json(sidecar_path)
    expected = {"schema": "projection_schedule_curves_provenance_v1", "curve_sha256": sha(curves_path),
        "checkpoint_sha256": sha(checkpoint_path), "recipe_sha256": digest, "cache_sha256": lock["source_sha256"]["cache"]}
    if sidecar != expected or summary["curve_provenance"]["sha256"] != sha(sidecar_path):
        raise ValueError("Final curve/checkpoint binding mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("recipe") != recipe or checkpoint.get("recipe_sha256") != digest or checkpoint.get("completed_epochs") != 18 or checkpoint.get("selection") != "none":
        raise ValueError("Final checkpoint recipe or fixed-epoch contract mismatch")
    curves = torch.load(curves_path, map_location="cpu", weights_only=False)
    if curves.get("noise_seeds") != [42, 123, 2026] or curves.get("decode_steps") != 12 or str(NOISE_SEED) not in curves.get("motion", {}):
        raise ValueError("Stored generation seed/decode protocol changed")
    return curves["motion"][str(NOISE_SEED)], recipe, expected


def centered_residual(prediction, split):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.train_predictable_renderer import center
    baseline = split["base"]["b0"] + split["identity"]["baseline"][:, None]
    if prediction.shape != baseline.shape or prediction.shape != split["q"]["motion"].shape:
        raise ValueError("Prediction/baseline/query shapes differ")
    observed = split["q"]["valid"][:, :, None] & split["q"]["channel_mask"][:, None]
    if not torch.isfinite(prediction[observed]).all() or not torch.isfinite(baseline[observed]).all():
        raise ValueError("Nonfinite observed coefficients")
    return center(prediction.double() - baseline.double(), split["q"]["valid"].double())


def plot_locked_examples(lock_file, cache_path, constant_run, rollout_run, output):
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.audit_projection_rollout_probe import validate_recipe_pair

    lock = read_json(lock_file)
    if lock != make_lock(lock["metadata_dir"]):
        raise ValueError("Metadata/selection/cache changed after the sample lock")
    if Path(cache_path).resolve() != Path(lock["metadata_dir"]) / "renderer_cache.pt" or sha(cache_path) != lock["source_sha256"]["cache"]:
        raise ValueError("Plotting cache must be the same data_locked source")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
    if cache.get("schema") != "predictable_renderer_cache_v1" or cache["provenance"].get("internal_split_lock_sha256") != lock["internal_split_lock_sha256"]:
        raise ValueError("Cache internal-split lineage mismatch")
    if cache["provenance"].get("manifest_hashes", {}).get("validation") != lock["source_sha256"]["validation_manifest"]:
        raise ValueError("Cache validation manifest binding mismatch")
    split = cache["splits"]["validation"]
    query = split["q"]
    validate_query_metadata(query, lock)
    constant, constant_recipe, constant_binding = load_final_curves(constant_run, lock, rollout=False)
    rollout, rollout_recipe, rollout_binding = load_final_curves(rollout_run, lock, rollout=True)
    validate_recipe_pair(rollout_recipe, constant_recipe)
    zero_delta = (constant["zero"] - rollout["zero"]).abs()
    valid_mask = query["valid"][:, :, None] & query["channel_mask"][:, None]
    zero_max = float(zero_delta[valid_mask].max())
    series = [("Target residual", query["motion"], "#111111", "-", 1.8),
        ("Zero local (rollout)", rollout["zero"], "#94a3b8", "--", 1.3),
        ("Constant flow: audio", constant["full"], "#d97706", "-", 1.2),
        ("Rollout: audio", rollout["full"], "#2563eb", "-", 1.4),
        ("Rollout: motion oracle", rollout["oracle"], "#16a34a", ":", 1.5)]
    if zero_max != 0:
        series.append(("Zero local (constant flow)", constant["zero"], "#7c3aed", "--", 1.0))
    centered = [(label, centered_residual(values, split), color, style, width) for label, values, color, style, width in series]
    limits = {}
    for channel, _ in CHANNELS:
        values = [curve[pick["index"], query["valid"][pick["index"]], channel]
            for _, curve, *_ in centered for pick in lock["clips"] if query["channel_mask"][pick["index"], channel]]
        if not values:
            continue
        values = torch.cat(values)
        low, high = float(values.min()), float(values.max())
        pad = max((high - low) * .06, 1e-5)
        limits[channel] = (low - pad, high + pad)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for pick in lock["clips"]:
        index = pick["index"]
        valid = query["valid"][index].bool()
        times = query["times"][index].double().numpy().copy()
        if not valid.any() or not np.isfinite(times[valid.numpy()]).all():
            raise ValueError("No finite valid native timestamps")
        times -= times[valid.numpy()][0]
        times[~valid.numpy()] = np.nan
        fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(12, 9), sharex=True)
        for ax, (channel, name) in zip(axes, CHANNELS):
            if query["channel_mask"][index, channel]:
                for label, curve, color, style, width in centered:
                    values = curve[index, :, channel].numpy().copy()
                    values[~valid.numpy()] = np.nan
                    ax.plot(times, values, color=color, linestyle=style, linewidth=width, label=label)
                ax.set_ylim(*limits[channel])
            else:
                ax.text(.5, .5, "Unobserved channel", transform=ax.transAxes, ha="center")
            ax.axhline(0, color="#e5e7eb", linewidth=.6)
            ax.set_ylabel(name)
            ax.grid(alpha=.2)
        axes[0].legend(ncol=3, fontsize=8, loc="upper right")
        axes[-1].set_xlabel("Native time (s); subtract cached B0 + identity, then each residual's valid-frame temporal mean")
        fig.suptitle(f"{pick['clip_id']} | {pick['emotion']} | noise42 | final epoch18\nCoefficient traces only; fixed metadata selection, no lag alignment or amplitude rescaling", fontsize=10)
        fig.tight_layout()
        path = output / f"{pick['speaker']}_{pick['emotion_id']}_{pick['emotion']}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        files.append({"path": path.name, "sha256": sha(path), "clip": pick})
    from PIL import Image
    thumbnails = []
    for item in files:
        with Image.open(output / item["path"]) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((840, 630))
            thumbnails.append(thumb.copy())
    montage = Image.new("RGB", (840 * 3, 630 * 3), "white")
    for index, thumb in enumerate(thumbnails):
        montage.paste(thumb, ((index % 3) * 840, (index // 3) * 630))
    montage.save(output / "montage.png")
    report = {"schema": "projection_schedule_coefficient_traces_v1", "selection_lock_sha256": sha(lock_file),
        "selection_seed": SELECTION_SEED, "noise_seed": NOISE_SEED, "final_epoch": 18,
        "curve_provenance": {"constant_flow": constant_binding, "rollout": rollout_binding},
        "zero_local_equal_exactly": zero_max == 0, "zero_local_max_abs_difference_observed": zero_max,
        "zero_local_display": "both zero curves shown separately" if zero_max != 0 else "rollout zero represents both identical controls",
        "centering": "center(prediction - cached_b0 - cached_identity_baseline, valid); same for target",
        "channel_y_limits_shared_across_clips": limits, "invalid_frames": "NaN gaps; no interpolation",
        "amplitude_rescaling": False, "lag_alignment": False, "files": files, "scope": SCOPE,
        "outer280_loaded": False, "new_identity439_loaded": False, "test_loaded": False}
    write_new_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--lock", action="store_true")
    mode.add_argument("--plot", action="store_true")
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--constant-run", type=Path)
    parser.add_argument("--rollout-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.lock:
        if args.metadata_dir is None or any(value is not None for value in (args.lock_file, args.cache, args.constant_run, args.rollout_run)):
            parser.error("--lock accepts only --metadata-dir and --output")
        value = make_lock(args.metadata_dir)
        write_new_json(args.output, value)
        print(json.dumps({"lock": str(args.output), "clips": value["clips"], "tensor_values_loaded": False}, ensure_ascii=False))
    else:
        if any(value is None for value in (args.lock_file, args.cache, args.constant_run, args.rollout_run)) or args.metadata_dir is not None:
            parser.error("--plot requires --lock-file --cache --constant-run --rollout-run --output")
        value = plot_locked_examples(args.lock_file, args.cache, args.constant_run, args.rollout_run, args.output)
        print(json.dumps({"output": str(args.output), "plots": len(value["files"]), "zero_local_equal_exactly": value["zero_local_equal_exactly"], "scope": SCOPE}))


if __name__ == "__main__":
    main()
