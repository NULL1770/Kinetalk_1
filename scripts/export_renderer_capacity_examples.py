"""Export fixed renderer-capacity traces and Blender-ready comparison arrays.

This is a CPU-only reader. It never loads a model or checkpoint state, selects
an epoch, aligns time, rescales coefficients, or chooses examples by outcome.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_multiseed_stochasticity import center
from kinetalk_b0.neutral_data import native_clip
from scripts.plot_projection_schedule_examples import make_lock, validate_query_metadata
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES, inspect_input
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.train_predictable_renderer import sha


SCHEMA = "renderer_capacity_example_export_v1"
ARM_SCHEMA = "renderer_capacity_probe_v1"
ARMS = ("oracle_local", "audio_local", "zero_local")
SEEDS = (42, 123, 2026, 7, 19, 73, 211, 997)
MODES = ("full", "zero", "reverse", "oracle")
NOISE_SEED = 42
DISPLAY_MODES = ("GT", "B0", "source", "audio", "oracle", "zero")
CHANNELS = ((41, "browDownLeft"), (43, "browInnerUp"),
            (5, "eyeSquintLeft"), (17, "jawOpen"))
COLORS = {"GT": "#111111", "B0": "#8c8c8c", "source": "#b66b10",
          "audio": "#2166ac", "oracle": "#18834b", "zero": "#7651a8"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def comparable_recipe(recipe):
    value = copy.deepcopy(recipe)
    value["args"].pop("arm", None)
    value["args"].pop("output", None)
    return value


def load_arm_metadata(run, arm):
    run = Path(run).resolve()
    provenance = read_json(run / "provenance.json")
    summary = read_json(run / "summary.json")
    recipe = provenance["recipe"]
    digest = canonical_hash(recipe)
    if (recipe.get("schema") != ARM_SCHEMA or summary.get("schema") != ARM_SCHEMA
            or recipe.get("args", {}).get("arm") != arm or summary.get("arm") != arm
            or provenance.get("recipe_sha256") != digest
            or summary.get("recipe_sha256") != digest):
        raise ValueError("Capacity recipe/schema/arm binding differs")
    expected = {"epochs": 8, "decode_steps": 12, "eval_noise_seeds": list(SEEDS),
                "eval_modes": list(MODES), "checkpoint_selection": "none; fixed final epoch8 diagnostic"}
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError("Capacity fixed-epoch sampling protocol differs")
    if (summary.get("completed_epochs") != 8 or summary.get("optimizer_steps") != 1160
            or summary.get("checkpoint_selection_performed") is not False
            or summary.get("frozen_unchanged") is not True
            or summary.get("head_unchanged") is not True):
        raise ValueError("Capacity run is incomplete or selected")
    for record in (recipe, summary):
        for flag in ("test_loaded", "default_replaced"):
            if record.get(flag) is not False:
                raise ValueError("Capacity run has forbidden test/default exposure")
    if any(recipe.get(flag) is not False for flag in ("outer280_loaded", "new_identity439_loaded")):
        raise ValueError("Capacity run has forbidden outer development exposure")
    inventory = read_json(run / "output_hashes.json")
    for name in ("provenance.json", "summary.json"):
        if inventory.get(name) != sha(run / name):
            raise ValueError("Capacity metadata differs from output inventory")
    return {"run": run, "recipe": recipe, "summary": summary,
            "inventory": inventory,
            "provenance_sha256": sha(run / "provenance.json"),
            "summary_sha256": sha(run / "summary.json"),
            "inventory_sha256": sha(run / "output_hashes.json")}


def load_curves(meta, epoch, cache_hash):
    run, recipe, summary = meta["run"], meta["recipe"], meta["summary"]
    stem = "epoch000" if epoch == 0 else "final_epoch008"
    curve_path = run / (stem + "_curves.pt")
    checkpoint_path = run / (stem + ".pt")
    sidecar_path = run / (stem + "_curves.provenance.json")
    sidecar = read_json(sidecar_path)
    expected = {"schema": "projection_schedule_curves_provenance_v1",
        "curve_sha256": sha(curve_path), "checkpoint_sha256": sha(checkpoint_path),
        "recipe_sha256": canonical_hash(recipe), "cache_sha256": cache_hash}
    if sidecar != expected:
        raise ValueError("Capacity curves are not bound to checkpoint/recipe/cache")
    for path, digest in ((curve_path, expected["curve_sha256"]),
                         (checkpoint_path, expected["checkpoint_sha256"]),
                         (sidecar_path, sha(sidecar_path))):
        if meta["inventory"].get(path.name) != digest:
            raise ValueError("Capacity curve/checkpoint differs from output inventory")
    binding = summary["epoch000_curve_provenance" if epoch == 0 else "curve_provenance"]
    if (Path(binding["path"]).resolve() != sidecar_path.resolve()
            or binding["sha256"] != sha(sidecar_path)):
        raise ValueError("Capacity summary/curve binding differs")
    curves = torch.load(curve_path, map_location="cpu", weights_only=False, mmap=True)
    validate_curve_protocol(curves)
    return curves, {**expected, "path": str(curve_path), "sidecar_sha256": sha(sidecar_path)}


def validate_curve_protocol(curves):
    if (curves.get("noise_seeds") != list(SEEDS) or curves.get("decode_steps") != 12
            or set(curves.get("motion", {})) != set(map(str, SEEDS))
            or any(set(row) != set(MODES) for row in curves["motion"].values())):
        raise ValueError("Capacity curves have missing seed or condition")


def validate_curve_shape(curves, reference):
    shape = tuple(reference.shape)
    for seed in SEEDS:
        for mode in MODES:
            value = curves["motion"][str(seed)][mode]
            if tuple(value.shape) != shape:
                raise ValueError("Capacity curve/reference shape differs")


def video_picks(clips):
    """Take the first locked entry for each speaker, before reading curves."""
    result, speakers = [], set()
    for pick in clips:
        if pick["speaker"] not in speakers:
            result.append(pick)
            speakers.add(pick["speaker"])
    if len(result) != 3:
        raise ValueError("Fixed video rule requires exactly three locked speakers")
    return result


def assemble_modes(split, index, source, audio, oracle, zero):
    q = split["q"]
    baseline = split["base"]["b0"] + split["identity"]["baseline"][:, None]
    values = torch.stack((q["motion"][index], baseline[index], source[index],
                          audio[index], oracle[index], zero[index])).float()
    if tuple(values.shape[1:]) != tuple(q["motion"][index].shape):
        raise ValueError("Exported modes do not share the native query shape")
    return values


def native_arrays(query, index, values):
    if query["valid"].dtype != torch.bool or query["channel_mask"].dtype != torch.bool:
        raise ValueError("Native valid and channel_mask must be boolean")
    times = query["times"][index].double().cpu().numpy()
    valid = query["valid"][index].cpu().numpy()
    channel_mask = query["channel_mask"][index].cpu().numpy()
    motion = values.cpu().numpy()
    if (times.ndim != 1 or valid.shape != times.shape or channel_mask.shape != (52,)
            or motion.shape != (len(DISPLAY_MODES), len(times), 52)
            or not valid.any() or not np.isfinite(times).all()
            or (len(times) > 1 and not np.allclose(np.diff(times), 1 / 25, atol=1e-7, rtol=1e-5))
            or not np.isfinite(motion[:, valid]).all()):
        raise ValueError("Invalid native clock, mask, or motion values")
    return times, valid, channel_mask, motion


def bind_native_audio(query, pick, row, native_root):
    """Verify the selected cache crop and its original synchronized waveform."""
    if row.get("clip_id") != pick["clip_id"]:
        raise ValueError("Selected manifest row differs from metadata lock")
    clip = native_clip(row, native_root)
    index = pick["index"]
    for key in ("motion", "times", "valid", "channel_mask"):
        if not torch.equal(query[key][index].cpu(), clip[key].cpu()):
            raise ValueError("Cached query differs from selected native artifact: " + key)
    provenance = clip["metadata"]["provenance"]
    audio_path = Path(provenance["audio_path"]).resolve()
    if not audio_path.is_file() or sha(audio_path) != provenance.get("audio_sha256"):
        raise ValueError("Selected native waveform hash differs")
    offset = float(provenance["audio_offset_s"])
    if not np.isfinite(offset):
        raise ValueError("Selected native audio offset is nonfinite")
    return {"artifact": clip["metadata"]["artifact"],
        "artifact_sha256": clip["metadata"]["artifact_sha256"],
        "audio_path": str(audio_path), "audio_sha256": provenance["audio_sha256"],
        "audio_offset_s": offset, "fps": float(provenance["fps"]),
        "clock_evidence": provenance["clock_evidence"],
        "crop_start": int(clip["metadata"]["crop_start"])}


def plot_nine(split, picks, fields, output, *, centered):
    q = split["q"]
    ids = [pick["index"] for pick in picks]
    observed = q["valid"][ids, :, None] & q["channel_mask"][ids, None, :]
    baseline = split["base"]["b0"][ids].double() + split["identity"]["baseline"][ids, None].double()
    plotted = {}
    for name, value in fields.items():
        selected = value[ids].double()
        plotted[name] = center(selected - baseline, observed) if centered else selected
    fig, axes = plt.subplots(len(CHANNELS), len(picks), figsize=(27, 8.4), sharey="row", squeeze=False)
    for col, pick in enumerate(picks):
        index = pick["index"]
        valid = q["valid"][index].bool().numpy()
        times = q["times"][index].double().numpy().copy()
        times -= times[0]
        times[~valid] = np.nan
        for row, (channel, label) in enumerate(CHANNELS):
            if not bool(q["channel_mask"][index, channel]):
                raise ValueError("Selected plot channel is unobserved")
            ax = axes[row, col]
            for mode in DISPLAY_MODES:
                values = plotted[mode][col, :, channel].numpy().copy()
                values[~valid] = np.nan
                ax.plot(times, values, color=COLORS[mode], linewidth=1.4 if mode in ("GT", "audio") else 1,
                        linestyle="--" if mode in ("B0", "zero") else "-", label=mode)
            ax.grid(alpha=.18)
            if col == 0:
                ax.set_ylabel(label + ("\ncentered residual" if centered else "\nraw coefficient"))
            if row == 0:
                ax.set_title(pick["speaker"].replace("mead_", "") + " / " + pick["emotion"], fontsize=9)
            if row == len(CHANNELS) - 1:
                ax.set_xlabel("Time from native clip start (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(DISPLAY_MODES), bbox_to_anchor=(.5, .965))
    fig.suptitle("Fixed metadata-locked examples | epoch8 | seed42 | no fitting or outcome selection", y=.997)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(output, dpi=145)
    plt.close(fig)


def export_examples(arm_paths, selection_path, output, *, native_root=None):
    output, selection_path = Path(output), Path(selection_path).resolve()
    if output.exists():
        raise FileExistsError("Fresh export directory required")
    if not selection_path.is_file():
        raise FileNotFoundError("Existing metadata-only selection is required")
    arms = {arm: load_arm_metadata(arm_paths[arm], arm) for arm in ARMS}
    recipes = [comparable_recipe(meta["recipe"]) for meta in arms.values()]
    if any(recipe != recipes[0] for recipe in recipes[1:]):
        raise ValueError("Capacity arms differ beyond arm/output")
    recipe = arms["audio_local"]["recipe"]
    cache_path = Path(recipe["source_recipe"]["args"]["cache"]).resolve()
    cache_hash = sha(cache_path)
    if cache_hash != recipe["input_sha256"]["cache"]:
        raise ValueError("Capacity source cache hash differs")
    lock = read_json(selection_path)
    if lock != make_lock(cache_path.parent) or len(lock.get("clips", [])) != 9:
        raise ValueError("Metadata-only nine-example selection changed")
    if lock["source_sha256"]["cache"] != cache_hash:
        raise ValueError("Selection and capacity cache differ")
    if lock["source_sha256"]["split_lock"] != recipe["input_sha256"]["split_lock"]:
        raise ValueError("Selection and capacity split differ")
    selected_videos = video_picks(lock["clips"])
    cache = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
    if (cache.get("schema") != "predictable_renderer_cache_v1"
            or cache["provenance"].get("internal_split_lock_sha256") != lock["internal_split_lock_sha256"]
            or cache["provenance"].get("manifest_hashes", {}).get("validation")
                != lock["source_sha256"]["validation_manifest"]):
        raise ValueError("Renderer cache metadata lineage differs")
    split, q = cache["splits"]["validation"], cache["splits"]["validation"]["q"]
    validate_query_metadata(q, lock)
    loaded, bindings = {}, {}
    for arm, meta in arms.items():
        loaded[(arm, 0)], bindings[(arm, 0)] = load_curves(meta, 0, cache_hash)
        loaded[(arm, 8)], bindings[(arm, 8)] = load_curves(meta, 8, cache_hash)
        validate_curve_shape(loaded[(arm, 0)], q["motion"])
        validate_curve_shape(loaded[(arm, 8)], q["motion"])
    for arm in ARMS[1:]:
        for seed in SEEDS:
            for mode in MODES:
                if not torch.equal(loaded[(ARMS[0], 0)]["motion"][str(seed)][mode],
                                   loaded[(arm, 0)]["motion"][str(seed)][mode]):
                    raise ValueError("Capacity arms do not share exact epoch0 curves")
    source = loaded[("audio_local", 0)]["motion"][str(NOISE_SEED)]["full"]
    audio = loaded[("audio_local", 8)]["motion"][str(NOISE_SEED)]["full"]
    oracle = loaded[("oracle_local", 8)]["motion"][str(NOISE_SEED)]["oracle"]
    zero = loaded[("zero_local", 8)]["motion"][str(NOISE_SEED)]["zero"]
    baseline = split["base"]["b0"] + split["identity"]["baseline"][:, None]
    fields = {"GT": q["motion"], "B0": baseline, "source": source,
              "audio": audio, "oracle": oracle, "zero": zero}
    # Validate all nine selected timelines and coefficients before creating files.
    for pick in lock["clips"]:
        values = assemble_modes(split, pick["index"], source, audio, oracle, zero)
        native_arrays(q, pick["index"], values)
    rows = [json.loads(line) for line in (cache_path.parent / "validation.jsonl").read_text(
        encoding="utf8").splitlines() if line.strip()]
    audio_bindings = {}
    if native_root is not None:
        for pick in selected_videos:
            audio_bindings[pick["clip_id"]] = bind_native_audio(
                q, pick, rows[pick["index"]], native_root)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(selection_path, output / "selection.json")
    plot_nine(split, lock["clips"], fields, output / "raw_seed42.png", centered=False)
    plot_nine(split, lock["clips"], fields, output / "centered_seed42.png", centered=True)
    videos = output / "video_npz"
    videos.mkdir()
    exported = []
    for pick in selected_videos:
        index = pick["index"]
        values = assemble_modes(split, index, source, audio, oracle, zero)
        times, valid, channel_mask, motions = native_arrays(q, index, values)
        path = videos / (pick["speaker"] + "_first_locked.npz")
        audio_binding = audio_bindings.get(pick["clip_id"])
        audio_metadata = ({"audio_path": np.asarray(audio_binding["audio_path"]),
                           "audio_sha256": np.asarray(audio_binding["audio_sha256"]),
                           "audio_offset_seconds": np.asarray(audio_binding["audio_offset_s"])}
                          if audio_binding else {})
        np.savez_compressed(path, channels=np.asarray(ARKIT_NAMES), times=times, valid=valid,
            channel_mask=channel_mask, mode_names=np.asarray(DISPLAY_MODES), motions=motions,
            clip_id=np.asarray(pick["clip_id"]), noise_seed=np.asarray(NOISE_SEED), **audio_metadata)
        inspect_input(path, 25)
        exported.append({"path": str(path.relative_to(output)), "sha256": sha(path), "clip": pick,
            "native_start_seconds": float(times[0]), "valid_frames": int(valid.sum()),
            "observed_channels": int(channel_mask.sum()), "audio_binding": audio_binding})
    provenance = {"schema": SCHEMA, "noise_seed": NOISE_SEED, "epoch": 8,
        "mode_names": list(DISPLAY_MODES), "mode_mapping": {
            "GT": "validation q.motion", "B0": "cached base.b0 + cached identity.baseline",
            "source": "shared epoch000 seed42 full", "audio": "audio_local epoch8 seed42 full",
            "oracle": "oracle_local epoch8 seed42 oracle (reads true motion; diagnostic only)",
            "zero": "zero_local epoch8 seed42 zero"},
        "selection_path": str(selection_path), "selection_sha256": sha(selection_path),
        "selection_copy_sha256": sha(output / "selection.json"), "selection_uses_metadata_only": True,
        "outcome_based_selection": False, "nine_plot_clips": lock["clips"],
        "video_selection_rule": "first locked clip for each speaker in original selection order",
        "video_clips": exported, "cache_path": str(cache_path), "cache_sha256": cache_hash,
        "arm_metadata": {arm: {key: meta[key] for key in ("provenance_sha256", "summary_sha256", "inventory_sha256")}
                         for arm, meta in arms.items()},
        "curves": {f"{arm}_epoch{epoch}": bindings[(arm, epoch)] for arm in ARMS for epoch in (0, 8)},
        "plots": {name: sha(output / name) for name in ("raw_seed42.png", "centered_seed42.png")},
        "native_clock": "NPZ times copied exactly from cached native q.times; no origin shift, interpolation, lag fitting, or resampling",
        "mouth_alignment": "All six motions use the identical cached native timeline and preserve cached B0 plus identity baseline",
        "audio": {"verified_native_bindings": bool(audio_bindings),
            "rule": "audio time = native motion time + native artifact audio_offset_s; no fitted offset",
            "waveform_exported": False,
            "warning": None if audio_bindings else "No waveform binding checked; pass --native-root before claiming audiovisual alignment"},
        "channel_mask": "Stored as an extra NPZ field; unobserved channels are not claimed as validated GT",
        "coefficient_modification": False, "display_clamp": "none in exported NPZ; renderer may clamp display only",
        "source_sha256": {str(path.resolve()): sha(path) for path in (
            Path(__file__), Path(__file__).with_name("plot_projection_schedule_examples.py"),
            Path(__file__).with_name("render_dynamic_rig_comparison.py"),
            Path(__file__).with_name("audit_multiseed_stochasticity.py"))},
        "training_or_generation_run": False, "test_loaded": False, "default_replaced": False}
    write_new_json(output / "provenance.json", provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument("--" + arm.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-root", type=Path,
                        help="Read and verify three selected native artifacts/audio hashes and exact crop clock")
    args = parser.parse_args()
    torch.set_num_threads(4)
    paths = {arm: getattr(args, arm) for arm in ARMS}
    result = export_examples(paths, args.selection, args.output, native_root=args.native_root)
    print(json.dumps({"complete": True, "output": str(args.output.resolve()),
        "plots": 2, "video_npz": len(result["video_clips"]), "noise_seed": NOISE_SEED}))


if __name__ == "__main__":
    main()
