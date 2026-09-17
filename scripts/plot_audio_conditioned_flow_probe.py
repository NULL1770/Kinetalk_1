"""CPU-only native-clock plots of the original metadata-locked nine examples.

Reads existing epoch0/8 curves and internal validation reference; no model,
training, generation, outcome-based selection, interpolation or recalibration.
The shaded min/max envelope is over eight saved seeds, not a confidence interval.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_multiseed_stochasticity import center
from scripts.plot_projection_schedule_examples import make_lock, validate_query_metadata
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.train_predictable_renderer import sha

SEEDS = (42, 123, 2026, 7, 19, 73, 211, 997)
LINE_SEEDS = (42, 123, 2026)
CHANNELS = ((43, "browInnerUp"), (44, "browOuterUpLeft"), (5, "eyeSquintLeft"))
STYLES = ("-", "--", ":")
COLORS = {"frozen-full": "#b66b10", "audio-adapted-full": "#2166ac", "zero-adapted-zero": "#26845b"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def load_metadata(run, arm):
    run = Path(run).resolve()
    provenance = read_json(run / "provenance.json")
    recipe, summary = provenance["recipe"], read_json(run / "summary.json")
    digest = canonical_hash(recipe)
    if (recipe.get("schema") != "audio_conditioned_flow_probe_v1" or summary.get("schema") != recipe["schema"]
            or recipe["args"].get("arm") != arm or summary.get("arm") != arm
            or digest != provenance.get("recipe_sha256") or digest != summary.get("recipe_sha256")
            or recipe.get("epochs") != 8 or summary.get("completed_epochs") != 8
            or recipe.get("eval_noise_seeds") != list(SEEDS) or recipe.get("decode_steps") != 12
            or summary.get("checkpoint_selection_performed") is not False):
        raise ValueError("Fixed epoch8/seed recipe binding differs")
    for key in ("test_loaded", "outer280_loaded", "new_identity439_loaded", "default_replaced"):
        if recipe.get(key) is not False:
            raise ValueError("Plot source has unexpected data/default exposure")
    return {"run": run, "recipe": recipe, "summary": summary,
            "provenance_sha256": sha(run / "provenance.json"), "summary_sha256": sha(run / "summary.json")}


def load_curves(meta, epoch, cache_hash):
    run, recipe, summary = meta["run"], meta["recipe"], meta["summary"]
    stem = "epoch000" if epoch == 0 else "final_epoch008"
    path, checkpoint = run / (stem + "_curves.pt"), run / (stem + ".pt")
    sidecar_path = run / (stem + "_curves.provenance.json")
    sidecar = read_json(sidecar_path)
    expected = {"schema": "projection_schedule_curves_provenance_v1", "curve_sha256": sha(path),
        "checkpoint_sha256": sha(checkpoint), "recipe_sha256": canonical_hash(recipe), "cache_sha256": cache_hash}
    if sidecar != expected:
        raise ValueError("Curves are not bound to their fixed checkpoint/cache")
    binding = summary["epoch000_curve_provenance" if epoch == 0 else "curve_provenance"]
    if binding["sha256"] != sha(sidecar_path) or Path(binding["path"]).resolve() != sidecar_path.resolve():
        raise ValueError("Summary/curve sidecar binding differs")
    curves = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if curves.get("noise_seeds") != list(SEEDS) or curves.get("decode_steps") != 12 or set(curves.get("motion", {})) != set(map(str, SEEDS)):
        raise ValueError("Saved seed/decoding protocol differs")
    if any(set(row) != {"full", "zero", "reverse", "oracle"} for row in curves["motion"].values()):
        raise ValueError("Missing saved condition")
    return curves, {**expected, "curves_path": str(path), "sidecar_sha256": sha(sidecar_path)}


def selected_fields(curves, mode, picks, query, baseline, *, centered):
    ids = [pick["index"] for pick in picks]
    observed = query["valid"][ids, :, None] & query["channel_mask"][ids, None, :]
    fields = []
    for seed in SEEDS:
        value = curves["motion"][str(seed)][mode]
        if value.shape != query["motion"].shape:
            raise ValueError("Curves/reference shape differs")
        value = value[ids].double()
        if not torch.isfinite(value[observed]).all():
            raise ValueError("Nonfinite observed selected trace")
        fields.append(center(value - baseline[ids], observed) if centered else value)
    return torch.stack(fields)


def plot_panels(query, picks, truth, fields, labels, limits, output, *, centered):
    fig, axes = plt.subplots(len(CHANNELS), len(picks), figsize=(27, 8.4), sharey="row", squeeze=False)
    for col, pick in enumerate(picks):
        index = pick["index"]
        valid = query["valid"][index].numpy()
        times = query["times"][index].double().numpy().copy()
        if not valid.any() or not np.isfinite(times[valid]).all() or not (np.diff(times[valid]) > 0).all():
            raise ValueError("Selected timestamps must be finite and increasing")
        times -= times[valid][0]
        times[~valid] = np.nan
        for row, (channel, name) in enumerate(CHANNELS):
            ax = axes[row, col]
            if not query["channel_mask"][index, channel]:
                raise ValueError("Selected plot channel is not observed")
            for label in labels:
                values = fields[label][:, col, :, channel].numpy().copy()
                values[:, ~valid] = np.nan
                ax.fill_between(times, np.min(values, axis=0), np.max(values, axis=0), color=COLORS[label], alpha=.12, linewidth=0)
                for seed, style in zip(LINE_SEEDS, STYLES):
                    ax.plot(times, values[SEEDS.index(seed)], color=COLORS[label], linestyle=style, linewidth=.9, alpha=.86)
            yy = truth[col, :, channel].numpy().copy(); yy[~valid] = np.nan
            ax.plot(times, yy, color="#181818", linewidth=1.3, zorder=9)
            ax.set_ylim(*limits[str(channel)]); ax.grid(alpha=.18)
            if col == 0:
                ax.set_ylabel(name + ("\ncentered residual" if centered else "\nraw coefficient"))
            if row == 0:
                ax.set_title(pick["speaker"].replace("mead_", "") + " / " + pick["emotion"], fontsize=10)
            if row == len(CHANNELS) - 1:
                ax.set_xlabel("Native time (s)")
    handles = [Line2D([], [], color="#181818", linewidth=1.4, label="GT")]
    handles += [Line2D([], [], color=COLORS[label], linewidth=1.5, label=label) for label in labels]
    handles += [Line2D([], [], color="#555555", linestyle=style, label=f"seed {seed}") for seed, style in zip(LINE_SEEDS, STYLES)]
    handles += [Patch(facecolor="#888888", alpha=.16, label="8-seed min–max envelope (not CI)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .955), ncol=len(handles), fontsize=10)
    fig.suptitle("Original nine metadata-locked examples | fixed epoch8 | native coefficient units, common row axes\n"
        "No lag alignment, interpolation, amplitude fitting, or best-seed selection; coefficient traces are not perceptual validation", y=.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .89))
    fig.savefig(output.with_suffix(".png"), dpi=145)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio-local", type=Path, required=True)
    p.add_argument("--zero-local", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--selection", type=Path, default=None)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Fresh plot output directory required")
    torch.set_num_threads(4)
    arms = {arm: load_metadata(getattr(args, arm), arm) for arm in ("audio_local", "zero_local")}
    audio_recipe = arms["audio_local"]["recipe"]
    common = []
    for meta in arms.values():
        recipe = copy.deepcopy(meta["recipe"])
        recipe["args"].pop("arm"); recipe["args"].pop("output")
        common.append(recipe)
    if common[0] != common[1]:
        raise ValueError("Matched runs differ beyond arm/output")
    cache_path = Path(audio_recipe["source_recipe"]["args"]["cache"]).resolve()
    metadata_dir = cache_path.parent
    # Selection is read and reconstructed using JSON metadata before any tensor
    # file is deserialized. This preserves the previous nine examples exactly.
    selection_path = args.selection or metadata_dir.parent / "trace_selection_seed20260923.json"
    existing_selection = selection_path.exists()
    lock = read_json(selection_path) if existing_selection else make_lock(metadata_dir)
    if lock != make_lock(metadata_dir) or len(lock["clips"]) != 9:
        raise ValueError("Previous metadata-only nine-example selection changed")
    cache_hash = sha(cache_path)
    if cache_hash != lock["source_sha256"]["cache"] or cache_hash != audio_recipe["input_sha256"]["cache"]:
        raise ValueError("Selection/run cache hashes differ")
    if sha(metadata_dir / "split_lock.json") != audio_recipe["input_sha256"]["split_lock"]:
        raise ValueError("Run does not use the metadata-locked split")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "selection.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf8")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
    if (cache.get("schema") != "predictable_renderer_cache_v1" or
            cache["provenance"].get("internal_split_lock_sha256") != lock["internal_split_lock_sha256"] or
            cache["provenance"].get("manifest_hashes", {}).get("validation") != lock["source_sha256"]["validation_manifest"]):
        raise ValueError("Cache metadata lineage differs")
    split, query = cache["splits"]["validation"], cache["splits"]["validation"]["q"]
    validate_query_metadata(query, lock)
    frozen, frozen_binding = load_curves(arms["audio_local"], 0, cache_hash)
    adapted, adapted_binding = load_curves(arms["audio_local"], 8, cache_hash)
    zero, zero_binding = load_curves(arms["zero_local"], 8, cache_hash)
    # Verify source initialization is identical between the two runs.
    zero_frozen, zero_frozen_binding = load_curves(arms["zero_local"], 0, cache_hash)
    for seed in SEEDS:
        for mode in ("full", "zero", "reverse", "oracle"):
            if not torch.equal(frozen["motion"][str(seed)][mode], zero_frozen["motion"][str(seed)][mode]):
                raise ValueError("Frozen initial curves differ across paired arms")
    del zero_frozen
    baseline = split["base"]["b0"].double() + split["identity"]["baseline"].double()[:, None]
    ids = [pick["index"] for pick in lock["clips"]]
    observed = query["valid"][ids, :, None] & query["channel_mask"][ids, None, :]
    target = query["motion"][ids].double()
    if not torch.isfinite(target[observed]).all() or not torch.isfinite(baseline[ids][observed]).all():
        raise ValueError("Nonfinite observed reference")
    all_limits, image_hashes = {}, {}
    for kind in ("raw", "centered"):
        centered = kind == "centered"
        truth = center(target - baseline[ids], observed) if centered else target
        fields = {"frozen-full": selected_fields(frozen, "full", lock["clips"], query, baseline, centered=centered),
            "audio-adapted-full": selected_fields(adapted, "full", lock["clips"], query, baseline, centered=centered),
            "zero-adapted-zero": selected_fields(zero, "zero", lock["clips"], query, baseline, centered=centered)}
        limits = {}
        for channel, _ in CHANNELS:
            parts = [truth[..., channel][observed[..., channel]]]
            parts += [value[..., channel][:, observed[..., channel]].flatten() for value in fields.values()]
            values = torch.cat(parts)
            lo, hi = float(values.min()), float(values.max()); pad = max((hi - lo) * .06, 1e-5)
            limits[str(channel)] = [lo - pad, hi + pad]
        all_limits[kind] = limits
        for variant, labels in (("frozen_vs_audio", ("frozen-full", "audio-adapted-full")),
                                ("audio_vs_zero", ("audio-adapted-full", "zero-adapted-zero"))):
            stem = args.output / (kind + "_" + variant)
            plot_panels(query, lock["clips"], truth, fields, labels, limits, stem, centered=centered)
            for suffix in (".png", ".svg"):
                image_hashes[stem.name + suffix] = sha(stem.with_suffix(suffix))
    sources = [Path(__file__), Path(__file__).with_name("plot_projection_schedule_examples.py"),
        Path(__file__).with_name("audit_multiseed_stochasticity.py")]
    provenance = {"schema": "audio_conditioned_flow_fixed9_traces_v1", "epoch": 8,
        "source_sha256": {str(path.resolve()): sha(path) for path in sources}, "cache_sha256": cache_hash,
        "selection_source_path": str(selection_path.resolve()), "selection_source_exists": existing_selection,
        "selection_source_sha256": sha(selection_path) if existing_selection else None,
        "selection_copy_sha256": sha(args.output / "selection.json"), "selection_verified_metadata_only": True,
        "new_outcome_based_selection": False, "clips": lock["clips"], "channels": CHANNELS,
        "line_noise_seeds": LINE_SEEDS, "envelope_noise_seeds": SEEDS,
        "envelope_definition": "pointwise minimum/maximum across all eight saved seeds; not confidence interval",
        "shared_y_limits": all_limits, "curves": {"frozen": frozen_binding, "audio_adapted": adapted_binding,
            "zero_adapted": zero_binding, "zero_arm_frozen": zero_frozen_binding},
        "run_metadata_hashes": {arm: {key: meta[key] for key in ("provenance_sha256", "summary_sha256")} for arm, meta in arms.items()},
        "image_sha256": image_hashes, "coordinates": {"raw": "unchanged native coefficient values",
            "centered": "per-clip observed-time center(prediction - cached_B0 - cached_identity), no amplitude rescaling"},
        "invalid_frames": "NaN gaps; no interpolation", "lag_alignment": False, "amplitude_rescaling": False,
        "training_or_generation_run": False, "test_loaded": False, "outer280_loaded": False, "new_identity439_loaded": False,
        "scope": "Internal405development, same nine metadata-locked clips. Curves are not rendered faces or perceptual validation."}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    print(json.dumps({"complete": True, "output": str(args.output.resolve()), "png_svg_pairs": 4, "clips": 9, "line_seeds": LINE_SEEDS, "envelope_seeds": SEEDS}), flush=True)


if __name__ == "__main__":
    main()
