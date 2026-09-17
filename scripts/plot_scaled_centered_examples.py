"""Read-only epoch8/noise42 coefficient montage on the original nine-clip lock.

No training, new decoding, example reselection, lag alignment or rescaling.
Rows share coefficient limits across all nine metadata-selected examples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_projection_rollout_probe import load_rollout
from scripts.audit_scaled_centered_probe import validate_metric_pair
from scripts.audit_teacher_schedule_probe import validate_curves
from scripts.plot_projection_schedule_examples import (
    CHANNELS, SCOPE, centered_residual, make_lock, validate_query_metadata,
)
from scripts.train_predictable_renderer import sha, state_hash
from scripts.train_projection_scaled_centered_probe import SCHEMA


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def draw_montage(query, picks, fields, output):
    """Draw native-clock traces with each channel's single shared y-axis range."""
    limits = {}
    for channel, _ in CHANNELS:
        values = [curve[pick["index"], query["valid"][pick["index"]], channel]
            for _, curve, *_ in fields for pick in picks
            if bool(query["channel_mask"][pick["index"], channel])]
        if not values:
            raise ValueError("Locked channel is unobserved in all examples")
        values = torch.cat(values)
        if not len(values) or not torch.isfinite(values).all():
            raise ValueError("Missing/nonfinite selected coefficient trace")
        low, high = float(values.min()), float(values.max())
        margin = max((high - low) * .06, 1e-5)
        limits[channel] = (low - margin, high + margin)
    fig, axes = plt.subplots(4, 9, figsize=(27, 10), sharey="row")
    for column, pick in enumerate(picks):
        index = pick["index"]
        valid = query["valid"][index].cpu().numpy().astype(bool)
        times = query["times"][index].double().cpu().numpy().copy()
        if not valid.any() or not np.isfinite(times[valid]).all() or not (np.diff(times[valid]) > 0).all():
            raise ValueError("Locked clip needs finite increasing native timestamps")
        times -= times[valid][0]
        times[~valid] = np.nan
        for row, (channel, name) in enumerate(CHANNELS):
            axis = axes[row, column]
            if bool(query["channel_mask"][index, channel]):
                for label, curve, color, style, width in fields:
                    values = curve[index, :, channel].cpu().numpy().copy()
                    values[~valid] = np.nan
                    axis.plot(times, values, label=label, color=color, linestyle=style, linewidth=width)
            else:
                axis.text(.5, .5, "Unobserved", ha="center", transform=axis.transAxes)
            axis.set_ylim(*limits[channel])
            axis.grid(alpha=.2)
            if column == 0:
                axis.set_ylabel(name)
            if row == 0:
                axis.set_title(pick["speaker"].replace("mead_", "") + " / " + pick["emotion"])
            if row == len(CHANNELS) - 1:
                axis.set_xlabel("Time (s)")
    handles = [plt.Line2D([], [], color=color, linestyle=style, linewidth=width, label=label)
        for label, _, color, style, width in fields]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(.5, .965))
    fig.suptitle("Original nine metadata-selected examples | epoch8, noise42 | centered residual coefficients, no lag or amplitude fitting\n"
        "Coefficient traces only; not rendered-face or perceptual validation. Raw mean errors require separate audit.", y=1., fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .925))
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Existing teacher_schedule_v1 root")
    parser.add_argument("--study", type=Path, default=None, help="Absolute directory or directory relative to --root; default scaled_centered_v1")
    args = parser.parse_args()
    root = args.root.resolve()
    study = args.study if args.study is not None else Path("scaled_centered_v1")
    study = study.resolve() if study.is_absolute() else (root / study).resolve()
    output = study / "trace_examples"
    if output.exists():
        raise FileExistsError("Use the absent study/trace_examples output; existing artifacts are never overwritten")
    torch.set_num_threads(4)
    lock_path = root / "trace_selection_seed20260923.json"
    lock = read_json(lock_path)
    if Path(lock["metadata_dir"]).resolve() != root / "data_locked" or lock != make_lock(root / "data_locked"):
        raise ValueError("Original selection lock differs in metadata, source hashes or selected values")
    if len(lock["clips"]) != 9 or lock["noise_seed"] != 42:
        raise ValueError("Require the original nine-clip/noise42 lock")
    cache_path = root / "data_locked" / "renderer_cache.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
    if (cache.get("schema") != "predictable_renderer_cache_v1"
            or cache["provenance"].get("internal_split_lock_sha256") != lock["internal_split_lock_sha256"]
            or cache["provenance"].get("manifest_hashes", {}).get("validation") != lock["source_sha256"]["validation_manifest"]):
        raise ValueError("Cache internal-split/validation-manifest provenance mismatch")
    split = cache["splits"]["validation"]
    query = split["q"]
    validate_query_metadata(query, lock)
    arms = {arm: load_rollout(study / arm, schema=SCHEMA, arm=arm, final_epoch=8) for arm in ("uniform", "train_rms")}
    validate_metric_pair(arms["train_rms"]["recipe"], arms["uniform"]["recipe"])
    for arm, data in arms.items():
        recipe = data["recipe"]
        if (recipe["input_sha256"]["cache"] != lock["source_sha256"]["cache"]
                or recipe["input_sha256"]["split_lock"] != lock["source_sha256"]["split_lock"]
                or Path(recipe["args"]["cache"]).resolve() != cache_path):
            raise ValueError("Run does not use the original locked cache/split")
        if data["summary"].get("generated_output_centered") is not False:
            raise ValueError("Require raw saved generated outputs")
        validate_curves(data["curves"][8], split)
        if data["checkpoints"][8]["step"] != 1160:
            raise ValueError("Require the fixed8epoch/1160step checkpoint")
    for key in ("head_sha256", "frozen_state_sha256"):
        if arms["uniform"]["checkpoints"][8][key] != arms["train_rms"]["checkpoints"][8][key]:
            raise ValueError("Compared runs changed frozen head/backbone")
    for epoch in range(1, 9):
        for key in ("minibatch_sha256", "noise_time_sha256", "teacher_choice_draw_sha256", "step", "samples_seen", "teacher_fraction"):
            if arms["uniform"]["records"][epoch][key] != arms["train_rms"]["records"][epoch][key]:
                raise ValueError("Matched runs used different training coverage/random draws")
    uniform, scaled = [arms[arm]["curves"][8]["motion"]["42"] for arm in ("uniform", "train_rms")]
    if not torch.equal(uniform["zero"], scaled["zero"]):
        raise ValueError("Frozen zero-local generation differs between arms")
    series = [("GT", query["motion"], "#171717", "-", 1.5),
        ("Zero local", scaled["zero"], "#94a3b8", "--", 1.),
        ("Uniform: audio", uniform["full"], "#d97706", "-", 1.15),
        ("Train RMS: audio", scaled["full"], "#2563eb", "-", 1.25),
        ("Train RMS: oracle", scaled["oracle"], "#16a34a", ":", 1.)]
    fields = [(label, centered_residual(values, split), color, style, width) for label, values, color, style, width in series]
    output.mkdir(parents=True, exist_ok=False)
    limits = draw_montage(query, lock["clips"], fields, output / "montage.png")
    sources = [Path(__file__), Path(__file__).with_name("plot_projection_schedule_examples.py"),
        Path(__file__).with_name("audit_projection_rollout_probe.py"), Path(__file__).with_name("audit_scaled_centered_probe.py"),
        Path(__file__).with_name("audit_teacher_schedule_probe.py"), Path(__file__).with_name("train_predictable_renderer.py")]
    provenance = {"schema": "scaled_centered_fixed9_coefficient_traces_v1", "epoch": 8, "noise_seed": 42,
        "source_sha256": {str(path.resolve()): sha(path) for path in sources},
        "selection_lock_sha256": sha(lock_path), "selection_lock_verified_value_for_value": True,
        "no_new_example_selection": True, "clips": lock["clips"], "channels": lock["channels"],
        "cache_sha256": lock["source_sha256"]["cache"], "split_lock_sha256": lock["source_sha256"]["split_lock"],
        "arms": {arm: {"artifact_hashes": data["hashes"], "recipe_sha256": data["summary"]["recipe_sha256"],
            "epoch8_curve_binding": read_json(study / arm / "final_epoch008_curves.provenance.json"),
            "frozen_state_sha256": data["checkpoints"][8]["frozen_state_sha256"],
            "head_sha256": state_hash(data["checkpoints"][8]["head"])} for arm, data in arms.items()},
        "matched_rng_all8epochs": True, "zero_local_equal_exactly": True,
        "series": [label for label, *_ in series], "shared_y_limits_by_channel": limits,
        "centering": "center(prediction-cached_B0-cached_identity, valid), including GT; no change to stored raw outputs",
        "invalid_frames": "NaN gaps; no interpolation", "lag_alignment": False, "amplitude_rescaling": False,
        "image_sha256": sha(output / "montage.png"), "scope": SCOPE,
        "outer280_loaded": False, "new_identity439_loaded": False, "test_loaded": False, "training_or_generation_run": False}
    with (output / "provenance.json").open("x", encoding="utf8") as handle:
        json.dump(provenance, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "epoch": 8, "noise": 42, "clips": 9, "scope": SCOPE}))


if __name__ == "__main__":
    main()
