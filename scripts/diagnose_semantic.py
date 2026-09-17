"""Fixed-condition v9 diagnostics; nonzero response alone is not disentanglement.

Run with ``python -m scripts.diagnose_semantic --config ... --checkpoint ...``.
The same target content, complete semantic condition and explicit initial noise
are reused for both same-person and different-person reference interventions.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from kinetalk_b0.models.semantic import MotionSemanticReadout, MultiReferenceStyleEncoder, SemanticGenerator
from kinetalk_b0.semantic_data import SemanticMotionDataset, semantic_collate
from kinetalk_b0.utils import load_yaml, move_to_device


ARKIT_NAMES = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft", "eyeBlinkRight",
    "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight",
    "eyeSquintRight", "eyeWideRight", "jawForward", "jawLeft", "jawRight",
    "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft",
    "mouthRight", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
]
BLENDER_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]
REGIONS = {
    "all": list(range(52)), "brow": list(range(41, 46)),
    "eye": list(range(14)), "mouth": list(range(14, 41)), "blink": [0, 7],
}


def write_csv_roundtrip(path: Path, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Actually serialize 51 named Blender channels, then map names back to 52.

    No clipping occurs here; callers explicitly choose raw/clamped output.
    tongueOut is absent from this CSV format and remains invalid after import.
    """
    if motion.ndim != 2 or motion.shape[-1] != 52:
        raise ValueError("CSV export requires raw [frames,52] ARKit-order motion")
    path.parent.mkdir(parents=True, exist_ok=True)
    values = motion.detach().float().cpu()
    index = {name: i for i, name in enumerate(ARKIT_NAMES)}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Frame", *BLENDER_NAMES])
        for frame, row in enumerate(values.tolist()):
            writer.writerow([frame, *(f"{row[index[name]]:.8f}" for name in BLENDER_NAMES)])
    restored = torch.zeros_like(values)
    channel_valid = torch.zeros(52, dtype=torch.bool)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Frame", *BLENDER_NAMES]:
            raise ValueError("CSV channel header does not match the declared Blender format")
        rows = list(reader)
        if len(rows) != len(values):
            raise ValueError("CSV roundtrip changed the frame count")
        for frame, row in enumerate(rows):
            if int(row["Frame"]) != frame:
                raise ValueError("CSV frame order changed")
            for name in BLENDER_NAMES:
                restored[frame, index[name]] = float(row[name])
                channel_valid[index[name]] = True
    return restored.to(motion.device), channel_valid.to(motion.device)


def region_delta(
    reference: torch.Tensor, changed: torch.Tensor,
    frame_valid: torch.Tensor, channel_valid: torch.Tensor,
) -> dict[str, Any]:
    """Separate a time-constant offset from the changing part of the response."""
    result: dict[str, Any] = {}
    delta = (changed - reference)[frame_valid]
    for name, indices in REGIONS.items():
        chosen = [i for i in indices if bool(channel_valid[i])]
        if not chosen or delta.shape[0] == 0:
            result[name] = {"observed_channels": len(chosen), "mae": None, "rms": None,
                            "max_abs": None, "static_rms": None, "temporal_rms": None}
            continue
        values = delta[:, chosen].float()
        static = values.mean(dim=0, keepdim=True)
        result[name] = {
            "observed_channels": len(chosen), "mae": float(values.abs().mean()),
            "rms": float(values.square().mean().sqrt()), "max_abs": float(values.abs().max()),
            "static_rms": float(static.square().mean().sqrt()),
            "temporal_rms": float((values - static).square().mean().sqrt()),
        }
    return result


def _content(branch: Mapping[str, Any], generator: SemanticGenerator) -> torch.Tensor:
    context = branch.get("content_context")
    expected = generator.stage1.native_aggregator.context
    if context is not None and context.shape[-2] == expected:
        return context
    return branch["content"]


def _residual(branch: Mapping[str, Any], generator: SemanticGenerator) -> torch.Tensor:
    motion, mask = branch["motion"], branch["valid"]
    content = _content(branch, generator)
    leading = motion.shape[:-2]
    flat_motion = motion.reshape(-1, *motion.shape[-2:])
    flat_mask = mask.reshape(-1, mask.shape[-1])
    content = content.reshape(-1, *content.shape[len(leading):])
    b0 = generator.stage1(content, flat_mask)["b0"]
    residual = (flat_motion - b0).reshape_as(motion)
    if "channel_mask" in branch:
        residual = residual * branch["channel_mask"].unsqueeze(-2).to(residual.dtype)
    return residual * mask.unsqueeze(-1).to(residual.dtype)


def _semantic_metrics(outputs: Mapping[str, torch.Tensor], branch: Mapping[str, Any], item: int) -> dict[str, Any]:
    expected = int(branch["emotion_id"][item])
    predicted = int(outputs["emotion_logits"][item].argmax())
    level_known = bool(branch["intensity_valid"][item]) and int(branch["intensity_id"][item]) >= 0
    va_valid = branch["va_valid"][item] & branch["valid"][item]
    weights = branch["va_confidence"][item] * va_valid
    observed = weights > 0
    error = (outputs["va"][item][observed] - branch["va"][item][observed]).abs()
    result = {
        "emotion_prediction": predicted, "emotion_correct": float(predicted == expected) if expected >= 0 else None,
        "intensity_prediction": int(outputs["intensity_logits"][item].argmax()),
        "intensity_correct": float(int(outputs["intensity_logits"][item].argmax()) == int(branch["intensity_id"][item])) if level_known else None,
        "va_mae": float((error * weights[observed, None]).sum() / (weights[observed].sum() * 2)) if observed.any() else None,
    }
    return result


def _average_scalars(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate numeric metrics with per-example (not per-frame) weighting."""
    if not rows:
        return {}
    result = {}
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if all(isinstance(value, dict) for value in values):
            result[key] = _average_scalars(values)
        else:
            finite = [float(value) for value in values if isinstance(value, (float, int))]
            if finite:
                result[key] = sum(finite) / len(finite)
    return result


@torch.no_grad()
def diagnose(
    generator: SemanticGenerator,
    style_encoder: MultiReferenceStyleEncoder,
    readout: MotionSemanticReadout,
    loader: Iterable[dict[str, Any]],
    device: torch.device | str,
    steps: int = 4,
    max_batches: int | None = 2,
    *,
    seed: int = 1234,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    device = torch.device(device)
    modules = (generator, style_encoder, readout)
    modes = [module.training for module in modules]
    for module in modules:
        module.eval()
    noise_rng = torch.Generator(device="cpu").manual_seed(seed)
    temporary = tempfile.TemporaryDirectory(prefix="semantic_csv_diagnostic_") if output_dir is None else None
    directory = Path(temporary.name if temporary is not None else output_dir)
    records: list[dict[str, Any]] = []
    try:
        for batch_index, cpu_batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_to_device(cpu_batch, device)
            query = batch["query"]
            if query["motion"].shape[-1] != 52:
                raise ValueError("Fixed-region/Blender diagnostics require ARKit-52 motion")
            styles = {
                key: style_encoder(_residual(batch[branch_name], generator), batch[branch_name]["valid"])
                for key, branch_name in (("baseline", "same_style_references"), ("same_person", "positive_style_references"), ("donor", "donor_references"))
            }
            noise = torch.randn(query["motion"].shape, generator=noise_rng).to(device=device, dtype=query["motion"].dtype)
            arguments = dict(
                content=_content(query, generator), mask=query["valid"],
                emotion_id=query["emotion_id"], intensity_id=query["intensity_id"], va=query["va"],
                va_valid=query["va_valid"], va_confidence=query["va_confidence"],
                intensity_valid=query["intensity_valid"], initial_noise=noise, steps=steps,
            )
            generated = {key: generator.generate(style=value, **arguments) for key, value in styles.items()}
            repeated = generator.generate(style=styles["baseline"], **arguments)
            channels = query.get("channel_mask", torch.ones(query["motion"].shape[0], 52, device=device, dtype=torch.bool)).bool()
            semantic = {
                key: readout(output["residual"] * channels.unsqueeze(1), query["valid"])
                for key, output in generated.items()
            }
            truth_semantic = readout(_residual(query, generator), query["valid"])
            generated_styles = {
                key: style_encoder(output["residual"] * channels.unsqueeze(1), query["valid"])
                for key, output in generated.items()
            }
            for item in range(query["motion"].shape[0]):
                mask, observed = query["valid"][item], channels[item]
                original = generated["baseline"]["motion"][item]
                row: dict[str, Any] = {
                    "batch": batch_index, "item": item, "valid_frames": int(mask.sum()),
                    "metadata": query.get("metadata", [{}] * len(channels))[item],
                    "donor_metadata": batch["donor_anchor"].get("metadata", [{}] * len(channels))[item],
                    "donor_intensity_matched": bool(batch.get("donor_intensity_matched", torch.zeros(len(channels), dtype=torch.bool))[item]),
                    "repeat_raw_max_abs": float((repeated["motion"][item] - original)[mask].abs().max()),
                    "real_motion_readout": _semantic_metrics(truth_semantic, query, item),
                    "baseline_readout": _semantic_metrics(semantic["baseline"], query, item),
                    "baseline_reconstruction": region_delta(query["motion"][item], original, mask, observed),
                    "b0_reconstruction": region_delta(query["motion"][item], generated["baseline"]["b0"][item], mask, observed),
                    "interventions": {},
                }
                for name in ("same_person", "donor"):
                    swapped = generated[name]["motion"][item]
                    left = styles["baseline"][item]
                    right = styles[name][item]
                    clipped_original, clipped_swapped = original.clamp(0, 1), swapped.clamp(0, 1)
                    csv_root = directory / f"batch_{batch_index:03d}_item_{item:03d}"
                    restored_original, exported = write_csv_roundtrip(csv_root / "baseline.csv", clipped_original)
                    restored_swapped, _ = write_csv_roundtrip(csv_root / f"{name}.csv", clipped_swapped)
                    applicable = exported & observed
                    frame_channels = mask[:, None] & observed[None]
                    preserved = {
                        field: float((generated[name]["conditions"][field][item] - generated["baseline"]["conditions"][field][item]).abs().max())
                        for field in ("global", "local", "intensity_value")
                    }
                    row["interventions"][name] = {
                        "style_l2": float((left - right).norm()),
                        "style_cosine_distance": float(1 - F.cosine_similarity(left[None], right[None])[0]),
                        "raw_delta": region_delta(original, swapped, mask, observed),
                        "clamped_delta": region_delta(clipped_original, clipped_swapped, mask, observed),
                        "csv_delta": region_delta(restored_original, restored_swapped, mask, applicable),
                        "csv_roundtrip_max_abs": float((restored_swapped - clipped_swapped)[mask][:, applicable].abs().max()) if applicable.any() else None,
                        "baseline_outside_unit_fraction": float(((original < 0) | (original > 1))[frame_channels].float().mean()),
                        "swapped_outside_unit_fraction": float(((swapped < 0) | (swapped > 1))[frame_channels].float().mean()),
                        "condition_max_abs_difference": preserved,
                        "b0_max_abs_difference": float((generated[name]["b0"][item] - generated["baseline"]["b0"][item]).abs().max()),
                        "readout": _semantic_metrics(semantic[name], query, item),
                        "readout_va_change_mae": float((semantic[name]["va"][item] - semantic["baseline"]["va"][item])[mask].abs().mean()),
                        "generated_style_target_cosine": float(F.cosine_similarity(generated_styles[name][item:item+1], right[None])[0]),
                        "generated_style_source_cosine": float(F.cosine_similarity(generated_styles[name][item:item+1], left[None])[0]),
                        "generated_style_prefers_target": float(
                            F.cosine_similarity(generated_styles[name][item:item+1], right[None])[0]
                            > F.cosine_similarity(generated_styles[name][item:item+1], left[None])[0]
                        ),
                    }
                records.append(row)
    finally:
        for module, mode in zip(modules, modes):
            module.train(mode)
        if temporary is not None:
            temporary.cleanup()
    if not records:
        raise ValueError("Diagnostic loader yielded no examples")
    return {
        "architecture_version": 9, "examples": len(records), "steps": steps, "seed": seed,
        "fixed_conditions": ["target content", "emotion category", "real/unknown intensity", "full VA trajectory", "VA confidence/validity", "initial noise"],
        "csv_export": {"order": "named Blender-51", "omitted_channel": "tongueOut", "clamped_before_export": True,
                       "files": str(directory.resolve()) if temporary is None else "temporary files verified and removed"},
        "interpretation": "Response and export diagnostic only. Nonzero style differences do not establish identity transfer or disentanglement. Compare same-person variation, donor response, static/temporal components, and real-motion readout quality before interpreting generated semantic scores.",
        "aggregate": _average_scalars([{key: row[key] for key in ("repeat_raw_max_abs", "real_motion_readout", "baseline_readout", "baseline_reconstruction", "b0_reconstruction", "interventions")} for row in records]),
        "samples": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/semantic_diagnostic.json"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    for name in ("generator", "readout", "style_encoder"):
        if name not in payload:
            raise ValueError(f"v9 diagnostic checkpoint missing {name!r}")
        if int(payload[name].get("architecture_version", -1)) != 9:
            raise ValueError(f"{name}: diagnostic requires architecture version 9")
    generator = SemanticGenerator(cfg).to(args.device)
    readout = MotionSemanticReadout(cfg).to(args.device)
    style_encoder = MultiReferenceStyleEncoder(cfg).to(args.device)
    for name, model in (("generator", generator), ("readout", readout), ("style_encoder", style_encoder)):
        model.load_state_dict(payload[name], strict=True)
    dataset = SemanticMotionDataset(cfg, split=args.split, random_crop=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=semantic_collate)
    report = diagnose(generator, style_encoder, readout, loader, args.device, args.steps, args.max_batches,
                      seed=args.seed, output_dir=args.output.with_suffix("").parent / f"{args.output.stem}_csv")
    report["checkpoint"] = str(args.checkpoint.resolve())
    report["split"] = args.split
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "examples": report["examples"], "aggregate": report["aggregate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
