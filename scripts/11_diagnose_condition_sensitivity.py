"""Diagnose whether the deployed renderer responds to Style or affect.

The comparison keeps query content/audio and the reference motion fixed. It is
an intervention diagnostic, not a quality score: Style and affect are replaced
inside the already-loaded deployment conditions and the bounded renderer is
evaluated with identical deterministic integration settings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.data import B0ResidualDataset, collate_b0_residual
from kinetalk_b0.models import Stage1Model, Stage2Model, Stage3Model, Stage4Model
from kinetalk_b0.protocol import audit_config
from kinetalk_b0.utils import load_checkpoint, load_yaml, move_to_device, seed_everything


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    weight = mask.to(value.dtype).unsqueeze(-1).expand_as(value)
    return float((value.abs() * weight).sum() / weight.sum().clamp_min(1.0))


def masked_velocity_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    if value.shape[1] < 2:
        return 0.0
    valid = mask[:, 1:] & mask[:, :-1]
    return masked_mean(torch.diff(value, dim=1), valid)


def load_generator(
    cfg: dict[str, Any],
    device: torch.device,
    stage4_checkpoint: Path | None = None,
) -> tuple[Stage4Model, dict[str, Any]]:
    """Load a generator, allowing only the intentional v9 -> v10 addition.

    Architecture v12 adds an isolated Style adapter. Reading a v9 checkpoint
    is useful for measuring the pre-change baseline, but must not turn into a
    general non-strict load. The model's audited base importer checks every
    shared key exactly and leaves only the new zero-initialized adapter fresh.
    """

    stage1 = Stage1Model(cfg)
    load_checkpoint(cfg["paths"]["stage1_ckpt"], stage1, map_location="cpu", strict=True)
    stage2 = Stage2Model(cfg)
    load_checkpoint(cfg["paths"]["stage2_ckpt"], stage2, map_location="cpu", strict=True)
    stage3 = Stage3Model(cfg, stage2)
    load_checkpoint(cfg["paths"]["stage3_ckpt"], stage3, map_location="cpu", strict=True)
    generator = Stage4Model(cfg, stage1, stage2, stage3)

    checkpoint = stage4_checkpoint or Path(cfg["paths"]["stage4_ckpt"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    version_value = state.get("architecture_version")
    version = int(version_value) if version_value is not None else None
    compatibility = "strict_v12"
    if version == 12:
        generator.load_state_dict(state, strict=True)
    elif version == 9:
        generator.load_frozen_base(checkpoint, map_location="cpu")
        compatibility = "v9_frozen_base_with_zero_style_adapter"
    else:
        raise RuntimeError(
            f"Unsupported Stage4 architecture version {version!r} in {checkpoint}; "
            "only strict v12 or the audited v9 baseline is accepted"
        )
    metadata = {
        "path": str(checkpoint.resolve()),
        "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "source_architecture_version": version,
        "load_mode": compatibility,
        "effective_style_gain": float(
            generator.style_adapter_max_gate * torch.sigmoid(generator.style_adapter_gate_logit.detach())
        ),
    }
    return generator.to(device).eval(), metadata


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--stage4-checkpoint",
        type=Path,
        default=None,
        help="Optional Stage4 checkpoint override; audited architecture-v9 baselines are supported",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    cfg = load_yaml(args.config)
    protocol = audit_config(cfg["data"])
    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    generator, checkpoint_metadata = load_generator(cfg, device, args.stage4_checkpoint)
    dataset = B0ResidualDataset(cfg, args.split, random_crop=False)
    if args.max_samples > 0:
        dataset.items = dataset.items[: args.max_samples]
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["data"].get("batch_size", 4)),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        collate_fn=collate_b0_residual,
    )
    mouth = torch.tensor(list(cfg["data"].get("neutral_output_indices", [])), device=device, dtype=torch.long)
    all_indices = torch.arange(int(cfg["data"]["motion_dim"]), device=device)
    upper = all_indices[~torch.isin(all_indices, mouth)]
    sums = {key: {"all": 0.0, "mouth": 0.0, "non_mouth": 0.0, "velocity_all": 0.0} for key in ("style_vs_zero", "style_half_vs_full", "affect_vs_zero")}
    counts = 0
    batches = 0
    range_counts = {"below_zero": 0, "above_one": 0, "valid": 0}

    for batch in loader:
        batch = move_to_device(batch, device)
        conditions = generator.conditions(batch)
        query_mask = batch["query"]["mask"]
        full_final, _ = generator.render(batch, conditions, steps=args.steps, stochastic=False)

        no_style = dict(conditions)
        no_style["style"] = torch.zeros_like(conditions["style"])
        no_style_final, _ = generator.render(batch, no_style, steps=args.steps, stochastic=False)

        half_style = dict(conditions)
        half_style["style"] = conditions["style"] * 0.5
        half_final, _ = generator.render(batch, half_style, steps=args.steps, stochastic=False)

        no_affect = dict(conditions)
        no_affect["global"] = torch.zeros_like(conditions["global"])
        no_affect["local"] = torch.zeros_like(conditions["local"])
        no_affect["intensity_value"] = torch.zeros_like(conditions["intensity_value"])
        no_affect_final, _ = generator.render(batch, no_affect, steps=args.steps, stochastic=False)

        variants = {
            "style_vs_zero": full_final - no_style_final,
            "style_half_vs_full": full_final - half_final,
            "affect_vs_zero": full_final - no_affect_final,
        }
        for name, delta in variants.items():
            sums[name]["all"] += masked_mean(delta, query_mask)
            sums[name]["mouth"] += masked_mean(delta[..., mouth], query_mask) if mouth.numel() else 0.0
            sums[name]["non_mouth"] += masked_mean(delta[..., upper], query_mask) if upper.numel() else 0.0
            sums[name]["velocity_all"] += masked_velocity_mean(delta, query_mask)
        range_counts["below_zero"] += int((full_final < 0).sum())
        range_counts["above_one"] += int((full_final > 1).sum())
        range_counts["valid"] += int(full_final.numel())
        counts += int(query_mask.shape[0])
        batches += 1
        if counts >= args.max_samples > 0:
            break

    result = {
        "schema_version": 1,
        "split": args.split,
        "samples": counts,
        "steps": args.steps,
        "checkpoint": checkpoint_metadata,
        "conditions": {
            "style": "reference residual encoded by frozen Stage2 Style encoder",
            "affect": "Stage3 global/local/intensity conditions",
            "style_half": "0.5 * deployed Style code",
        },
        "mean_abs_intervention": {
            name: {key: value / max(batches, 1) for key, value in values.items()}
            for name, values in sums.items()
        },
        "range_counts": range_counts,
        "protocol": protocol,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "samples": counts, "mean_abs_intervention": result["mean_abs_intervention"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
