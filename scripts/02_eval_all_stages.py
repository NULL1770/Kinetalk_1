from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from kinetalk_b0.data import B0ResidualDataset, collate_b0_residual
from kinetalk_b0.models import Stage1Model, Stage2Model, Stage3Model, Stage4Model
from kinetalk_b0.utils import load_checkpoint, load_yaml, move_to_device, seed_everything


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).to(pred.dtype).expand_as(pred)
    return (pred - target).abs().mul(weight).sum() / weight.sum().clamp_min(1.0)


def _masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).to(pred.dtype).expand_as(pred)
    return (((pred - target).square().mul(weight).sum()) / weight.sum().clamp_min(1.0)).sqrt()


def _masked_channel_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).to(pred.dtype) * channels.view(1, 1, -1).to(pred.dtype)
    return (pred - target).abs().mul(weight).sum() / weight.sum().clamp_min(1.0)


def _masked_velocity_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_zeros(())
    return _masked_mae(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1], mask[:, 1:] & mask[:, :-1])


def _velocity_channel_corr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
    """Per-sample Pearson correlation between the velocity of two BS streams.

    Restricted to the active BS channels (e.g. mouth) so overall body/head
    style differences don't dominate the metric.  Content-preserving style
    intervention should keep this high (typically > 0.5).
    """
    if a.shape[1] < 2:
        return a.new_zeros(())
    va = (a[:, 1:] - a[:, :-1])[..., channels]
    vb = (b[:, 1:] - b[:, :-1])[..., channels]
    m = (mask[:, 1:] & mask[:, :-1]).to(a.dtype).unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1.0)
    va_mean = (va * m).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
    vb_mean = (vb * m).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
    va_c = (va - va_mean) * m
    vb_c = (vb - vb_mean) * m
    num = (va_c * vb_c).sum(dim=1)
    denom_std = (va_c.square().sum(dim=1).sqrt() * vb_c.square().sum(dim=1).sqrt()).clamp_min(1e-8)
    return (num / denom_std).mean()


def _cosine(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(F.cosine_similarity(pred, target, dim=-1).mean().item())


def _add(store: dict[str, list[float]], key: str, value: torch.Tensor | float) -> None:
    store.setdefault(key, []).append(float(value.item() if isinstance(value, torch.Tensor) else value))


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end evaluation for trained KineTalk stages")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val",
                        help="Manifest split; validation is the default and test is for final frozen evaluation")
    parser.add_argument("--per-emotion", type=int, default=24)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["data"]["num_workers"] = min(int(cfg["data"].get("num_workers", 0)), 2)
    dataset = B0ResidualDataset(cfg, split=args.split, random_crop=False)
    names = [str(item).lower() for item in cfg["data"]["emotion_classes"]]
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.items):
        raw = str(record.get("emotion", "neutral")).lower()
        aliases = {"disgusted": "disgust", "fearful": "fear", "surprised": "surprise"}
        label = names.index(aliases.get(raw, raw)) if aliases.get(raw, raw) in names else 0
        grouped[label].append(index)
    selected: list[int] = []
    for label in range(len(names)):
        indices = grouped[label]
        if args.per_emotion and len(indices) > args.per_emotion:
            stride = max(1, len(indices) // args.per_emotion)
            indices = indices[::stride][: args.per_emotion]
        selected.extend(indices)
    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=int(cfg["data"].get("batch_size", 16)),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg["data"].get("num_workers", 0)) > 0,
        collate_fn=collate_b0_residual,
    )

    stage1 = Stage1Model(cfg).to(device)
    p1 = load_checkpoint(cfg["paths"]["stage1_ckpt"], stage1, map_location=device)
    stage1.eval()
    stage2 = Stage2Model(cfg).to(device)
    p2 = load_checkpoint(cfg["paths"]["stage2_ckpt"], stage2, map_location=device, strict=True)
    stage2.eval()
    stage3 = Stage3Model(cfg, stage2).to(device)
    p3 = load_checkpoint(cfg["paths"]["stage3_ckpt"], stage3, map_location=device, strict=True)
    stage3.eval()
    stage4 = Stage4Model(cfg, stage1, stage2, stage3).to(device)
    p4 = load_checkpoint(cfg["paths"]["stage4_ckpt"], stage4, map_location=device, strict=True)
    stage4.eval()

    metrics: dict[str, list[float]] = {}
    by_emotion: dict[str, dict[str, list[float]]] = defaultdict(dict)
    render_steps = int(cfg["model"].get("render_steps", 4))
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            query = batch["query"]
            q_out = stage1(query["content"], query["mask"])
            b0_mae = _masked_mae(q_out["b0"], query["b0_gt"], query["mask"])
            _add(metrics, "stage1_b0_mae", b0_mae)

            factors = stage2.encode_factors(query["residual_gt"], query["residual_mask"], query.get("audio_emotion", query.get("audio")))
            flow_pred, flow_target = stage2.flow_prediction(query["residual_gt"], q_out["h0"], factors, query["residual_mask"])
            _add(metrics, "stage2_flow_rmse", _masked_rmse(flow_pred, flow_target, query["residual_mask"]))
            _add(metrics, "stage2_emotion_acc", (factors["emotion_logits"].argmax(-1) == query["emotion_id"]).float().mean())
            _add(metrics, "stage2_intensity_acc", (factors["intensity_logits"].argmax(-1) == query["intensity_id"]).float().mean())
            stage2_render = stage2.render(
                q_out["h0"],
                {"global": factors["global"], "intensity_value": factors["intensity_value"], "style": factors["style"]},
                query["mask"],
                steps=render_steps,
            )
            _add(metrics, "stage2_render_residual_mae", _masked_mae(stage2_render, query["residual_gt"], query["residual_mask"]))
            _add(metrics, "stage2_render_final_mae", _masked_mae(stage2_render + query["b0_gt"], query["motion"], query["mask"]))

            audio = stage3(query["audio_emotion"], query["mask"])
            teacher = stage3.teacher(
                query["residual_gt"],
                query["residual_mask"],
                reference_audio=query.get("audio_emotion", query.get("audio")),
            )
            _add(metrics, "stage3_emotion_acc", (audio["emotion_logits"].argmax(-1) == query["emotion_id"]).float().mean())
            _add(metrics, "stage3_intensity_acc", (audio["intensity_logits"].argmax(-1) == query["intensity_id"]).float().mean())
            _add(metrics, "stage3_global_cosine", _cosine(audio["global"], teacher["global"]))
            # Stage 3 no longer trains a render-closure objective; retain only
            # the semantic audio-to-emotion evaluation above.

            conditions = stage4.conditions(batch)
            final, residual = stage4.render(batch, conditions, steps=render_steps)
            mouth = torch.zeros(query["motion"].shape[-1], dtype=torch.bool, device=device)
            mouth[list(cfg["data"]["neutral_output_indices"])] = True
            # Style intervention: replace the reference style with query style.
            query_style = factors["style"]
            alt_conditions = dict(conditions)
            alt_conditions["style"] = query_style
            alt_final, _ = stage4.render(batch, alt_conditions, steps=render_steps)
            _add(metrics, "style_intervention_delta", _masked_mae(final, alt_final, query["mask"]))
            # Up to three independent references from the current batch, plus a
            # random-style null baseline so pairwise spread has something to be
            # measured against.
            style_outputs = [final]
            donor_styles = [conditions["style"]]
            re_encoded_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
            ref = batch["style_reference"]
            for ref_shift in (1, 2):
                if ref["residual_gt"].shape[0] <= ref_shift:
                    continue
                donor = stage2.encode_factors(
                    ref["residual_gt"][ref_shift:ref_shift+1], ref["residual_mask"][ref_shift:ref_shift+1],
                    ref.get("audio_emotion", ref.get("audio"))[ref_shift:ref_shift+1]
                )["style"].expand(query["motion"].shape[0], -1)
                c = dict(conditions); c["style"] = donor
                donor_final, donor_residual = stage4.render(batch, c, steps=render_steps)
                style_outputs.append(donor_final)
                donor_styles.append(donor)
                # Re-encoding cosine: pass the rendered residual back through the
                # style encoder and check that donor_style is recovered.  Uses
                # query audio for the encoder's audio hint because the audio
                # channel is meant to help distinguish content vs habit only.
                re_style = stage2.encode_factors(
                    donor_residual, query["mask"],
                    query.get("audio_emotion", query.get("audio")),
                )["style"]
                re_encoded_pairs.append((donor, re_style))
            if len(style_outputs) > 1:
                spread = sum(float(_masked_mae(style_outputs[0], x, query["mask"]).item()) for x in style_outputs[1:]) / (len(style_outputs) - 1)
                _add(metrics, "style_three_reference_spread", spread)
                # Null baseline: random style vector with matching norm.
                null_style = torch.randn_like(conditions["style"])
                null_style = null_style / null_style.norm(dim=-1, keepdim=True).clamp_min(1e-6) * conditions["style"].norm(dim=-1, keepdim=True)
                null_c = dict(conditions); null_c["style"] = null_style
                null_final, _ = stage4.render(batch, null_c, steps=render_steps)
                _add(metrics, "style_null_spread", _masked_mae(final, null_final, query["mask"]))
                # Mouth-content preservation: two style-swapped renders should keep
                # the query's phoneme timing on mouth channels.  Absolute amplitude
                # can differ; velocity direction should not.
                if len(style_outputs) >= 2:
                    mouth_corr = _velocity_channel_corr(style_outputs[0], style_outputs[1], query["mask"], mouth)
                    _add(metrics, "style_mouth_velocity_corr", mouth_corr)
            if re_encoded_pairs:
                cos = sum(F.cosine_similarity(re, donor, dim=-1).mean() for donor, re in re_encoded_pairs) / len(re_encoded_pairs)
                _add(metrics, "style_reencoding_cosine", cos)
            base = conditions["b0_pred"]
            active = torch.zeros(query["motion"].shape[-1], dtype=torch.bool, device=device)
            active[list(cfg["data"]["neutral_output_indices"])] = True
            _add(metrics, "stage4_base_b0_active_mae", _masked_channel_mae(base, query["b0_gt"], query["mask"], active))
            _add(metrics, "stage4_base_full_mae", _masked_mae(base, query["motion"], query["mask"]))
            _add(metrics, "stage4_zero_residual_baseline_mae", _masked_mae(base, query["motion"], query["mask"]))
            _add(metrics, "stage4_final_mae", _masked_mae(final, query["motion"], query["mask"]))
            _add(metrics, "stage4_final_rmse", _masked_rmse(final, query["motion"], query["mask"]))
            _add(metrics, "stage4_residual_mae", _masked_mae(residual, query["motion"] - base, query["mask"]))
            _add(metrics, "stage4_velocity_mae", _masked_velocity_mae(final, query["motion"], query["mask"]))
            _add(metrics, "stage4_mouth_velocity_mae", _masked_velocity_mae(final[..., mouth], query["motion"][..., mouth], query["mask"]))
            _add(metrics, "stage4_final_vs_base_gain", _masked_mae(base, query["motion"], query["mask"]) - _masked_mae(final, query["motion"], query["mask"]))

            labels = query["emotion_id"].tolist()
            for row, label in enumerate(labels):
                name = names[int(label)] if 0 <= int(label) < len(names) else str(label)
                local = by_emotion[name]
                local.setdefault("stage3_emotion_correct", []).append(float(audio["emotion_logits"][row].argmax().eq(query["emotion_id"][row]).item()))
                local.setdefault("stage3_intensity_correct", []).append(float(audio["intensity_logits"][row].argmax().eq(query["intensity_id"][row]).item()))
                local.setdefault("stage4_final_mae", []).append(float(_masked_mae(
                    final[row : row + 1], query["motion"][row : row + 1], query["mask"][row : row + 1]
                ).item()))

    print(f"split={args.split} samples={len(selected)} device={device} render_steps={render_steps}")
    print(f"checkpoint_epochs stage1={p1.get('epoch')} stage2={p2.get('epoch')} stage3={p3.get('epoch')} stage4={p4.get('epoch')}")
    for key in sorted(metrics):
        values = metrics[key]
        print(f"{key}={sum(values) / max(len(values), 1):.6f}")
    print("per_emotion:")
    for name in names:
        local = by_emotion.get(name, {})
        if not local:
            continue
        parts = []
        for key in ("stage3_emotion_correct", "stage3_intensity_correct", "stage4_final_mae"):
            values = local.get(key, [])
            parts.append(f"{key}={sum(values) / max(len(values), 1):.6f}")
        print(f"{name} count={len(local.get('stage4_final_mae', []))} " + " ".join(parts))


if __name__ == "__main__":
    main()
