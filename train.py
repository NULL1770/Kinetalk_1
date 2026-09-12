from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from kinetalk_b0.data import B0ResidualDataset, CanonicalStage1Dataset, collate_b0_residual, collate_stage1
from kinetalk_b0.losses import (
    stage2_loss,
    stage2_swap_loss,
    stage2_factor_cycle_loss,
    stage2_style_supcon_loss,
    style_factor_invariance_loss,
    style_content_invariance_loss,
    style_probe_loss,
    stage2_style_triplet_loss,
    stage2_style_cross_emotion_loss,
    stage3_loss,
    stage4_loss,
    stage1_clean_loss,
)
from kinetalk_b0.models import Stage1Model, Stage2Model, Stage3Model, Stage4Model
from kinetalk_b0.utils import load_checkpoint, load_yaml, move_to_device, save_checkpoint, seed_everything


def _device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "cuda"))
    return torch.device(requested if requested != "cuda" or torch.cuda.is_available() else "cpu")


def _configure_torch(config: dict[str, Any], device: torch.device) -> None:
    """Enable safe CUDA throughput settings without changing the raw-BS contract."""
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def _amp_dtype(config: dict[str, Any], device: torch.device) -> torch.dtype | None:
    if not bool(config.get("amp", False)) or device.type != "cuda":
        return None
    requested = str(config.get("amp_dtype", "bfloat16")).lower()
    if requested in {"float16", "fp16", "half"}:
        return torch.float16
    return torch.bfloat16


def _autocast(config: dict[str, Any], device: torch.device):
    dtype = _amp_dtype(config, device)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _scaler(config: dict[str, Any], device: torch.device):
    dtype = _amp_dtype(config, device)
    enabled = dtype == torch.float16 and device.type == "cuda"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def _loader(config: dict[str, Any], stage: str) -> DataLoader:
    if stage == "neutral":
        random_crop = bool(config["data"].get("stage1_random_crop", True))
        dataset = CanonicalStage1Dataset(config, split="train", random_crop=random_crop)
        batch_size = int(config["data"].get("batch_size", 4))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(config["data"].get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=int(config["data"].get("num_workers", 0)) > 0,
            collate_fn=collate_stage1,
        )
    dataset = B0ResidualDataset(config, split="train", random_crop=True)
    workers = int(config["data"].get("num_workers", 0))
    sampler = None
    if bool(config["data"].get("speaker_balanced", True)) and stage in {"factors", "generator"}:
        counts = {}
        for item in dataset.items:
            sp = str(item.get("speaker", "unknown")); counts[sp] = counts.get(sp, 0) + 1
        weights = torch.tensor([1.0 / counts[str(x.get("speaker", "unknown"))] for x in dataset.items], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return DataLoader(
        dataset,
        batch_size=int(config["data"].get("batch_size", 4)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=collate_b0_residual,
    )


def _optimizer(config: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    optim_cfg = config["optim"]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters remain for this stage")
    return torch.optim.AdamW(
        parameters,
        lr=float(optim_cfg.get("lr", 3e-4)),
        weight_decay=float(optim_cfg.get("weight_decay", 1e-5)),
    )


def _step_optimizer(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    config: dict[str, Any],
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["optim"].get("grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["optim"].get("grad_clip", 1.0)))
        optimizer.step()


def _stage1(config: dict[str, Any], device: torch.device) -> None:
    model = Stage1Model(config).to(device)
    optimizer = _optimizer(config, model)
    loader = _loader(config, "neutral")
    loss_cfg = config["loss"]
    scaler = _scaler(config, device)
    # Stage1 structurally emits only these channels. Counting the remaining
    # channels creates irreducible loss because their predictions are fixed at
    # zero, which dilutes the mouth gradients and makes the scalar loss look
    # better than the actual articulation fidelity.
    channel_weights = torch.zeros(int(config["data"]["motion_dim"]), device=device)
    channel_weights[list(config["data"]["neutral_output_indices"])] = float(loss_cfg.get("stage1_active_weight", 1.0))
    for epoch in range(int(config["optim"].get("stage1_epochs", config["optim"].get("epochs", 1)))):
        totals = 0.0
        for batch in loader:
            batch = move_to_device(batch, device)
            content, target, mask = batch["content"], batch["target"], batch["mask"]
            cross_start = int(loss_cfg.get("stage1_cross_emotion_start_epoch", 0))
            if epoch < cross_start:
                identity = torch.tensor(
                    [str(kind) == "identity" for kind in batch.get("pair_type", [])],
                    device=mask.device, dtype=torch.bool,
                )
                if not bool(identity.any()):
                    continue
                mask = mask & identity.unsqueeze(1)
            with _autocast(config, device):
                output = model(
                    content, mask,
                )
                losses = stage1_clean_loss(output["b0"], target, mask, channel_weights,
                                           velocity_weight=float(loss_cfg.get("stage1_velocity", 0.5)),
                                           frame_weight=batch.get("quality"),
                                           sample_weight=batch.get("pair_weight"),
                                           acceleration_weight=float(loss_cfg.get("stage1_acceleration", 0.0)))
            _step_optimizer(losses["total"], optimizer, model, config, scaler)
            totals += float(losses["total"].detach())
        print(f"stage1 epoch={epoch + 1} loss={totals / max(len(loader), 1):.6f}", flush=True)
        save_checkpoint(config["paths"]["stage1_ckpt"], model, optimizer, epoch + 1, stage="stage1")
        epoch_dir = config["paths"].get("stage1_epoch_ckpt_dir")
        if epoch_dir:
            save_checkpoint(Path(epoch_dir) / f"stage1_epoch_{epoch + 1:03d}.pt", model, optimizer, epoch + 1, stage="stage1")


def _load_stage1(config: dict[str, Any], device: torch.device) -> Stage1Model:
    model = Stage1Model(config).to(device)
    load_checkpoint(config["paths"]["stage1_ckpt"], model, map_location=device, strict=True, expected_architecture_version=6)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def _stage2(config: dict[str, Any], device: torch.device) -> None:
    stage1 = _load_stage1(config, device)
    model = Stage2Model(config).to(device)
    optimizer = _optimizer(config, model)
    loader = _loader(config, "factors")
    loss_cfg = config["loss"]
    cross_weight = float(loss_cfg.get("stage2_cross", 1.0))
    cycle_weight = float(loss_cfg.get("stage2_cycle", 0.5))
    supcon_weight = float(loss_cfg.get("stage2_style_supcon", 0.5))
    supcon_temperature = float(loss_cfg.get("stage2_style_supcon_temperature", 0.1))
    scaler = _scaler(config, device)
    for epoch in range(int(config["optim"].get("epochs", 1))):
        totals = 0.0
        for batch in loader:
            batch = move_to_device(batch, device)
            query, emotion = batch["query"], batch["emotion_pair"]
            with torch.no_grad():
                query_stage1 = stage1(
                    query["content"], query["mask"],
                )
                h0 = query_stage1["h0"]
                query_style_residual = query["motion"] - query_stage1["b0"]
            with _autocast(config, device):
                factors = model.encode_factors(
                    query["residual_gt"], query["residual_mask"],
                    query.get("audio_emotion", query.get("audio")),
                    style_residual=query_style_residual,
                )
                flow_prediction, flow_target = model.flow_prediction(query["residual_gt"], h0, factors, query["residual_mask"])
                losses = stage2_loss(
                    flow_prediction,
                    flow_target,
                    query["residual_gt"],
                    query["residual_mask"],
                    factors,
                    query["emotion_id"],
                    query["intensity_id"],
                    velocity_weight=float(loss_cfg.get("stage2_velocity", 0.05)),
                    classification_weight=float(loss_cfg.get("stage2_classification", 0.5)),
                    intensity_weight=float(loss_cfg.get("stage2_intensity", 0.25)),
                )
                if supcon_weight > 0:
                    speakers = batch["query"].get("speaker", [])
                    if len(speakers) == factors["style"].shape[0] and len(speakers) > 1:
                        same_speaker = torch.tensor(
                            [[a == b for b in speakers] for a in speakers],
                            device=factors["style"].device,
                            dtype=torch.bool,
                        )
                        supcon = stage2_style_supcon_loss(
                            factors["style"], same_speaker, temperature=supcon_temperature
                        )
                        losses["style_supcon"] = supcon
                        losses["total"] = losses["total"] + supcon_weight * supcon
                losses["style_invariance"] = style_factor_invariance_loss(
                    factors["style"], factors["emotion_logits"], query["emotion_id"]
                )
                losses["total"] = losses["total"] + float(loss_cfg.get("stage2_style_invariance", 0.05)) * losses["style_invariance"]
                losses["style_content_invariance"] = style_content_invariance_loss(factors["style"], query["content"])
                losses["total"] = losses["total"] + float(loss_cfg.get("stage2_style_content_invariance", 0.02)) * losses["style_content_invariance"]
                probe_weight = float(loss_cfg.get("stage2_style_probe", 0.0))
                if probe_weight > 0.0:
                    # These logits pass through GRL: the probe learns to predict
                    # emotion/content while the style encoder receives reversed
                    # gradients and is discouraged from carrying those signals.
                    losses["style_probe"] = style_probe_loss(factors, query["emotion_id"], query["content"])
                    losses["total"] = losses["total"] + probe_weight * losses["style_probe"]
                emotion_factors = None
                if bool(batch["relations"]["emotion"].any()):
                    with torch.no_grad():
                        emotion_stage1 = stage1(
                            emotion["content"], emotion["mask"],
                        )
                        emotion_style_residual = emotion["motion"] - emotion_stage1["b0"]
                    emotion_factors = model.encode_factors(
                        emotion["residual_gt"], emotion["residual_mask"],
                        emotion.get("audio_emotion", emotion.get("audio")),
                        style_residual=emotion_style_residual,
                    )
                    swapped = model.render(
                        h0,
                        {"global": emotion_factors["global"], "intensity_value": emotion_factors["intensity_value"], "style": factors["style"]},
                        query["mask"],
                        steps=int(config["model"].get("render_steps", 4)),
                    )
                    swap_mask = query["mask"] & emotion["mask"] & batch["relations"]["emotion"].unsqueeze(1)
                    cross = stage2_swap_loss(swapped, emotion["motion"], emotion["b0_gt"], swap_mask)
                    swapped_factors = model.encode_factors(swapped, query["mask"], query.get("audio_emotion", query.get("audio")))
                    expected = {
                        "global": emotion_factors["global"],
                        "style": factors["style"],
                    }
                    emotion_cycle = stage2_factor_cycle_loss(swapped_factors, expected, batch["relations"]["emotion"])
                    losses["cross"] = cross
                    losses["emotion_cycle"] = emotion_cycle
                    losses["total"] = losses["total"] + cross_weight * cross + cycle_weight * emotion_cycle
                if bool(batch["relations"]["style"].any()):
                    style_reference = batch["style_reference"]
                    with torch.no_grad():
                        style_stage1 = stage1(
                            style_reference["content"], style_reference["mask"],
                        )
                        style_reference_residual = style_reference["motion"] - style_stage1["b0"]
                    style_factors = model.encode_factors(
                        style_reference["residual_gt"], style_reference["residual_mask"],
                        style_reference.get("audio_emotion", style_reference.get("audio")),
                        style_residual=style_reference_residual,
                    )
                    style_swapped = model.render(
                        h0,
                        {"global": factors["global"], "intensity_value": factors["intensity_value"], "style": style_factors["style"]},
                        query["mask"],
                        steps=int(config["model"].get("render_steps", 4)),
                    )
                    style_reencoded = model.encode_factors(style_swapped, query["mask"], query.get("audio_emotion", query.get("audio")))
                    expected = {
                        "global": factors["global"],
                        "style": style_factors["style"],
                    }
                    style_cycle = stage2_factor_cycle_loss(style_reencoded, expected, batch["relations"]["style"])
                    losses["style_cycle"] = style_cycle
                    losses["total"] = losses["total"] + cycle_weight * style_cycle
                # Triplet uses the dedicated same-speaker `style_positive`
                # branch (different sentence + different emotion when possible).
                # `emotion_pair` shares the query sentence and would reward
                # content leak; `style_reference` follows `cross_speaker_style`
                # and can become cross-speaker, inverting the pull direction.
                triplet_weight = float(loss_cfg.get("stage2_style_triplet", 0.0))
                cross_emotion_weight = float(loss_cfg.get("stage2_style_cross_emotion", 0.0))
                positive_available = "style_positive" in batch and bool(batch["relations"].get("style_positive", torch.zeros(1, dtype=torch.bool)).any())
                negative_available = "style_negative" in batch and bool(batch["relations"].get("style_negative", torch.zeros(1, dtype=torch.bool)).any())
                if (triplet_weight > 0.0 or cross_emotion_weight > 0.0) and positive_available:
                    positive = batch["style_positive"]
                    with torch.no_grad():
                        positive_stage1 = stage1(
                            positive["content"], positive["mask"],
                        )
                        positive_style_residual = positive["motion"] - positive_stage1["b0"]
                    positive_factors = model.encode_factors(
                        positive["residual_gt"],
                        positive["residual_mask"],
                        positive.get("audio_emotion", positive.get("audio")),
                        style_residual=positive_style_residual,
                    )
                    if cross_emotion_weight > 0.0:
                        cross_emotion = stage2_style_cross_emotion_loss(
                            factors["style"], positive_factors["style"], batch["relations"]["style_positive"]
                        )
                        losses["style_cross_emotion"] = cross_emotion
                        losses["total"] = losses["total"] + cross_emotion_weight * cross_emotion
                    if triplet_weight > 0.0 and negative_available:
                        negative = batch["style_negative"]
                        with torch.no_grad():
                            negative_stage1 = stage1(
                                negative["content"], negative["mask"],
                            )
                            negative_style_residual = negative["motion"] - negative_stage1["b0"]
                        negative_factors = model.encode_factors(
                            negative["residual_gt"],
                            negative["residual_mask"],
                            negative.get("audio_emotion", negative.get("audio")),
                            style_residual=negative_style_residual,
                        )
                        triplet_valid = batch["relations"]["style_positive"] & batch["relations"]["style_negative"]
                        triplet = stage2_style_triplet_loss(
                            factors["style"],
                            positive_factors["style"],
                            negative_factors["style"],
                            triplet_valid,
                            margin=float(loss_cfg.get("stage2_style_triplet_margin", 0.2)),
                        )
                        losses["style_triplet"] = triplet
                        losses["total"] = losses["total"] + triplet_weight * triplet
            _step_optimizer(losses["total"], optimizer, model, config, scaler)
            totals += float(losses["total"].detach())
        print(f"stage2 epoch={epoch + 1} loss={totals / max(len(loader), 1):.6f}", flush=True)
        save_checkpoint(config["paths"]["stage2_ckpt"], model, optimizer, epoch + 1, stage="stage2")
    


def _load_stage2(config: dict[str, Any], device: torch.device) -> Stage2Model:
    model = Stage2Model(config).to(device)
    load_checkpoint(config["paths"]["stage2_ckpt"], model, map_location=device, strict=True, expected_architecture_version=6)
    return model


def _stage3(config: dict[str, Any], device: torch.device) -> None:
    stage2 = _load_stage2(config, device)
    model = Stage3Model(config, stage2).to(device)
    optimizer = _optimizer(config, model)
    loader = _loader(config, "audio_emotion")
    loss_cfg = config["loss"]
    scaler = _scaler(config, device)
    for epoch in range(int(config["optim"].get("epochs", 1))):
        totals = 0.0
        for batch in loader:
            batch = move_to_device(batch, device)
            query = batch["query"]
            with _autocast(config, device):
                audio_factors = model(query["audio_emotion"], query["mask"])
                teacher = model.teacher(
                    query["residual_gt"], query["residual_mask"],
                )
                losses = stage3_loss(
                    audio_factors,
                    teacher,
                    query["emotion_id"],
                    query["intensity_id"],
                    global_weight=float(loss_cfg.get("stage3_global", 1.0)),
                    classification_weight=float(loss_cfg.get("stage3_classification", 1.0)),
                    intensity_weight=float(loss_cfg.get("stage3_intensity", 0.5)),
                )
            _step_optimizer(losses["total"], optimizer, model, config, scaler)
            totals += float(losses["total"].detach())
        print(f"stage3 epoch={epoch + 1} loss={totals / max(len(loader), 1):.6f}", flush=True)
        save_checkpoint(config["paths"]["stage3_ckpt"], model, optimizer, epoch + 1, stage="stage3")


def _load_stage3(config: dict[str, Any], device: torch.device, stage2: Stage2Model) -> Stage3Model:
    model = Stage3Model(config, stage2).to(device)
    load_checkpoint(config["paths"]["stage3_ckpt"], model, map_location=device, strict=True, expected_architecture_version=6)
    return model


def _stage4(config: dict[str, Any], device: torch.device) -> None:
    stage1 = _load_stage1(config, device)
    stage2 = _load_stage2(config, device)
    stage3 = _load_stage3(config, device, stage2)
    model = Stage4Model(config, stage1, stage2, stage3).to(device)
    optimizer = _optimizer(config, model)
    loader = _loader(config, "generator")
    loss_cfg = config["loss"]
    scaler = _scaler(config, device)
    for epoch in range(int(config["optim"].get("epochs", 1))):
        totals = 0.0
        for batch in loader:
            batch = move_to_device(batch, device)
            with _autocast(config, device):
                conditions = model.conditions(batch, training_target=True)
                flow_pred, flow_target, _ = model.flow_prediction(batch, conditions)
                losses = stage4_loss(
                    flow_pred,
                    flow_target,
                    batch["query"]["mask"],
                )
            _step_optimizer(losses["total"], optimizer, model, config, scaler)
            totals += float(losses["total"].detach())
        print(f"stage4 epoch={epoch + 1} loss={totals / max(len(loader), 1):.6f}", flush=True)
        save_checkpoint(config["paths"]["stage4_ckpt"], model, optimizer, epoch + 1, stage="stage4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the standalone KineTalk b0 + residual DiT stages")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--stage", choices=["neutral", "factors", "audio_emotion", "generator"], required=True)
    parser.add_argument("--pipeline", action="store_true", help="Run stages 1-4 sequentially after contract checks")
    args = parser.parse_args()
    config = load_yaml(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = _device(config)
    _configure_torch(config, device)
    print(f"amp_dtype={_amp_dtype(config, device) or 'off'} batch_size={config['data'].get('batch_size', 4)} workers={config['data'].get('num_workers', 0)}")
    print(f"stage={args.stage} device={device} raw_bs=true aligned_dtw=v3 safe_supervision=true training_requested=true")
    if args.pipeline:
        _stage1(config, device)
        # _loader() fail-fast protects against using canonical neutral-only
        # artifacts as emotional residual supervision.
        _stage2(config, device)
        _stage3(config, device)
        _stage4(config, device)
    else:
        {"neutral": _stage1, "factors": _stage2, "audio_emotion": _stage3, "generator": _stage4}[args.stage](config, device)


if __name__ == "__main__":
    main()
