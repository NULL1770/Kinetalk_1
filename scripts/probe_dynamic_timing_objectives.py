"""Matched FiLM probes using one of MSE, clip-shape MSE, or event CE.

This changes supervision only. The pretrained system stays frozen. No motion
target or sentence-specific scale is used when making a heldout prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.dynamic_timing import (
    event_loss, event_metrics, fit_event_threshold, fit_positive_gain,
    normalize_target_shape,
)
from kinetalk_b0.gated_audio_dynamic import ContentGatedDynamicPredictor
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.neutral_data import NeutralAffectDataset
from kinetalk_b0.utils import freeze_module
from scripts.probe_audio_content_gating import (
    field_prediction, intervene_content, normalize_inputs,
)
from scripts.probe_dynamic_predictability import (
    fit_target_scale, sentence_split, weighted_mse,
)
from scripts.probe_expression_energy_field import energy_metrics, fit_expression_energy_target
from scripts.train_neutral_affect_audio_ablation import state_hash
from scripts.train_neutral_affect_pilot import (
    affect_residual, cached_identity, device_batch, observed, prepare, select,
    sha, write_json,
)


def reverse_field(values, weight):
    result = torch.zeros_like(values)
    for i in range(len(values)):
        ids = (weight[i] > 0).nonzero(as_tuple=True)[0]
        result[i, ids] = values[i, ids.flip(0)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "data", "checkpoint", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--objective", choices=("mse", "clip_rms", "events"), default="clip_rms")
    parser.add_argument("--mode", choices=("audio_only", "film", "content_only"), default="film")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--bottleneck-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()
    if args.steps < 0 or min(args.hidden_dim, args.bottleneck_dim) < 1:
        raise ValueError("steps must be nonnegative and dimensions positive")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "audio":
        raise ValueError("checkpoint must be an audio-stage checkpoint")
    cache_hash = sha(args.data / "train.pt")
    if cache_hash != checkpoint.get("provenance", {}).get("train_sha256"):
        raise ValueError("training cache differs from checkpoint provenance")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(cfg.get("device", "cuda"))
    dataset = NeutralAffectDataset(args.data / "train.pt")
    train_cpu, heldout_cpu = sentence_split([q["sentence_id"] for q in dataset.queries], args.seed)
    train_ids, heldout_ids = train_cpu.to(device), heldout_cpu.to(device)
    system = NeutralAffectSystem(cfg).to(device).eval()
    system.load_state_dict(checkpoint["model"], strict=True)
    freeze_module(system)
    frozen_hash = state_hash(system.state_dict())
    with torch.no_grad():
        query, _ = prepare(system, dataset, device)
        if query["audio"].shape[-1] != 768 or query["content"].shape[-1] != 768:
            raise ValueError("768-D emotion2vec and content caches are required")
        refs = dataset.identity_references
        if sorted(refs) != list(range(len(refs))) or len({len(v) for v in refs.values()}) != 1:
            raise ValueError("contiguous speaker IDs and equal reference counts required")
        reference = device_batch([r for speaker in sorted(refs) for r in refs[speaker]], device)
        ref_residual = torch.where(observed(reference), reference["motion"] -
            system.base(reference["content"], reference["valid"])["b0"], 0)
        identity = cached_identity(system, {"residual": ref_residual.reshape(
            len(refs), len(refs[0]), *ref_residual.shape[1:]),
            "valid": reference["valid"].reshape(len(refs), len(refs[0]), -1)})
        residual = affect_residual(query, select(identity, query["speaker_id"]))
        if not query["channel_mask"][:, list(range(14)) + list(range(41, 46))].all():
            raise ValueError("upper_l1 probe requires observed upper-face channels")
        stride = system.motion_teacher.stride
        energy, weight, _ = fit_expression_energy_target(
            residual, query["valid"], query["channel_mask"], query["speaker_id"],
            query["emotion_id"], query["intensity_id"], train_ids, stride, mode="upper_l1")
        scale = fit_target_scale(energy, weight, train_ids)
        target = energy / scale
        shape_target, shape_floor = normalize_target_shape(target, weight, train_ids)
        event_threshold = fit_event_threshold(target, weight, train_ids)
        query, feature_stats = normalize_inputs(query, train_ids)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    model = ContentGatedDynamicPredictor(hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim, output_dim=1, mode=args.mode).to(device)
    initial_hash = state_hash(model.state_dict())
    lr = args.learning_rate if args.learning_rate is not None else float(cfg["training"].get("audio_lr", 5e-4))
    if lr <= 0:
        raise ValueError("learning rate must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    losses = []
    model.train()
    for step in range(args.steps):
        sampled = torch.randint(len(train_cpu), (int(cfg["training"].get("batch_size", 8)),), generator=rng)
        ids = train_cpu[sampled].to(device)
        pred, _ = field_prediction(model, select(query, ids), stride)
        if args.objective == "events":
            loss = event_loss(pred, target[ids], weight[ids], event_threshold)
        else:
            supervision = shape_target if args.objective == "clip_rms" else target
            loss = weighted_mse(pred, supervision[ids], weight[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 200 == 0:
            print(json.dumps({"step": step + 1, "loss": losses[-1]}), flush=True)
    model.eval()
    with torch.no_grad():
        pred, _ = field_prediction(model, query, stride)
        gain = fit_positive_gain(pred, target, weight, train_ids)
        calibrated = pred * gain
        reverse = reverse_field(calibrated, weight)
        zero = torch.zeros_like(pred)
        def metrics(p, ids):
            return energy_metrics(p, target, weight, query, ids, scale)
        heldout_query = select(query, heldout_ids)
        local_ids = torch.arange(len(heldout_ids), device=device)
        interventions = {}
        for mode in ("zero", "reverse", "shuffle"):
            altered, _ = field_prediction(model, intervene_content(heldout_query, mode), stride)
            interventions[mode] = energy_metrics(altered * gain, target[heldout_ids],
                weight[heldout_ids], heldout_query, local_ids, scale)
        frozen_after_hash = state_hash(system.state_dict())
        if frozen_hash != frozen_after_hash:
            raise RuntimeError("frozen main model state changed")
        event_rows = {name: event_metrics(value[heldout_ids], target[heldout_ids],
                      weight[heldout_ids], event_threshold) for name, value in
                      (("raw", pred), ("calibrated", calibrated), ("zero", zero), ("reverse", reverse))}
    summary = {"schema": "dynamic_timing_objective_probe_v1", "seed": args.seed,
        "mode": args.mode, "objective": args.objective, "steps": args.steps, "stride": stride,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "train": metrics(pred, train_ids), "heldout": metrics(pred, heldout_ids),
        "train_calibrated": metrics(calibrated, train_ids),
        "heldout_calibrated": metrics(calibrated, heldout_ids),
        "heldout_calibrated_reverse": metrics(reverse, heldout_ids),
        "heldout_events": event_rows,
        "heldout_shape": energy_metrics(pred, shape_target, weight, query,
                                         heldout_ids, torch.ones_like(scale)),
        "heldout_content_interventions_calibrated": interventions,
        "training_fit_gain": gain.cpu().tolist(),
        "training_fit_event_threshold": event_threshold.cpu().tolist(),
        "training_fit_shape_floor": shape_floor.cpu().tolist(),
        "last_loss": losses[-1] if losses else None,
        "initial_state_sha256": initial_hash, "final_state_sha256": state_hash(model.state_dict()),
        "main_model_unchanged": frozen_hash == frozen_after_hash,
        "source_sha256": {"script": sha(Path(__file__)), "objective_module": sha(
            Path(__file__).resolve().parents[1] / "kinetalk_b0/dynamic_timing.py"),
            "predictor": sha(Path(__file__).resolve().parents[1] / "kinetalk_b0/gated_audio_dynamic.py")}}
    write_json(args.output / "summary.json", summary)
    serialized_stats = {key: {k: v.tolist() if torch.is_tensor(v) else v for k, v in value.items()}
                        for key, value in feature_stats.items()}
    write_json(args.output / "provenance.json", {
        "schema": summary["schema"], "checkpoint_sha256": sha(args.checkpoint),
        "train_cache_sha256": cache_hash, "main_model_sha256": frozen_hash,
        "feature_stats": serialized_stats, "train_indices": train_cpu.tolist(),
        "heldout_indices": heldout_cpu.tolist(),
        "train_sentences": sorted(set(query["sentence_id"][int(i)] for i in train_cpu)),
        "heldout_sentences": sorted(set(query["sentence_id"][int(i)] for i in heldout_cpu)),
        "target_scale": scale.cpu().tolist(), "target_mode": "upper_l1",
        "calibration": "one nonnegative gain fit on training outputs/labels only; no heldout/sentence/identity adaptation",
        "shape_metric": "heldout target RMS is used only to score shape, never to set generated amplitude",
        "selection": "fixed final step; no heldout model selection"})
    effective = {**cfg, "probe": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (args.output / "effective_config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf8")
    torch.save({"prediction": pred.cpu(), "calibrated_prediction": calibrated.cpu(),
        "target": target.cpu(), "shape_target": shape_target.cpu(), "weight": weight.cpu(),
        "model": {k: v.cpu() for k, v in model.state_dict().items()}, "gain": gain.cpu(),
        "scale": scale.cpu(), "event_threshold": event_threshold.cpu(),
        "shape_floor": shape_floor.cpu(), "feature_stats": feature_stats}, args.output / "curves.pt")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
