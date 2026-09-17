"""Probe a compact temporal head over frozen emotion2vec frame features.

This is an isolated diagnostic experiment.  The renderer, B0, identity and
global-affect paths are untouched.  The only trainable parameters are a
96-channel dilated temporal head and a scalar output used for the existing
upper-face intensity target.  It works with the current 768-D final
emotion2vec cache; if intermediate layers become available later, their
features can be projected into the same input contract without changing the
training/evaluation code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.multiscale_audio import FrozenEmotion2VecTemporalPredictor
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.neutral_data import NeutralAffectDataset
from kinetalk_b0.utils import freeze_module
from scripts.probe_dynamic_predictability import (
    bin_frames, center_controls, fit_target_scale, sentence_split,
    target_metrics, weighted_mse,
)
from scripts.probe_expression_energy_field import fit_expression_energy_target
from scripts.train_neutral_affect_audio_ablation import state_hash
from scripts.train_neutral_affect_pilot import (
    affect_residual, cached_identity, device_batch, observed, prepare,
    select, sha, write_json,
)


def binned_prediction(frame_prediction: torch.Tensor, valid: torch.Tensor,
                      stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the existing low-rate mean-removed control contract."""
    if frame_prediction.ndim != 3 or frame_prediction.shape[-1] != 1:
        raise ValueError("frame_prediction must be [batch,time,1]")
    clean = torch.where(valid[..., None], frame_prediction, 0)
    binned, weight = bin_frames(clean, valid, stride)
    return center_controls(binned, weight), weight


def _metrics(pred, target, weight, query, ids, scale):
    q = select(query, ids)
    p, t, w = pred[ids], target[ids], weight[ids]
    out = target_metrics(p, t, w, q["clip_id"], q["sentence_id"], scale)
    mse = weighted_mse(p * scale, t * scale, w)
    zero = weighted_mse(torch.zeros_like(t) * scale, t * scale, w)
    out.update({"native_mse": float(mse), "native_zero_mse": float(zero),
                "native_r2_against_zero": float(1 - mse / zero)
                if float(zero) > 1e-12 else None})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--hidden-dim", type=int, default=96)
    args = parser.parse_args()
    if args.steps < 0 or args.hidden_dim < 1:
        raise ValueError("steps and hidden-dim must be positive")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "audio":
        raise ValueError("checkpoint must be an audio-stage checkpoint")
    source_hash = sha(args.data / "train.pt")
    expected_hash = checkpoint.get("provenance", {}).get("train_sha256")
    if expected_hash and source_hash != expected_hash:
        raise ValueError("Training cache hash differs from checkpoint")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(cfg.get("device", "cuda"))
    dataset = NeutralAffectDataset(args.data / "train.pt")
    train_cpu, heldout_cpu = sentence_split([q["sentence_id"] for q in dataset.queries], args.seed)
    train_ids, heldout_ids = train_cpu.to(device), heldout_cpu.to(device)
    system = NeutralAffectSystem(cfg).to(device).eval()
    system.load_state_dict(checkpoint["model"], strict=True)
    freeze_module(system)
    with torch.no_grad():
        query, _ = prepare(system, dataset, device)
        refs = dataset.identity_references
        reference = device_batch([r for speaker in sorted(refs) for r in refs[speaker]], device)
        ref_residual = torch.where(observed(reference), reference["motion"] -
                                   system.base(reference["content"], reference["valid"])["b0"], 0)
        identities = cached_identity(system, {"residual": ref_residual.reshape(
            len(refs), len(refs[0]), *ref_residual.shape[1:]),
            "valid": reference["valid"].reshape(len(refs), len(refs[0]), -1)})
        residual = affect_residual(query, select(identities, query["speaker_id"]))
        energy, weight, target_state = fit_expression_energy_target(
            residual, query["valid"], query["channel_mask"], query["speaker_id"],
            query["emotion_id"], query["intensity_id"], train_ids,
            system.motion_teacher.stride, mode="upper_l1")
        scale = fit_target_scale(energy, weight, train_ids)
    args.output.mkdir(parents=True, exist_ok=False)
    features = query["audio"]
    if features.shape[-1] != 768:
        raise ValueError(f"Expected frozen emotion2vec 768-D features, got {features.shape[-1]}")
    predictor = FrozenEmotion2VecTemporalPredictor(768, args.hidden_dim).to(device)
    opt = torch.optim.AdamW(predictor.parameters(), lr=float(cfg.get("training", {}).get("audio_lr", 5e-4)), weight_decay=1e-5)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    target = energy / scale
    predictor.train()
    losses = []
    for _ in range(args.steps):
        sampled = torch.randint(len(train_cpu), (int(cfg.get("training", {}).get("batch_size", 8)),), generator=rng)
        ids = train_cpu[sampled].to(device)
        frame = torch.tanh(predictor(features[ids], query["valid"][ids]))
        pred, pred_weight = binned_prediction(frame, query["valid"][ids], system.motion_teacher.stride)
        w = weight[ids]
        loss = ((pred - target[ids]).square() * w[..., None]).sum() / w.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    predictor.eval()
    with torch.no_grad():
        pred, pred_weight = binned_prediction(torch.tanh(predictor(features, query["valid"])), query["valid"], system.motion_teacher.stride)
    summary = {"schema": "emotion2vec_multiscale_probe_v1", "seed": args.seed,
               "steps": args.steps, "input_dim": 768, "hidden_dim": args.hidden_dim,
               "dilations": [1, 2, 4], "target": "upper-face L1, stride4, clip-centered",
               "train": _metrics(pred, target, weight, query, train_cpu, scale),
               "heldout": _metrics(pred, target, weight, query, heldout_cpu, scale),
               "last_loss": losses[-1] if losses else None,
               "state_sha256": state_hash(predictor.state_dict()),
               "source_sha256": {"script": sha(Path(__file__))}}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "provenance.json", {"schema": "emotion2vec_multiscale_probe_v1",
        "checkpoint_sha256": sha(args.checkpoint), "train_cache_sha256": source_hash,
        "feature_extractor": "frozen emotion2vec final frame embedding (768-D)",
        "train_indices": train_cpu.tolist(), "heldout_indices": heldout_cpu.tolist()})
    torch.save({"prediction": pred.cpu(), "target": target.cpu(), "weight": weight.cpu(),
                "model": predictor.state_dict()}, args.output / "curves.pt")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
