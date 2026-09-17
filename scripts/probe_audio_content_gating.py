"""Sentence-disjoint FiLM probe: frozen emotion2vec plus frozen content.

One target MSE; no additional losses and no main-model edits. The same backbone
can be run with audio_only, film and content_only for matched controls. Input
and target scales are fit exclusively on training sentences. Heldout content
zero/reverse/shuffle interventions diagnose whether timing information helps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.gated_audio_dynamic import ContentGatedDynamicPredictor
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.neutral_data import NeutralAffectDataset
from kinetalk_b0.utils import freeze_module
from scripts.probe_dynamic_predictability import (
    bin_frames, center_controls, fit_target_scale, sentence_split, weighted_mse,
)
from scripts.probe_expression_energy_field import (
    axis_metrics, energy_metrics, fit_expression_energy_target,
)
from scripts.train_neutral_affect_audio_ablation import state_hash
from scripts.train_neutral_affect_feature_probe import fit_feature_stats
from scripts.train_neutral_affect_pilot import (
    affect_residual, cached_identity, device_batch, observed, prepare, select,
    sha, write_json,
)


def normalize_inputs(query, train_ids):
    """Refit both affine transforms on training frames, preserving B0 inputs.

    Re-standardizing cached emotion2vec features cancels any prior positive
    affine cache standardization, so no heldout statistics remain in the probe
    inputs (up to floating-point rounding and the fixed small std floor).
    """
    result, stats = dict(query), {}
    for source, key in (("acoustic", "audio"), ("content", "content")):
        fitted = fit_feature_stats(select(query, train_ids), source)
        mean, std = [fitted[k].to(query[key]) for k in ("mean", "std")]
        result[key] = torch.where(query["valid"][..., None],
                                  (query[key] - mean) / std, 0)
        stats[key] = fitted
    return result, stats


def prepare_dynamic_audio(query, mode="raw"):
    """Optional removal of per-utterance DC using audio alone, at inference too."""
    if mode == "raw":
        return query
    if mode != "clip_centered":
        raise ValueError("audio input must be raw or clip_centered")
    valid = query["valid"]
    clean = torch.where(valid[..., None], query["audio"], 0)
    mean = clean.sum(1, keepdim=True) / valid.sum(1).clamp_min(1)[:, None, None]
    return {**query, "audio": torch.where(valid[..., None], clean - mean, 0)}


def field_prediction(model, query, stride, readout="legacy_tanh"):
    logits = model(query["audio"], query["content"], query["valid"])
    if readout == "legacy_tanh":
        frames = 3.0 * logits.tanh()
    elif readout == "linear":
        # Same slope at zero, but a constant logit bias cannot saturate a
        # whole utterance and then disappear in clip centering.
        frames = 3.0 * logits
    else:
        raise ValueError("readout must be legacy_tanh or linear")
    values, weight = bin_frames(frames, query["valid"], stride)
    return center_controls(values, weight), weight


@torch.no_grad()
def readout_diagnostics(model, query):
    logits = model(query["audio"], query["content"], query["valid"])
    valid = query["valid"]
    saturation = (logits.abs() > 3).all(-1) & valid
    return {"fraction_frames_abs_logit_gt3": float(saturation.sum() / valid.sum()),
            "all_saturated_clips": int((saturation.sum(1) == valid.sum(1)).sum()),
            "mean_tanh_derivative": float((1 - logits[valid].tanh().square()).mean()),
            "raw_logit_min": float(logits[valid].min()),
            "raw_logit_max": float(logits[valid].max())}


def intervene_content(query, mode):
    """Keep frame masks fixed; interventions use no target or training clips."""
    content, valid = query["content"], query["valid"]
    if mode == "zero":
        altered = torch.zeros_like(content)
    elif mode == "reverse":
        altered = torch.zeros_like(content)
        for i in range(len(content)):
            idx = valid[i].nonzero(as_tuple=True)[0]
            altered[i, idx] = content[i, idx.flip(0)]
    elif mode == "shuffle":
        if len(content) < 2:
            raise ValueError("content shuffle needs at least two clips")
        # Resample only valid frames so another clip's padding cannot become
        # the explanation for a shuffle degradation.
        altered = torch.zeros_like(content)
        for i in range(len(content)):
            destination = valid[i].nonzero(as_tuple=True)[0]
            source = content[(i + 1) % len(content)][valid[(i + 1) % len(content)]]
            if len(destination) and len(source):
                positions = torch.linspace(0, len(source) - 1, len(destination),
                                            device=content.device).round().long()
                altered[i, destination] = source[positions]
    else:
        raise ValueError("mode must be zero, reverse or shuffle")
    return {**query, "content": altered}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "data", "checkpoint", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--mode", choices=("audio_only", "film", "content_only"), default="film")
    parser.add_argument("--target", choices=("upper_l1", "energy_velocity"), default="upper_l1")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--split-seed", type=int, default=None,
                        help="Separate sentence split from model/minibatch seed")
    parser.add_argument("--readout", choices=("legacy_tanh", "linear"), default="legacy_tanh")
    parser.add_argument("--audio-input", choices=("raw", "clip_centered"), default="raw")
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
    split_seed = args.seed if args.split_seed is None else args.split_seed
    train_cpu, heldout_cpu = sentence_split([q["sentence_id"] for q in dataset.queries], split_seed)
    train_ids, heldout_ids = train_cpu.to(device), heldout_cpu.to(device)
    system = NeutralAffectSystem(cfg).to(device).eval()
    system.load_state_dict(checkpoint["model"], strict=True)
    freeze_module(system)
    frozen_hash = state_hash(system.state_dict())
    with torch.no_grad():
        query, _ = prepare(system, dataset, device)
        if query["audio"].shape[-1] != 768 or query["content"].shape[-1] != 768:
            raise ValueError("this probe requires 768-D emotion2vec and content caches")
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
        selected = list(range(14)) + list(range(41, 46))
        if not query["channel_mask"][:, selected].all():
            raise ValueError("upper_l1 probe requires observed upper-face channels")
        stride = system.motion_teacher.stride
        energy, weight, target_state = fit_expression_energy_target(
            residual, query["valid"], query["channel_mask"], query["speaker_id"],
            query["emotion_id"], query["intensity_id"], train_ids, stride, mode=args.target)
        scale = fit_target_scale(energy, weight, train_ids)
        query, feature_stats = normalize_inputs(query, train_ids)
        query = prepare_dynamic_audio(query, args.audio_input)
    args.output.mkdir(parents=True, exist_ok=False)
    # System construction consumes RNG; reset before creating each matched arm.
    torch.manual_seed(args.seed)
    model = ContentGatedDynamicPredictor(hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim, output_dim=energy.shape[-1], mode=args.mode).to(device)
    initial_hash = state_hash(model.state_dict())
    lr = args.learning_rate if args.learning_rate is not None else float(cfg["training"].get("audio_lr", 5e-4))
    if lr <= 0:
        raise ValueError("learning rate must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    target = energy / scale
    losses = []
    model.train()
    for step in range(args.steps):
        sampled = torch.randint(len(train_cpu), (int(cfg["training"].get("batch_size", 8)),), generator=rng)
        ids = train_cpu[sampled].to(device)
        pred, _ = field_prediction(model, select(query, ids), stride, args.readout)
        loss = weighted_mse(pred, target[ids], weight[ids])
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
        pred, _ = field_prediction(model, query, stride, args.readout)
        heldout_query = select(query, heldout_ids)
        interventions = {}
        local_ids = torch.arange(len(heldout_ids), device=device)
        for mode in ("zero", "reverse", "shuffle"):
            altered, _ = field_prediction(model, intervene_content(heldout_query, mode), stride, args.readout)
            interventions[mode] = energy_metrics(altered, target[heldout_ids], weight[heldout_ids],
                heldout_query, local_ids, scale)
            interventions[mode]["prediction_change_mse"] = float(weighted_mse(
                altered * scale, pred[heldout_ids] * scale, weight[heldout_ids]))
        audio_interventions = {}
        for mode in ("zero", "reverse", "shuffle"):
            # Reuse the mask-safe temporal operation on the audio stream.
            altered_audio = intervene_content({**heldout_query, "content": heldout_query["audio"]}, mode)["content"]
            altered, _ = field_prediction(model, {**heldout_query, "audio": altered_audio}, stride, args.readout)
            audio_interventions[mode] = energy_metrics(altered, target[heldout_ids], weight[heldout_ids],
                heldout_query, local_ids, scale)
        frozen_after_hash = state_hash(system.state_dict())
        if frozen_hash != frozen_after_hash:
            raise RuntimeError("frozen main model state changed")
    summary = {"schema": "audio_content_gating_probe_v1", "seed": args.seed,
        "split_seed": split_seed, "readout": args.readout, "audio_input": args.audio_input,
        "mode": args.mode, "target": args.target, "steps": args.steps, "stride": stride,
        "hidden_dim": args.hidden_dim, "bottleneck_dim": args.bottleneck_dim,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "train": energy_metrics(pred, target, weight, query, train_ids, scale),
        "heldout": energy_metrics(pred, target, weight, query, heldout_ids, scale),
        "heldout_axes": axis_metrics(pred, target, weight, query, heldout_ids, scale),
        "heldout_content_interventions": interventions,
        "heldout_audio_interventions": audio_interventions,
        "readout_diagnostics": {s: readout_diagnostics(model, select(query, ids))
                                 for s, ids in (("train", train_ids), ("heldout", heldout_ids))},
        "last_loss": losses[-1] if losses else None,
        "initial_state_sha256": initial_hash, "final_state_sha256": state_hash(model.state_dict()),
        "main_model_unchanged": frozen_hash == frozen_after_hash,
        "source_sha256": {"script": sha(Path(__file__)), "module": sha(
            Path(__file__).resolve().parents[1] / "kinetalk_b0/gated_audio_dynamic.py")}}
    write_json(args.output / "summary.json", summary)
    serialized_stats = {key: {k: v.tolist() if torch.is_tensor(v) else v for k, v in value.items()}
                        for key, value in feature_stats.items()}
    write_json(args.output / "provenance.json", {
        "schema": summary["schema"], "checkpoint_sha256": sha(args.checkpoint),
        "train_cache_sha256": cache_hash, "main_model_sha256": frozen_hash,
        "feature_stats": serialized_stats, "train_indices": train_cpu.tolist(),
        "split_seed": split_seed, "readout": args.readout, "audio_input": args.audio_input,
        "heldout_indices": heldout_cpu.tolist(),
        "train_sentences": sorted(set(query["sentence_id"][int(i)] for i in train_cpu)),
        "heldout_sentences": sorted(set(query["sentence_id"][int(i)] for i in heldout_cpu)),
        "target_scale": scale.cpu().tolist(), "target_mode": args.target,
        "selection": "fixed final step; no heldout model selection"})
    effective = {**cfg, "probe": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (args.output / "effective_config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf8")
    torch.save({"prediction": pred.cpu(), "target": target.cpu(), "weight": weight.cpu(),
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "scale": scale.cpu(), "feature_stats": feature_stats}, args.output / "curves.pt")
    print(json.dumps({k: summary[k] for k in ("seed", "split_seed", "mode", "readout", "audio_input")}
        | {s: {k: summary[s][k] for k in ("r2_against_zero", "temporal_correlation")}
           for s in ("train", "heldout")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
