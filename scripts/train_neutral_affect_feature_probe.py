"""Frozen-teacher feature probe: acoustic features versus pretrained content.

This changes only the audio-student input; B0 always receives original content.
Statistics use training-query valid frames only. No generation is evaluated.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import LowRateAffectEncoder, NeutralAffectSystem
from kinetalk_b0.neutral_data import NeutralAffectDataset
from kinetalk_b0.utils import freeze_module
from scripts.train_neutral_affect_audio_ablation import state_hash, summarize
from scripts.train_neutral_affect_pilot import (
    affect_residual, cached_identity, device_batch, distill, observed, optimize,
    prepare, select, semantics, sha, write_json,
)


def _features(query, source):
    if source not in ("acoustic", "content"):
        raise ValueError("source must be acoustic or content")
    feature = query["audio" if source == "acoustic" else "content"]
    if feature.ndim != 3 or query["valid"].shape != feature.shape[:2]:
        raise ValueError("Features require [B,T,D] and matching valid mask")
    if not torch.isfinite(feature[query["valid"]]).all():
        raise ValueError("Observed feature values must be finite")
    return feature


def fit_feature_stats(query, source):
    """Fit on original training-query tensors, excluding invalid/padded frames."""
    feature = _features(query, source)
    values = feature[query["valid"]].detach()
    if len(values) < 2:
        raise ValueError("At least two training frames are needed for feature statistics")
    return {"source": source, "mean": values.mean(0).cpu(),
            "std": values.std(0).clamp_min(1e-3).cpu(), "count": len(values),
            "fitted_on": "training_query_valid_frames_only", "input_dim": feature.shape[-1]}


def prepare_feature_batch(query, source, stats):
    """Return a new query with normalized student ``audio``; preserve content.

    Apply once to an original cache batch, for both training and evaluation.
    Acoustic input means the cache's acoustic features (which may already have
    its original cache normalization); saved probe statistics are a second,
    explicitly recorded training-only transform. The input batch is not mutated.
    """
    feature = _features(query, source)
    if stats.get("source") != source or int(stats["input_dim"]) != feature.shape[-1]:
        raise ValueError("Feature statistics do not match the selected source/dimension")
    mean, std = [torch.as_tensor(stats[k], device=feature.device, dtype=feature.dtype) for k in ("mean", "std")]
    if mean.shape != (feature.shape[-1],) or std.shape != mean.shape:
        raise ValueError("Invalid feature mean/std shape")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Feature statistics must be finite with positive std")
    return {**query, "audio": torch.where(query["valid"].unsqueeze(-1), (feature - mean) / std, 0)}


def build_feature_system(cfg, checkpoint, source, seed, input_dim=None):
    """Strictly validate all teacher weights before replacing only audio."""
    if source not in ("acoustic", "content"):
        raise ValueError("source must be acoustic or content")
    if input_dim is not None and (source != "acoustic" or isinstance(input_dim, bool)
                                  or not isinstance(input_dim, int) or input_dim < 1):
        raise ValueError("input_dim must be a positive integer override for acoustic source only")
    original = checkpoint.get("provenance", {}).get("config", {})
    if checkpoint.get("stage") != "motion_teacher" or any(original.get(k) != cfg.get(k) for k in ("data", "model")):
        raise ValueError("Matching model/data config and motion_teacher checkpoint are required")
    system = NeutralAffectSystem(cfg)
    system.load_state_dict(checkpoint["model"], strict=True)
    feature_cfg = copy.deepcopy(cfg)
    dimension = int(input_dim if input_dim is not None else
                    (cfg["data"]["audio_dim"] if source == "acoustic" else cfg["data"]["content_dim"]))
    feature_cfg["data"]["audio_emotion_dim"] = dimension
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        common = LowRateAffectEncoder(feature_cfg, int(cfg["data"]["audio_dim"]), motion=False)
        torch.manual_seed(seed)
        system.audio_encoder = LowRateAffectEncoder(feature_cfg, dimension, motion=False)
    # Match every equal-shaped initial tensor across arms; only the different
    # input projection shape necessarily prevents exact initial-state equality.
    initial = system.audio_encoder.state_dict()
    initial.update({k: v for k, v in common.state_dict().items() if v.shape == initial[k].shape})
    system.audio_encoder.load_state_dict(initial, strict=True)
    for name in ("emotion_classifier", "intensity_classifier"):
        getattr(system.audio_encoder, name).load_state_dict(getattr(system.motion_teacher, name).state_dict(), strict=True)
    freeze_module(system)
    system.audio_encoder.requires_grad_(True)
    return system.eval(), feature_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "teacher-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--audio-source", choices=("acoustic", "content"), required=True)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()
    if args.steps < 0:
        raise ValueError("steps must be nonnegative")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(cfg.get("device", "cuda"))
    data = {name: NeutralAffectDataset(args.data / (name + ".pt")) for name in ("train", "heldout")}
    input_dim = int(data["train"].queries[0]["audio"].shape[-1])
    for name, dataset in data.items():
        if any(q["audio"].ndim != 2 or q["audio"].shape[-1] != input_dim for q in dataset.queries):
            raise ValueError(f"{name} cached audio dimensions must match train audio dimension {input_dim}")
    system, feature_cfg = build_feature_system(cfg, checkpoint, args.audio_source, args.seed,
                                                input_dim=input_dim if args.audio_source == "acoustic" else None)
    system.to(device)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "effective_config.yaml").write_text(yaml.safe_dump(feature_cfg, sort_keys=False), encoding="utf8")
    frozen = lambda: {k: v for k, v in system.state_dict().items() if not k.startswith("audio_encoder.")}
    frozen_hash = state_hash(frozen())
    queries = {name: prepare(system, dataset, device)[0] for name, dataset in data.items()}
    stats = fit_feature_stats(queries["train"], args.audio_source)
    queries = {name: prepare_feature_batch(q, args.audio_source, stats) for name, q in queries.items()}
    refs = data["train"].identity_references
    if sorted(refs) != list(range(len(refs))) or len({len(v) for v in refs.values()}) != 1:
        raise ValueError("Enrollment requires contiguous IDs and equal reference counts")
    reference = device_batch([r for speaker in sorted(refs) for r in refs[speaker]], device)
    with torch.no_grad():
        residual = torch.where(observed(reference), reference["motion"] - system.base(reference["content"], reference["valid"])["b0"], 0)
        identities = cached_identity(system, {"residual": residual.reshape(len(refs), len(refs[0]), *residual.shape[1:]),
                                            "valid": reference["valid"].reshape(len(refs), len(refs[0]), -1)})
        targets = {name: system.encode_motion(affect_residual(q, select(identities, q["speaker_id"])), q["valid"]) for name, q in queries.items()}
        teacher = targets["train"]
        scales = {"global": teacher["global"].std(0).clamp_min(0.05),
                  "controls": teacher["controls"][teacher["control_mask"]].std(0).clamp_min(0.05)}
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), root / "scripts/train_neutral_affect_audio_ablation.py", root / "scripts/train_neutral_affect_pilot.py",
               *[root / "kinetalk_b0" / p for p in ("neutral_data.py", "utils.py", "models/neutral_affect.py", "models/dit.py", "models/model.py", "models/encoders.py")]]
    provenance = {"schema": "neutral_affect_feature_probe_v1", "audio_source": args.audio_source,
                  "seed": args.seed, "steps": args.steps, "config": feature_cfg, "original_config_sha256": sha(args.config),
                  "teacher_checkpoint": str(args.teacher_checkpoint.resolve()), "teacher_sha256": sha(args.teacher_checkpoint),
                  "train_sha256": sha(args.data / "train.pt"), "heldout_sha256": sha(args.data / "heldout.pt"),
                  "source_sha256": {str(p.relative_to(root)): sha(p) for p in sources},
                  "initial_state_sha256": state_hash(system.state_dict()), "initial_audio_state_sha256": state_hash(system.audio_encoder.state_dict()),
                  "initial_frozen_state_sha256": frozen_hash, "scales_sha256": state_hash(scales),
                  "feature_stats": {k: v.tolist() if torch.is_tensor(v) else v for k, v in stats.items()},
                  "cached_audio_dim": input_dim, "student_input_dim": feature_cfg["data"]["audio_emotion_dim"],
                  "normalization": "Preserve cache preprocessing and audio_stats; fit an additional mean/std transform on training-query valid frames only, then reuse it on heldout.",
                  "heads": "Both classifiers from teacher; fresh seed shared across arms; equal-shaped audio tensors matched",
                  "scope": "Same enrolled people, heldout sentences; content is pretrained speech content, not a claimed emotion embedding.",
                  "torch": torch.__version__}
    write_json(args.output / "provenance.json", provenance)
    report = {"before": {name: summarize(system, q, targets[name], scales) for name, q in queries.items()}}
    params = list(system.audio_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(cfg["training"].get("audio_lr", 0.0005)), weight_decay=1e-5)
    rng, batch_hash = torch.Generator(device="cpu").manual_seed(args.seed), hashlib.sha256()
    system.audio_encoder.train()
    for step in range(args.steps):
        cpu_ids = torch.randint(len(data["train"]), (int(cfg["training"].get("batch_size", 8)),), generator=rng)
        batch_hash.update(cpu_ids.numpy().tobytes())
        ids = cpu_ids.to(device)
        q, target = select(queries["train"], ids), select(targets["train"], ids)
        audio = system.encode_audio(q["audio"], q["valid"])
        latent, semantic = distill(audio, target, scales), semantics(audio, q)
        loss = latent + .1 * semantic
        norm = optimize(loss, optimizer, params)
        if step % 100 == 0 or step + 1 == args.steps:
            if state_hash(frozen()) != frozen_hash or any(p.grad is not None for n, p in system.named_parameters() if not n.startswith("audio_encoder.")):
                raise RuntimeError("Frozen model changed or accumulated gradients")
            record = {"time": time.time(), "step": step + 1, "loss": float(loss), "distill": float(latent), "semantic": float(semantic), "grad_norm": norm}
            with (args.output / "training.jsonl").open("a", encoding="utf8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record, allow_nan=False), flush=True)
    if state_hash(frozen()) != frozen_hash:
        raise RuntimeError("Frozen model state changed")
    report.update(after={name: summarize(system, q, targets[name], scales) for name, q in queries.items()},
                  minibatch_sha256=batch_hash.hexdigest(), frozen_state_unchanged=True)
    write_json(args.output / "summary.json", report)
    torch.save({"model": system.state_dict(), "config": feature_cfg, "provenance": provenance, "stage": "audio", "steps": args.steps,
                "audio_source": args.audio_source, "feature_stats": stats, "audio_stats": data["train"].cache["audio_stats"],
                "scales": scales, "optimizer": optimizer.state_dict(), "minibatch_sha256": batch_hash.hexdigest()}, args.output / "audio.pt")


if __name__ == "__main__":
    main()
